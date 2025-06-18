import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer, EncoderLayerWithBatchNorm
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_Patch
import numpy as np
torch.autograd.set_detect_anomaly(True)


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)        

        self.conv1 = nn.Conv1d(in_channels=nf, out_channels=2*nf, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2*nf, out_channels=nf, kernel_size=1)
        self.norm1 = nn.LayerNorm(nf)
        self.norm2 = nn.LayerNorm(nf)
        self.dropout = nn.Dropout(head_dropout)
        self.activation = F.gelu
        self.proj = nn.Linear(nf, target_window)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        # B, C, embedding_dim, N
        x = self.flatten(x) # B, C, embedding_dim * N
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.proj(self.norm2(x + y))


class OnlyAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super(OnlyAttention, self).__init__()
        assert input_dim % num_heads == 0, "input_dim必须能被num_heads整除"
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.out_projection = nn.Linear(output_dim, output_dim)
        self.scale_factor = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, input_tensor):
        batch_size, time_steps, input_dim = input_tensor.shape
        queries = self.query_projection(input_tensor).view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1,2)  # (B, heads, T, head_dim)
        keys = self.key_projection(input_tensor).view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1,2)
        values = self.value_projection(input_tensor).view(batch_size, time_steps, self.num_heads, -1).transpose(1,2)  # (B, heads, T, v_dim)
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2)) / self.scale_factor  # (B, heads, T, T)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        weighted_output = torch.matmul(attention_weights, values)  # (B, heads, T, v_dim)
        weighted_output = weighted_output.transpose(1,2).contiguous().view(batch_size, time_steps, -1)
        return weighted_output


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.padding = configs.stride
        # Embedding
        self.patch_embedding = DataEmbedding_Patch(configs.d_model, 
                configs.L, configs.stride, self.padding, configs.dropout
                )
        self.L = configs.L # 需要传L参数
        self.num_patches = int((self.seq_len + self.padding - self.L) / configs.stride + 1)

        self.act = nn.GELU() if configs.activation == 'gelu' else nn.ReLU()
        self.z_n = nn.ModuleList([
            nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            self.act
            ) for _ in range(self.num_patches)
        ])
        self.h_t_hat = nn.Linear(configs.d_model, configs.d_model)

        self.encoderlayer = EncoderLayerWithBatchNorm if configs.normalization == 'batch' else EncoderLayer
        self.normalization = nn.LayerNorm(configs.d_model) if configs.normalization == 'layer' else nn.BatchNorm1d(configs.d_model)
        
        # PatchEncoder
        self.enc1 = Encoder(
                    [
                        self.encoderlayer(
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for l in range(configs.e_layers)
                    ],
                    norm_layer=self.normalization
                # )
            )

        self.enc2 = OnlyAttention(configs.d_model, configs.d_model)
        self.dec = nn.Linear(configs.d_model, configs.L)

        # Decoder
        self.head_nf = configs.d_model * \
                       int((self.seq_len - self.L) / configs.stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

                                    
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        seq_mean = torch.mean(x_enc, dim=1).unsqueeze(1)
        x_enc = (x_enc - seq_mean).permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x_enc)

        h_t = torch.zeros_like(x)
        h = 1.0
        for i in range(h_t.shape[2]):
            x_n_enc, attn = self.enc1(x[:, :, i, :])
            z_n = self.z_n[i](x[:, :, i, :])
            tmp = z_n * h + (1 - z_n) * x_n_enc
            h_t[:, :, i, :] = tmp
            h = tmp
        enc_out = h_t

        enc_out = self.enc2(enc_out.reshape((-1, enc_out.shape[-2], enc_out.shape[-1])))
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1) + seq_mean

        return dec_out



    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None




