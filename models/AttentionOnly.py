import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.Embed import DataEmbedding_inverted, DataEmbedding_Patch, DataEmbedding_proj_inverted
import numpy as np
from layers.Autoformer_EncDec import series_decomp
torch.autograd.set_detect_anomaly(True)


class OnlyAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OnlyAttention, self).__init__()
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.scale_factor = torch.sqrt(torch.tensor(input_dim))


    def forward(self, input_tensor):
        batch_size, time_steps, input_dim = input_tensor.shape
        queries = self.query_projection(input_tensor)
        keys = self.key_projection(input_tensor)
        values = self.value_projection(input_tensor)
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2)) / self.scale_factor
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        weighted_output = torch.matmul(attention_weights, values)
        return weighted_output
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        input_len=configs.seq_len # 720
        patch_length=configs.L # 144
        embedding_dim=configs.d_model # 64
        ffn_dim=configs.d_ff # 64
        n_encoder=configs.e_layers # 1
        n_decoder=configs.e2_layers # 1
        n_heads=configs.n_heads # 8
        pred_len=configs.pred_len # 720
        drop_out=configs.dropout # 0.1
        flatten_dropout=configs.flatten_dropout # 0.1
        
        self.input_len = input_len
        self.patch_length = patch_length
        self.n_patches = input_len // patch_length
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_length, embedding_dim),
            nn.ReLU()
        )
        
        # First Transformer with CLS token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False),
                        embedding_dim, n_heads # d_model, nums heads
                    ),
                    embedding_dim, # ffn的输入维度
                    d_ff=ffn_dim, # ffn的中间层维度
                    dropout=drop_out,
                    activation='gelu'
                ) for _ in range(n_encoder)
            ],
            norm_layer=torch.nn.LayerNorm(embedding_dim)
        )
        
        # Second Transformer
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False),
                        embedding_dim, n_heads # d_model, nums heads
                    ),
                    embedding_dim, # ffn的输入维度
                    d_ff=ffn_dim, # ffn的中间层维度
                    dropout=drop_out,
                    activation='gelu'
                ) for _ in range(n_decoder)
            ],
            norm_layer=torch.nn.LayerNorm(embedding_dim)
        )

        self.enc2 = OnlyAttention(self.embedding_dim, self.embedding_dim)
        
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.flatten_linear = nn.Linear(self.embedding_dim * self.n_patches, self.pred_len)
        self.flatten_dropout = nn.Dropout(flatten_dropout)

    

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x.shape = [B, input_len, C]
        B, T, C = x.shape
        # 1. Patch division and reshape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        n_patches = T // self.patch_length
        start_idx = T - (n_patches * self.patch_length)
        x = x[:, start_idx:, :].reshape(B, n_patches, self.patch_length, C)
        
        # 2. Patch embedding and dimension adjustment
        x = x.permute(0, 1, 3, 2)  # [B, n_patches, C, patch_length]
        x = self.patch_embed(x)  # [B, n_patches, C, embedding_dim]
        x = x.reshape(B * n_patches, C, self.embedding_dim)  # [(B*n_patches), C, embedding_dim]
        
        # 3. Add CLS token and apply first Transformer
        # cls_tokens = self.cls_token.expand(B * n_patches, -1, -1)  # [(B*n), 1, embedding_dim]
        # x = torch.cat((cls_tokens, x), dim=1)  # [(B*n), C+1, embedding_dim]
        x, _ = self.encoder(x)

        # x = x.reshape(B, self.n_patches, C + 1, self.embedding_dim).transpose(2, 1).reshape(B * (C + 1), self.n_patches, self.embedding_dim)
        x = x.reshape(B, self.n_patches, C, self.embedding_dim).transpose(2, 1).reshape(B * C, self.n_patches, self.embedding_dim)
        # x = self.enc2(x)
        x, _ = self.decoder(x)

        # x = x.reshape(B, C + 1, self.embedding_dim * self.n_patches)[:, 1:, :]
        x = x.reshape(B, C, self.embedding_dim * self.n_patches)
        x = self.flatten_linear(x) # [B, C, pred_len]
        x = self.flatten_dropout(x)
        outputs = x.transpose(1, 2) + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return outputs
        