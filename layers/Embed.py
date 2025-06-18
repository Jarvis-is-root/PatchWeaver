import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0) # torch.Size([1, 5000, 512])
        self.register_buffer('pe', pe)

    def forward(self, x): # x.shape [32, 96, 7]
        return self.pe[:, :x.size(1)] # torch.Size([1, 96, 512])


class PositionalEmbedding_4D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding_4D, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 5000, 512])
        self.register_buffer('pe', pe)

    def forward(self, x): # x.shape [32, 7, 12, 16]
        return self.pe[:, :, :x.size(2)] # torch.Size([1, 1, 12, 512])



class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x # torch.Size([32, 96, 512])


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time] torch.Size([32, 7, 96])
        # x_mark.shape: torch.Size([32, 96, 4])
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_proj_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_proj_inverted, self).__init__()
        self.value_embedding = nn.Sequential(
            nn.Linear(c_in, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # print(x.shape, x_mark.shape)
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x torch.Size([32, 7, 96])
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x) # torch.Size([32, 7, 104]), padding前0后8
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # self.patch_len= 16, self.stride=8 -> x torch.Size([32, 7, 12, 16])
        # unfold用于实现滑动窗口，104以8为步长滑动，窗口大小16，一共有12个窗口
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # torch.Size([224, 12, 16])
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) # torch.Size([224, 12, 512])
        return self.dropout(x), n_vars


class DataEmbedding_Patch(nn.Module):
    # 参考了PatchTST之后的embedding
    def __init__(self, d_model, patch_len, stride, padding, dropout=0.1, 
            embed_type='fixed', freq='h', embedding=None):
        super(DataEmbedding_Patch, self).__init__()
        self.patch_len = patch_len # 以pathc_len为12举例
        self.stride = stride # stride == patch_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding)) # 表示在1维序列左侧填充0个，右侧填充padding个，补齐的内容是边界值的复制。这是用于避免序列长度无法被patch_len整除

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # self.value_embedding = nn.Sequential(
        #     nn.Linear(patch_len, d_model, bias=False),
        #     nn.ReLU()
        # )
        # self.value_embedding = nn.Sequential(
        #     nn.Linear(patch_len, d_model*2),
        #     nn.GELU(),
        #     nn.Linear(d_model*2, d_model)
        # )
        # Positional embedding
        self.position_embedding = PositionalEmbedding_4D(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x torch.Size([32, 7, 96]) B, C, T
        # do patching
        # import pdb; pdb.set_trace()
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x) # torch.Size([32, 7, 108]), padding前0后12
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # B, C, Patch_num, Patch_len
        # unfold用于实现滑动窗口，以size为窗口大小，stride为步长滑动。必须能整除
        x = self.value_embedding(x) + self.position_embedding(x) 
        # x = self.value_embedding(x)
        return self.dropout(x), n_vars


class DataActEmbedding_Patch(nn.Module):
    # 参考了PatchTST之后的embedding
    def __init__(self, d_model, patch_len, stride, padding, dropout=0.1, 
            embed_type='fixed', freq='h', embedding=None):
        super(DataActEmbedding_Patch, self).__init__()
        self.patch_len = patch_len # 以pathc_len为12举例
        self.stride = stride # stride == patch_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding)) # 表示在1维序列左侧填充0个，右侧填充padding个，补齐的内容是边界值的复制。这是用于避免序列长度无法被patch_len整除

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        # self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.value_embedding = nn.Sequential(
            nn.Linear(patch_len, d_model, bias=False),
            nn.ReLU()
        )
        # self.value_embedding = nn.Sequential(
        #     nn.Linear(patch_len, d_model*2),
        #     nn.GELU(),
        #     nn.Linear(d_model*2, d_model)
        # )
        # Positional embedding
        self.position_embedding = PositionalEmbedding_4D(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x torch.Size([32, 7, 96]) B, C, T
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x) # torch.Size([32, 7, 108]), padding前0后12
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # B, C, Patch_num, Patch_len
        # unfold用于实现滑动窗口，以size为窗口大小，stride为步长滑动。必须能整除
        x = self.value_embedding(x) + self.position_embedding(x) 
        # x = self.value_embedding(x)
        return self.dropout(x), n_vars

class DataEmbedding_Patch2(nn.Module):
    # 参考了PatchTST之后的embedding
    def __init__(self, d_model, patch_len, stride, padding, dropout=0.1, 
            embed_type='fixed', freq='h', embedding=None):
        super(DataEmbedding_Patch2, self).__init__()
        self.embedding = embedding
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        # self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding_4D(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x torch.Size([32, 7, 96])
        # do patching
        n_vars = x.shape[1]
        # x = self.padding_patch_layer(x) # torch.Size([32, 7, 104]), padding前0后8
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # self.patch_len= 16, self.stride=8 -> x torch.Size([32, 7, 12, 16])
        # unfold用于实现滑动窗口，104以8为步长滑动，窗口大小16，一共有12个窗口

        x = x.reshape((x.shape[0], n_vars, -1, self.patch_len))
        # import pdb; pdb.set_trace()
        x = self.value_embedding(x) + self.position_embedding(x) # torch.Size([224, 12, 512])
        return self.dropout(x), n_vars


# class DataEmbedding_Patch(nn.Module):
#     def __init__(self, D, embed_type='fixed', freq='h', dropout=0.1, embedding=None):
#         super(DataEmbedding_Patch, self).__init__()
#         self.embedding = embedding
#         self.position_embedding = PositionalEmbedding(d_model=D)
#         self.temporal_embedding = TemporalEmbedding(d_model=D, embed_type=embed_type,
#                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
#             d_model=D, embed_type=embed_type, freq=freq)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x, x_mark):
#         if self.embedding == 'temporal':
#             assert x_mark is not None
#             x += self.temporal_embedding(x_mark)
#         elif self.embedding == 'pos':
#             x += self.position_embedding(x)
#         elif self.embedding == 'temporal_pos':
#             assert x_mark is not None
#             x = x + self.temporal_embedding(x_mark) + self.position_embedding(x)
#         elif self.embedding == None:
#             return self.dropout(x)
#         else:
#             raise "Invalid embedding type!"
#         return self.dropout(x)




class DataEmbedding_Patch_TemporalEmbedding(nn.Module):
    # 参考了PatchTST之后的embedding
    def __init__(self, d_model, patch_len, stride, padding, dropout=0.1, 
            embed_type='fixed', freq='h', embedding=None):
        super(DataEmbedding_Patch_TemporalEmbedding, self).__init__()
        self.patch_len = patch_len # 以patch_len为12举例
        self.stride = stride # stride == patch_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Positional embedding
        self.position_embedding = PositionalEmbedding_4D(d_model)
        # 新增：可学习时间戳编码
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type='fixed', freq=freq)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        # x: [B, C, T], x_mark: [B, 4, T]
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x) # torch.Size([B, C, T_pad])
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B, C, num_patches, patch_len]
        x = self.value_embedding(x) # [B, C, num_patches, d_model]
        x = x + self.position_embedding(x) # [B, C, num_patches, d_model]

        if x_mark is not None:
            # 处理x_mark
            x_mark = self.padding_patch_layer(x_mark) # [B, 4, T_pad]
            x_mark = x_mark.permute(0, 2, 1) # [B, T_pad, 4]
            # patch化
            x_mark = x_mark.unfold(dimension=1, size=self.patch_len, step=self.stride) # [B, num_patches, patch_len, 4]
            # 合并patch内时间特征
            B, num_patches, patch_len, num_feats = x_mark.shape
            x_mark = x_mark.reshape(B * num_patches, patch_len, num_feats) # [B*num_patches, patch_len, 4]
            # 时间特征编码
            t_embed = self.temporal_embedding(x_mark) # [B*num_patches, patch_len, d_model]
            # 聚合patch内时间特征（均值池化）
            t_embed = t_embed.mean(dim=1) # [B*num_patches, d_model]
            t_embed = t_embed.view(B, 1, num_patches, -1) # [B, 1, num_patches, d_model]
            x = x + t_embed # 广播加法

        return self.dropout(x), n_vars


class DataEmbedding_Patch_Proj(nn.Module):
    # 参考了PatchTST之后的embedding
    def __init__(self, d_model, patch_len, stride, padding, enc_in=7, dropout=0.1, 
            embed_type='fixed', freq='h', embedding=None):
        super(DataEmbedding_Patch_Proj, self).__init__()
        self.patch_len = patch_len # 以patch_len为12举例
        self.stride = stride # stride == patch_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Positional embedding
        self.position_embedding = PositionalEmbedding_4D(d_model)
        # 新增：可学习时间戳编码
        self.temporal_embedding = nn.Linear(4, enc_in)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        # x: [B, C, T], x_mark: [B, 4, T]
        n_vars = x.shape[1]
        if x_mark is not None:
            # print('OhYeah!')
            x = x + self.temporal_embedding(x_mark).transpose(1, 2)

        x = self.padding_patch_layer(x) # torch.Size([B, C, T_pad])
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B, C, num_patches, patch_len]
        x = self.value_embedding(x) # [B, C, num_patches, d_model]
        x = x + self.position_embedding(x) # [B, C, num_patches, d_model]

        return self.dropout(x), n_vars
