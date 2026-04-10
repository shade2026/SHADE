import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: B, L, d
        # padding on the both ends of time series
        # 保持输出结果形状不变
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x  # B, L, d


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # x: B, L, d
        moving_mean = self.moving_avg(x)  # 趋势项
        res = x - moving_mean  # 周期项
        return res, moving_mean  # B, L, d


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


# # 只取周期项
# class EncoderLayer(nn.Module):
#     """
#     Autoformer encoder layer with the progressive decomposition architecture
#     """
#
#     def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, attn_mask=None):
#         # x: B, seq_len, d_model
#         # AutoCorrelation
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )  # B, seq_len, d_model
#         x = x + self.dropout(new_x)  # B, seq_len, d_model
#         # SeriesDecomposition
#         # 只取周期项
#         x, _ = self.decomp1(x)  # B, seq_len, d_model
#         # FeedForward
#         y = x  # B, seq_len, d_model
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # B, d_model, seq_len
#         y = self.dropout(self.conv2(y).transpose(-1, 1))  # B, seq_len, d_model
#         # SeriesDecomposition
#         # 只取周期项
#         res, _ = self.decomp2(x + y)  # B, seq_len, d_model
#         # res: B, seq_len, d_model
#         # attn: B, H, D, seq_len
#         return res, attn
#
#
# class Encoder(nn.Module):
#     """
#     Autoformer encoder
#     """
#
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer
#
#     def forward(self, x, attn_mask=None):
#         # x: B, L*S, D
#         attns = []
#         if self.conv_layers is not None:
#             for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
#                 x, attn = attn_layer(x, attn_mask=attn_mask)
#                 x = conv_layer(x)
#                 attns.append(attn)
#             x, attn = self.attn_layers[-1](x)
#             attns.append(attn)
#         else:
#             for attn_layer in self.attn_layers:
#                 x, attn = attn_layer(x, attn_mask=attn_mask)  # B, L*S, D
#                 attns.append(attn)
#
#         if self.norm is not None:
#             x = self.norm(x)
#
#         # x: B, L*S, D
#         return x, attns

# 取周期项和趋势项
class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x: B, seq_len, d_model
        # AutoCorrelation
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )  # B, seq_len, d_model
        x = x + self.dropout(new_x)  # B, seq_len, d_model
        # SeriesDecomposition
        # 只取周期项
        x, trend1 = self.decomp1(x)  # B, seq_len, d_model
        # FeedForward
        y = x  # B, seq_len, d_model
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # B, d_model, seq_len
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # B, seq_len, d_model
        # SeriesDecomposition
        # 只取周期项
        res, trend2 = self.decomp2(x + y)  # B, seq_len, d_model
        # res: B, seq_len, d_model
        # attn: B, H, D, seq_len
        residual_trend = trend1 + trend2  # B, seq_len, d_model
        return res, residual_trend, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x: B, L*S, D
        attns = []
        trend = 0.0
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, residual_trend, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                trend += residual_trend
                attns.append(attn)
            x, residual_trend, attn = self.attn_layers[-1](x)
            trend += residual_trend
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, residual_trend, attn = attn_layer(x, attn_mask=attn_mask)  # B, L*S, D
                trend += residual_trend  # B, L*S, D
                attns.append(attn)

        x = x + trend  # B, L*S, D

        if self.norm is not None:
            x = self.norm(x)

        # x: B, L*S, D
        return x, attns
