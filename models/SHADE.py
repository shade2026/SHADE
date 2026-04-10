import numpy as np
import torch
import torch.nn as nn
from layers.Ada_HGCN import AdaptiveHypergraphAttention
from layers.RevIN import RevIN
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.embed_with_space import DataEmbedding


class Intra_Patch_Attention(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(Intra_Patch_Attention, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x, queries):
        x = x.permute(0, 2, 1, 4, 3).contiguous()
        B, S, patch_nums, patch_size, D = x.shape
        H = self.n_heads

        queries = queries.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        queries = self.query_projection(queries).view(B * S * patch_nums, 1, H, -1)
        keys = self.key_projection(x).view(B * S * patch_nums, patch_size, H, -1)
        values = self.value_projection(x).view(B * S * patch_nums, patch_size, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=None
        )
        out = self.out_projection(out.view(B, S, patch_nums, 1, -1))
        out = out.squeeze(-2)

        return out, attn


class Inter_Patch_Attention(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(Inter_Patch_Attention, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x):
        x = x.permute(0, 2, 1, 4, 3).contiguous()
        B, S, patch_nums, patch_size, D = x.shape
        H = self.n_heads
        x = x.reshape(B * S, patch_nums, -1)
        queries = self.query_projection(x).view(B * S, patch_nums, H, -1)
        keys = self.key_projection(x).view(B * S, patch_nums, H, -1)
        values = self.value_projection(x).view(B * S, patch_nums, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=None
        )
        out = self.out_projection(out.view(B, S, patch_nums, -1))

        return out, attn


class FuseLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, num_nodes, patch_nums, patch_size, hyperedge_num, k_hyperedge):
        super(FuseLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.num_nodes = num_nodes
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.stride = patch_size
        self.hyperedge_num = hyperedge_num
        self.k_hyperedge = k_hyperedge

        # adaptive hyper gcn
        self.spatial_attention = AdaptiveHypergraphAttention(self.num_nodes, self.hyperedge_num, self.d_model,
                                                             self.k_hyperedge, dropout=dropout)

        # intra_patch_attention
        self.intra_patch_attention = Intra_Patch_Attention(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            self.d_model,
            self.n_heads)
        self.queries = nn.Parameter(torch.rand(self.num_nodes, self.patch_nums, 1, self.d_model),
                                    requires_grad=True)
        self.intra_linear = nn.Linear(self.patch_nums, self.patch_nums * self.patch_size)

        # inter_patch_attention
        self.inter_d_model = self.d_model * self.patch_size
        self.inter_patch_attention = Inter_Patch_Attention(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            self.inter_d_model,
            self.n_heads)

        # FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.linear1 = nn.Linear(self.d_model, self.d_ff, bias=True)
        self.linear2 = nn.Linear(self.d_ff, self.d_model, bias=True)

    def forward(self, x):
        # adaptive hyper gat
        spatial_out, constraint_loss = self.spatial_attention(x)
        x = spatial_out + x
        new_x = x

        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        B, patch_nums, S, D, patch_size = x.shape
        # intra attention
        intra_out, _ = self.intra_patch_attention(x, self.queries)
        intra_out = intra_out.permute(0, 1, 3, 2)
        intra_out = self.intra_linear(intra_out).permute(0, 3, 1, 2)
        # inter attention
        inter_out, _ = self.inter_patch_attention(x)
        inter_out = inter_out.reshape(B, S, patch_nums * patch_size, D).permute(0, 2, 1, 3)

        out = new_x + intra_out + inter_out

        # ffn
        y = x = self.norm1(self.dropout(out))
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y), constraint_loss


class AMS(nn.Module):
    def __init__(self, configs, win_size, num_experts, num_nodes=32, d_model=32, d_ff=64, patch_size=[8, 6, 4, 2],
                 noisy_gating=True, k=4, residual_connection=1, loss_coef=1e-2, loss_coef1=1e-2):
        super(AMS, self).__init__()
        self.num_experts = num_experts
        self.win_size = win_size
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.hyperedge_num = configs.hyperedge_num
        self.k_hyperedge = configs.k_hyperedge
        self.k = k

        self.experts = nn.ModuleList()
        for patch in patch_size:
            patch_nums = int(win_size / patch)
            self.experts.append(
                FuseLayer(d_model=d_model, d_ff=d_ff, n_heads=self.n_heads,
                          dropout=self.dropout, num_nodes=num_nodes,
                          patch_nums=patch_nums, patch_size=patch, hyperedge_num=self.hyperedge_num,
                          k_hyperedge=self.k_hyperedge))

        self.w_gate = nn.Parameter(torch.zeros(d_model, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d_model, num_experts), requires_grad=True)
        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()
        assert (self.k <= self.num_experts)

        self.residual_connection = residual_connection
        self.loss_coef = loss_coef
        self.loss_coef1 = loss_coef1

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def noisy_top_k_gating(self, x, noise_epsilon=1e-5):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        kth_largest_val, _ = torch.kthvalue(logits, self.num_experts - self.k + 1, dim=-1)
        kth_largest_mat = kth_largest_val.unsqueeze(-1).repeat(1, 1, 1, self.num_experts)
        logits = logits.masked_fill(logits < kth_largest_mat, -np.inf)
        gates = torch.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        # multi-scale router
        gates = self.noisy_top_k_gating(x)
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) * self.loss_coef

        # Ada_HGCN
        expert_outputs = []
        constrain_loss = 0
        for i in range(self.num_experts):
            expert_output, loss = self.experts[i](x)
            expert_outputs.append(expert_output)
            constrain_loss += loss
        constrain_loss = constrain_loss * self.loss_coef1

        expert_outputs = torch.stack(expert_outputs, dim=-1)
        gates = gates.unsqueeze(-2)
        output = expert_outputs * gates
        output = torch.sum(output, dim=-1)

        if self.residual_connection:
            output = output + x

        return output, balance_loss + constrain_loss


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.layer_nums = configs.e_layers
        self.num_nodes = configs.num_nodes
        self.task_name = configs.task_name
        self.win_size = configs.seq_len
        self.moving_avg = configs.moving_avg
        self.top_k = configs.top_k
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.c_out = configs.c_out
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.loss_coef = configs.loss_coef
        self.loss_coef1 = configs.loss_coef1

        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.mask_projection = nn.Linear(1, self.d_model)

        self.AMS_lists = nn.ModuleList()
        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.configs, self.win_size, self.num_experts_list[num], k=self.top_k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, residual_connection=self.residual_connection,
                    loss_coef=self.loss_coef, loss_coef1=self.loss_coef1))

        self.out_projection = nn.Linear(self.d_model, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_mark_space_enc, mask=None):
        B, L, S, _ = x_enc.shape
        balance_loss = 0
        # norm
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')

        out = self.enc_embedding(x_enc, x_mark_enc, x_mark_space_enc)

        mask = mask.float()
        mask = self.mask_projection(mask)
        out = out + mask

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss
        out = self.out_projection(out)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out, balance_loss
