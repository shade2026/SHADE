import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax


class Adaptive_Hypergraph(nn.Module):
    def __init__(self, num_nodes, hyperedge_num, d_model, k_hyperedge=3, alpha=1, beta=0.5, dropout=0.1):
        super(Adaptive_Hypergraph, self).__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.hyperedge_num = hyperedge_num
        self.alpha = alpha
        self.beta = beta
        self.eta = k_hyperedge
        self.embed_hyper = nn.Embedding(self.hyperedge_num, self.d_model)
        self.embed_node = nn.Embedding(self.num_nodes, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: B*L, S, D
        hyperedge_idx = torch.arange(self.hyperedge_num).to(x.device)
        node_idx = torch.arange(self.num_nodes).to(x.device)
        hyperedge_embed = self.embed_hyper(hyperedge_idx)
        node_embed = self.embed_node(node_idx)

        a = torch.mm(node_embed, hyperedge_embed.transpose(1, 0))
        adj = F.softmax(F.relu(self.alpha * a))
        mask = torch.zeros_like(a).to(x.device)
        top_indices = adj.topk(min(adj.size(1), self.eta), 1).indices
        mask.scatter_(1, top_indices, 1)
        adj = adj * mask
        adj = torch.where(adj > self.beta, 1, 0)
        adj = adj[:, (adj != 0).any(dim=0)]
        matrix_array = torch.tensor(adj, dtype=torch.int)
        nonzero_list = torch.nonzero(matrix_array.T)
        hyperedge_list = nonzero_list[:, 0]
        node_list = nonzero_list[:, 1]
        hypergraph = torch.vstack([node_list, hyperedge_list])
        return hypergraph


class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, gamma=0.5, negative_slope=0.02, dropout=0.1,
                 bias=True):
        super(HypergraphConv, self).__init__(aggr='add', node_dim=-3)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gamma = gamma
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.a = nn.Parameter(torch.Tensor(2 * out_channels, 1))
        self.b = nn.Parameter(torch.Tensor(2 * out_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        # glorot(self.att)
        glorot(self.a)
        glorot(self.b)
        zeros(self.bias)

    def message(self, x_j, edge_index_j, norm, H):
        out = norm[edge_index_j].view(-1, 1, 1) * x_j
        if H is not None:
            out = H * out
        return out

    def forward(self, x, hyperedge_index):
        # x: B*L, S, D

        x = torch.matmul(x, self.weight) + self.bias
        x = x.transpose(0, 1)

        D = degree(hyperedge_index[0], x.size(0), x.dtype).pow(-1)
        D[D == float("inf")] = 0
        B = degree(hyperedge_index[1], hyperedge_index[1].max().item() + 1, x.dtype).pow(-1)
        B[B == float("inf")] = 0

        v_features = torch.index_select(x, dim=0, index=hyperedge_index[0])

        num_hyperedges = hyperedge_index[1].max().item() + 1
        hyperedge_idx_expanded = hyperedge_index[1].view(-1, 1, 1).expand(-1, x.size(1), x.size(2))
        result_list = torch.zeros((num_hyperedges, x.size(1), x.size(2)), device=x.device, dtype=x.dtype)
        result_list.scatter_add_(0, hyperedge_idx_expanded, v_features)
        result_list = result_list * B.unsqueeze(-1).unsqueeze(-1)
        e_features = torch.index_select(result_list, dim=0, index=hyperedge_index[1])

        # Node Constraint
        node_constraint = abs(v_features - e_features)
        loss_node = torch.mean(node_constraint)

        # Hyperedge Constraint
        edge_features_norm = F.normalize(result_list, p=2, dim=-1)
        cos_sim = torch.einsum('ebd,mbd->ebm', edge_features_norm, edge_features_norm)

        dist_sq = torch.sum((result_list.unsqueeze(1) - result_list.unsqueeze(0)) ** 2, dim=-1)
        distance = torch.sqrt(dist_sq + 1e-8)

        alpha = cos_sim.permute(0, 2, 1)
        loss_items = alpha * distance + (1 - alpha) * torch.clamp(self.gamma - distance,min=0.0)
        loss_hyperedge = torch.sum(torch.abs(torch.mean(loss_items, dim=-1))) / (len(result_list) ** 2)

        # HGAT
        H_n = (torch.cat([v_features, e_features], dim=-1) @ self.a)
        H_n = F.leaky_relu(H_n, self.negative_slope)
        H_n = softmax(H_n, hyperedge_index[1], num_nodes=x.size(0))
        H_n = F.dropout(H_n, p=self.dropout, training=self.training)

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=D.pow(0.5), H=H_n)

        e_features = torch.index_select(out, dim=0, index=hyperedge_index[1])
        H_e = (torch.cat([e_features, v_features], dim=-1) @ self.b)
        H_e = F.leaky_relu(H_e, self.negative_slope)
        H_e = softmax(H_e, hyperedge_index[0], num_nodes=x.size(0))
        H_e = F.dropout(H_e, p=self.dropout, training=self.training)

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=B, H=H_e)

        out = out * D.view(-1, 1, 1).pow(0.5)
        out = out.transpose(0, 1)

        return out, loss_node + loss_hyperedge


class AdaptiveHypergraphAttention(nn.Module):
    def __init__(self, num_nodes, hyperedge_num, d_model, k_hyperedge=3, alpha=1, beta=0.5, gamma=0.5, dropout=0.1):
        super(AdaptiveHypergraphAttention, self).__init__()
        self.adp_hyper = Adaptive_Hypergraph(num_nodes, hyperedge_num, d_model, k_hyperedge, alpha, beta, dropout)
        self.hyper_conv = HypergraphConv(d_model, d_model, gamma)

    def forward(self, x):
        # x: B, L, S, D
        B, L, S, D = x.shape
        x = x.reshape(B * L, S, -1)
        adj_matrix = self.adp_hyper(x)
        output, constraint_loss = self.hyper_conv(x, adj_matrix)
        return output.reshape(B, L, S, D), constraint_loss
