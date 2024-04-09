import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax


# class multi_shallow_embedding(nn.Module):
    
#     def __init__(self, num_nodes, k_neighs, num_graphs):
#         super().__init__()
        
#         self.num_nodes = num_nodes
#         self.k = k_neighs
#         self.num_graphs = num_graphs

#         self.emb_s = Parameter(Tensor(num_graphs, num_nodes, 1))
#         self.emb_t = Parameter(Tensor(num_graphs, 1, num_nodes))
        
#     def reset_parameters(self):
#         init.xavier_uniform_(self.emb_s)
#         init.xavier_uniform_(self.emb_t)
        
        
#     def forward(self, device):
        
#         # adj: [G, N, N]
#         adj = torch.matmul(self.emb_s, self.emb_t).to(device)
        
#         # remove self-loop
#         adj = adj.clone()
#         idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
#         adj[:, idx, idx] = float('-inf')
        
#         # top-k-edge adj
#         adj_flat = adj.reshape(self.num_graphs, -1)
#         indices = adj_flat.topk(k=self.k)[1].reshape(-1)
        
#         idx = torch.tensor([ i//self.k for i in range(indices.size(0)) ], device=device)
        
#         adj_flat = torch.zeros_like(adj_flat).clone()
#         adj_flat[idx, indices] = 1.
#         adj = adj_flat.reshape_as(adj)
        
#         return adj

# MultiShallowEmbeddingEnhanced with Attention
class multi_shallow_embedding(nn.Module):
   
    def __init__(self, num_nodes, k_neighs, num_graphs):
        super().__init__()
       
        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs

        self.emb_s = Parameter(torch.Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(torch.Tensor(num_graphs, 1, num_nodes))
       
        # Attention weights
        self.attention_weights = Parameter(torch.Tensor(num_graphs, num_nodes, num_nodes))
        self.reset_parameters()
       
    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)
        init.xavier_uniform_(self.attention_weights)
       
    def forward(self, device):
        # Basic adjacency matrix
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)
       
        # Calculate scaled attention
        attention_scores = softmax(self.attention_weights, dim=-1).to(device)
       
        # # Scale attention by the square root of the degree
        # degree = attention_scores.sum(dim=-1, keepdim=True).sqrt()
        scaled_attention = attention_scores / np.sqrt(self.num_nodes)
       
        # adj = adj * attention_scores
        adj = adj * scaled_attention
       
        # remove self-loop
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')
       
        # top-k-edge adj
        adj_flat = adj.reshape(self.num_graphs, -1)
        indices = adj_flat.topk(k=self.k)[1].reshape(-1)
       
        idx = torch.tensor([ i//self.k for i in range(indices.size(0)) ], device=device)
       
        adj_flat = torch.zeros_like(adj_flat).clone()
        adj_flat[idx, indices] = 1.
        adj = adj_flat.reshape_as(adj)
       
        return adj

class Group_Linear(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()
                
        self.out_channels = out_channels
        self.groups = groups
        
        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.group_mlp.reset_parameters()
        
        
    def forward(self, x: Tensor, is_reshape: False):
        """
        Args:
            x (Tensor): [B, C, N, F] (if not is_reshape), [B, C, G, N, F//G] (if is_reshape)
        """
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups
        
        if not is_reshape:
            # x: [B, C_in, G, N, F//G]
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # x: [B, G*C_in, N, F//G]
        x = x.transpose(1, 2).reshape(B, G*C, N, -1)
        
        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)
        
        # out: [B, C_out, G, N, F//G]
        return out


class DenseGCNConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin = Group_Linear(in_channels, out_channels, groups, bias=False)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        init.zeros_(self.bias)
        
    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[:, idx, idx] += 1
        
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        return adj
        
        
    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [B, G, N, N]
        """
        adj = self.norm(adj, add_loop).unsqueeze(1)

        # x: [B, C, G, N, F//G]
        x = self.lin(x, False)
        
        out = torch.matmul(adj, x)
        
        # out: [B, C, N, F]
        B, C, _, N, _ = out.size()
        out = out.transpose(2, 3).reshape(B, C, N, -1)
        
        if self.bias is not None:
            out = out.transpose(1, -1) + self.bias
            out = out.transpose(1, -1)
        
        return out


class DenseGINConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, eps=0, train_eps=True):
        super().__init__()
        
        # TODO: Multi-layer model
        self.mlp = Group_Linear(in_channels, out_channels, groups, bias=False)
        
        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))
            
        self.reset_parameters()
            
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)
        
    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1
        
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        return adj
        
        
    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        B, C, N, _ = x.size()
        G = adj.size(0)
        
        # adj-norm
        adj = self.norm(adj, add_loop=False)
        
        # x: [B, C, G, N, F//G]
        x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        
        out = torch.matmul(adj, x)
        
        # DYNAMIC
        x_pre = x[:, :, :-1, ...]
        
        # out = x[:, :, 1:, ...] + x_pre
        out[:, :, 1:, ...] = out[:, :, 1:, ...] + x_pre
        # out = torch.cat( [x[:, :, 0, ...].unsqueeze(2), out], dim=2 )
        
        if add_loop:
            out = (1 + self.eps) * x + out
        
        # out: [B, C, G, N, F//G]
        out = self.mlp(out, True)
        
        # out: [B, C, N, F]
        C = out.size(1)
        out = out.transpose(2, 3).reshape(B, C, N, -1)
        
        return out


class Dense_TimeDiffPool2d(nn.Module):
    
    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()
        
        # TODO: add Normalization
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))
        self.re_param = Parameter(Tensor(kern_size, 1))

        ##added for attention in this module
        # Adjusted attention layers for query, key 
        # Todo (and optionally value) -does it make sense
        self.query = nn.Linear(63, 63)
        self.key = nn.Linear(63, 63)
        ##todo- remove specific initialization add parameter to init
        # self.query = nn.Linear(1, 1)
        # self.key = nn.Linear(1, 1)


        self.reset_parameters()
        
    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')

        ######################################## for the attention
        init.xavier_uniform_(self.query.weight)
        init.xavier_uniform_(self.key.weight)
        
        
    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        B, C, N, F = x.shape ## added lines  # Assuming x is [B, C, N, F]
        #################################################################
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)

        out_features = out
        ##################################################################

        # s: [ N^(l+1), N^l, 1, K ]
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        # TODO: fully-connect, how to decrease time complexity
        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))
        ###################################################################
        
        G = adj.size(0)
        # Reshape and transpose to include the graph dimension G
        # xlater = out_features.reshape(B, C, self.G, N, -1).transpose(2, 3)  # [B, C, N, G, F']
        out_features = out_features.view(B, C, N, G, -1).permute(0, 1, 3, 2, 4)  # Reshape to [B, C, G, N, F']
        out_features_flat = out_features.reshape(B, C, G, N, -1)
        # print(out_features_flat.shape)
        out_features_flatpool = torch.mean(out_features_flat, axis=1)
        # print(out_features_flatpool.shape)
        # Apply query and key transformations
        queries = self.query(out_features_flatpool)  # Shape: [B, C, G, N, pooled_nodes]
        keys = self.key(out_features_flatpool)  # Shape: [B, C, G, N, pooled_nodes]

        # Calculate attention scores
        attention_scores = torch.einsum('bgnl,bgml->bgnm', (queries, keys))  # [B, C, G, N, N]
        # print('att score shape before sfmax', attention_scores.shape)
        # temperature = 0.5
        # attention_scores = softmax(attention_scores / temperature, dim=-1)

        # attention_scores = softmax(attention_scores, dim=-1)  # Normalize over the last dimension
        # print('att score shape', attention_scores.shape)

        ### Apply modifications here ###

        # # Example: Thresholding to reduce connectivity
        # threshold = 0.01  # Example threshold value
        # attention_scores = torch.where(attention_scores > threshold, attention_scores, torch.zeros_like(attention_scores))
        
        # k = 5  # Keep top 5 connections per node
        # topk_scores, indices = torch.topk(attention_scores, k=k, dim=-1)
        # attention_scores_sparse = torch.zeros_like(attention_scores).scatter(dim=-1, index=indices, src=topk_scores)
        # attention_scores = attention_scores_sparse  # Use the sparse scores

        # attention_scores = torch.mean(attention_scores, dim=1)

        # Combine attention_scores with the adjacency matrix
        # Assuming adj is already [B, G, N, N]
        adj_expanded = out_adj.unsqueeze(0).repeat(B, 1, 1, 1)
        
        attention_scores = attention_scores.detach().cpu()
        adj_expanded     = adj_expanded.detach().cpu()
        
        # print('attention_scores', attention_scores.shape, 'adj_expanded shape', adj_expanded.shape)
        weighted_adj = attention_scores * adj_expanded  # Element-wise multiplication
        # weighted_adj = weighted_adj.cpu()
     
        return out, out_adj, weighted_adj

##################################################################################################################    



