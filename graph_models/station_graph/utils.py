import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(in_channels * K, out_channels)

    def forward(self, x, laplacian):
        # x: [B, T, N, F]
        B, T, N, F_in = x.shape

        cheb_polys = [x]

        if self.K > 1:
            cheb_polys.append(torch.einsum("ij,btjf->btif", laplacian, x))

        for k in range(2, self.K):
            cheb_polys.append(
                2 * torch.einsum("ij,btjf->btif", laplacian, cheb_polys[-1])
                - cheb_polys[-2]
            )

        x_cat = torch.cat(cheb_polys, dim=-1)
        return self.linear(x_cat)


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x: [B, T, N, F]
        weights = torch.softmax(self.fc(x), dim=-1)
        return x * weights

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, T, N, F]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.einsum("btnd,bsnd->btns", Q, K)
        scores = scores / (x.shape[-1] ** 0.5)

        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("btns,bsnd->btnd", attn, V)

class STBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, K):
        super().__init__()

        self.feature_att = FeatureAttention(in_channels)
        self.temporal_att = TemporalAttention(in_channels)

        self.graph_conv = ChebGraphConv(in_channels, hidden_dim, K)

        self.temporal_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(3, 1),
            padding=(1, 0)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, laplacian):
        # x: [B, T, N, F]

        x = self.feature_att(x)
        x = self.temporal_att(x)

        x = self.graph_conv(x, laplacian)

        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 3, 1)

        return self.norm(x)
    
def compute_laplacian(adj):
    D = torch.diag(torch.sum(adj, dim=1))
    L = D - adj
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(adj, dim=1) + 1e-6))
    return D_inv_sqrt @ L @ D_inv_sqrt