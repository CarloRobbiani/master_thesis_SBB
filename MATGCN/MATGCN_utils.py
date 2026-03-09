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
        
        cheb_polynomials = [x]
        if self.K > 1:
            cheb_polynomials.append(torch.einsum("ij,btjf->btif", laplacian, x))

        for k in range(2, self.K):
            cheb_polynomials.append(
                2 * torch.einsum("ij,btjf->btif", laplacian, cheb_polynomials[-1])
                - cheb_polynomials[-2]
            )

        x_concat = torch.cat(cheb_polynomials, dim=-1)
        return self.linear(x_concat)
    
class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x: [B, T, N, F]
        scores = torch.tanh(self.proj(x))
        scores = self.score(scores)
        weights = torch.softmax(scores, dim=-2)  # attention over features
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

        attention = torch.softmax(
            torch.einsum("btnd,bsnd->btns", Q, K) / (x.shape[-1] ** 0.5),
            dim=-1
        )

        out = torch.einsum("btns,bsnd->btnd", attention, V)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, T, N, F]
        Q = self.query(x)
        K = self.key(x)

        attention = torch.softmax(
            torch.einsum("btnd,btmd->btnm", Q, K) / (x.shape[-1] ** 0.5),
            dim=-1
        )

        return attention
    
class STBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, K):
        super().__init__()

        self.feature_attention = FeatureAttention(in_channels)
        self.temporal_attention = TemporalAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels)

        self.cheb_conv = ChebGraphConv(in_channels, hidden_dim, K)
        self.temporal_conv = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=(3, 1),
            padding=(1, 0)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, laplacian):
        x = self.feature_attention(x)
        x = self.temporal_attention(x)

        spatial_att = self.spatial_attention(x)
        laplacian = laplacian * spatial_att.mean(dim=0)

        x = self.cheb_conv(x, laplacian)

        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 3, 1)

        return self.norm(x)