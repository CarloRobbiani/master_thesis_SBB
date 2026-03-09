import torch
import torch.nn as nn
from MATGCN_utils import STBlock

class MATGCN(nn.Module):
    def __init__(
        self,
        num_features,
        external_features,
        hidden_dim,
        K,
        num_blocks,
        horizon
    ):
        super().__init__()

        self.external_proj = nn.Linear(
            num_features + external_features,
            hidden_dim
        )

        self.blocks = nn.ModuleList([
            STBlock(hidden_dim, hidden_dim, K)
            for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, horizon)

    def forward(self, x, external, laplacian):
        # x: [B, T, N, F]
        # external: [B, T, E]

        B, T, N, F = x.shape

        external = external.unsqueeze(2).repeat(1, 1, N, 1)
        x = torch.cat([x, external], dim=-1)

        x = self.external_proj(x)

        for block in self.blocks:
            x = block(x, laplacian)

        out = self.output_layer(x[:, -1])
        return out