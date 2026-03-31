import torch.nn as nn
import torch
from utils import STBlock

class StationMATGCN(nn.Module):
    def __init__(
        self,
        num_station_features,
        num_external_features,
        hidden_dim=64,
        K=3,
        num_blocks=2,
        horizon=12
    ):
        super().__init__()

        self.input_proj = nn.Linear(
            num_station_features + num_external_features,
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

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, laplacian)

        #out = self.output_layer(x[:, -1])
        out = self.output_layer(x.mean(dim=1))
        return out.transpose(1, 2)