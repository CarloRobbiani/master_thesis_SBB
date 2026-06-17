import torch.nn as nn
import torch
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from graph_models.station_graph.utils import STBlock
#from utils import STBlock

class StationMATGCN(nn.Module):
    def __init__(
        self,
        num_station_features,
        num_external_features,
        hidden_dim=32,
        K=2,
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
        self.horizon = horizon
        self.output_layer = nn.Linear(hidden_dim, horizon * 2) # predict two values per station. Arrival + Departure

    def forward(self, x, external, laplacian, return_att=False):
        # x: [B, T, N, F]
        # external: [B, T, E]

        B, T, N, F = x.shape

        external = external.unsqueeze(2).repeat(1, 1, N, 1)
        x = torch.cat([x, external], dim=-1)

        x = self.input_proj(x)
        feat_weigths_all = []

        for block in self.blocks:
            if return_att: 
                x, feat_w = block(x, laplacian, return_att=True)
                feat_weigths_all.append(feat_w)
            else:
                x = block(x, laplacian, return_att=False)

        out = self.output_layer(x[:, -1])          # (B, N, horizon*2)
        out = out.view(B, N, self.horizon, 2) 
        #out = self.output_layer(x.mean(dim=1))
        #out = self.output_layer(x)
        #return out.transpose(1, 2)
        if return_att:
            return out, feat_weigths_all
        return out