import torch
import torch.nn as nn
#from torch_geometric_temporal.nn.recurrent import STConv
from torch_geometric_temporal.nn.attention import STConv


class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_channels=32, horizon=12):
        super().__init__()

        self.stconv1 = STConv(
            num_nodes=num_nodes,
            in_channels=num_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            K=3
        )

        self.stconv2 = STConv(
            num_nodes=num_nodes,
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            K=3
        )

        self.fc = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index):
        # x: (batch, num_nodes, num_features, seq_len)

        batch_size = x.shape[0]
        outputs = []

        for b in range(batch_size):
            xb = x[b]  # (nodes, features, seq_len)

            h = self.stconv1(xb, edge_index)
            h = self.stconv2(h, edge_index)

            # output shape: (nodes, hidden_channels, seq_len_out)
            h_last = h[:, :, -1]  # (nodes, hidden_channels)

            pred = self.fc(h_last)  # (nodes, horizon)
            outputs.append(pred)

        return torch.stack(outputs, dim=0)  # (batch, nodes, horizon)


def train_gnn(model, train_loader, val_loader, edge_index, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    edge_index = edge_index.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X, Y in train_loader:
            X = X.to(device)  # (batch, nodes, seq_len, features)
            Y = Y.to(device)

            # STConv expects (batch, nodes, features, seq_len)
            X = X.permute(0, 1, 3, 2)

            pred = model(X, edge_index)
            loss = loss_fn(pred, Y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Train Loss={train_loss/len(train_loader):.4f}")
