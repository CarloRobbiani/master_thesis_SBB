import torch
from stationMATGCN import StationMATGCN
from utils import compute_laplacian

class StationDataset(torch.utils.data.Dataset):
    def __init__(self, X, external, Y, T, H):
        """
        X: [total_time, N, F]
        external: [total_time, E_ext]
        Y: [total_time, N]
        """
        self.X = X
        self.external = external
        self.Y = Y
        self.T = T
        self.H = H

    def __len__(self):
        return len(self.X) - self.T - self.H

    def __getitem__(self, idx):
        x = self.X[idx : idx+self.T]
        e = self.external[idx : idx+self.T]
        y = self.Y[idx+self.T : idx+self.T+self.H]

        return x, e, y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

F = 12 # Node feature dimension
T = 12 # History length
E = 12 # external feature dimension
H = 12 # Prediction horizon
B = 64 # Batch size
N = 5 # Number of nodes
epochs = 10

model = StationMATGCN(
    num_station_features=F,
    num_external_features=E,
    hidden_dim=64,
    K=3,
    num_blocks=2,
    horizon=H
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.L1Loss()

laplacian = compute_laplacian(adj).to(device)

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for x, e, y in train_loader:
        x = x.to(device).float()
        e = e.to(device).float()
        y = y.to(device).float()

        optimizer.zero_grad()

        pred = model(x, e, laplacian)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, e, y in val_loader:
            x = x.to(device).float()
            e = e.to(device).float()
            y = y.to(device).float()

            pred = model(x, e, laplacian)
            val_loss += criterion(pred, y).item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f}")