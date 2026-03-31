import torch
from stationMATGCN import StationMATGCN
from utils import compute_laplacian, create_df_tensors
from torch.utils.data import Dataset
import sys
import os
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pandas as pd
from torch.utils.data import DataLoader
from adjacency import create_adj_matrix


#Uses data coming from data_preprocessing.py
class StationMATGCNDataset(Dataset):

    def __init__(self, station_tensor, external_tensor, target_tensor, T, H):

        self.X = torch.tensor(station_tensor, dtype=torch.float32)
        self.E = torch.tensor(external_tensor, dtype=torch.float32)
        self.Y = torch.tensor(target_tensor, dtype=torch.float32)

        self.T = T
        self.H = H

        self.length = self.X.shape[0] - T - H

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        x = self.X[idx : idx + self.T]          # [T, N, F]
        e = self.E[idx : idx + self.T]          # [T, E]
        y = self.Y[idx + self.T : idx + self.T + self.H]   # [H, N]

        return x, e, y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

F = 5 # Node feature dimension
T = 12 # History length
E = 6 # external feature dimension
H = 12 # Prediction horizon
B = 64 # Batch size
N = 12 # Number of nodes
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


station_list_path = os.path.join("data", "station_list.csv")
adj = torch.tensor(create_adj_matrix(station_list_path))


laplacian = compute_laplacian(adj).float().to(device)

#TODO also change here to have different loaders
training_data_path = os.path.join("data", "train_data_weather.parquet")
df = pd.read_parquet(training_data_path)
df = df.sort_values("OPERATION_PLANNED_TIMESTAMP")
df = df.reset_index(drop=True)

train_end = "2025-02-28"
val_end   = "2025-03-31"

train_df = df[df["OPERATION_PLANNED_TIMESTAMP"] < train_end]
val_df   = df[(df["OPERATION_PLANNED_TIMESTAMP"] >= train_end) & (df["OPERATION_PLANNED_TIMESTAMP"] < val_end)]
test_df  = df[df["OPERATION_PLANNED_TIMESTAMP"] >= val_end]

# Sanity check
print(train_df["OPERATION_PLANNED_TIMESTAMP"].min(), train_df["OPERATION_PLANNED_TIMESTAMP"].max())
print(val_df["OPERATION_PLANNED_TIMESTAMP"].min(), val_df["OPERATION_PLANNED_TIMESTAMP"].max())
print(test_df["OPERATION_PLANNED_TIMESTAMP"].min(), test_df["OPERATION_PLANNED_TIMESTAMP"].max())

train_station, train_ext, train_target = create_df_tensors(train_df)
val_station, val_ext, val_target = create_df_tensors(val_df)
test_station, test_ext, test_target = create_df_tensors(test_df)

train_dataset = StationMATGCNDataset(
    train_station, train_ext, train_target, T, H
)

val_dataset = StationMATGCNDataset(
    val_station, val_ext, val_target, T, H
)

test_dataset = StationMATGCNDataset(
    test_station, test_ext, test_target, T, H
)

# Create different dataloaders for this
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def evaluate(model, dataloader, laplacian, criterion, device):

    model.eval()

    total_loss = 0
    total_samples = 0

    with torch.no_grad():

        for x, e, y in dataloader:

            x = x.to(device)
            e = e.to(device)
            y = y.to(device)

            pred = model(x, e, laplacian)

            loss = criterion(pred, y)

            batch_size = x.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

train_losses = []
val_losses = []

for epoch in range(epochs):

    model.train()

    running_loss = 0
    total_samples = 0

    for x, e, y in train_loader:

        x = x.to(device)
        e = e.to(device)
        y = y.to(device)
        y = y.permute(0, 2, 1)  # [B, N, H]

        optimizer.zero_grad()

        pred = model(x, e, laplacian)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    train_loss = running_loss / total_samples

    val_loss = evaluate(
        model,
        val_loader,
        laplacian,
        criterion,
        device
    )
    #maybe include early stopping
    """ if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pt")

    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break """

    

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

torch.save(model.state_dict(), "matgcn_model.pt")

# Plot curves:

import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Curve")

plt.show()