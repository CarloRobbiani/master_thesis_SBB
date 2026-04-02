import torch
from stationMATGCN import StationMATGCN
from utils import create_df_tensors, prepare_laplacian, filter_tensors, evaluate
from torch.utils.data import Dataset
import sys
import os
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pandas as pd
from torch.utils.data import DataLoader
from adjacency import create_adj_matrix
import numpy as np


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

F = 10 # Node feature dimension
T = 12 # History length
E = 6 # external feature dimension
H = 12 # Prediction horizon
B = 64 # Batch size
N = 12 # Number of nodes
epochs = 20

model = StationMATGCN(
    num_station_features=F,
    num_external_features=E,
    hidden_dim=32,
    K=2,
    num_blocks=2,
    horizon=H
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
criterion = torch.nn.L1Loss()


station_list_path = os.path.join("data", "station_list.csv")
laplacian = prepare_laplacian(station_list_path, device)

training_data_path = os.path.join("data", "train_data_weather.parquet")
df = pd.read_parquet(training_data_path)

df["hour_sin"] = np.sin(2 * np.pi * df["OPERATION_PLANNED_TIMESTAMP"].dt.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["OPERATION_PLANNED_TIMESTAMP"].dt.hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["OPERATION_PLANNED_TIMESTAMP"].dt.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["OPERATION_PLANNED_TIMESTAMP"].dt.dayofweek / 7)
df = df.sort_values("OPERATION_PLANNED_TIMESTAMP")
df = df.reset_index(drop=True)

station_tensor, external_tensor, target_tensor, timestamps = create_df_tensors(df)
print("NaNs in station:", np.isnan(station_tensor).sum())
print("NaNs in external:", np.isnan(external_tensor).sum())
print("NaNs in target:", np.isnan(target_tensor).sum())
timestamps = pd.to_datetime(timestamps, utc=True)      # parse correctly

#print(timestamps)


train_end = pd.Timestamp("2025-02-28")
val_end   = pd.Timestamp("2025-03-31")

train_station, val_station, test_station = filter_tensors(station_tensor, train_end, val_end, timestamps)
train_ext, val_ext, test_ext = filter_tensors(external_tensor, train_end, val_end, timestamps)
train_target, val_target, test_target = filter_tensors(target_tensor, train_end, val_end, timestamps)

# Normalize tensors
station_mean = train_station.mean()
station_std  = train_station.std() + 1e-6
train_station = (train_station - station_mean) / station_std
val_station   = (val_station - station_mean) / station_std
test_station  = (test_station - station_mean) / station_std


# Normalize if RMSE 0.3 its very good
mean = np.nanmean(train_target)
std = np.nanstd(train_target) + 1e-6
train_target = (train_target - mean) / std
val_target   = (val_target - mean) / std
test_target  = (test_target - mean) / std

train_dataset = StationMATGCNDataset(
    train_station, train_ext, train_target, T, H
)

val_dataset = StationMATGCNDataset(
    val_station, val_ext, val_target, T, H
)

test_dataset = StationMATGCNDataset(
    test_station, test_ext, test_target, T, H
)

# Create different dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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

        #loss = criterion(pred, y)
        loss = torch.nn.functional.smooth_l1_loss(pred, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch_size = x.shape[0]

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    train_loss = running_loss / total_samples

    val_loss, val_rmse = evaluate(
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
        f"Val Loss: {val_loss:.4f} | "
        f"Val RMSE: {val_rmse:.4f} (Normalized: {val_rmse * std})"
    )

torch.save(model.state_dict(), "matgcn_model.pt")

test_loss, test_rmse = evaluate(
    model,
    test_loader,
    laplacian,
    criterion,
    device
)

print(f"Test Loss: {test_loss:.4f} | Test RMSE: {test_rmse:.4f} (Normalized: {test_rmse * std})")



# Plot curves:

import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Curve")

plt.show()