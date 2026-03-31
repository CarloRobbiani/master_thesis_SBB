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
from sklearn.metrics import root_mean_squared_error
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
lambda_max = torch.linalg.eigvals(laplacian).real.max()
laplacian = (2 / lambda_max) * laplacian - torch.eye(laplacian.size(0))

training_data_path = os.path.join("data", "train_data_weather.parquet")
df = pd.read_parquet(training_data_path)
df["hour_sin"] = np.sin(2 * np.pi * df["OPERATION_PLANNED_TIMESTAMP"].dt.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["OPERATION_PLANNED_TIMESTAMP"].dt.hour / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["OPERATION_PLANNED_TIMESTAMP"].dt.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["OPERATION_PLANNED_TIMESTAMP"].dt.dayofweek / 7)
df = df.sort_values("OPERATION_PLANNED_TIMESTAMP")
df = df.reset_index(drop=True)

station_tensor, external_tensor, target_tensor, timestamps = create_df_tensors(df)
# Normalize tensors
station_tensor = (station_tensor - station_tensor.mean()) / (station_tensor.std() + 1e-6)
external_tensor = (external_tensor - external_tensor.mean()) / (external_tensor.std() + 1e-6)
print("NaNs in station:", np.isnan(station_tensor).sum())
print("NaNs in external:", np.isnan(external_tensor).sum())
print("NaNs in target:", np.isnan(target_tensor).sum())
timestamps = pd.to_datetime(timestamps, utc=True)      # parse correctly

#print(timestamps)

# Remove timezone if present
if isinstance(timestamps, pd.DatetimeIndex):
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert(None)
elif isinstance(timestamps, pd.Series):
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert(None)
else:
    # If it's a plain Index, try to convert to DatetimeIndex
    timestamps = pd.to_datetime(timestamps, utc=True)
    if hasattr(timestamps, 'tz') and timestamps.tz is not None:
        timestamps = timestamps.tz_convert(None)

train_end = pd.Timestamp("2025-02-28")
val_end   = pd.Timestamp("2025-03-31")


train_idx = timestamps < train_end
val_idx   = (timestamps >= train_end) & (timestamps < val_end)
test_idx  = timestamps >= val_end

train_station = station_tensor[train_idx]
val_station   = station_tensor[val_idx]
test_station  = station_tensor[test_idx]

train_ext = external_tensor[train_idx]
val_ext   = external_tensor[val_idx]
test_ext  = external_tensor[test_idx]

train_target = target_tensor[train_idx]
val_target   = target_tensor[val_idx]
test_target  = target_tensor[test_idx]

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

# Create different dataloaders for this
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def evaluate(model, dataloader, laplacian, criterion, device):

    model.eval()

    total_loss = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for x, e, y in dataloader:

            x = x.to(device)
            e = e.to(device)
            y = y.to(device)
            y = y.permute(0, 2, 1)

            pred = model(x, e, laplacian)


            #loss = criterion(pred, y)
            loss = torch.nn.functional.smooth_l1_loss(pred, y)

            batch_size = x.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Collect ONLY valid values for RMSE
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    all_preds = all_preds.reshape(-1)
    all_targets = all_targets.reshape(-1)

    rmse = root_mean_squared_error(all_targets, all_preds)

    return total_loss / total_samples, rmse

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
        pred = pred * std + mean
        y    = y * std + mean   

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