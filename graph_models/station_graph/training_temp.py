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

        # Mask targets for computing loss only where valid
        self.mask = ~torch.isnan(self.Y)

        self.T = T
        self.H = H

        self.length = self.X.shape[0] - T - H

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        x = self.X[idx : idx + self.T]          # [T, N, F]
        e = self.E[idx : idx + self.T]          # [T, E]
        y = self.Y[idx + self.T : idx + self.T + self.H]   # [H, N]

        m = self.mask[idx + self.T : idx + self.T + self.H]

        # Replace NaNs in y (only for numerical stability)
        y = torch.nan_to_num(y, nan=0.0)

        return x, e, y, m


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

F = 10 # Node feature dimension
T = 12 # History length
E = 6 # external feature dimension
H = 12 # Prediction horizon
B = 64 # Batch size
N = 10 # Number of nodes
epochs = 50

model = StationMATGCN(
    num_station_features=F,
    num_external_features=E,
    hidden_dim=32,
    K=3,
    num_blocks=2,
    horizon=H
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()


# -- Laplacian --
station_list_path = os.path.join("data", "station_list.csv")
laplacian = prepare_laplacian(station_list_path, device)


# -- Load and feature Engineer Dataset --
training_data_path = os.path.join("data", "train_data_weather.parquet")
df = pd.read_parquet(training_data_path)

print(f"\n[DEBUG] Raw dataframe shape: {df.shape}")
print(f"[DEBUG] Timestamp range: {df['OPERATION_ACTUAL_TIMESTAMP'].min()} → {df['OPERATION_ACTUAL_TIMESTAMP'].max()}")
print(f"[DEBUG] Unique stations: {df['OPERATING_POINT_ABBREVIATION'].unique()}")

df["hour_sin"] = np.sin(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.dayofweek / 7)
df = df.sort_values("OPERATION_ACTUAL_TIMESTAMP")
df = df.reset_index(drop=True)


# -- Build tensors --
station_tensor, external_tensor, target_tensor, timestamps = create_df_tensors(df)

print(f"\n[DEBUG] Tensor shapes — station: {station_tensor.shape}, external: {external_tensor.shape}, target: {target_tensor.shape}")
print(f"[DEBUG] NaNs — station: {np.isnan(station_tensor).sum()}, external: {np.isnan(external_tensor).sum()}, target: {np.isnan(target_tensor).sum()}")
valid_ratio = (~np.isnan(target_tensor)).mean()
print(f"[DEBUG] Target valid ratio: {valid_ratio:.4f}")


print("NaNs in station:", np.isnan(station_tensor).sum())
print("NaNs in external:", np.isnan(external_tensor).sum())
print("NaNs in target:", np.isnan(target_tensor).sum())
valid_ratio = (~np.isnan(target_tensor)).mean()
print(f"Valid ratio: {valid_ratio}")
timestamps = pd.to_datetime(timestamps, utc=True)      # parse correctly

valid_counts = np.sum(~np.isnan(target_tensor), axis=1)

#-- Keep only stations where there are enough known data points --
threshold = 3   # try 3–5
keep_idx = valid_counts >= threshold

station_tensor = station_tensor[keep_idx]
external_tensor = external_tensor[keep_idx]
target_tensor = target_tensor[keep_idx]
timestamps = np.array(timestamps)[keep_idx]


train_end = pd.Timestamp("2025-02-28")
val_end   = pd.Timestamp("2025-03-30")

train_station, val_station, test_station = filter_tensors(station_tensor, train_end, val_end, timestamps)
train_ext, val_ext, test_ext = filter_tensors(external_tensor, train_end, val_end, timestamps)
train_target, val_target, test_target = filter_tensors(target_tensor, train_end, val_end, timestamps)

# Normalize station features
station_mean = train_station.mean()
station_std  = train_station.std() + 1e-6
train_station = (train_station - station_mean) / station_std
val_station   = (val_station - station_mean) / station_std
test_station  = (test_station - station_mean) / station_std

# Normalize external features
ext_mean = train_ext.mean(axis=0)
ext_std  = train_ext.std(axis=0) + 1e-6
train_ext = (train_ext - ext_mean) / ext_std
val_ext   = (val_ext   - ext_mean) / ext_std
test_ext  = (test_ext  - ext_mean) / ext_std


# Normalize if RMSE 0.3 its very good
mean = np.nanmean(train_target)
std = np.nanstd(train_target) + 1e-6
train_target = (train_target - mean) / std
val_target   = (val_target - mean) / std
test_target  = (test_target - mean) / std

print(f"\n[DEBUG] Target normalisation — mean: {mean:.4f}  std: {std:.4f}")
print(f"[DEBUG] Normalised train_target — min: {np.nanmin(train_target):.4f}  max: {np.nanmax(train_target):.4f}")


# --- Datasets and Dataloader ---
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


# DEBUG: check a single batch comes out with the right shapes before training
print("\n[DEBUG] Checking a single train batch...")
_x, _e, _y, _m = next(iter(train_loader))
print(f"  x: {_x.shape}  e: {_e.shape}  y: {_y.shape}  m: {_m.shape}")
print(f"  x NaNs: {_x.isnan().sum().item()}  e NaNs: {_e.isnan().sum().item()}  y NaNs: {_y.isnan().sum().item()}")
print(f"  mask True fraction: {_m.float().mean().item():.3f}")
del _x, _e, _y, _m


# --- Training Loop ---
train_losses = []
val_losses = []

for epoch in range(epochs):

    model.train()

    running_loss = 0
    total_samples = 0

    for batch_idx, (x, e, y, m) in enumerate(train_loader):

        x = x.to(device)
        e = e.to(device)
        y = y.to(device)
        y = y.permute(0, 2, 1)  # [B, N, H]
        m = m.to(device)
        m = m.permute(0, 2, 1)   # [B, N, H]

        


        optimizer.zero_grad()
        pred = model(x, e, laplacian)
        #print(pred.mean().item(), pred.std().item())

        if epoch == 0 and batch_idx == 0:
            print(f"\n[DEBUG epoch=1 batch=0] pred shape: {pred.shape}")
            print(f"  pred  — mean: {pred.mean().item():.4f}  std: {pred.std().item():.4f}  NaN: {pred.isnan().sum().item()}  Inf: {pred.isinf().sum().item()}")
            print(f"  y     — mean: {y.mean().item():.4f}  std: {y.std().item():.4f}")
            print(f"  mask coverage: {m.float().mean().item():.3f}")
 
        if pred.isnan().any() or pred.isinf().any():
            print(f"[DEBUG] WARNING: NaN/Inf in pred at epoch {epoch+1}!")
        

        # Masked loss
        loss = ((pred - y) ** 2) * m
        loss = loss.sum() / (m.sum() + 1e-6)

        #loss = criterion(pred, y)
        #loss = torch.nn.functional.smooth_l1_loss(pred, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[DEBUG] WARNING: loss is {loss.item()} at epoch {epoch+1} — skipping update")
            continue

        loss.backward()

        if batch_idx == 0 and epoch % 10 == 0:
            total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            print(f"[DEBUG epoch={epoch+1}] Grad norm (before clip): {total_norm:.4f}")
 
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