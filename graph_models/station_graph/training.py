"""
Training script for StationMATGCN on train delay data.

Expected parquet schema (adjust column lists to match your actual data):
  - STATION_ID            : station identifier (str/int)
  - DATE / TIMESTAMP      : temporal index
  - DAILY_PLAN_OPERATIONAL_DELAY_SEC : target (seconds of delay)
  - station feature cols  : numeric per-station operational features
  - weather / external cols: numeric context features (temperature, precipitation, …)

Run:
    pip install torch pyarrow pandas scikit-learn scipy
    python train_matgcn.py
"""
import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import math
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from stationMATGCN import StationMATGCN
from utils import load_and_pivot, normalize, prepare_laplacian
from delay_dataset import DelayDataset

# ──────────────────────────────────────────────
# 0.  CONFIGURATION
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_path = os.path.join("data", "train_data_weather.parquet")
DATA_PATH = training_data_path

# Column that identifies each station
STATION_COL = "OPERATING_POINT_ABBREVIATION" 

# Date / time column used to sort and group
DATE_COL    = "OPERATION_PLANNED_TIMESTAMP"                

# Target variable
TARGET_COL  = "DAILY_PLAN_OPERATIONAL_DELAY_SEC"

# Station-level feature columns  (everything that varies per station per timestep)
# Leave empty [] to auto-detect: all numeric cols except TARGET and EXTERNAL_COLS
STATION_FEATURE_COLS = [
        "EVENT_TYPE",
        "EVENT_SERVED",
        "PLAN_STOP_TYPE", 
        "OPERATION_DAY_PERIOD_IDENTIFIER_COARSE",
        'OPERATION_TRAFFIC_CATEGORY_ABBREVIATION',
        'PLAN_FORMATION_MAXIMAL_VELOCITY',
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos"]

# External / weather feature columns  (same value for all stations at a timestep)
EXTERNAL_COLS = [ 
        'tre200s0', 'fkl010z1', 'fu3010z0', 'rre150z0',
       'htoauts0', 'hto000d0']                  

# ---- Model hyper-parameters ----
HIDDEN_DIM  = 32
K           = 3      # Chebyshev filter order
NUM_BLOCKS  = 3
HORIZON     = 1      # number of future steps to predict

# ---- Sequence lengths ----
SEQ_LEN     = 28      # lookback window (timesteps fed to the model)

# ---- Training ----
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-3
TRAIN_RATIO = 0.8
DEVICE      = device


# ──────────────────────────────────────────────
# 3.  TRAINING LOOP
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, laplacian, device):
    model.train()
    total_loss = 0.0
    for x, ext, y in loader:
        x, ext, y = x.to(device), ext.to(device), y.to(device)
        L = laplacian.to(device)

        optimizer.zero_grad()
        # model output: (B, N, HORIZON)
        pred = model(x, ext, L)             # → (B, N, HORIZON)
        # collapse horizon dim to match y (B, N)
        pred = pred.mean(dim=-1)            # (B, N)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, laplacian, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for x, ext, y in loader:
        x, ext, y = x.to(device), ext.to(device), y.to(device)
        L = laplacian.to(device)
        pred = model(x, ext, L).mean(dim=-1)
        total_loss += criterion(pred, y).item() * x.size(0)
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)
    mae  = (preds - trues).abs().mean().item()
    rmse = ((preds - trues) ** 2).mean().sqrt().item()
    return avg_loss, mae, rmse


# ──────────────────────────────────────────────
# 4.  MAIN
# ──────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    # ── load data ────────────────────────────────────────────────────
    print("Loading data …")
    station_arr, external_arr, target_arr, stations = load_and_pivot(DATA_PATH,
                                                                    STATION_FEATURE_COLS,
                                                                    EXTERNAL_COLS)
    N = len(stations)
    T = len(station_arr)
    F = station_arr.shape[-1]   # includes target as last feature
    E = external_arr.shape[-1]

    # ── train / val / test split (temporal) ──────────────────────────
    t_train = int(T * TRAIN_RATIO)
    t_val   = int(T * (TRAIN_RATIO + (1 - TRAIN_RATIO) / 2))

    tr_st = station_arr [:t_train];  tr_ex = external_arr[:t_train];  tr_tg = target_arr[:t_train]
    va_st = station_arr [t_train:t_val]; va_ex = external_arr[t_train:t_val]; va_tg = target_arr[t_train:t_val]
    te_st = station_arr [t_val:];  te_ex = external_arr[t_val:];  te_tg = target_arr[t_val:]

    print(f"\nSplit sizes → train: {len(tr_st)}, val: {len(va_st)}, test: {len(te_st)}")

    # ── normalize ─────────────────────────────────────────────────────
    tr_st, va_st, te_st, feat_scaler = normalize(tr_st, va_st, te_st)

    #  normalize targets using the log scaler
    # (last column of station_arr is TARGET_COL)
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(tr_tg.reshape(-1, 1))
    def scale_tgt(a): return tgt_scaler.transform(a.reshape(-1, 1)).reshape(a.shape)
    tr_tg, va_tg, te_tg = scale_tgt(tr_tg), scale_tgt(va_tg), scale_tgt(te_tg)

    """ tr_tg = np.log1p(np.clip(tr_tg, 0, None))
    va_tg = np.log1p(np.clip(va_tg, 0, None))
    te_tg = np.log1p(np.clip(te_tg, 0, None)) """

    # ── datasets & loaders ───────────────────────────────────────────
    train_ds = DelayDataset(tr_st, tr_ex, tr_tg, SEQ_LEN, HORIZON)
    val_ds   = DelayDataset(va_st, va_ex, va_tg, SEQ_LEN, HORIZON)
    test_ds  = DelayDataset(te_st, te_ex, te_tg, SEQ_LEN, HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── graph Laplacian ───────────────────────────────────────────────
    station_list_path = os.path.join("data", "station_list.csv")
    laplacian = prepare_laplacian(station_list_path, device)

    # ── model ─────────────────────────────────────────────────────────
    # Station features: F (includes target as last col)
    # External features: E
    model = StationMATGCN(
        num_station_features  = F,
        num_external_features = E,
        hidden_dim = HIDDEN_DIM,
        K          = K,
        num_blocks = NUM_BLOCKS,
        horizon    = HORIZON,
    ).to(DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5 #verbose=True
    )
    criterion = nn.HuberLoss()   # robust to outlier delays

    # ── training ──────────────────────────────────────────────────────
    best_val_loss = math.inf
    print(f"\n{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val MAE':>9}  {'Val RMSE':>10}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, laplacian, DEVICE)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, criterion, laplacian, DEVICE)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_matgcn.pt")

        print(f"{epoch:>5}  {train_loss:>12.4f}  {val_loss:>10.4f}  {val_mae:>9.2f}  {val_rmse:>10.2f}")

    # ── test evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load("best_matgcn.pt", map_location=DEVICE))
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, criterion, laplacian, DEVICE)

    # Un-scale metrics back to seconds
    test_mae_sec  = test_mae  * tgt_scaler.scale_[0]
    test_rmse_sec = test_rmse * tgt_scaler.scale_[0]
    """ test_mae_sec  = float(np.expm1(test_mae))
    test_rmse_sec = float(np.expm1(test_rmse)) """

    print(f"\n{'─'*55}")
    print(f"Test  MAE : {test_mae_sec:>8.1f} sec")
    print(f"Test RMSE : {test_rmse_sec:>8.1f} sec")
    print("Best model saved to best_matgcn.pt")


if __name__ == "__main__":
    main()