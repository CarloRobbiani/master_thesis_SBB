"""
Training script for StationMATGCN on train delay data.

Changes vs. original
─────────────────────
1. load_and_pivot now correctly handles (timestamp, station) collisions by
   aggregating arrivals and departures separately — see utils.py for details.
   Arrival delay is added as an extra station input feature.

2. Target is log1p-transformed to compress the heavy tail of the delay
   distribution. Spikes no longer dominate gradients; rare large delays
   still get a gradient signal proportional to their log magnitude.
   Metrics are inverse-transformed (expm1) before printing.

3. Loss: HuberLoss(delta=2.0) — more robust than MSE on the remaining
   variance after log-transform.

4. LR schedule: CosineAnnealingLR for smoother decay without the noisy
   plateau-detection of ReduceLROnPlateau.

5. Dropout(0.1) added inside STBlock (see utils.py STBlock definition).

Run:
    python training.py
"""
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from stationMATGCN import StationMATGCN
from utils import load_and_pivot, normalize, prepare_laplacian
from delay_dataset import DelayDataset
import json
import pickle

# ──────────────────────────────────────────────
# 0.  CONFIGURATION
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_path = os.path.join("data", "train_data_weather.parquet")
DATA_PATH = training_data_path

STATION_COL = "OPERATING_POINT_ABBREVIATION"
DATE_COL    = "OPERATION_PLANNED_TIMESTAMP"
TARGET_COL  = "DAILY_PLAN_OPERATIONAL_DELAY_SEC"

# Station-level feature columns (EVENT_TYPE removed — arrival/departure are
# now handled structurally; arrival delay is added as "target_arr" by
# load_and_pivot automatically)
STATION_FEATURE_COLS = [
    "EVENT_TYPE",
    "EVENT_SERVED",
    "PLAN_STOP_TYPE",
    "OPERATION_DAY_PERIOD_IDENTIFIER_COARSE",
    "OPERATION_TRAFFIC_CATEGORY_ABBREVIATION",
    "PLAN_FORMATION_MAXIMAL_VELOCITY",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

EXTERNAL_COLS = [
    "tre200s0", "fkl010z1", "fu3010z0", "rre150z0",
    "htoauts0", "hto000d0",
]

# ---- Model hyper-parameters ----
HIDDEN_DIM  = 64
K           = 3
NUM_BLOCKS  = 3
HORIZON     = 1

# ---- Sequence lengths ----
SEQ_LEN     = 28

# ---- Training ----
EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 1e-3
TRAIN_RATIO = 0.7
DEVICE      = device


# ──────────────────────────────────────────────
# 1.  TARGET TRANSFORMS
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# 2.  TRAIN / EVAL LOOPS
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, laplacian, device):
    model.train()
    total_loss = 0.0
    for x, ext, y in loader:
        x, ext, y = x.to(device), ext.to(device), y.to(device)
        L = laplacian.to(device)

        optimizer.zero_grad()
        #pred = model(x, ext, L).mean(dim=-1)   # (B, N)
        pred = model(x, ext, L).squeeze(2) # (B, N, 2)
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
        #pred = model(x, ext, L).mean(dim=-1)
        pred = model(x, ext, L).squeeze(2)     # (B, N, 2)
        total_loss += criterion(pred, y).item() * x.size(0)
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_pred).numpy()
    trues = torch.cat(all_true).numpy()
    mae  = float(np.abs(preds - trues).mean())
    rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
    return avg_loss, mae, rmse


# ──────────────────────────────────────────────
# 3.  MAIN
# ──────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    # ── load data ─────────────────────────────────────────────────────
    print("Loading data …")
    station_arr, external_arr, target_arr, stations = load_and_pivot(
        DATA_PATH, STATION_FEATURE_COLS, EXTERNAL_COLS
    )
    N = len(stations)
    T = len(station_arr)
    #F = station_arr.shape[-1]
    F = station_arr.shape[-1] + 2  # +2 for lagged dep/arr targets
    E = external_arr.shape[-1]

    # ── train / val / test split (temporal) ──────────────────────────
    t_train = int(T * TRAIN_RATIO)
    t_val   = int(T * (TRAIN_RATIO + (1 - TRAIN_RATIO) / 2))

    tr_st = station_arr[:t_train];   tr_ex = external_arr[:t_train];  tr_tg = target_arr[:t_train]
    va_st = station_arr[t_train:t_val]; va_ex = external_arr[t_train:t_val]; va_tg = target_arr[t_train:t_val]
    te_st = station_arr[t_val:];     te_ex = external_arr[t_val:];    te_tg = target_arr[t_val:]

    print(f"\nSplit sizes → train: {len(tr_st)}, val: {len(va_st)}, test: {len(te_st)}")

    # ── normalize station features ────────────────────────────────────
    tr_st, va_st, te_st, feat_scaler = normalize(tr_st, va_st, te_st)


    # ── log1p-transform targets ───────────────────────────────────────
    #
    # Delays are in seconds and can be negative (early arrivals).
    # Strategy: shift by the training-set minimum so all values >= 0,
    # then apply log1p.  This compresses the heavy positive tail while
    # keeping the relative ordering and allowing exact inversion.
    #
    tg_min = float(tr_tg.min())   # computed on train split only

    # Save stats for simulation evaluation
    with open("data/train_stats.json", "w") as f:
        json.dump({"tg_min": tg_min}, f)

    with open("data/feat_scaler.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)


    def to_log(a):
        return np.log1p(np.clip(a - tg_min, 0, None))

    def from_log(a):
        return np.expm1(a) + tg_min

    tr_tg = to_log(tr_tg)
    va_tg = to_log(va_tg)
    te_tg = to_log(te_tg)

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

    # Cosine schedule: smoothly decays LR from LR → eta_min over EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # Huber with delta=2.0: stronger gradient on moderately large errors
    # than the default delta=1.0, without MSE's instability on outliers
    criterion = nn.HuberLoss(delta=2.0)

    # ── training loop ─────────────────────────────────────────────────
    best_val_loss = math.inf
    print(f"\n{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val MAE':>9}  {'Val RMSE':>10}  {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, laplacian, DEVICE)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, criterion, laplacian, DEVICE)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_matgcn.pt")

        print(f"{epoch:>5}  {train_loss:>12.4f}  {val_loss:>10.4f}  {val_mae:>9.4f}  {val_rmse:>10.4f}  {current_lr:>10.2e}")

    # ── test evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load("best_matgcn.pt", map_location=DEVICE))
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, criterion, laplacian, DEVICE)

    # Invert log transform to get metrics in seconds
    # MAE in log space → convert a representative prediction back
    # More accurate: collect all preds and invert, then compute MAE/RMSE
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, ext, y in test_loader:
            x, ext, y = x.to(DEVICE), ext.to(DEVICE), y.to(DEVICE)
            pred = model(x, ext, laplacian.to(DEVICE)).mean(dim=-1)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())

    preds_log = np.concatenate(all_pred)
    trues_log = np.concatenate(all_true)

    preds_sec = from_log(preds_log)
    trues_sec = from_log(trues_log)

    test_mae_sec  = float(np.abs(preds_sec - trues_sec).mean())
    test_rmse_sec = float(np.sqrt(((preds_sec - trues_sec) ** 2).mean()))

    print(f"\n{'─'*65}")
    print(f"Test MAE dep : {np.abs(preds_sec[...,0] - trues_sec[...,0]).mean():.1f} sec")
    print(f"Test MAE arr : {np.abs(preds_sec[...,1] - trues_sec[...,1]).mean():.1f} sec")
    print(f"Test  MAE : {test_mae_sec:>8.1f} sec")
    print(f"Test RMSE : {test_rmse_sec:>8.1f} sec")
    print("Best model saved to best_matgcn.pt")


if __name__ == "__main__":
    main()