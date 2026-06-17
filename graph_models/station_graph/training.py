import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import math
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from stationMATGCN import StationMATGCN
from utils import (
    load_and_pivot, normalize, normalize_targets,
    prepare_laplacian, permutation_importance,
)
from delay_dataset import DelayDataset


# -- Configuration ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DATA_PATH = os.path.join("data", "train_data_weather.parquet")
#DATA_PATH = os.path.join("simulator", "data", "sim_training.parquet")
DATA_PATH = os.path.join("data", "train_data_augmented.parquet") # The data to train on

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

HIDDEN_DIM  = 64
K           = 3
NUM_BLOCKS  = 3
HORIZON     = 1
SEQ_LEN     = 28
EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 1e-3
TRAIN_RATIO = 0.7
DEVICE      = device



# -- Train / Eval loops -------
def train_epoch(model, loader, optimizer, criterion, laplacian, device):
    model.train()
    total_loss = 0.0
    for x, ext, y in loader:
        x, ext, y = x.to(device), ext.to(device), y.to(device)
        L = laplacian.to(device)
        optimizer.zero_grad()
        pred = model(x, ext, L).squeeze(2)    # (B, N, 2)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, laplacian, device,
             tg_min, target_scaler):
    """
    Returns (avg_loss_in_normalised_space, MAE_seconds, RMSE_seconds).
    Loss is in normalised log-space 
    MAE/RMSE are in seconds
    """
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for x, ext, y in loader:
        x, ext, y = x.to(device), ext.to(device), y.to(device)
        L    = laplacian.to(device)
        pred = model(x, ext, L).squeeze(2)          # (B, N, 2)
        total_loss += criterion(pred, y).item() * x.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_pred) # (total, N, 2)
    trues = np.concatenate(all_true)

    # Invert target normalisation to log space
    sh    = preds.shape
    preds = target_scaler.inverse_transform(preds.reshape(-1, 2)).reshape(sh)
    trues = target_scaler.inverse_transform(trues.reshape(-1, 2)).reshape(sh)

    # Invert log transform to seconds
    preds_sec = np.expm1(preds) + tg_min
    trues_sec = np.expm1(trues) + tg_min

    mae  = float(np.abs(preds_sec - trues_sec).mean())
    rmse = float(np.sqrt(((preds_sec - trues_sec) ** 2).mean()))
    return avg_loss, mae, rmse


def main():
    print(f"Device: {DEVICE}\n")

    print("Loading data …")
    station_arr, external_arr, target_arr, stations = load_and_pivot(
        DATA_PATH, STATION_FEATURE_COLS, EXTERNAL_COLS, sim=False
    )
    N = len(stations)
    T = len(station_arr)
    F_raw = station_arr.shape[-1]
    F = F_raw + 2  # +2 lagged dep/arr channels added by DelayDataset
    E = external_arr.shape[-1]

    # -- temporal split ---------
    t_train = int(T * TRAIN_RATIO)
    t_val = int(T * (TRAIN_RATIO + (1 - TRAIN_RATIO) / 2))

    tr_st = station_arr[:t_train]; tr_ex = external_arr[:t_train]; tr_tg = target_arr[:t_train]
    va_st = station_arr[t_train:t_val]; va_ex = external_arr[t_train:t_val]; va_tg = target_arr[t_train:t_val]
    te_st = station_arr[t_val:]; te_ex = external_arr[t_val:]; te_tg = target_arr[t_val:]

    print(f"\nSplit sizes train: {len(tr_st)}, val: {len(va_st)}, test: {len(te_st)}")

    # -- normalise station features -------
    tr_st, va_st, te_st, feat_scaler = normalize(tr_st, va_st, te_st)

    # -- log1p-transform targets ---------
    tg_min = float(tr_tg.min())

    # Use same stats as in training
    os.makedirs("data", exist_ok=True)
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

    # -- normalise targets (z-score in log space) --------
    tr_tg, va_tg, te_tg, target_scaler = normalize_targets(tr_tg, va_tg, te_tg)

    with open("data/target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    # -- datasets and loaders ----------
    train_ds = DelayDataset(tr_st, tr_ex, tr_tg, SEQ_LEN, HORIZON)
    val_ds   = DelayDataset(va_st, va_ex, va_tg, SEQ_LEN, HORIZON)
    test_ds  = DelayDataset(te_st, te_ex, te_tg, SEQ_LEN, HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -- graph Laplacian ----------
    station_list_path = os.path.join("data", "station_list.csv")
    laplacian = prepare_laplacian(station_list_path, device)

    # -- model -------------------
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    criterion = nn.HuberLoss(delta=2.0)

    # -- training loop ----------
    best_val_loss = math.inf
    print(f"\n{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  "
          f"{'Val MAE(s)':>11}  {'Val RMSE(s)':>12}  {'LR':>10}")
    print("-" * 75)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, laplacian, DEVICE
        )
        val_loss, val_mae, val_rmse = evaluate(
            model, val_loader, criterion, laplacian, DEVICE,
            tg_min, target_scaler
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_matgcn_augmented.pt")

        print(f"{epoch:>5}  {train_loss:>12.4f}  {val_loss:>10.4f}  "
              f"{val_mae:>11.1f}  {val_rmse:>12.1f}  {current_lr:>10.2e}")

    # -- test evaluation -------
    model.load_state_dict(torch.load("best_matgcn.pt", map_location=DEVICE))
    _, test_mae, test_rmse = evaluate(
        model, test_loader, criterion, laplacian, DEVICE,
        tg_min, target_scaler
    )

    print(f"\n{'-'*75}")
    print(f"Test  MAE (both) : {test_mae:>8.1f} sec")
    print(f"Test RMSE (both) : {test_rmse:>8.1f} sec")

    # Per-channel breakdown
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, ext, y in test_loader:
            x, ext = x.to(DEVICE), ext.to(DEVICE)
            pred = model(x, ext, laplacian.to(DEVICE)).squeeze(2).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())

    preds_norm = np.concatenate(all_pred)
    trues_norm = np.concatenate(all_true)
    sh = preds_norm.shape
    preds_log = target_scaler.inverse_transform(preds_norm.reshape(-1, 2)).reshape(sh)
    trues_log = target_scaler.inverse_transform(trues_norm.reshape(-1, 2)).reshape(sh)
    preds_sec = np.expm1(preds_log) + tg_min
    trues_sec = np.expm1(trues_log) + tg_min

    print(f"Test MAE dep : {np.abs(preds_sec[...,0] - trues_sec[...,0]).mean():>8.1f} sec")
    print(f"Test MAE arr : {np.abs(preds_sec[...,1] - trues_sec[...,1]).mean():>8.1f} sec")
    print("Best model saved to best_matgcn.pt")

    # -- permutation importance on test set ------
    print("Running permutation importance on test set...")
    permutation_importance(
        model       = model,
        loader      = test_loader,
        laplacian   = laplacian,
        station_feature_names  = STATION_FEATURE_COLS,
        external_feature_names = EXTERNAL_COLS,
        tg_min         = tg_min,
        target_scaler  = target_scaler,
        n_repeats      = 3,
    )


if __name__ == "__main__":
    main()