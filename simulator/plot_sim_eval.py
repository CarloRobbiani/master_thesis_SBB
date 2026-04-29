"""
Evaluation + plotting script for the trained StationMATGCN model.

Must be kept consistent with training.py:
  - Same STATION_FEATURE_COLS / EXTERNAL_COLS
  - Same target transform (log1p with training-set shift)
  - Same split boundaries
"""
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import json
import pickle

import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from graph_models.station_graph.stationMATGCN import StationMATGCN
from graph_models.station_graph.utils import load_and_pivot, normalize, prepare_laplacian
from graph_models.station_graph.delay_dataset import DelayDataset

# ── config (keep in sync with training.py) ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "simulator/normal_weather.csv"


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
BATCH_SIZE  = 32
TRAIN_RATIO = 0.7
DEVICE      = device

# ── load & split ──────────────────────────────────────────────────────────────
print("Loading data …")
station_arr, external_arr, target_arr, stations = load_and_pivot(
    DATA_PATH, STATION_FEATURE_COLS, EXTERNAL_COLS
)

# Load stats fitted on REAL training data
with open("data/train_stats.json") as f:
    tg_min = json.load(f)["tg_min"]

with open("data/feat_scaler.pkl", "rb") as f:
    feat_scaler = pickle.load(f)


N = len(stations)
T = len(station_arr)
F = station_arr.shape[-1]
E = external_arr.shape[-1]

station_arr_norm = feat_scaler.transform(
    station_arr.reshape(-1, F)
    ).reshape(T, N, F)

t_train = int(T * TRAIN_RATIO)
t_val   = int(T * (TRAIN_RATIO + (1 - TRAIN_RATIO) / 2))

tr_st = station_arr[:t_train];      tr_tg = target_arr[:t_train]
va_st = station_arr[t_train:t_val]
te_st = station_arr[t_val:];        te_ex = external_arr[t_val:]; te_tg = target_arr[t_val:]

# ── normalize station features (fit on train only) ───────────────────────────
#tr_st, va_st, te_st, _ = normalize(tr_st, va_st, te_st)

# ── log1p target transform (fit shift on train only) ─────────────────────────

def to_log(a):
    return np.log1p(np.clip(a - tg_min, 0, None))

def from_log(a):
    return np.expm1(a) + tg_min

te_tg_log = to_log(target_arr)

# ── dataset / loader ──────────────────────────────────────────────────────────
#test_ds     = DelayDataset(te_st, te_ex, te_tg_log, SEQ_LEN, HORIZON)
test_ds = DelayDataset(station_arr_norm, external_arr, te_tg_log, SEQ_LEN, HORIZON)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── model ─────────────────────────────────────────────────────────────────────
model = StationMATGCN(
    num_station_features  = F,
    num_external_features = E,
    hidden_dim = HIDDEN_DIM,
    K          = K,
    num_blocks = NUM_BLOCKS,
    horizon    = HORIZON,
).to(DEVICE)

model.load_state_dict(torch.load("graph_models\station_graph/best_matgcn.pt", map_location=DEVICE))
model.eval()

station_list_path = os.path.join("data", "station_list.csv")
laplacian = prepare_laplacian(station_list_path, DEVICE)

# ── inference ─────────────────────────────────────────────────────────────────
all_preds, all_trues = [], []

with torch.no_grad():
    for x, ext, y in test_loader:
        x, ext, y = x.to(DEVICE), ext.to(DEVICE), y.to(DEVICE)
        pred = model(x, ext, laplacian).mean(dim=-1)   # (B, N)
        all_preds.append(pred.cpu().numpy())
        all_trues.append(y.cpu().numpy())

preds_log = np.concatenate(all_preds)   # (num_samples, N)
trues_log = np.concatenate(all_trues)

# ── invert log transform → seconds ───────────────────────────────────────────
preds = from_log(preds_log)
trues = from_log(trues_log)

# ── metrics ───────────────────────────────────────────────────────────────────
mae  = float(np.abs(preds - trues).mean())
rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
print(f"\nTest  MAE : {mae:.1f} sec")
print(f"Test RMSE : {rmse:.1f} sec")

# ── plot one station ──────────────────────────────────────────────────────────
STATION_IDX  = 1   # change to inspect different stations
WINDOW       = None  # number of timesteps to show (None = all)

pred_series = preds[:, STATION_IDX]
true_series = trues[:, STATION_IDX]

if WINDOW is not None:
    pred_series = pred_series[:WINDOW]
    true_series = true_series[:WINDOW]

time_axis = np.arange(len(pred_series))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: scatter plot (predicted vs actual)
axes[0].scatter(true_series, pred_series, alpha=0.5, s=20, color="steelblue", label="Predictions")

# Add diagonal line (perfect prediction)
min_val = min(true_series.min(), pred_series.min())
max_val = max(true_series.max(), pred_series.max())
axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction (y=x)")

axes[0].set_xlabel("Actual Delay (seconds)")
axes[0].set_ylabel("Predicted Delay (seconds)")
axes[0].set_title(f"Predicted vs Actual Delays — {stations[STATION_IDX]}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: error distribution histogram
error = pred_series - true_series
axes[1].hist(error, bins=30, alpha=0.7, color="orange", edgecolor="black", label="Prediction error")
axes[1].axvline(0, color="red", linewidth=2, linestyle="--", label="Zero error")
axes[1].axvline(error.mean(), color="green", linewidth=2, linestyle="--", label=f"Mean error: {error.mean():.1f}s")
axes[1].set_xlabel("Error (seconds)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Prediction Error Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig("simulator/sim_eval_plot_scatter.png", dpi=150)
plt.show()
print("Plot saved to eval_plot.png")