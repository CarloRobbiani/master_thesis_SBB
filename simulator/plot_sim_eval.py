import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import json
import pickle

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from graph_models.station_graph.stationMATGCN import StationMATGCN
from graph_models.station_graph.utils import load_and_pivot, prepare_laplacian
from graph_models.station_graph.delay_dataset import DelayDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "simulator/data/normal_weather.csv"

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
DEVICE      = device

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
station_arr, external_arr, target_arr, stations = load_and_pivot(
    DATA_PATH, STATION_FEATURE_COLS, EXTERNAL_COLS, sim=True
)
# target_arr is now (T, N, 2): ch0=departure, ch1=arrival

N = len(stations)
T = len(station_arr)
F = station_arr.shape[-1]
E = external_arr.shape[-1]

# ── load stats fitted on REAL training data ───────────────────────────────────
# We never refit these on simulator data — the model was trained with these
# exact transforms and must see the same distribution at inference time.
with open("data/train_stats.json") as f:
    tg_min = json.load(f)["tg_min"]

with open("data/feat_scaler.pkl", "rb") as f:
    feat_scaler = pickle.load(f)

# Apply real-data scaler to simulator station features
station_arr_norm = feat_scaler.transform(
    station_arr.reshape(-1, F)
).reshape(T, N, F)

# ── log1p transform targets with real-data shift ──────────────────────────────
def to_log(a):
    return np.log1p(np.clip(a - tg_min, 0, None))

def from_log(a):
    return np.expm1(a) + tg_min

# Evaluate on the full simulator dataset (no train/val/test split needed here)
target_log = to_log(target_arr)     # (T, N, 2)

# ── dataset / loader ──────────────────────────────────────────────────────────
# DelayDataset appends lagged targets internally → x becomes (T, N, F+2)
sim_ds     = DelayDataset(station_arr_norm, external_arr, target_log, SEQ_LEN, HORIZON)
sim_loader = DataLoader(sim_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── model ─────────────────────────────────────────────────────────────────────
model = StationMATGCN(
    num_station_features  = F + 2,      # +2 for lagged dep/arr targets
    num_external_features = E,
    hidden_dim = HIDDEN_DIM,
    K          = K,
    num_blocks = NUM_BLOCKS,
    horizon    = HORIZON,
).to(DEVICE)

model.load_state_dict(torch.load(
    "graph_models/station_graph/best_matgcn.pt", map_location=DEVICE
))
model.eval()

station_list_path = os.path.join("data", "station_list.csv")
laplacian = prepare_laplacian(station_list_path, DEVICE)

# ── inference ─────────────────────────────────────────────────────────────────
all_preds, all_trues = [], []

with torch.no_grad():
    for x, ext, y in sim_loader:
        x, ext, y = x.to(DEVICE), ext.to(DEVICE), y.to(DEVICE)
        pred = model(x, ext, laplacian)     # (B, N, horizon, 2)
        pred = pred.squeeze(2)              # (B, N, 2)
        all_preds.append(pred.cpu().numpy())
        all_trues.append(y.cpu().numpy())   # y is (B, N, 2)

preds_log = np.concatenate(all_preds)       # (num_samples, N, 2)
trues_log = np.concatenate(all_trues)

preds = from_log(preds_log)                 # (num_samples, N, 2)
trues = from_log(trues_log)

# ── metrics per channel ───────────────────────────────────────────────────────
for ch, ch_name in enumerate(["Departure", "Arrival"]):
    mae  = float(np.abs(preds[..., ch] - trues[..., ch]).mean())
    rmse = float(np.sqrt(((preds[..., ch] - trues[..., ch]) ** 2).mean()))
    print(f"\n{ch_name}  MAE : {mae:.1f} sec")
    print(f"{ch_name} RMSE : {rmse:.1f} sec")

# ── plots: scatter + error histogram per channel ──────────────────────────────
STATION_IDX = None  # set to int to inspect a single station
WINDOW      = None  # set to int to limit timesteps shown

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ch, ch_name in enumerate(["Departure", "Arrival"]):
    if STATION_IDX is None:
        pred_series = preds[..., ch].ravel()
        true_series = trues[..., ch].ravel()
    else:
        pred_series = preds[:, STATION_IDX, ch]
        true_series = trues[:, STATION_IDX, ch]

    if WINDOW is not None:
        pred_series = pred_series[:WINDOW]
        true_series = true_series[:WINDOW]

    # Scatter: predicted vs actual
    ax = axes[ch, 0]
    ax.scatter(true_series, pred_series, alpha=0.5, s=20, color="steelblue")
    min_val = min(true_series.min(), pred_series.min())
    max_val = max(true_series.max(), pred_series.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect (y=x)")
    ax.set_xlabel("Actual Delay (seconds)")
    ax.set_ylabel("Predicted Delay (seconds)")
    ax.set_title(f"{ch_name}: Predicted vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error histogram
    ax = axes[ch, 1]
    error = pred_series - true_series
    abs_error = np.abs(pred_series - true_series)
    ax.hist(error, bins=30, alpha=0.7, color="orange", edgecolor="black")
    ax.axvline(0, color="red", linewidth=2, linestyle="--", label="Zero error")
    ax.axvline(abs_error.mean(), color="green", linewidth=2, linestyle="--",
               label=f"Mean abs error: {abs_error.mean():.1f}s")
    ax.set_xlabel("Error (seconds)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{ch_name}: Error Distribution")
    ax.legend()

plt.tight_layout()
plt.savefig("simulator/images/sim_eval_plot_scatter_normal.png", dpi=150)
plt.show()
print("Plot saved to simulator/images/sim_eval_plot_scatter_normal.png")