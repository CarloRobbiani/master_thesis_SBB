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

from stationMATGCN import StationMATGCN
from utils import load_and_pivot, normalize, prepare_laplacian
from delay_dataset import DelayDataset

# ── config (keep in sync with training.py) ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join("data", "train_data_weather.parquet")

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
N = len(stations)
T = len(station_arr)
F = station_arr.shape[-1]
E = external_arr.shape[-1]

t_train = int(T * TRAIN_RATIO)
t_val   = int(T * (TRAIN_RATIO + (1 - TRAIN_RATIO) / 2))

tr_st = station_arr[:t_train];      tr_tg = target_arr[:t_train]
va_st = station_arr[t_train:t_val]
te_st = station_arr[t_val:];        te_ex = external_arr[t_val:]; te_tg = target_arr[t_val:]

# ── normalize station features (fit on train only) ───────────────────────────
tr_st, va_st, te_st, _ = normalize(tr_st, va_st, te_st)

# ── log1p target transform (fit shift on train only) ─────────────────────────
tg_min = float(tr_tg.min())

def to_log(a):
    return np.log1p(np.clip(a - tg_min, 0, None))

def from_log(a):
    return np.expm1(a) + tg_min

te_tg_log = to_log(te_tg)

# ── dataset / loader ──────────────────────────────────────────────────────────
test_ds     = DelayDataset(te_st, te_ex, te_tg_log, SEQ_LEN, HORIZON)
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
STATION_IDX  = 0   # change to inspect different stations
WINDOW       = 500  # number of timesteps to show (None = all)

pred_series = preds[:, STATION_IDX]
true_series = trues[:, STATION_IDX]

if WINDOW is not None:
    pred_series = pred_series[:WINDOW]
    true_series = true_series[:WINDOW]

time_axis = np.arange(len(pred_series))

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top: time series overlay
axes[0].plot(time_axis, true_series, label="Actual Delay",    color="steelblue", alpha=0.7, linewidth=0.8)
axes[0].plot(time_axis, pred_series, label="Predicted Delay", color="tomato",    alpha=0.9, linewidth=1.2)
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Delay (seconds)")
axes[0].set_title(f"MATGCN — Station: {stations[STATION_IDX]}")
axes[0].legend()

# Bottom: error
error = pred_series - true_series
axes[1].fill_between(time_axis, error, alpha=0.5, color="orange", label="Prediction error")
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Error (seconds)")
axes[1].set_title("Prediction error (predicted − actual)")
axes[1].legend()

plt.tight_layout()
plt.savefig("images/eval_plot.png", dpi=150)
plt.show()
print("Plot saved to eval_plot.png")