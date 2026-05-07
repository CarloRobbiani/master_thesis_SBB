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
from utils import load_and_pivot, normalize, prepare_laplacian, permutation_importance
from delay_dataset import DelayDataset
import pandas as pd

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

station_list_path = os.path.join("data", "station_list.csv")
laplacian = prepare_laplacian(station_list_path, DEVICE)

def eval_model(model, loader, laplacian):
    model.eval()

    # ── inference ─────────────────────────────────────────────────────────────────
    all_preds, all_trues, all_weigths = [], [], []

    with torch.no_grad():
        for x, ext, y in loader:
            x, ext, y = x.to(DEVICE), ext.to(DEVICE), y.to(DEVICE)
            pred, feat_weights = model(x, ext, laplacian, return_att=True)   # (B, N)
            pred = pred.mean(dim=-1)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
            all_weigths.append(feat_weights)

    preds_log = np.concatenate(all_preds)   # (num_samples, N)
    trues_log = np.concatenate(all_trues)

    # ── invert log transform → seconds ───────────────────────────────────────────
    preds = from_log(preds_log)
    trues = from_log(trues_log)

    weights = []
    for batch in all_weigths:
        for block_w in batch:
            weights.append(block_w.cpu())

    weights = torch.cat(weights, dim=0)
    feat_importance = weights.mean(dim=(0,1,2))
    all_station_cols = STATION_FEATURE_COLS + ["target_arr"]
    feature_names = STATION_FEATURE_COLS + ["target_arr"] + EXTERNAL_COLS
    """for name, score in zip(feature_names, feat_importance):
        print(f"{name}: {score:.4f}") """

    # ── metrics ───────────────────────────────────────────────────────────────────
    mae  = float(np.abs(preds - trues).mean())
    rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
    print(f"\nTest  MAE : {mae:.1f} sec")
    print(f"Test RMSE : {rmse:.1f} sec")

    return preds, trues, mae

preds, trues, mae = eval_model(model, test_loader, laplacian)
station_feature_names = STATION_FEATURE_COLS + ["target_arr"]
external_feature_names = EXTERNAL_COLS
importances = permutation_importance(
    model,
    test_loader,
    laplacian,
    station_feature_names,
    external_feature_names,
    tg_min
)
print(importances)

# ── plot one station ──────────────────────────────────────────────────────────
STATION_IDX  = None   # change to inspect different stations, or set to None to plot all errors
WINDOW       = None  # number of timesteps to show (None = all)

station_df = pd.read_csv("data/station_list.csv", header=None)
station_names = station_df.iloc[0].tolist()   # full names
station_codes = station_df.iloc[1].tolist()   # short codes

if STATION_IDX is None:
    pred_series = preds.ravel()
    true_series = trues.ravel()
else:
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
axes[0].set_title(f"Predicted vs Actual Delays")
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
plt.savefig("images/eval_plot_scatter.png", dpi=150)
plt.show()
print("Plot saved to eval_plot.png")

# --- Hourly delay and error analysis ---
# Reconstruct timestamps for the test split
raw_df = pd.read_parquet(DATA_PATH)
raw_df = raw_df.sort_values("OPERATION_PLANNED_TIMESTAMP")
timestamps = raw_df["OPERATION_PLANNED_TIMESTAMP"].iloc[t_val + SEQ_LEN:].reset_index(drop=True)

# preds and trues are (num_samples, N) — average across stations for hourly analysis
pred_mean = preds.mean(axis=1)   # (num_samples,)
true_mean = trues.mean(axis=1)

# Trim timestamps to match (dataset drops last HORIZON rows)
timestamps = timestamps.iloc[:len(pred_mean)].reset_index(drop=True)

hourly_df = pd.DataFrame({
    "hour":      timestamps.dt.hour,
    "predicted": pred_mean,
    "actual":    true_mean,
    "abs_error": np.abs(pred_mean - true_mean)
})

hourly = hourly_df.groupby("hour").agg(
    avg_actual=("actual", "mean"),
    avg_predicted=("predicted", "mean"),
    avg_error=("abs_error", "mean")
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(hourly["hour"], hourly["avg_actual"], label="Actual", marker="o")
axes[0].plot(hourly["hour"], hourly["avg_predicted"], label="Predicted", marker="o")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Average Delay (seconds)")
axes[0].set_title("Average Delay by Hour of Day")
axes[0].set_xticks(range(0, 24))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(hourly["hour"], hourly["avg_error"], color="tomato", alpha=0.8)
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Mean Absolute Error (seconds)")
axes[1].set_title("Model Error by Hour of Day")
axes[1].set_xticks(range(0, 24))
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("images/hourly_delay_analysis_matgcn.png")
plt.show()


# ---- per-station metrics -----
mae_per_station  = np.abs(preds - trues).mean(axis=0)   # (N,)
rmse_per_station = np.sqrt(((preds - trues) ** 2).mean(axis=0))

# Sort stations by error (optional but useful)
sorted_idx = np.argsort(mae_per_station)[::-1]  # descending
mae_sorted = mae_per_station[sorted_idx]
station_names_sorted = [stations[i] for i in sorted_idx]
errors = preds - trues   # shape: (num_samples, N)

plt.figure(figsize=(16, 6))

plt.boxplot(
    errors,
    labels=[station_names[i] for i in sorted_idx],
    showfliers=False   # optional: hides extreme outliers (cleaner)
)

plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Prediction Error (seconds)")
plt.title("Error Distribution per Station")

plt.tight_layout()
plt.savefig("images/boxplot_per_station.png", dpi=150)
plt.show()