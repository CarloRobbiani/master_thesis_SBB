import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from stationMATGCN import StationMATGCN
from utils import load_and_pivot, normalize, normalize_targets, prepare_laplacian, permutation_importance
from delay_dataset import DelayDataset
import pandas as pd
import pickle

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

# -- load & split ----------------------------------------------
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
va_st = station_arr[t_train:t_val]; va_tg = target_arr[t_train:t_val]
te_st = station_arr[t_val:];        te_ex = external_arr[t_val:]; te_tg = target_arr[t_val:]

# -- normalize station features ------------------
tr_st, va_st, te_st, _ = normalize(tr_st, va_st, te_st)

# -- log1p target transform ---------------------------
import json
with open("data/train_stats.json") as f:
    tg_min = json.load(f)["tg_min"]

def to_log(a):
    return np.log1p(np.clip(a - tg_min, 0, None))

def from_log(a):
    return np.expm1(a) + tg_min

te_tg_log = to_log(te_tg)

tr_tg = to_log(tr_tg)
va_tg = to_log(va_tg)
te_tg = to_log(te_tg)

with open("data/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

def invert_targets(arr_norm):
    sh = arr_norm.shape
    return target_scaler.inverse_transform(arr_norm.reshape(-1, 2)).reshape(sh)



# ── normalise targets (z-score in log space) ──────────────────────
tr_tg, va_tg, te_tg, target_scaler = normalize_targets(tr_tg, va_tg, te_tg)

T_te, N_te, C = te_tg_log.shape
te_tg_scaled = target_scaler.transform(te_tg_log.reshape(-1, C)).reshape(T_te, N_te, C)


# -- dataset / loader ------------------------------------------
# DelayDataset now concatenates lagged targets internally → x is (T, N, F+2)
test_ds     = DelayDataset(te_st, te_ex, te_tg_scaled, SEQ_LEN, HORIZON)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -- model ---------------------------------------------------
model = StationMATGCN(
    num_station_features  = F + 2, #  F+2 because DelayDataset appends 2 lagged target channels
    num_external_features = E,
    hidden_dim = HIDDEN_DIM,
    K          = K,
    num_blocks = NUM_BLOCKS,
    horizon    = HORIZON,
).to(DEVICE)

model.load_state_dict(torch.load("graph_models/station_graph/best_matgcn.pt", map_location=DEVICE))

station_list_path = os.path.join("data", "station_list.csv")
laplacian = prepare_laplacian(station_list_path, DEVICE)


def eval_model(model, loader, laplacian):
    model.eval()
    all_preds, all_trues, all_weights = [], [], []

    with torch.no_grad():
        for x, ext, y in loader:
            x, ext, y = x.to(DEVICE), ext.to(DEVICE), y.to(DEVICE)
            pred, feat_weights = model(x, ext, laplacian, return_att=True)
            # CHANGED: pred is (B, N, horizon, 2); squeeze horizon dim (=1)
            pred = pred.squeeze(2)          # → (B, N, 2)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())   # y is (B, N, 2)
            all_weights.append(feat_weights)

    preds_log = np.concatenate(all_preds)   # (num_samples, N, 2)
    trues_log = np.concatenate(all_trues)   # (num_samples, N, 2)

    sh = preds_log.shape

    preds_log = invert_targets(preds_log)
    trues_log = invert_targets(trues_log)
    preds_sec = from_log(preds_log)          # (num_samples, N, 2)
    trues_sec = from_log(trues_log)


    # CHANGED: report metrics per channel (ch0=departure, ch1=arrival)
    for ch, name in enumerate(["Departure", "Arrival"]):
        mae  = float(np.abs(preds_sec[..., ch] - trues_sec[..., ch]).mean())
        rmse = float(np.sqrt(((preds_sec[..., ch] - trues_sec[..., ch]) ** 2).mean()))
        print(f"\n{name}  MAE : {mae:.1f} sec")
        print(f"{name} RMSE : {rmse:.1f} sec")

    # Feature attention weights (unchanged logic, updated feature names)
    weights = []
    for batch in all_weights:
        for block_w in batch:
            weights.append(block_w.cpu())
    weights = torch.cat(weights, dim=0)
    feat_importance = weights.mean(dim=(0, 1, 2))

    # lagged_dep and lagged_arr replace old target_arr input
    feature_names = STATION_FEATURE_COLS + ["lagged_dep", "lagged_arr"] + EXTERNAL_COLS
    for name, score in zip(feature_names, feat_importance):
        print(f"{name}: {score:.4f}")

    mae_overall = float(np.abs(preds_sec - trues_sec).mean())
    return preds_sec, trues_sec, mae_overall


preds, trues, mae = eval_model(model, test_loader, laplacian)

# updated station feature names to match new F+2 input
station_feature_names  = STATION_FEATURE_COLS #+ ["lagged_dep", "lagged_arr"]
external_feature_names = EXTERNAL_COLS

# permutation_importance calls compute_mae_seconds internally;
# that function uses pred.mean(dim=-1) which collapses the horizon dim.
# compute_mae_seconds in utils.py needs updating too
importances = permutation_importance(
    model,
    test_loader,
    laplacian,
    station_feature_names,
    external_feature_names,
    tg_min,
    target_scaler=target_scaler,
    n_repeats=3

)
print("\nPermutation importances (delta MAE seconds):")
for k, v in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {k:45s}  {v:+.2f} s")


# -- plotting ------------------------------------------------------------------
# preds/trues are (num_samples, N, 2). Plot each channel separately.
STATION_IDX = None
WINDOW      = None

station_df    = pd.read_csv("data/station_list.csv", header=None)
station_names = station_df.iloc[0].tolist()
station_codes = station_df.iloc[1].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ch, ch_name in enumerate(["Departure", "Arrival"]):
    if STATION_IDX is None:
        # index last dim for channel
        pred_series = preds[..., ch].ravel()
        true_series = trues[..., ch].ravel()
    else:
        pred_series = preds[:, STATION_IDX, ch]
        true_series = trues[:, STATION_IDX, ch]

    if WINDOW is not None:
        pred_series = pred_series[:WINDOW]
        true_series = true_series[:WINDOW]

    # Scatter
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
    ax.axvline(0, color="red", linewidth=2, linestyle="--")
    ax.axvline(abs_error.mean(), color="green", linewidth=2, linestyle="--",
               label=f"Mean abs error: {abs_error.mean():.1f}s")
    ax.set_xlabel("Error (seconds)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{ch_name}: Error Distribution")
    ax.legend()

plt.tight_layout()
plt.savefig("images/eval_plot_scatter.png", dpi=150)
plt.show()

# -- hourly analysis ---------------------------------------
raw_df = pd.read_parquet(DATA_PATH)
raw_df = raw_df.sort_values("OPERATION_PLANNED_TIMESTAMP")
timestamps = raw_df["OPERATION_PLANNED_TIMESTAMP"].iloc[
    t_val + SEQ_LEN : t_val + SEQ_LEN + len(preds)
].reset_index(drop=True)

num_samples, num_stations, _ = preds.shape  # unpack 3 dims
expanded_hours = np.repeat(timestamps.dt.hour.values, num_stations)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ch, ch_name in enumerate(["Departure", "Arrival"]):
    # index last dim for channel
    hourly_df = pd.DataFrame({
        "hour":      expanded_hours,
        "predicted": preds[..., ch].reshape(-1),
        "actual":    trues[..., ch].reshape(-1),
    })
    hourly_df["abs_error"] = np.abs(hourly_df["predicted"] - hourly_df["actual"])
    hourly = hourly_df.groupby("hour").agg(
        avg_actual=("actual", "mean"),
        avg_predicted=("predicted", "mean"),
        avg_error=("abs_error", "mean")
    ).reset_index()

    axes[ch, 0].plot(hourly["hour"], hourly["avg_actual"],    label="Actual",    marker="o")
    axes[ch, 0].plot(hourly["hour"], hourly["avg_predicted"], label="Predicted", marker="o")
    axes[ch, 0].set_xlabel("Hour of Day")
    axes[ch, 0].set_ylabel("Average Delay (seconds)")
    axes[ch, 0].set_title(f"{ch_name}: Average Delay by Hour")
    axes[ch, 0].set_xticks(range(0, 24))
    axes[ch, 0].legend()
    axes[ch, 0].grid(True, alpha=0.3)

    axes[ch, 1].bar(hourly["hour"], hourly["avg_error"], color="tomato", alpha=0.8)
    axes[ch, 1].set_xlabel("Hour of Day")
    axes[ch, 1].set_ylabel("Mean Absolute Error (seconds)")
    axes[ch, 1].set_title(f"{ch_name}: Model Error by Hour")
    axes[ch, 1].set_xticks(range(0, 24))
    axes[ch, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("images/hourly_delay_analysis_matgcn.png")
plt.show()

# -- per-station boxplot -------------------------------------------------------
# plot one boxplot figure per channel
for ch, ch_name in enumerate(["Departure", "Arrival"]):
    errors = preds[..., ch] - trues[..., ch]   # (num_samples, N)
    mae_per_station = np.abs(errors).mean(axis=0)
    sorted_idx = np.argsort(mae_per_station)[::-1]

    plt.figure(figsize=(16, 6))
    plt.boxplot(
        errors[:, sorted_idx],
        labels=[station_names[i] for i in sorted_idx],
        showfliers=False
    )
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Prediction Error (seconds)")
    plt.title(f"{ch_name} Error Distribution per Station (MATGCN)")
    plt.tight_layout()
    plt.savefig(f"images/boxplot_per_station_{ch_name.lower()}_matgcn.png", dpi=150)
    plt.show()