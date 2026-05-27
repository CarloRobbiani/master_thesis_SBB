import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FILE_NAME = "sim_with_SBI_long"
BASELINE_NAME = "normal_weather.csv"

df_sim = pd.read_csv(f"simulator\data\{FILE_NAME}.csv")
#df_sim = pd.read_csv("simulator/data/normal_weather.csv")
df_sim["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(df_sim["OPERATION_PLANNED_TIMESTAMP"], format="%Y-%m-%d %H:%M:%S", utc=True)
#df_sim = df_sim[df_sim["SIMULATED_DELAY"] < 5000]
df_sim = df_sim.sort_values(by="OPERATION_PLANNED_TIMESTAMP")
print(df_sim.shape)


# ── Load baseline (no injection) for comparison ───────────────────────────────
df_base = pd.read_csv(f"simulator/data/{BASELINE_NAME}")
df_base["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(
    df_base["OPERATION_PLANNED_TIMESTAMP"], format="%Y-%m-%d %H:%M:%S", utc=True
)
df_base = df_base.sort_values("OPERATION_PLANNED_TIMESTAMP")

df_true = pd.read_parquet("data/train_data_weather.parquet")
df_true = df_true[
    df_true["OPERATION_PLANNED_TIMESTAMP"].dt.date.isin(
        df_sim["OPERATION_PLANNED_TIMESTAMP"].dt.date)]
df_true["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(df_true["OPERATION_PLANNED_TIMESTAMP"], format="%Y-%m-%d %H:%M:%S.%f %z", utc=True)
df_true = df_true.sort_values(by="OPERATION_PLANNED_TIMESTAMP")


# Merge on the closest timestamp by train_number
df_merged = pd.merge_asof(
    df_true,
    df_sim,
    left_on="OPERATION_PLANNED_TIMESTAMP",
    right_on="OPERATION_PLANNED_TIMESTAMP",
    by="TRAIN_NUMBER",
    direction="nearest",
    tolerance=pd.Timedelta('10min'),

)

print(df_merged.shape)

true_series = df_merged["DAILY_PLAN_OPERATIONAL_DELAY_SEC_x"]
pred_series = df_merged["SIMULATED_DELAY"]

PUNCTUALITY_SEC = 180
LINE_ORDER = ["BI", "TUE", "TWN", "LIG", "NV", "LD", "CRNE", "CORN", "SBL", "NE"]
KM = {"BI": 0, "TUE": 9.5, "TWN": 14.2, "LIG": 16.8, "NV": 20.3,
      "LD": 24.1, "CRNE": 27.0, "CORN": 29.8, "SBL": 33.2, "NE": 38.0}
SINGLE_TRACK = (KM["TWN"], KM["LIG"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: scatter plot (predicted vs actual)
axes[0].scatter(true_series, pred_series, alpha=0.5, s=20, color="steelblue", label="Predictions")

# Add diagonal line (perfect prediction)
min_val = min(true_series.min(), pred_series.min())
max_val = max(true_series.max(), pred_series.max())
axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction (y=x)")

axes[0].set_xlabel("Actual Delay (seconds)")
axes[0].set_ylabel("Simulated Delay (seconds)")
axes[0].set_title(f"Simulated vs Actual Delays")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: error distribution histogram
error = pred_series - true_series
abs_error = np.abs(pred_series - true_series)
axes[1].hist(error, bins=30, alpha=0.7, color="orange", edgecolor="black", label="Prediction error")
axes[1].axvline(0, color="red", linewidth=2, linestyle="--", label="Zero error")
axes[1].axvline(abs_error.mean(), color="green", linewidth=2, linestyle="--", label=f"Mean abs error: {abs_error.mean():.1f}s")
axes[1].set_xlabel("Error (seconds)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Prediction Error Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"simulator\images/eval_{FILE_NAME}.png")
plt.show()


# Station heatmap: mean additional delay per station

def mean_dep_delay_by_station(df):
    return (
        df[df["EVENT_TYPE"] == "departure"]
        .groupby("OPERATING_POINT_ABBREVIATION")["SIMULATED_DELAY"]
        .mean()
    )
 
st_base = mean_dep_delay_by_station(df_base)
st_inj  = mean_dep_delay_by_station(df_sim)
st_delta = (st_inj - st_base).reindex(LINE_ORDER).dropna()
 
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
 
# Left: stacked mean delays (baseline vs injected)
x = np.arange(len(st_delta))
w = 0.35
axes4[0].bar(x - w/2, st_base.reindex(st_delta.index), w,
             label="Baseline", color="steelblue", alpha=0.8)
axes4[0].bar(x + w/2, st_inj.reindex(st_delta.index), w,
             label="Injected", color="tomato", alpha=0.8)
axes4[0].axhline(0, color="black", linewidth=0.8)
axes4[0].set_xticks(x)
axes4[0].set_xticklabels(st_delta.index, rotation=45, ha="right", fontsize=9)
axes4[0].set_ylabel("Mean departure delay (seconds)")
axes4[0].set_title("Mean departure delay by station")
axes4[0].legend()
axes4[0].grid(True, axis="y", alpha=0.3)
 
# Shade single-track stations
for idx, st in enumerate(st_delta.index):
    if st in ("TWN", "LIG"):
        axes4[0].axvspan(idx - 0.6, idx + 0.6, alpha=0.07, color="red")
 
# Right: delta (additional delay from injection)
bar_colors = ["tomato" if d > 0 else "steelblue" for d in st_delta]
axes4[1].bar(st_delta.index, st_delta.values, color=bar_colors, alpha=0.85, edgecolor="none")
axes4[1].axhline(0, color="black", linewidth=0.8)
axes4[1].axhline(PUNCTUALITY_SEC, color="orange", linewidth=1, linestyle=":",
                 label=f"+{PUNCTUALITY_SEC}s threshold")
for st, val in st_delta.items():
    axes4[1].text(st, val + (4 if val >= 0 else -10),
                  f"{val:+.0f}s", ha="center", fontsize=8,
                  color="darkred" if val > 0 else "darkblue")
axes4[1].set_ylabel("Additional delay (injected − baseline, seconds)")
axes4[1].set_title("Delay increase by station (injection effect)")
axes4[1].tick_params(axis="x", rotation=45)
axes4[1].legend()
axes4[1].grid(True, axis="y", alpha=0.3)
 
plt.tight_layout()
#plt.savefig(f"simulator/images/heatmap_{FILE_NAME}.png")
plt.show()
