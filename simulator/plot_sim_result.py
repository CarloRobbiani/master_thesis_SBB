import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_sim = pd.read_csv("simulator\sbi_calibrated_2025-01-01.csv")
df_sim["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(df_sim["OPERATION_PLANNED_TIMESTAMP"], format="%Y-%m-%d %H:%M:%S", utc=True)
df_sim = df_sim.sort_values(by="OPERATION_PLANNED_TIMESTAMP")

df_true = pd.read_parquet("data/train_data_weather.parquet")


df_true = df_true[df_true["OPERATIONAL_DAY"] == "2025-01-01"]
df_true["TRAIN_NUMBER"] = np.int64(df_true["TRAIN_NUMBER"])
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
    tolerance=pd.Timedelta('10min')
)

true_series = df_merged["DAILY_PLAN_OPERATIONAL_DELAY_SEC_x"]
pred_series = df_merged["SIMULATED_DELAY"]

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
axes[1].hist(error, bins=30, alpha=0.7, color="orange", edgecolor="black", label="Prediction error")
axes[1].axvline(0, color="red", linewidth=2, linestyle="--", label="Zero error")
axes[1].axvline(error.mean(), color="green", linewidth=2, linestyle="--", label=f"Mean error: {error.mean():.1f}s")
axes[1].set_xlabel("Error (seconds)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Prediction Error Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig("simulator\eval_simulator_vs_actual.png")
plt.show()