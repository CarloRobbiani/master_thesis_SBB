import pandas as pd
from XGBoost import XGBoostBaseline, summarize_dependence
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data_preprocessing import preprocess_train, time_split
my_xgboost = XGBoostBaseline()
import matplotlib.pyplot as plt


df = pd.read_parquet("data/train_data_weather.parquet")

df = df.sort_values("OPERATION_PLANNED_TIMESTAMP")

df["hour_sin"] = np.sin(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.dayofweek / 7)

df["hto000d0"] = df["hto000d0"].fillna(0)
df = df.drop(["date", "days"], axis=1)



# Reduce data types to save space
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

target_col = 'DAILY_PLAN_OPERATIONAL_DELAY_SEC'
X,Y = preprocess_train(df, target_column=target_col)

X_train, X_val, X_test, y_train, y_val, y_test = time_split(X, Y)
my_xgboost.fit(X, Y, X_val, y_val)

prediction = my_xgboost.predict(X_test)

my_xgboost.plot_loss()

plot_len = None # Number of entries to show None = all

val_end = int(len(X) * (0.7 + 0.15))
actual_timestamps = df.iloc[val_end:]["OPERATION_PLANNED_TIMESTAMP"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))


if plot_len is not None:
    pred_series = prediction[:plot_len]
    true_series = y_test[:plot_len]
else:
    pred_series = prediction
    true_series = y_test


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

error = pred_series[:,0] - true_series

axes[1].hist(error, bins=30, alpha=0.7, color="orange", edgecolor="black", label="Prediction error")
axes[1].axvline(0, color="red", linewidth=2, linestyle="--", label="Zero error")
axes[1].axvline(error.mean(), color="green", linewidth=2, linestyle="--", label=f"Mean error: {error.mean():.1f}s")
axes[1].set_xlabel("Error (seconds)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Prediction Error Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig("images/Xgboost_comparison.png")
plt.show()


# --- Hourly delay and error analysis ---
test_df = df.iloc[val_end:].copy()
test_df["predicted"] = prediction[:, 0]
test_df["actual"] = y_test.values
test_df["abs_error"] = np.abs(test_df["predicted"] - test_df["actual"])
test_df["hour"] = test_df["OPERATION_PLANNED_TIMESTAMP"].dt.hour
test_df["errors"] = test_df["predicted"] - test_df["actual"]

hourly = test_df.groupby("hour").agg(
    avg_actual=("actual", "mean"),
    avg_predicted=("predicted", "mean"),
    avg_error=("abs_error", "mean")
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: actual vs predicted delay by hour
axes[0].plot(hourly["hour"], hourly["avg_actual"], label="Actual", marker="o")
axes[0].plot(hourly["hour"], hourly["avg_predicted"], label="Predicted", marker="o")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Average Delay (seconds)")
axes[0].set_title("Average Delay by Hour of Day")
axes[0].set_xticks(range(0, 24))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: average absolute error by hour
axes[1].bar(hourly["hour"], hourly["avg_error"], color="tomato", alpha=0.8)
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Mean Absolute Error (seconds)")
axes[1].set_title("Model Error by Hour of Day")
axes[1].set_xticks(range(0, 24))
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("images/hourly_delay_analysis.png")
plt.show()

# ---- per-station metrics -----

per_station = test_df.groupby("OPERATING_POINT_ABBREVIATION").agg(
    avg_actual=("actual", "mean"),
    avg_predicted=("predicted", "mean"),
    avg_error=("abs_error", "mean")
).reset_index()

# Collect errors per station
station_groups = test_df.groupby("OPERATING_POINT_ABBREVIATION")["errors"]

data = [group.values for _, group in station_groups]
labels = [station for station, _ in station_groups]

plt.figure(figsize=(16, 6))

plt.boxplot(
    data,
    labels=labels,
    showfliers=False
)

plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Prediction Error (seconds)")
plt.title("Error Distribution per Station (XGBoost)")

plt.tight_layout()
plt.savefig("images/boxplot_per_station.png", dpi=150)
plt.show()



mae_error = mean_absolute_error(y_test,prediction)
rmse_error = root_mean_squared_error(y_test, prediction)
print(f"Mean absolute error: {mae_error}")
print(f"Root mean squared error: {rmse_error}")

""" for feature in ["fu3010z0", "fkl010z1", "hour_cos"]:
    dep_df = my_xgboost.shap_dependence(X_test[:500], feature)
    summary = summarize_dependence(dep_df)
    print(summary)

    plt.figure()
    plt.scatter(dep_df["feature_value"], dep_df["shap_value"], alpha=0.2)
    plt.plot(summary["feature_value"], summary["shap_value"], color="red")
    plt.title(feature)
    plt.show() """


importance_df = my_xgboost.permutation_importance(
    X_sample=X_test,
    Y_sample=y_test,
    n_repeats=5
)

print(importance_df.head(10))
