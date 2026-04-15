import pandas as pd
from XGBoost import XGBoostBaseline
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

val_end = int(len(X) * (0.7 + 0.15))
actual_timestamps = df.iloc[val_end:]["OPERATION_PLANNED_TIMESTAMP"]
plt.plot(actual_timestamps, y_test)
plt.plot(actual_timestamps, prediction)
plt.savefig("images/Xgboost_comparison.png")
plt.show()

mae_error = mean_absolute_error(y_test,prediction)
rmse_error = root_mean_squared_error(y_test, prediction)
print(f"Mean absolute error: {mae_error}")
print(f"Root mean squared error: {rmse_error}")

importance_df = my_xgboost.feature_importance(
    feature_names=X.columns.tolist(),
    importance_type="gain"
)

print(importance_df.head(10))
