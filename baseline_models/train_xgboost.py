import pandas as pd
import polars as pl
from XGBoost import XGBoostBaseline
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data_preprocessing import preprocess_train
my_xgboost = XGBoostBaseline()


df = pd.read_parquet("data/train_data.parquet")

# Reduce data types to save space
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

target_col = 'DAILY_PLAN_OPERATIONAL_DELAY_SEC'
X,Y = preprocess_train(df, target_column=target_col)



my_xgboost.fit(X, Y)

prediction = my_xgboost.predict(X)
print(prediction)

""" error = prediction-Y

print(error) """

importance_df = my_xgboost.feature_importance(
    feature_names=X.columns.tolist(),
    importance_type="gain"
)

print(importance_df.head(10))
