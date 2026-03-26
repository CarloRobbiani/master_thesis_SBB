import pandas as pd
from XGBoost import XGBoostBaseline
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data_preprocessing import preprocess_train
my_xgboost = XGBoostBaseline()

df = pd.read_csv("data/train_data.csv")

df.fillna(0)

target_col = 'DAILY_PLAN_OPERATIONAL_DELAY_SEC'
X,Y, encoders = preprocess_train(df, target_column=target_col)



my_xgboost.fit(X, Y)

prediction = my_xgboost.predict(X)

""" error = prediction-Y

print(error) """

importance_df = my_xgboost.feature_importance(
    feature_names=X.columns.tolist(),
    importance_type="gain"
)

print(importance_df.head(10))
