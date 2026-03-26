# Expects all the preprocessed data in one file

import pandas as pd
#from graph_models.station_graph.training import StationMATGCNDataset
from sklearn.preprocessing import LabelEncoder

def preprocess_train(df, target_column="OPERATIONAL_PUNCTUAL"):
    df = df.copy()

    # -----------------------
    # 1. Drop useless columns
    # -----------------------
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # -----------------------
    # 2. Handle timestamps
    # -----------------------
    time_cols = ["OPERATION_PLANNED_TIMESTAMP", "OPERATION_ACTUAL_TIMESTAMP"]

    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        df[f"{col}_hour"] = df[col].dt.hour
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_weekday"] = df[col].dt.weekday

    # Delay feature (very useful!)
    df["delay_seconds"] = (
        (df["OPERATION_ACTUAL_TIMESTAMP"] - df["OPERATION_PLANNED_TIMESTAMP"])
        .dt.total_seconds()
    )

    # Drop original timestamps
    df = df.drop(columns=time_cols)

    # -----------------------
    # 3. Boolean → int
    # -----------------------
    df["EVENT_SERVED"] = df["EVENT_SERVED"].astype(int)

    # -----------------------
    # 4. Categorical encoding
    # -----------------------
    cat_cols = df.select_dtypes(include="object").columns

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # -----------------------
    # 5. Handle missing values
    # -----------------------
    df = df.fillna(-1)

    # -----------------------
    # 6. Split X / y
    # -----------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y, encoders





def create_df_tensors(df: pd.DataFrame):
    # Define feature groups for station MATGCN:

    station_feature_cols = [
        "delay_mean",
        "num_trains",
        "platform_occ"
    ]

    external_cols = [
        "temperature",
        "precipitation",
        "hour_sin",
        "hour_cos"
    ]

    target_col = "target_delay"

    # Sort the data

    df = df.sort_values(["timestamp", "station_id"])

    # Extract dimensions
    timestamps = df["timestamp"].unique()
    stations = df["station_id"].unique()

    T_total = len(timestamps)
    N = len(stations)

    F = len(station_feature_cols)
    E = len(external_cols)

    station_tensor = (
        df.pivot_table(
            index="timestamp",
            columns="station_id",
            values=station_feature_cols
        )
        .values
        .reshape(T_total, N, F)
    )

    target_tensor = (
        df.pivot_table(
            index="timestamp",
            columns="station_id",
            values=target_col
        )
        .values
    )

    external_df = (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
    )

    external_tensor = external_df[external_cols].values

    return station_tensor, external_tensor, target_tensor


