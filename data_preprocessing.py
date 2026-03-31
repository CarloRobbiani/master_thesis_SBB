# Expects all the preprocessed data in one file

import pandas as pd
#from graph_models.station_graph.training import StationMATGCNDataset
from sklearn.preprocessing import LabelEncoder

def convert_to_parquet(filepath: str):
    print("starting conversion...")
    df = pd.read_csv(filepath)
    df.to_parquet("data/train_data.parquet")
    print("conversion finished!")

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow
import numpy as np

def filter_stations(station_list = "data\station_list.csv", file_path = "data/train_data.parquet"):
    """
    Filter the dataframe based on the stations
    """
    print("filtering stations...")
    train_df = pd.read_parquet(file_path)

    station_df = pd.read_csv(station_list, header=None)

    stations = station_df.iloc[1].tolist() # second row is list of station abbreviations
    train_df = train_df[train_df["OPERATING_POINT_ABBREVIATION"].isin(stations)]

    train_df.to_parquet("data/train_data.parquet")

def filter_parquet_file(filepath: str):
    """
    Filter the dataframe based on the date
    """
    print("starting filtering...")

    parquet_file = pq.ParquetFile(filepath)
    writer = None
    
    for batch in parquet_file.iter_batches():
        table = pyarrow.Table.from_batches([batch])

        mask = pc.and_(
            pc.greater_equal(table['OPERATIONAL_DAY'], '2025-01-01'),
            pc.less_equal(table['OPERATIONAL_DAY'], '2025-06-01')
        )

        filtered = table.filter(mask)

        if writer is None:
            writer = pq.ParquetWriter("data/train_data_small.parquet", filtered.schema)

        writer.write_table(filtered)

    if writer:
        writer.close()

    print("Done filtering!")


def full_pipeline_preparing(csv_filepath):
    """
    This function starts from the .csv file and filters it based on stations and dates and saves it as .parquet
    """
    convert_to_parquet(csv_filepath)

    filter_stations()

    filter_parquet_file("data/train_data.parquet")



def preprocess_train(df, target_column="OPERATIONAL_PUNCTUAL"):
    """
    Preprocess the dataframe for training
    """

    # -----------------------
    # 1. Drop useless columns
    # -----------------------
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # -----------------------
    # 2. Handle timestamps
    # -----------------------
    print("handling timestamps...")
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
    print("converting boolean to int...")
    df["EVENT_SERVED"] = df["EVENT_SERVED"].astype(int)

    # -----------------------
    # 4. Categorical encoding
    # -----------------------
    print("categoircal encoding...")
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    """ encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le """

    # -----------------------
    # 5. Handle missing values
    # -----------------------
    print("handling missing values...")
    df = df.fillna(-1)

    # -----------------------
    # 6. Split X / y
    # -----------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]
    print("finished preprocessing...")

    return X, y



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
    stations =["BI","TUE","TWN","LIG","CHAV","POU","NV","LD","CRNE","CORN","SBLB","NE"]

    T_total = len(timestamps)
    N = len(stations)

    F = len(station_feature_cols)
    E = len(external_cols)

    station_tensor = np.stack([
        df.pivot_table(index="timestamp", columns="station_id", values=station_feature_cols)
          .reindex(index=timestamps, columns=stations)
          .values
        for col in station_feature_cols
    ], axis=-1)

    target_tensor = (
    df.pivot_table(index="timestamp", columns="station_id", values=target_col)
    .reindex(index=timestamps, columns=stations)
    .values
    )

    external_df = (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
    )

    external_tensor = external_df[external_cols].values

    return station_tensor, external_tensor, target_tensor


if __name__ == "__main__":

    #filter_parquet_file("data/train_data.parquet")
    #filter_stations()
    full_pipeline_preparing("data/train_data.csv")