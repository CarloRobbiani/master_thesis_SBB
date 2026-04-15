# Expects all the preprocessed data in one file

import pandas as pd
#from graph_models.station_graph.training import StationMATGCNDataset
from sklearn.preprocessing import LabelEncoder

def convert_to_parquet(filepath: str):
    print("starting conversion...")
    df = pd.read_csv(filepath)
    print(df["OPERATIONAL_DAY"].max())
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

    print(train_df["OPERATIONAL_DAY"].max())

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
            pc.less_equal(table['OPERATIONAL_DAY'], '2025-12-31')
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
    Preprocess the dataframe for training of the XGBoost model
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

    # Delay feature
    df["delay_seconds"] = df["DAILY_PLAN_OPERATIONAL_DELAY_SEC"] 

    # Drop original timestamps
    df = df.drop(columns=time_cols)

    # -----------------------
    # 3. Boolean to int
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

    # -----------------------
    # 5. Handle missing values
    # -----------------------
    print("handling missing values...")

    # -----------------------
    # 6. Split X / y
    # -----------------------
    print("Splitting features...")
    FEATURE_COLS = [
        "EVENT_TYPE",
        "EVENT_SERVED",
        "PLAN_STOP_TYPE", 
        "OPERATION_DAY_PERIOD_IDENTIFIER_COARSE",
        'OPERATION_TRAFFIC_CATEGORY_ABBREVIATION',
        'PLAN_FORMATION_MAXIMAL_VELOCITY',
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        'tre200s0', 'fkl010z1', 'fu3010z0', 'rre150z0',
        'htoauts0', 'hto000d0']
    X = df[FEATURE_COLS]
    y = df[target_column]
    print("finished preprocessing...")

    return X, y

def time_split(X, y, train_size=0.7, val_size=0.15):
    """
    Generates a time split for the XGBoost dataframe.
    X,y: time sorted dataframe splits
    """
    n = len(X)
    
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    #filter_parquet_file("data/train_data.parquet")
    #filter_stations()
    full_pipeline_preparing("data/train_data.csv")