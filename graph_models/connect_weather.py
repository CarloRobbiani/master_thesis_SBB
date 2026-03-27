import pandas as pd

def connect_weather_stations(weather_file = None, connectionfile=None, train_file="data/train_data.parquet"):
    json_weather = pd.read_json("graph_models\weather_connection.json")

    neu_weather = prepare_neuchatel()
    neu_weather = neu_weather.sort_values("reference_timestamp")
    grenchen_weather = prepare_grenchen()
    grenchen_weather = grenchen_weather.sort_values("reference_timestamp")

    train_df = pd.read_parquet(train_file)
    train_df["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(train_df["OPERATION_PLANNED_TIMESTAMP"], format="%Y-%m-%d %H:%M:%S.%f %z", utc=True)
    train_df["OPERATION_PLANNED_TIMESTAMP"] = train_df["OPERATION_PLANNED_TIMESTAMP"].dt.tz_localize(None) # remove timezone
    train_df["OPERATION_PLANNED_TIMESTAMP"] = train_df["OPERATION_PLANNED_TIMESTAMP"].dt.floor("min") # round down to minute
    train_df = train_df.sort_values("OPERATION_PLANNED_TIMESTAMP")
    train_df = train_df.dropna()
    

    #split datasets
    train_df_ne = train_df[train_df["OPERATING_POINT_ABBREVIATION"].isin(json_weather["Neuchatel"])]
    train_df_gre = train_df[train_df["OPERATING_POINT_ABBREVIATION"].isin(json_weather["Grenchen"])]

    merged_ne = pd.merge_asof(
        train_df_ne,
        neu_weather,
        left_on='OPERATION_PLANNED_TIMESTAMP',
        right_on='reference_timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('10min')
    )
    merged_gre = pd.merge_asof(
        train_df_gre,
        grenchen_weather,
        left_on='OPERATION_PLANNED_TIMESTAMP',
        right_on='reference_timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('10min')
    )
    
    final = pd.concat([merged_gre, merged_ne])
    final.to_parquet("data/train_data_weather.parquet")
    return final


def prepare_neuchatel():

    # read in data
    df = pd.read_csv("data\weather/neu_pre_temp_wind_sun.csv", delimiter=";", header=0)

    cols_to_keep = ["station_abbr", "reference_timestamp", "date", "tre200s0", "fkl010z1", "fu3010z0", "rre150z0"]

    #filter date
    df['reference_timestamp'] = pd.to_datetime(df['reference_timestamp'], format="%d.%m.%Y %H:%M")
    df['date'] = df['reference_timestamp']
    mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-01')
    filtered_df = df[mask]

    filtered_df = filtered_df[cols_to_keep]
    filtered_df["days"] = filtered_df['date'].dt.date
    #filtered_df.head()

    # Merge with snow data
    neu_snow = pd.read_csv("data\weather/neu_snow.csv", delimiter=";", header=0)

    cols_to_keep = ["station_abbr", "date", "reference_timestamp", "hto000d0"] # hto000d0 Schneehöhe um 6 UTC

    neu_snow['reference_timestamp'] = pd.to_datetime(neu_snow['reference_timestamp'], format="%d.%m.%Y %H:%M")
    neu_snow['date'] = neu_snow['reference_timestamp']
    mask_snow = (neu_snow['date'] >= '2025-01-01') & (neu_snow['date'] <= '2025-06-01')
    filtered_df_snow = neu_snow[mask_snow]
    filtered_neu_snow = filtered_df_snow[cols_to_keep]
    filtered_neu_snow["days"] = filtered_neu_snow["date"].dt.date
    #print(filtered_neu_snow.head())


    merged_neu = pd.merge(filtered_df, filtered_neu_snow[['days', 'hto000d0']], on='days', how='left') 
    return merged_neu

def prepare_grenchen():

    # read in data
    df = pd.read_csv("data\weather/grenchen_pre_temp_wind_sun_snow.csv", delimiter=";", header=0)

    cols_to_keep = ["station_abbr", "reference_timestamp", "tre200s0", "fkl010z1", "fu3010z0", "rre150z0", "htoauts0"]

    #filter date
    df['reference_timestamp'] = pd.to_datetime(df['reference_timestamp'], format="%d.%m.%Y %H:%M")
    df['date'] = df['reference_timestamp']
    mask = (df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-01')
    filtered_df = df[mask]

    filtered_df = filtered_df[cols_to_keep]
    #filtered_df.head()

    

    return filtered_df


if __name__ == "__main__":
    
    df = connect_weather_stations()
