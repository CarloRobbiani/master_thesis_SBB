import pandas as pd
import zipfile
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import shutil


# === CONFIG ===
GTFS_ZIP = "C:\Master\Masterarbeit\sumo_files\GTFS.zip" #pd.read_csv("CONFIG.txt")
OUTPUT_ZIP = "GTFS_filtered.zip"

# Your station names (adjust if needed!)
station_abbrev = pd.read_csv("data/station_list.csv")
train_stations = pd.read_csv("data/train_stations.csv", delimiter=";")
stations = station_abbrev.iloc[0].tolist()

TARGET_STOPS = train_stations[train_stations["OPERATING_POINT_ABBREVIATION"].isin(stations)]["BP_Name"]
TARGET_STOPS = TARGET_STOPS.str.strip()

# === STEP 1: unzip ===
with zipfile.ZipFile(GTFS_ZIP, 'r') as zip_ref:
    zip_ref.extractall("gtfs_tmp")

# === STEP 2: load files ===
stops = pd.read_csv("gtfs_tmp/stops.txt")
stop_times = pd.read_csv("gtfs_tmp/stop_times.txt")
trips = pd.read_csv("gtfs_tmp/trips.txt")
routes = pd.read_csv("gtfs_tmp/routes.txt")
stops["stop_name_clean"] = stops["stop_name"].str.strip()

# === STEP 3: filter stops ===
filtered_stops = stops[stops["stop_name_clean"].isin(TARGET_STOPS)]

stop_ids = set(filtered_stops["stop_id"])

# === STEP 5: find stop_times touching your target stops ===
touching_stop_times = stop_times[stop_times["stop_id"].isin(stop_ids)]

# === STEP: find trips that serve at least N of your target stops ===
trip_stop_counts = touching_stop_times.groupby("trip_id")["stop_id"].nunique()
VALID_TRIPS = trip_stop_counts[trip_stop_counts >= 2].index  # lower threshold too

# === STEP 5b: keep ALL stop_times for valid trips (not just target stops) ===
filtered_stop_times = stop_times[stop_times["trip_id"].isin(VALID_TRIPS)]
filtered_stop_times = filtered_stop_times.sort_values(["trip_id", "stop_sequence"])

all_stop_ids_used = set(filtered_stop_times["stop_id"])
filtered_stops = stops[stops["stop_id"].isin(all_stop_ids_used)]

# === STEP 6: filter trips ===
trip_ids = set(filtered_stop_times["trip_id"])
filtered_trips = trips[trips["trip_id"].isin(trip_ids)]

# === STEP 7: filter routes ===
route_ids = set(filtered_trips["route_id"])
filtered_routes = routes[routes["route_id"].isin(route_ids)]

# === STEP 8: save cleaned GTFS ===
os.makedirs("gtfs_filtered", exist_ok=True)

filtered_stops.to_csv("gtfs_filtered/stops.txt", index=False)
filtered_stop_times.to_csv("gtfs_filtered/stop_times.txt", index=False)
filtered_trips.to_csv("gtfs_filtered/trips.txt", index=False)
filtered_routes.to_csv("gtfs_filtered/routes.txt", index=False)

# copy required files if present
for file in ["agency.txt", "calendar.txt", "calendar_dates.txt"]:
    src = f"gtfs_tmp/{file}"
    if os.path.exists(src):
        shutil.copy(src, f"gtfs_filtered/{file}")

# === STEP 9: re-zip ===
with zipfile.ZipFile(OUTPUT_ZIP, 'w') as zipf:
    for root, dirs, files in os.walk("gtfs_filtered"):
        for file in files:
            zipf.write(os.path.join(root, file), file)

print("Filtered GTFS written to", OUTPUT_ZIP)