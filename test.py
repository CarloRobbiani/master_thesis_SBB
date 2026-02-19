import pandas as pd

# maybe install pip install gtfs-realtime-bindings protobuf pandas
from google.transit import gtfs_realtime_pb2
from google.protobuf.json_format import MessageToDict

# Load .pb file
feed = gtfs_realtime_pb2.FeedMessage()

with open("GTFSR_20260217100801.pb", "rb") as f:
    feed.ParseFromString(f.read())

print(len(feed.entity))
import pandas as pd

records = []

for entity in feed.entity:
    if entity.HasField("trip_update"):
        trip = entity.trip_update.trip
        
        for stop_time in entity.trip_update.stop_time_update:
            
            record = {
                "trip_id": trip.trip_id,
                "route_id": trip.route_id,
                "stop_id": stop_time.stop_id,
                "arrival_delay": (
                    stop_time.arrival.delay 
                    if stop_time.HasField("arrival") else None
                ),
                "departure_delay": (
                    stop_time.departure.delay 
                    if stop_time.HasField("departure") else None
                )
            }
            
            records.append(record)

df_rt = pd.DataFrame(records)
df_rt["arrival_delay_min"] = df_rt["arrival_delay"] / 60
#print(df_rt[df_rt["arrival_delay_min"]  > 0.0])

stop_times = pd.read_csv("stop_times.txt")

df = df_rt.merge(
    stop_times,
    on=["trip_id", "stop_id"],
    how="left"
)

df["delay_min"] = df["arrival_delay"] / 60
df = df[df["delay_min"].notna()]
df = df.sort_values(["trip_id", "stop_sequence"])

df["prev_delay"] = (
    df.groupby("trip_id")["delay_min"]
    .shift(1)
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

features = [
    "stop_sequence",
    "prev_delay"
]

df = df.dropna(subset=features + ["delay_min"])

X = df[features]
y = df["delay_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

print("Score:", model.score(X_test, y_test))


