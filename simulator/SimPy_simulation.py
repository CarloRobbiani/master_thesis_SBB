import pandas as pd
import numpy as np

np.random.seed(42)

stations = ["ZUE", "BS", "OL", "BE"]
trains = ["T1", "T2", "T3"]

rows = []

for train_id in trains:
    delay = 0
    for seq, station in enumerate(stations):
        for event_type in ["arrival", "departure"]:
            
            planned_time = seq * 300 + (0 if event_type == "arrival" else 60)
            
            # Weather features
            rain = np.random.choice([0, 1])
            temp = np.random.uniform(-5, 30)

            # Generate delay (ground truth)
            base_delay = np.random.normal(30, 20)
            weather_impact = rain * 40
            delay = max(0, delay + base_delay + weather_impact)

            rows.append({
                "train_id": train_id,
                "sequence": seq,
                "station": station,
                "event_type": event_type,
                "planned_time": planned_time,
                "delay": delay,
                "rain": rain,
                "temp": temp,
                "velocity": 200
            })

df = pd.DataFrame(rows)

from sklearn.preprocessing import LabelEncoder

df = df.sort_values(["train_id", "sequence"])

# Encode categorical features
le_station = LabelEncoder()
le_event = LabelEncoder()

df["station_enc"] = le_station.fit_transform(df["station"])
df["event_enc"] = le_event.fit_transform(df["event_type"])

# Previous delay per train
df["prev_delay"] = df.groupby("train_id")["delay"].shift(1).fillna(0)

# Features and target
features = [
    "station_enc",
    "event_enc",
    "planned_time",
    "rain",
    "temp",
    "velocity",
    "prev_delay"
]

X = df[features]
y = df["delay"]

import xgboost as xgb

model = XGBoostBaseline()
model.fit(X, y)

import simpy

env = simpy.Environment()

# One resource per station
stations_sim = {
    s: simpy.Resource(env, capacity=1)
    for s in stations
}

def build_features(row, prev_delay):
    return [
        le_station.transform([row["station"]])[0],
        le_event.transform([row["event_type"]])[0],
        row["planned_time"],
        row["rain"],
        row["temp"],
        row["velocity"],
        prev_delay
    ]

def train_process(env, train_id, train_df, model):
    prev_delay = 0

    for _, row in train_df.iterrows():
        station = row["station"]

        with stations_sim[station].request() as req:
            yield req

            features = build_features(row, prev_delay)
            pred_delay = model.predict(np.array(features).reshape(1, -1))[0][0]

            prev_delay = pred_delay

            yield env.timeout(pred_delay)

            print(f"{env.now:.1f} | {train_id} at {station} | delay={pred_delay:.1f}")

for train_id, group in df.groupby("train_id"):
    env.process(train_process(env, train_id, group, model))

env.run(until=5000)