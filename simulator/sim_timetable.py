from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
from sim_topology import LINE_ORDER


# ══════════════════════════════════════════════════════════════════════════════
# TIMETABLE
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class StopEntry:
    """One planned stop from the historical data."""
    train_number:  int
    station:       str
    event_type:    str        # "arrival" | "departure"
    event_served:  bool
    stop_type:     str        # "commercialStop" | "pass"
    planned_ts:    datetime
    actual_delay:  float      # ground-truth delay in seconds
    sequence:      float
    line:          str
    category:      str        # "FV" | "RV"
    max_speed_kmh: int
    period_id:     str
    direction:     str = ""   # "BI_to_NE" | "NE_to_BI" | "" (unknown)
 
 
@dataclass
class TrainSchedule:
    """Ordered sequence of stops for one train on one day."""
    train_number: int
    line:         str
    category:     str
    max_speed_kmh: int
    event_served: bool
    period_id:    str
    stops:        list[StopEntry]
    direction:    str   # "BI_to_NE" | "NE_to_BI"
 
    @property
    def origin_time(self) -> datetime:
        deps = [s for s in self.stops if s.event_type == "departure"]
        return deps[0].planned_ts if deps else self.stops[0].planned_ts
 
 
class Timetable:
    """
    Container for all train schedules on one operational day,
    loaded from the parquet/CSV produced by the data pipeline.
    """
 
    def __init__(self, schedules: list[TrainSchedule], day: str):
        self.schedules = schedules
        self.day       = day
 
    @classmethod
    def from_parquet(cls, path: str | Path, day: str) -> "Timetable":
        p = Path(path)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        return cls._build(df, day)
 
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, day: str) -> "Timetable":
        return cls._build(df, day)
 
    @classmethod
    def _build(cls, df: pd.DataFrame, day: str) -> "Timetable":
        df = df.copy()
        df["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(df["OPERATION_PLANNED_TIMESTAMP"])
        df["OPERATION_ACTUAL_TIMESTAMP"]  = pd.to_datetime(df["OPERATION_ACTUAL_TIMESTAMP"])
        df["OPERATIONAL_DAY"]             = pd.to_datetime(df["OPERATIONAL_DAY"]).dt.date.astype(str)
 
        day_df = df[
            (df["OPERATIONAL_DAY"] == day) &
            (df["OPERATING_POINT_ABBREVIATION"].isin(LINE_ORDER))
        ].copy()
 
        if day_df.empty:
            available = sorted(df["OPERATIONAL_DAY"].unique())
            raise ValueError(f"No data for day '{day}'. Available: {available}")
 
        # Get delay in seconds
        day_df["delay_sec"] = day_df["DAILY_PLAN_OPERATIONAL_DELAY_SEC"]
 
        schedules = []
        for tn, grp in day_df.groupby("TRAIN_NUMBER"):
            grp = grp.sort_values(["OPERATION_TRAIN_RUN_SEQUENCE_NUMBER", "EVENT_TYPE"])
 
            # Drop stale stops from a previous run that leaked into this operational day.
            # These appear as very low sequence numbers (e.g. seq=1 BI departure from the
            # previous NE->BI run) before the actual corridor run begins (seq=30+).
            # Strategy: find the longest monotonically-consistent sequence of stops that
            # forms a single corridor traversal, by detecting a sequence number gap > 5
            # between the stale prefix and the real run.
            seq_vals = grp["OPERATION_TRAIN_RUN_SEQUENCE_NUMBER"].values
            if len(seq_vals) > 1:
                gaps = [seq_vals[i+1] - seq_vals[i] for i in range(len(seq_vals)-1)]
                big_gap = next((i+1 for i, g in enumerate(gaps) if g > 5), None)
                if big_gap is not None:
                    # Check if the prefix (before gap) is a single stale stop/pair
                    # vs the real run being in the prefix. Use timestamp ordering:
                    # the real run starts at the earliest planned_ts cluster.
                    prefix_ts = grp.iloc[:big_gap]["OPERATION_PLANNED_TIMESTAMP"].min()
                    suffix_ts = grp.iloc[big_gap:]["OPERATION_PLANNED_TIMESTAMP"].min()
                    if prefix_ts > suffix_ts:
                        # Prefix is later in the day → it's the stale previous-run stop
                        grp = grp.iloc[big_gap:]

            stops = []
            for _, row in grp.iterrows():
                stops.append(StopEntry(
                    train_number  = int(tn),
                    station       = row["OPERATING_POINT_ABBREVIATION"],
                    event_type    = row["EVENT_TYPE"],
                    event_served  = row["EVENT_SERVED"],
                    stop_type     = row.get("PLAN_STOP_TYPE", "commercialStop"),
                    planned_ts    = row["OPERATION_PLANNED_TIMESTAMP"],
                    period_id     = row["OPERATION_DAY_PERIOD_IDENTIFIER_COARSE"],
                    actual_delay  = float(row["delay_sec"])
                                    if pd.notna(row["delay_sec"]) else float("nan"),
                    sequence      = float(row["OPERATION_TRAIN_RUN_SEQUENCE_NUMBER"]),
                    line          = str(row.get("COMMERCIAL_LINE_NUMBER_DESIGNATION", "")),
                    category      = str(row.get("OPERATION_TRAFFIC_CATEGORY_ABBREVIATION", "")),
                    max_speed_kmh = int(row.get("PLAN_FORMATION_MAXIMAL_VELOCITY", 140)),
                ))
 
            if not stops:
                continue
 
            # Infer direction from station sequence
            station_sequence = [s.station for s in stops]
            direction = cls._infer_direction(station_sequence)
            for s in stops:
                s.direction = direction
 
            line_val = stops[0].line
            cat_val  = stops[0].category
            spd_val  = stops[0].max_speed_kmh
 
            schedules.append(TrainSchedule(
                train_number  = int(tn),
                line          = line_val,
                category      = cat_val,
                max_speed_kmh = spd_val,
                event_served  = stops[0].event_served,
                period_id     = stops[0].period_id,
                stops         = stops,
                direction     = direction,
            ))
 
        schedules.sort(key=lambda s: s.origin_time)
        return cls(schedules, day)
 
    @staticmethod
    def _infer_direction(stations: list[str]) -> str:
        bi_idx = [LINE_ORDER.index(s) for s in stations if s in LINE_ORDER]
        if not bi_idx:
            return "unknown"
        return "BI_to_NE" if bi_idx[-1] > bi_idx[0] else "NE_to_BI"
 
    def available_trains(self) -> list[int]:
        return [s.train_number for s in self.schedules]