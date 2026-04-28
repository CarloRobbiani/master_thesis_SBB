from sim_weather import WeatherConditions
from sim_events import SimEvent, ConflictEvent
from sim_topology import SEGMENTS, build_planned_segment_times
from sim_timetable import Timetable
from sim_result import SimResult
from sim_processes import TrainProcess
from typing import Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
import simpy
import pickle
import os


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
 
class RailwaySimulator:
    """
    SimPy-based railway simulator for the Biel/Bienne ↔ Neuchâtel corridor.
 
    Parameters
    ──────────
    timetable : Timetable
        Loaded via Timetable.from_parquet() or Timetable.from_dataframe().
    weather   : WeatherConditions
        Weather for this run.  Defaults to clear/calm if not provided.
    seed      : int | None
        Random seed for switch failure draws (for reproducibility).
 
    Example
    ───────
        tt  = Timetable.from_parquet("data/train_data_weather.parquet", "2025-01-15")
        sim = RailwaySimulator(tt, WeatherConditions(temp_c=-5, snow_cm=12))
        result = sim.run()
        print(result.summary())
        result.plot()
        result.plot_space_time()
        result.to_csv("output.csv")
    """
 
    def __init__(
        self,
        PLANNED_SEGMENT_TIMES,
        timetable: Timetable,
        weather:   WeatherConditions = WeatherConditions(),
        seed:      Optional[int]     = None,
    ):
        self.timetable = timetable
        self.weather   = weather
        if seed is not None:
            random.seed(seed)
        self.PLANNED_SEGMENT_TIMES = PLANNED_SEGMENT_TIMES
 
    def run(self) -> SimResult:
        """Execute the simulation and return a SimResult."""
        tt        = self.timetable
        day_start = datetime.strptime(tt.day, "%Y-%m-%d")
 
        env = simpy.Environment()
 
        # ── Build segment resources ────────────────────────────────────────────
        # Single-track segments get capacity=1 (only one train at a time).
        # Double-track segments are unconstrained (capacity = large number).
        resources: dict[tuple[str, str], simpy.Resource] = {}
        single_track_pairs: set[frozenset] = set()
 
        for key, seg in SEGMENTS.items():
            if not seg.double_track:
                # Use a single shared resource for both directions of a single-track segment
                pair = frozenset(key)
                if pair not in single_track_pairs:
                    single_track_pairs.add(pair)
                    res = simpy.Resource(env, capacity=1)
                    resources[key]             = res
                    resources[(key[1], key[0])] = res   # same resource for reverse direction
 
        # ── Shared output lists ────────────────────────────────────────────────
        sim_events:   list[SimEvent]     = []
        conflict_log: list[ConflictEvent] = []
 
        # ── Launch one process per train ────────────────────────────────────────
        for schedule in tt.schedules:
            """ if schedule.train_number != 1506:
                continue """
            proc = TrainProcess(
                PLANNED_SEGMENT_TIMES = self.PLANNED_SEGMENT_TIMES,
                env          = env,
                schedule     = schedule,
                resources    = resources,
                weather      = self.weather,
                sim_events   = sim_events,
                conflict_log = conflict_log,
                day_start    = day_start,
            )
            env.process(proc.run())
 
        # Run until all processes complete (or 30 hours to be safe)
        env.run(until=30 * 3600)
 
        # Sort events chronologically
        sim_events.sort(key=lambda e: e.simulated_ts)
 
        return SimResult(
            events    = sim_events,
            conflicts = conflict_log,
            weather   = self.weather,
            day       = tt.day,
        )
    


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: SENSITIVITY SWEEP
# ══════════════════════════════════════════════════════════════════════════════
 
def weather_sensitivity(
    timetable: Timetable,
    param:     str = "wind_ms",
    values:    Optional[list] = None,
    seed:      int = 42,
) -> pd.DataFrame:
    """
    Run the simulator across a range of one weather parameter and return
    a summary DataFrame showing how MAE, RMSE and punctuality change.
 
    Parameters
    ──────────
    timetable : Timetable
    param     : one of "temp_c", "wind_ms", "precip_mm", "snow_cm", "visibility_m"
    values    : list of values to try; if None, uses sensible defaults per param
    seed      : random seed
 
    Example
    ───────
        df = weather_sensitivity(tt, param="snow_cm", values=[0, 5, 10, 20, 40])
        print(df)
    """
    defaults = {
        "temp_c":       [15, 5, 0, -5, -10, -20],
        "wind_ms":      [0, 5, 10, 15, 20, 25, 30],
        "precip_mm":    [0, 1, 3, 5, 10, 20],
        "snow_cm":      [0, 2, 5, 10, 20, 40],
        "visibility_m": [10000, 2000, 1000, 500, 200, 100],
    }
    if values is None:
        values = defaults.get(param, [0, 5, 10])
 
    rows = []
    for v in values:
        kwargs = {param: v}
        w   = WeatherConditions(**kwargs)
        sim = RailwaySimulator(PLANNED_SEGMENT_TIMES, timetable, w, seed=seed)
        res = sim.run()
        acc = res.accuracy()
        pct = res.punctuality()
        rows.append({
            param:             v,
            "mae_sec":         acc["mae"],
            "rmse_sec":        acc["rmse"],
            "punctuality_sim": pct["simulated"],
            "n_conflicts":     len(res.conflicts),
            "n_events":        acc["n"],
        })
 
    return pd.DataFrame(rows)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — demo when run directly
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    import sys
 
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_data_weather.parquet"
    day_arg   = sys.argv[2] if len(sys.argv) > 2 else None
 
    print(f"Loading timetable from {data_path} …")
 
    # Use CSV if parquet not available
    p = Path(data_path)
    if p.suffix == ".parquet":
        df_raw = pd.read_parquet(p)
    else:
        df_raw = pd.read_csv(p)
 
    df_raw["OPERATIONAL_DAY"] = pd.to_datetime(
        df_raw["OPERATIONAL_DAY"]
    ).dt.date.astype(str)
    available = sorted(df_raw["OPERATIONAL_DAY"].unique())
    day = day_arg or available[0]
    #day = "2025-01-06"


    if os.path.isfile("simulator/timetable.pkl"):
        with open("simulator/timetable.pkl", "rb") as f:
            PLANNED_SEGMENT_TIMES = pickle.load(f)
    else: # file does not exist
        PLANNED_SEGMENT_TIMES = build_planned_segment_times(df_raw)
        with open("simulator/timetable.pkl", "wb") as fp:
            pickle.dump(PLANNED_SEGMENT_TIMES, fp)

    # Manual corrections for asymmetric segment times caused by cross-run data contamination.
    # In each case the correct value is taken from the cleaner direction.
    PLANNED_SEGMENT_TIMES.update({
    # IC5
    ("TUE", "BI",  "IC5"): 240,  # was wrongly set to 180; real planned = 240s
    ("NE",  "SBL", "IC5"): 180,  # was wrongly set to 120; real planned = 180s
    ("LIG", "TWN", "IC5"): 120,  # correct ✓
    ("TWN", "LIG", "IC5"): 120,
    ("LIG", "NV", "IC5"): 120,
    ("NV", "LIG", "IC5"): 120,
    ("SBL", "CORN","IC5"): 120,  # correct ✓
    # R13
    ("NV",  "LIG", "R13"): 180,
    ("NE",  "SBL", "R13"): 180,
    # R16
    ("NV",  "LIG", "R16"): 180,
})

    
    
    print(f"Simulating day: {day}")
 
    tt = Timetable.from_dataframe(df_raw, day)
    print(f"Loaded {len(tt.schedules)} train schedules\n")
 
    # ── Run 1: clear weather ──────────────────────────────────────────────────
    print("══ Run 1: clear weather ══")
    sim1 = RailwaySimulator(PLANNED_SEGMENT_TIMES, tt, WeatherConditions(), seed=42)
    r1   = sim1.run()
    r1.to_csv("simulator/normal_weather.csv")
    print(r1.summary())
 
    # ── Run 2: winter storm ───────────────────────────────────────────────────
    print("══ Run 2: winter storm ══")
    storm = WeatherConditions(temp_c=-4, wind_ms=22, precip_mm=8, snow_cm=15, visibility_m=300)
    sim2  = RailwaySimulator(PLANNED_SEGMENT_TIMES, tt, storm, seed=42)
    r2    = sim2.run()
    #r2.to_csv("winter_strom.csv")
    print(r2.summary())
 
    # ── Run 3: wind sensitivity sweep ────────────────────────────────────────
    print("══ Wind sensitivity sweep ══")
    sweep = weather_sensitivity(tt, param="wind_ms", seed=42)
    print(sweep.to_string(index=False))