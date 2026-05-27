from sim_weather import WeatherConditions, WeatherTimeline
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
import json
from sim_GCN import GCNPredictor


# ------------------------------------------------------------------------------
# MAIN SIMULATOR
# ------------------------------------------------------------------------------
 
class RailwaySimulator:
    """
    SimPy-based railway simulator for the Biel/Bienne ↔ Neuchâtel corridor.
 
    Parameters
    ----------
    timetable : Timetable
        Loaded via Timetable.from_parquet() or Timetable.from_dataframe().
    weather   : WeatherConditions
        Weather for this run.  Defaults to clear/calm if not provided.
    seed      : int | None
        Random seed for switch failure draws (for reproducibility).
    param_type : str
        Type of params to load from json file. Either "normal" or "learned"
    inject_delay : tuple
        Tuple that specifies where to inject how much delay (Station, Train, Delay)
    use_GCN : Optional[bool]
        Boolean that specifies if we want to use the GCN to predict travel times
 
    Example
    -------
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
        weather:   WeatherTimeline | WeatherConditions = WeatherConditions(),
        seed:      Optional[int]     = None,
        param_type: str = "normal" ,
        inject_delay : Optional[tuple] = None,
        GCN = None,
        df_day: Optional[pd.DataFrame] = None,
        ):
        self.timetable = timetable
        # Normalise to WeatherTimeline so the rest of the code is uniform
        if isinstance(weather, WeatherConditions):
            self.weather = WeatherTimeline.from_single(weather, speed_factors)
        else:
            self.weather = weather
        if seed is not None:
            random.seed(seed)
        self.PLANNED_SEGMENT_TIMES = PLANNED_SEGMENT_TIMES
        self.param_type = param_type

        self.inject_delay = inject_delay

        self.GCN = GCN
        self.df_day = df_day
 
    def run(self) -> SimResult:
        """Execute the simulation and return a SimResult."""
        tt        = self.timetable
        day_start = datetime.strptime(tt.day, "%Y-%m-%d")
 
        env = simpy.Environment()
 
        # -- Build segment resources --------------------------------------------
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
 
        # -- Shared output lists ------------------------------------------------
        sim_events:   list[SimEvent]     = []
        conflict_log: list[ConflictEvent] = []
 
        # -- Launch one process per train ----------------------------------------
        # Pre-compute GCN entry delays once, before launching processes
        station_priors: dict[str, float] | None = None
        if self.GCN is not None and self.df_day is not None:
            station_priors = self.GCN.predict_station_priors(self.df_day)
            # Diagnostic: print what the GCN predicts per station
            #for st, delay in station_priors.items():
             #   print(f"  GCN prior | {st}: {delay:+.1f}s")
        else: entry_delays = None

     
        for schedule in tt.schedules:            
            
            proc = TrainProcess(
                PLANNED_SEGMENT_TIMES = self.PLANNED_SEGMENT_TIMES,
                env          = env,
                schedule     = schedule,
                resources    = resources,
                weather      = self.weather,   # WeatherTimeline
                sim_events   = sim_events,
                conflict_log = conflict_log,
                day_start    = day_start,
                param_type   = self.param_type,
                inject_delay = self.inject_delay,
                entry_delays_sec = station_priors
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
    


# ------------------------------------------------------------------------------
# CONVENIENCE: SENSITIVITY SWEEP
# ------------------------------------------------------------------------------
 
def weather_sensitivity(
    timetable: Timetable,
    param:     str = "fu3010z0",
    values:    Optional[list] = None,
    seed:      int = 42,
) -> pd.DataFrame:
    """
    Run the simulator across a range of one weather parameter and return
    a summary DataFrame showing how MAE, RMSE and punctuality change.
 
    Parameters
    ----------
    timetable : Timetable
    param     : one of "temp_c", "wind_ms", "precip_mm", "snow_cm", "visibility_m"
    values    : list of values to try; if None, uses sensible defaults per param
    seed      : random seed
 
    Example
    -------
        df = weather_sensitivity(tt, param="snow_cm", values=[0, 5, 10, 20, 40])
        print(df)
    """
    defaults = {
        "tree200s0":       [15, 5, 0, -5, -10, -20],
        "fu3010z0":      [0, 5, 10, 15, 20, 25, 30],
        "rre150z0":    [0, 1, 3, 5, 10, 20],
        "htoauts0":      [0, 2, 5, 10, 20, 40],
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

def run_ablation(
    PLANNED_SEGMENT_TIMES: dict,
    timetable:   Timetable,
    weather:     WeatherTimeline,
    gcn:         "GCNPredictor",
    df_day:      pd.DataFrame,
    seed:        int = 42,
    n_runs:      int = 5,
) -> pd.DataFrame:

    """
    Three-way ablation to quantify what each component contributes.
    Conditions
    cold_start      : zero entry delay, SBI-tuned propagation only
    gcn_prior       : GCN station prior seeds entry delay, same propagation
    gcn_prior_sbi   : same as gcn_prior but with learned SBI parameters
    Each condition is repeated n_runs times (different random seeds) so that
    the stochastic dwell/travel noise averages out.  Returns a tidy DataFrame
    with one row per (condition, run).
    """


    conditions = [
        ("cold_start",    None,  "normal"),
        ("gcn_prior",     gcn,   "normal"),
        ("gcn_prior_sbi", gcn,   "learned"),
    ]
    rows = []
    for condition, gcn_arg, param_type in conditions:
        for i in range(n_runs):
            run_seed = seed + i
            sim = RailwaySimulator(
                PLANNED_SEGMENT_TIMES = PLANNED_SEGMENT_TIMES,
                timetable  = timetable,
                weather    = weather,
                seed       = run_seed,
                param_type = param_type,
                GCN        = gcn_arg,
                df_day     = df_day,
          )
            result = sim.run()
            acc = result.accuracy()
            pct = result.punctuality()
            rows.append({
                "condition":       condition,
                "run":             i,
                "seed":            run_seed,
                "mae_sec":         acc["mae"],
                "rmse_sec":        acc["rmse"],
                "punctuality_sim": pct["simulated"],
                "punctuality_act": pct["actual"],
                "n_conflicts":     len(result.conflicts),
                "n_events":        acc["n"],
            })
        print(f"  {condition}: done ({n_runs} runs)")
    return pd.DataFrame(rows)


 
 
# ------------------------------------------------------------------------------
# ENTRY POINT — demo when run directly
# ------------------------------------------------------------------------------
 
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
    else: # file does not exist built it from a new
        PLANNED_SEGMENT_TIMES = build_planned_segment_times(df_raw)
        with open("simulator/timetable.pkl", "wb") as fp:
            pickle.dump(PLANNED_SEGMENT_TIMES, fp)

    # Manual corrections for asymmetric segment times caused by cross-run data contamination.
    # In each case the correct value is taken from the cleaner direction.
    PLANNED_SEGMENT_TIMES.update({
    # IC5
    ("TUE", "BI",  "IC5"): 240, 
    ("NE",  "SBL", "IC5"): 180, 
    ("LIG", "TWN", "IC5"): 120,  
    ("TWN", "LIG", "IC5"): 120,
    ("LIG", "NV", "IC5"): 120,
    ("NV", "LIG", "IC5"): 120,
    ("SBL", "CORN","IC5"): 120, 
    # R13
    ("NV",  "LIG", "R13"): 180,
    ("NE",  "SBL", "R13"): 180,
    # R16
    ("NV",  "LIG", "R16"): 180,
})

    
    
    print(f"Simulating day: {day}")
 
    tt = Timetable.from_dataframe(df_raw, day)
    print(f"Loaded {len(tt.schedules)} train schedules")

    # Load the json file containing information on the speed factors
    
    with open(os.path.join("simulator", "weather_factors.json")) as f:
        speed_factors = json.load(f)

 
    # -- Build time-varying weather timeline from real MeteoSwiss data ---------
    # Each snapshot corresponds to one unique timestamp in the day's data,
    # giving sub-hourly resolution that matches your measurement cadence.
    weather_timeline = WeatherTimeline(speed_factors=speed_factors, snapshots=None).from_day_dataframe(df_raw, day, speed_factors)
    print(f"Weather: {weather_timeline}")

    # -- Run 1: real time-varying weather from dataset -------------------------
    print("-- Run 1: real weather timeline --")
    """ # Take the mean weather from that day
    weather_row = day_rows[["tre200s0", "fkl010z1", "fu3010z0", 
                            "rre150z0", "htoauts0", "hto000d0"]].mean()
    weather = WeatherConditions.from_meteoswiss_row(weather_row) 
    """
    sim1 = RailwaySimulator(PLANNED_SEGMENT_TIMES, tt, weather_timeline, seed=42)
    r1   = sim1.run()
    r1.to_csv("simulator/data/normal_weather.csv")
    #print(r1.summary())
 
    # -- Run 2: winter storm (static, for comparison) --------------------------
    print("-- Run 2: winter storm --")
    storm = WeatherConditions(tre200s0=-4, fu3010z0=22, rre150z0=8, htoauts0=10, speed_factors=speed_factors)
    sim2  = RailwaySimulator(PLANNED_SEGMENT_TIMES, tt, storm, seed=42)
    r2    = sim2.run()
    r2.to_csv("simulator/data/winter_storm.csv")
    #print(r2.summary())

    # -- Run 3: inject Delay --------------------------
    print("-- Run 3: Inject delay--")
    inject_delay = ("TWN", None, 120) # (Station, Train/Line, Delay(s))
    storm = WeatherConditions(tre200s0=15, fu3010z0=0, rre150z0=0, htoauts0=0, speed_factors=speed_factors)
    sim2  = RailwaySimulator(PLANNED_SEGMENT_TIMES, tt, storm, seed=42, inject_delay=inject_delay)
    r2    = sim2.run()
    r2.to_csv("simulator/data/injected_delay.csv")
    #print(r2.summary())

    # -- Run 4: use model for initial delay -----------------
    gcn = GCNPredictor(
    model_path        = os.path.join("graph_models", "station_graph", "best_matgcn.pt"),
    scaler_path       = os.path.join("data", "feat_scaler.pkl"),
    stats_path        = os.path.join("data", "train_stats.json"),
    target_scaler_path= os.path.join("data", "target_scaler.pkl"),
    station_list_path = os.path.join("data", "station_list.csv"),
    )

    sim4 = RailwaySimulator(PLANNED_SEGMENT_TIMES, tt, weather_timeline, 
                            seed=42, GCN = gcn, df_day = df_raw[df_raw["OPERATIONAL_DAY"] == day])
    r4   = sim4.run()
    r4.to_csv("simulator/data/sim_with_GCN.csv")
    #print(r4.summary())

    # -- Run 5: use learned SBI params --------------------

    #days = df_raw.iloc[:int(len(df_raw) *0.3)]
    days_unique = available[:int(len(available) *0.5)]
    #days_unique = days["OPERATIONAL_DAY"].unique()

    df_list = []
    for day in days_unique:
        print(day)
        tt_day = Timetable.from_dataframe(df_raw, day)   #rebuild per day
        weather_timeline_sbi = WeatherTimeline(
            speed_factors=speed_factors, snapshots=None, param_type="normal"
        ).from_day_dataframe(df_raw, day, speed_factors)
        
        sim5 = RailwaySimulator(
            PLANNED_SEGMENT_TIMES, tt_day, weather_timeline_sbi, GCN=None,
            param_type="normal", seed=42, df_day = df_raw[df_raw["OPERATIONAL_DAY"] == day]
        )
        r5 = sim5.run()
        df = r5.to_dataframe()
        df_list.append(df)

    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    final_df.to_csv("simulator/data/sim_normal_long.csv", index=False)
    #r5.to_csv("simulator/data/sim_with_sbi.csv")

    # -- Run 6: three-way ablation ------------------------------
    """ print("\n-- Run 5: ablation study --")
    df_day = df_raw[df_raw["OPERATIONAL_DAY"] == day]
    ablation_df = run_ablation(
        PLANNED_SEGMENT_TIMES = PLANNED_SEGMENT_TIMES,
        timetable = tt,
        weather   = weather_timeline,
        gcn       = gcn,
        df_day    = df_day,
        seed      = 42,
        n_runs    = 5,
    )
    print("\nAblation results (mean over runs):")
    print(
        ablation_df
        .groupby("condition")[["mae_sec", "rmse_sec", "punctuality_sim"]]
        .mean()
        .round(1)
        .to_string()
    )
    ablation_df.to_csv("simulator/data/ablation.csv", index=False) """
