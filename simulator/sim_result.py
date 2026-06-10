from sim_weather import WeatherConditions, WeatherTimeline
from sim_events import SimEvent, ConflictEvent
from sim_topology import SEGMENTS, LINE_ORDER, STATIONS, PUNCTUALITY_SEC
from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

 
class SimResult:
    """
    Output of one simulation run.

    """
 
    def __init__(
        self,
        events:    list[SimEvent], # All simulated events
        conflicts: list[ConflictEvent], # All recorded conflicts
        weather:   WeatherConditions | WeatherTimeline, # The weather used
        day:       str, # The day for which the simulation was made
    ):
        self.events    = events
        self.conflicts = conflicts
        # Store as WeatherTimeline internally
        if isinstance(weather, WeatherConditions):
            self.weather = WeatherTimeline.from_single(weather)
        else:
            self.weather = weather
        self.day       = day
        self._df: Optional[pd.DataFrame] = None
 
    def to_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            rows = []
            # Use the day-median weather for the scalar export columns.
            # Per-segment weather is already baked into simulated_delay values.
            w = self.weather.representative()
            for e in self.events:
                rows.append({
                    "TRAIN_NUMBER":                         e.train_number,
                    "COMMERCIAL_LINE_NUMBER_DESIGNATION":   e.line,
                    "OPERATING_POINT_ABBREVIATION":         e.station,
                    "OPERATION_TRAFFIC_CATEGORY_ABBREVIATION": e.traffic_category,
                    "EVENT_TYPE":      e.event_type,
                    "EVENT_SERVED": True,
                    "PLAN_STOP_TYPE":       e.stop_type,
                    "PLAN_FORMATION_MAXIMAL_VELOCITY":  e.max_velocity,
                    "OPERATION_DAY_PERIOD_IDENTIFIER_COARSE": e.period_id,
                    "OPERATION_PLANNED_TIMESTAMP":      e.planned_ts,
                    "OPERATION_ACTUAL_TIMESTAMP":       e.actual_ts,
                    "SIMULATED_TIMESTAMP":    e.simulated_ts,
                    "SIMULATED_DELAY": e.simulated_delay,
                    "DAILY_PLAN_OPERATIONAL_DELAY_SEC":    e.actual_delay,
                    "tre200s0": w.tre200s0,
                    "fkl010z1": w.fkl010z1,
                    "fu3010z0": w.fu3010z0,
                    "rre150z0": w.rre150z0,
                    "htoauts0": w.htoauts0,
                    "hto000d0": w.hto000d0,
                    "causes":  " | ".join(e.causes),
                })
            self._df = pd.DataFrame(rows)
        return self._df
 
    def accuracy(self) -> dict:
        df = self.to_dataframe().dropna(subset=["DAILY_PLAN_OPERATIONAL_DELAY_SEC"])
        if df.empty:
            return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
        err = df["SIMULATED_DELAY"] - df["DAILY_PLAN_OPERATIONAL_DELAY_SEC"]
        return {
            "mae":  float(err.abs().mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "n":    len(df),
        }
 
    def punctuality(self) -> dict:
        df = self.to_dataframe()
        deps = df[df["EVENT_TYPE"] == "departure"]
        if deps.empty:
            return {"simulated": float("nan"), "actual": float("nan")}
        return {
            "simulated": float((deps["SIMULATED_DELAY"].abs() <= PUNCTUALITY_SEC).mean()),
            "actual":    float((deps["DAILY_PLAN_OPERATIONAL_DELAY_SEC"].abs() <= PUNCTUALITY_SEC).mean()),
        }
 
    def conflict_summary(self) -> pd.DataFrame:
        if not self.conflicts:
            return pd.DataFrame(columns=["segment", "n_conflicts", "total_wait_sec", "mean_wait_sec"])
        rows = [{"segment": f"{c.segment[0]}→{c.segment[1]}",
                 "wait_sec": c.waited_sec} for c in self.conflicts]
        df = pd.DataFrame(rows)
        return (df.groupby("segment")["wait_sec"]
                  .agg(n_conflicts="count", total_wait_sec="sum", mean_wait_sec="mean")
                  .reset_index())
 
    def delay_causes(self) -> pd.Series:
        """Frequency count of delay cause labels."""
        all_causes = []
        for e in self.events:
            all_causes.extend(e.causes)
        if not all_causes:
            return pd.Series(dtype=int)
        # Categorise
        cats = {"weather": 0, "switch_failure": 0, "conflict": 0,
                "propagated": 0, "other": 0}
        for c in all_causes:
            if c.startswith("weather"):        cats["weather"] += 1
            elif c.startswith("switch"):       cats["switch_failure"] += 1
            elif c.startswith("conflict"):     cats["conflict"] += 1
            elif c.startswith("propagated"):   cats["propagated"] += 1
            else:                              cats["other"] += 1
        return pd.Series(cats)
 
    def summary(self) -> str:
        acc = self.accuracy()
        pct = self.punctuality()
        cf  = self.conflict_summary()
        causes = self.delay_causes()
 
        lines = [
            f"\n{'-' * 100}",
            f"  SimPy Railway Simulation — {self.day}",
            f"  Weather: {self.weather}",
            f"{'-' * 100}",
            f"  Accuracy vs ground truth:  MAE={acc['mae']:.1f}s  RMSE={acc['rmse']:.1f}s  n={acc['n']}",
            f"  Punctuality (≤{PUNCTUALITY_SEC}s):   simulated={pct['simulated']:.1%}  actual={pct['actual']:.1%}",
            f"{'-' * 100}",
        ]
 
        # Single-track conflicts
        if not cf.empty:
            lines.append("  Single-track conflicts:")
            for _, row in cf.iterrows():
                lines.append(f"    {row['segment']:<15}  "
                             f"{int(row['n_conflicts'])} conflict(s)  "
                             f"total wait {row['total_wait_sec']:.0f}s  "
                             f"mean {row['mean_wait_sec']:.0f}s")
        else:
            lines.append("  Single-track conflicts: none")
 
        lines.append(f"{'-' * 100}")
 
        # Delay cause breakdown
        if causes.sum() > 0:
            lines.append("  Delay causes:")
            for cat, cnt in causes[causes > 0].items():
                lines.append(f"    {cat:<18} {cnt}")
 
        lines.append(f"{'-' * 100}")
 
        # Event table
        lines.append(
            f"  {'Train':>6}  {'Line':<5}  {'Station':<6}  {'Type':<9}  "
            f"{'Planned':<8}  {'Sim delay':>9}  {'Act delay':>9}  "
            f"{'Error':>7}  Causes"
        )
        lines.append(f"  {'-' * 93}")
 
        df = self.to_dataframe().sort_values("OPERATION_PLANNED_TIMESTAMP")
        for _, row in df.iterrows():
            err_s = ""
            if pd.notna(row["DAILY_PLAN_OPERATIONAL_DELAY_SEC"]):
                err_s = f"{row['SIMULATED_DELAY'] - row['DAILY_PLAN_OPERATIONAL_DELAY_SEC']:+.0f}s"
            ad_s = f"{row['DAILY_PLAN_OPERATIONAL_DELAY_SEC']:+.0f}s" if pd.notna(row["DAILY_PLAN_OPERATIONAL_DELAY_SEC"]) else "  n/a"
            lines.append(

                f"  {int(row['TRAIN_NUMBER']):>6}  "
                f"{row['COMMERCIAL_LINE_NUMBER_DESIGNATION']:<5}  "
                f"{row['OPERATING_POINT_ABBREVIATION']:<6}  "
                f"{row['EVENT_TYPE']:<9}  "
                f"{row['OPERATION_PLANNED_TIMESTAMP'].strftime('%H:%M'):<8}  "
                f"{row['SIMULATED_DELAY']:>+8.0f}s  "
                f"{ad_s:>9}  "
                f"{err_s:>7}  "
                f"{row['causes'][:60]}"
            )
 
        lines.append(f"{'-' * 100}\n")
        return "\n".join(lines)
 
    def to_csv(self, path: str | Path):
        """Export event log to CSV."""
        self.to_dataframe().to_csv(path, index=False)
        print(f"Saved to {path}")
 
    def to_parquet(self, path: str | Path):
        """Export event log to Parquet."""
        self.to_dataframe().to_parquet(path, index=False)
        print(f"Saved to {path}")

