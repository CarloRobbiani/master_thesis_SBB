
from sim_weather import WeatherConditions
from sim_events import SimEvent, ConflictEvent
from sim_topology import SEGMENTS, LINE_ORDER, STATIONS, PUNCTUALITY_SEC
from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION RESULT
# ══════════════════════════════════════════════════════════════════════════════
 
class SimResult:
    """
    Output of one simulation run.
 
    Attributes
    ──────────
    events        : list[SimEvent]      — all simulated stop events
    conflicts     : list[ConflictEvent] — single-track conflict waits
    weather       : WeatherConditions   — weather used in this run
    day           : str                 — operational day
    """
 
    def __init__(
        self,
        events:    list[SimEvent],
        conflicts: list[ConflictEvent],
        weather:   WeatherConditions,
        day:       str,
    ):
        self.events    = events
        self.conflicts = conflicts
        self.weather   = weather
        self.day       = day
        self._df: Optional[pd.DataFrame] = None
 
    def to_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            rows = []
            for e in self.events:
                w = self.weather
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
                    "tre200s0": w.tree200s0,
                    "fkl010z1": w.fkl010z1,
                    "fu3010z0": w.fu3010z0,
                    "rre150z0": w.rre150z0,
                    "htoauts0": w.htoauts0,
                    "hto000d0": w.hto000d0,
                    "causes":          " | ".join(e.causes),
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
            f"\n{'═' * 100}",
            f"  SimPy Railway Simulation — {self.day}",
            f"  Weather: {self.weather}",
            f"{'─' * 100}",
            f"  Accuracy vs ground truth:  MAE={acc['mae']:.1f}s  RMSE={acc['rmse']:.1f}s  n={acc['n']}",
            f"  Punctuality (≤{PUNCTUALITY_SEC}s):   simulated={pct['simulated']:.1%}  actual={pct['actual']:.1%}",
            f"{'─' * 100}",
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
 
        lines.append(f"{'─' * 100}")
 
        # Delay cause breakdown
        if causes.sum() > 0:
            lines.append("  Delay causes:")
            for cat, cnt in causes[causes > 0].items():
                lines.append(f"    {cat:<18} {cnt}")
 
        lines.append(f"{'─' * 100}")
 
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
 
        lines.append(f"{'═' * 100}\n")
        return "\n".join(lines)
 
    def to_csv(self, path: str | Path):
        """Export event log to CSV."""
        self.to_dataframe().to_csv(path, index=False)
        print(f"Saved to {path}")
 
    def to_parquet(self, path: str | Path):
        """Export event log to Parquet."""
        self.to_dataframe().to_parquet(path, index=False)
        print(f"Saved to {path}")
 
    def plot(
        self,
        station:    Optional[str] = None,
        train:      Optional[int] = None,
        show_causes: bool         = True,
    ):
        """
        Plot simulated vs actual departure delays.
 
        Parameters
        ──────────
        station     : filter to one station abbreviation
        train       : filter to one train number
        show_causes : annotate bars with cause labels
        """
 
        df = self.to_dataframe()
        df = df[df["event_type"] == "departure"].copy()
        if station:
            df = df[df["station"] == station]
        if train:
            df = df[df["train"] == train]
        df = df.sort_values("planned_ts")
 
        if df.empty:
            print("No matching events.")
            return
 
        fig, axes = plt.subplots(2, 1, figsize=(max(12, len(df) * 0.7), 9),
                                 gridspec_kw={"height_ratios": [3, 1]})
 
        # ── top: delay bars ───────────────────────────────────────────────────
        ax = axes[0]
        x  = np.arange(len(df))
        ax.bar(x - 0.2, df["actual_delay"],    0.35, label="Actual",    color="steelblue", alpha=0.8)
        ax.bar(x + 0.2, df["simulated_delay"], 0.35, label="Simulated", color="tomato",    alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline( PUNCTUALITY_SEC, color="orange", lw=1, ls=":",
                    label=f"±{PUNCTUALITY_SEC}s punctuality")
        ax.axhline(-PUNCTUALITY_SEC, color="orange", lw=1, ls=":")
 
        if show_causes:
            for i, (_, row) in enumerate(df.iterrows()):
                c = row["causes"]
                if c and row["simulated_delay"] > 10:
                    label = c[:20]
                    ax.text(i + 0.2, row["simulated_delay"] + 5, label,
                            fontsize=5, ha="center", va="bottom", rotation=60, color="darkred")
 
        labels = [f"{int(r.train)}\n{r.station}\n{r.planned_ts.strftime('%H:%M')}"
                  for _, r in df.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Delay (seconds)")
        title = f"Railway Simulation — {self.day}  |  {self.weather}"
        if station: title += f"  |  Station: {station}"
        if train:   title += f"  |  Train: {train}"
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
 
        # ── bottom: topology schematic ────────────────────────────────────────
        ax2 = axes[1]
        ax2.set_xlim(0, 38)
        ax2.set_ylim(-0.5, 1.5)
        ax2.axis("off")
        ax2.set_title("Line topology  (══ double track │ ── single track)", fontsize=8)
 
        prev_km = None
        for abbr in LINE_ORDER:
            st   = STATIONS[abbr]
            km   = st.km
            seg  = SEGMENTS.get((LINE_ORDER[max(0, LINE_ORDER.index(abbr)-1)], abbr))
            color = "navy" if (seg and seg.double_track) else "red"
            if prev_km is not None:
                lw = 3 if (seg and seg.double_track) else 1.5
                ax2.plot([prev_km, km], [0.5, 0.5], color=color, lw=lw)
            ax2.plot(km, 0.5, "s", ms=8, color="black")
            ax2.text(km, 0.0, abbr, ha="center", fontsize=7)
            prev_km = km
 
        # Highlight single-track section
        ax2.axvspan(STATIONS["TWN"].km, STATIONS["NV"].km,
                    alpha=0.15, color="red", label="single track")
        ax2.text((STATIONS["TWN"].km + STATIONS["NV"].km) / 2, 1.1,
                 "SINGLE TRACK\n(Ligerz bottleneck)", ha="center",
                 fontsize=7, color="red")
 
        plt.tight_layout()
        plt.show()
 
    def plot_space_time(self):
        """
        Draw a space-time diagram: x=km along line, y=simulated clock time.
        Each train is one line; conflicts show as flat sections.
        """
 
        df = self.to_dataframe()
        trains = df["train"].unique()
        cmap   = plt.cm.get_cmap("tab20", len(trains))
 
        fig, ax = plt.subplots(figsize=(14, 8))
 
        for i, tn in enumerate(trains):
            tdf = df[df["train"] == tn].sort_values("planned_ts")
            xs, ys = [], []
            for _, row in tdf.iterrows():
                st = STATIONS.get(row["station"])
                if st:
                    xs.append(st.km)
                    ys.append(row["simulated_ts"])
            if xs:
                ax.plot(xs, ys, marker="o", ms=4, lw=1.2,
                        color=cmap(i), label=str(tn))
 
        # Mark single-track section
        ax.axvspan(STATIONS["TWN"].km, STATIONS["NV"].km,
                   alpha=0.1, color="red")
        ax.text((STATIONS["TWN"].km + STATIONS["NV"].km) / 2,
                ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else df["simulated_ts"].min(),
                "SINGLE\nTRACK", ha="center", fontsize=7, color="red")
 
        # Station labels on x-axis
        ax.set_xticks([STATIONS[a].km for a in LINE_ORDER])
        ax.set_xticklabels([f"{a}\n{STATIONS[a].km:.0f}km" for a in LINE_ORDER], fontsize=8)
        ax.set_xlabel("Distance from Biel (km)")
        ax.set_ylabel("Simulated time")
        ax.set_title(f"Space-time diagram — {self.day}  |  {self.weather}")
        ax.legend(title="Train", fontsize=6, ncol=2, loc="upper left")
        plt.tight_layout()
        plt.show()