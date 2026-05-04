from dataclasses import dataclass
from sim_topology import Segment
import random
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# WEATHER
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class WeatherConditions:
    """
    Weather inputs for the simulation.  All fields have safe defaults
    (clear, warm, calm) so you only need to set what matters.
 
    Fields
    ──────
    temp_c        : air temperature (°C).  ≤ 0 triggers ice/frost rules.
    wind_ms       : wind speed (m/s).  ≥ 20 triggers lakeside speed cap.
    precip_mm     : precipitation in mm/h.  > 0 extends braking distances.
    snow_cm       : fresh snow depth (cm).  > 5 increases switch failure risk.
    visibility_m  : visibility (m).  < 200 triggers fog speed restriction.
    """
    tre200s0:  float = 15.0 # Air temp
    fkl010z1:   float = 0.0  # Gust peak
    fu3010z0:   float = 0.0  # Wind velocity
    rre150z0:   float = 0.0  # Precipation
    htoauts0:   float = 0.0  # Snow height Biel
    hto000d0:   float = 0.0  # Snow height Neuchatel
 
    # ── derived speed factors ─────────────────────────────────────────────────
 
    def speed_factor(self, segment: Segment) -> float:
        factor = 1.0

        if not segment.tunnel:
            # Temperature — keep existing thresholds, they match the data
            if self.tre200s0 <= -5:
                factor = min(factor, 0.9)

            # Wind — only apply on exposed segments, raise thresholds significantly
            # Data shows gusts up to 30 m/s on LOW delay days, so wind effect
            # only kicks in at extreme values on this corridor
            if segment.exposed:
                if self.fu3010z0 >= 40:      # extreme storm (p99+ in data)
                    factor = min(factor, 0.70)
                elif self.fu3010z0 >= 30:    # severe (p99 in data)
                    factor = min(factor, 0.85)
                elif self.fu3010z0 >= 25:    # strong (p95 in data)
                    factor = min(factor, 0.93)
                # Below 25 m/s: no effect — data shows no correlation

            # Precipitation — strongest signal, keep but recalibrate units
            # rre150z0 is mm/10min, convert *6 for mm/h before passing in
            if self.rre150z0 * 6 >= 8:         # heavy (max in data ~10.8 mm/h)
                factor = min(factor, 0.85)
            elif self.rre150z0 * 6 >= 3:
                factor = min(factor, 0.93)
            elif self.rre150z0 * 6 >= 1:
                factor = min(factor, 0.97)

            # Snow
            if self.htoauts0 > 20:
                factor = min(factor, 0.70)
            elif self.htoauts0 > 10:
                factor = min(factor, 0.85)

        return factor
 
    def switch_failure_prob(self) -> float:
        """
        Probability that a switch failure adds extra dwell time at a station.
        Driven by snow depth.
        """
        if self.htoauts0 >= 20:
            return 0.15
        if self.htoauts0 >= 10:
            return 0.08
        if self.htoauts0 >= 5:
            return 0.03
        return 0.0
 
    def switch_failure_delay_sec(self) -> float:
        """Extra dwell seconds if a switch failure occurs (random 60–300 s)."""
        return random.uniform(60, 300)
 
    def travel_time(self, segment: Segment, planned_sec: int) -> float:
        """
        Compute realistic travel time for a segment given weather.
        Returns seconds (float).
        """
        factor = self.speed_factor(segment)
        # Travel time scales inversely with speed factor
        return planned_sec / factor
    

    @classmethod
    def from_meteoswiss_row(cls, row: pd.Series) -> "WeatherConditions":
        """Create WeatherConditions from a MeteoSwiss data row matching training features."""
        return cls(
            tre200s0       = float(row.get("tre200s0", 15.0)),  # temp C
            fu3010z0      = float(row.get("fu3010z0", 0.0)),   # wind velocity km/h
            fkl010z1    = float(row.get("fkl010z1", 0.0)),    # gust peak
            rre150z0    = float(row.get("rre150z0", 0.0)),  # rain fall
            htoauts0       = float(row.get("htoauts0", 0.0)),   # snow height at the moment
            hto000d0    = float(row.get("hto000d0", 0.0))   # snow at 6am in Neuchatel
            
        )
 
    def __str__(self) -> str:
        parts = [f"T={self.tre200s0}°C", f"wind={self.fu3010z0}m/s",
                 f"precip={self.rre150z0}mm/h", f"snow={self.htoauts0}cm",
                ]
        return "  ".join(parts)


class WeatherTimeline:

    """
    Holds a sequence of WeatherConditions snapshots indexed by seconds-from-
    midnight (SimPy clock units).  Call ``at(sim_time)`` from any TrainProcess
    to get the conditions that apply at that moment.

    Lookup is a simple step-function: returns the most-recent snapshot whose
    timestamp is ≤ sim_time.  This matches how MeteoSwiss 10-minute data works
    (each row is valid until the next row).
    Construction
    ────────────
    Use the classmethod ``from_day_dataframe`` to build a timeline directly
    from your raw dataset for a given operational day.  The DataFrame must
    contain an ``OPERATION_PLANNED_TIMESTAMP`` (or ``time`` / ``datetime``)
    column and the six MeteoSwiss feature columns.

    Fallback
    ────────
    If sim_time is before the first snapshot, the first snapshot is returned.
    If the DataFrame has no usable rows, a single clear-weather snapshot at
    t=0 is inserted so the simulation never crashes.

    Example
    ───────
        timeline = WeatherTimeline.from_day_dataframe(df_raw, day="2025-01-15")
        sim = RailwaySimulator(PLANNED_SEGMENT_TIMES, tt, weather=timeline)
        result = sim.run()
    """



    WEATHER_COLS = ["tre200s0", "fkl010z1", "fu3010z0",
                    "rre150z0", "htoauts0", "hto000d0"]


    def __init__(self, snapshots: list[tuple[float, WeatherConditions]]):

        if not snapshots:
            snapshots = [(0.0, WeatherConditions())]

        # Ensure sorted
        self._times = [t for t, _ in snapshots]
        self._conds = [c for _, c in snapshots]

    def at(self, sim_time: float) -> WeatherConditions:

        """Return the WeatherConditions applicable at *sim_time* (seconds from midnight)."""
        import bisect
        idx = bisect.bisect_right(self._times, sim_time) - 1
        idx = max(0, idx)
        return self._conds[idx]



    def representative(self) -> WeatherConditions:

        """Day-median conditions — useful for summary strings and single-value output."""
        if len(self._conds) == 1:
            return self._conds[0]

        fields = WeatherTimeline.WEATHER_COLS
        medians = {}
        for f in fields:
            medians[f] = float(np.median([getattr(c, f) for c in self._conds]))
        return WeatherConditions(**medians)

    def __str__(self) -> str:
        return f"WeatherTimeline({len(self._conds)} snapshots) median: {self.representative()}"



    @classmethod
    def from_day_dataframe(
        cls,
        df: pd.DataFrame,
        day: str,
        time_col: str = "OPERATION_PLANNED_TIMESTAMP",
        resample_minutes: int | None = None,

    ) -> "WeatherTimeline":

        """
        Build a WeatherTimeline from the raw operational DataFrame.
        Parameters
        ----------
        df               : full raw DataFrame (all days are fine; we filter to ``day``)
        day              : operational day string, e.g. ``"2025-01-15"``
        time_col         : column whose datetime values become the snapshot times.
                           Defaults to ``OPERATION_PLANNED_TIMESTAMP``.
                           Alternatively pass ``"time"`` if your dataset has a
                           dedicated weather-measurement timestamp column.
        resample_minutes : if not None, resample/interpolate to this interval
                           (e.g. 10 to get one snapshot every 10 minutes).
                           If None, one snapshot is built per unique timestamp
                           in the filtered data (after de-duplication).
        Returns
        -------
        WeatherTimeline
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Filter to the requested day
        day_mask = df[time_col].dt.date.astype(str) == day
        day_df   = df[day_mask].copy()

        if day_df.empty:
            return cls([(0.0, WeatherConditions())])

        # Keep only weather columns + time; drop rows missing all weather cols
        weather_cols_present = [c for c in cls.WEATHER_COLS if c in day_df.columns]
        if not weather_cols_present:
            return cls([(0.0, WeatherConditions())])


        day_df = day_df[[time_col] + weather_cols_present].copy()
        day_df = day_df.dropna(subset=weather_cols_present, how="all")

        if day_df.empty:
            return cls([(0.0, WeatherConditions())])

        # Compute seconds from midnight for each row
        midnight = pd.Timestamp(day)
        day_df["_sim_time"] = (day_df[time_col] - midnight).dt.total_seconds()
        day_df = day_df.sort_values("_sim_time")

        if resample_minutes is not None:
            # Set a DatetimeIndex and resample, then interpolate missing values
            day_df = day_df.set_index(time_col)
            rule = f"{resample_minutes}min"
            day_df = (
                day_df[weather_cols_present]
                .resample(rule)
                .mean()
                .interpolate("time")
                .ffill()
                .bfill()
                .reset_index()
            )
            midnight = pd.Timestamp(day)
            day_df["_sim_time"] = (day_df[time_col] - midnight).dt.total_seconds()
        else:
            # De-duplicate by sim_time: take the mean weather per unique timestamp
            day_df = (
                day_df
                .groupby("_sim_time")[weather_cols_present]
                .mean()
                .reset_index()
            )

        snapshots: list[tuple[float, WeatherConditions]] = []

        for _, row in day_df.iterrows():
            cond = WeatherConditions.from_meteoswiss_row(row)
            snapshots.append((float(row["_sim_time"]), cond))

        return cls(snapshots)



    @classmethod
    def from_single(cls, cond: WeatherConditions) -> "WeatherTimeline":
        """Wrap a single WeatherConditions so old call-sites still work."""
        return cls([(0.0, cond)])
