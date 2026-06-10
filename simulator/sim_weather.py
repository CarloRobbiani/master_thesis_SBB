from dataclasses import dataclass
from sim_topology import Segment
import random
import pandas as pd
import numpy as np
import json
 
@dataclass
class WeatherConditions:
    """ 
    Paramters
    ------
    temp_c : air temperature (°C).  ≤ 0 triggers ice/frost rules.
    wind_ms : wind speed (m/s).  ≥ 20 triggers lakeside speed cap.
    precip_mm : precipitation in mm/h.  > 0 extends braking distances.
    snow_cm : fresh snow depth (cm).  > 5 increases switch failure risk.
    visibility_m  : visibility (m).  < 200 triggers fog speed restriction.
    weather_type : the type of weather (normal, snow, learned)
    weather_factor : points to the json file where the weather factors are encoded
    """
    tre200s0:  float = 15.0 # Air temp
    fkl010z1:   float = 0.0  # Gust peak
    fu3010z0:   float = 0.0  # Wind velocity
    rre150z0:   float = 0.0  # Precipation
    htoauts0:   float = 0.0  # Snow height Biel
    hto000d0:   float = 0.0  # Snow height Neuchatel
    param_type: str = "normal"
    speed_factors : json = None
 
    # -- derived speed factors -----
 
    def speed_factor(self, segment: Segment) -> float:

        sf = self.speed_factors
        pt = self.param_type
        factor = 1.0

        if not segment.tunnel:
            if self.tre200s0 <= -5:
                factor = min(factor,sf[pt]["air_temp"])

            # Wind — only apply on exposed segments, raise thresholds significantly
            # Data shows gusts up to 30 m/s on LOW delay days, so wind effect
            # only kicks in at extreme values on this corridor
            if segment.exposed:
                if self.fu3010z0 >= 40:      # extreme storm (p99+ in data)
                    factor = min(factor, sf[pt]["wind_high_exposed"])
                elif self.fu3010z0 >= 30:    # severe (p99 in data)
                    factor = min(factor, sf[pt]["wind_high"])
                elif self.fu3010z0 >= 25:    # strong (p95 in data)
                    factor = min(factor, sf[pt]["wind_moderate"])
                # Below 25 m/s: no effect — data shows no correlation

            # rre150z0 is mm/10min, convert *6 for mm/h before passing in
            if self.rre150z0 * 6 >= 8:   
                factor = min(factor, sf[pt]["rain_high"])
            elif self.rre150z0 * 6 >= 3:
                factor = min(factor, sf[pt]["rain_moderate"])
            elif self.rre150z0 * 6 >= 1:
                factor = min(factor, sf[pt]["rain_low"])

            # Snow
            if self.htoauts0 > 20:
                factor = min(factor, sf[pt]["snow_high"])
            elif self.htoauts0 > 10:
                factor = min(factor, sf[pt]["snow_low"])

        return factor
 
    def switch_failure_prob(self) -> float:
        """
        Probability that a switch failure adds extra dwell time at a station.
        """
        sf = self.speed_factors
        pt = self.param_type

        if self.htoauts0 >= 20:
            return sf[pt]["switch_fail_high"]
        if self.htoauts0 >= 10:
            return sf[pt]["switch_fail_moderate"]
        if self.htoauts0 >= 5:
            return sf[pt]["switch_fail_low"]
        return 0.0
 
    def switch_failure_delay_sec(self) -> float:
        """Extra dwell seconds if a switch failure occurs (random 60-300 s)."""
        return random.uniform(60, 300)
 
    def travel_time(self, segment: Segment, planned_sec: int) -> float:
        """
        Compute realistic travel time for a segment given weather.
        """
        factor = self.speed_factor(segment)
        return planned_sec / factor
    

    @classmethod
    def from_meteoswiss_row(cls, row: pd.Series, speed_factors=None) -> "WeatherConditions":
        """Create WeatherConditions from a MeteoSwiss data row matching training features."""
        return cls(
            tre200s0       = float(row.get("tre200s0", 15.0)),  # temp C
            fu3010z0      = float(row.get("fu3010z0", 0.0)),   # wind velocity km/h
            fkl010z1    = float(row.get("fkl010z1", 0.0)),    # gust peak
            rre150z0    = float(row.get("rre150z0", 0.0)),  # rain fall
            htoauts0       = float(row.get("htoauts0", 0.0)),   # snow height at the moment
            hto000d0    = float(row.get("hto000d0", 0.0)),   # snow at 6am in Neuchatel
            speed_factors = speed_factors
        )
 
    def __str__(self) -> str:
        parts = [f"T={self.tre200s0}°C", f"wind={self.fu3010z0}m/s",
                 f"precip={self.rre150z0}mm/h", f"snow={self.htoauts0}cm",
                ]
        return "  ".join(parts)


class WeatherTimeline:

    """
    Holds a sequence of WeatherConditions snapshots indexed by seconds-from-
    midnight (SimPy clock units)

    Example
    -------
        timeline = WeatherTimeline.from_day_dataframe(df_raw, day="2025-01-15")
        sim = RailwaySimulator(PLANNED_SEGMENT_TIMES, tt, weather=timeline)
        result = sim.run()
    """

    WEATHER_COLS = ["tre200s0", "fkl010z1", "fu3010z0",
                    "rre150z0", "htoauts0", "hto000d0"]


    def __init__(self,speed_factors, 
                 snapshots: list[tuple[float, WeatherConditions]],
                 param_type = "normal"):

        if not snapshots:
            snapshots = [(0.0, WeatherConditions(
                param_type=param_type, speed_factors=speed_factors
            ))]
        # Ensure sorted
        self._times = [t for t, _ in snapshots]
        self._conds = [c for _, c in snapshots]
        self.sf = speed_factors

    def at(self, sim_time: float) -> WeatherConditions:
        """Return the WeatherConditions applicable at sim_time"""
        import bisect
        idx = bisect.bisect_right(self._times, sim_time) - 1
        idx = max(0, idx)
        return self._conds[idx]



    def representative(self) -> WeatherConditions:
        """
        Day-median conditions
        """
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
        speed_factors,
        time_col: str = "OPERATION_PLANNED_TIMESTAMP",
        resample_minutes: int | None = None,

    ) -> "WeatherTimeline":

        """
        Build a WeatherTimeline from the raw operational DataFrame.
        Parameters
        ----------
        df               : full raw DataFrame (all days are fine; we filter to day)
        day              : operational day string, e.g. "2025-01-15"
        time_col         : column whose datetime values become the snapshot times.
        resample_minutes : if not None, resample/interpolate to this interval
                           (e.g. 10 to get one snapshot every 10 minutes).
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Filter to the day
        day_mask = df[time_col].dt.date.astype(str) == day
        day_df   = df[day_mask].copy()

        if day_df.empty:
            return cls([(0.0, WeatherConditions())])

        # Keep only weather columns + time and drop rows missing all weather cols
        weather_cols_present = [c for c in cls.WEATHER_COLS if c in day_df.columns]
        if not weather_cols_present:
            return cls(speed_factors, [(0.0, WeatherConditions(speed_factors=speed_factors))])


        day_df = day_df[[time_col] + weather_cols_present].copy()
        day_df = day_df.dropna(subset=weather_cols_present, how="all")

        if day_df.empty:
            return cls(speed_factors, [(0.0, WeatherConditions(speed_factors=speed_factors))])


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
            # Deduplicate by sim_time: take the mean weather per unique timestamp
            day_df = (
                day_df
                .groupby("_sim_time")[weather_cols_present]
                .mean()
                .reset_index()
            )

        snapshots: list[tuple[float, WeatherConditions]] = []

        for _, row in day_df.iterrows():
            cond = WeatherConditions.from_meteoswiss_row(row, speed_factors)
            snapshots.append((float(row["_sim_time"]), cond))

        return cls(speed_factors, snapshots)



    @classmethod
    def from_single(cls, cond: WeatherConditions, speed_factors) -> "WeatherTimeline":
        """Wrap a single WeatherConditions"""
        return cls(speed_factors, [(0.0, cond)])
