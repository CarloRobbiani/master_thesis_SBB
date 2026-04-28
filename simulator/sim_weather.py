from dataclasses import dataclass
from sim_topology import Segment
import random
import pandas as pd

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
    tree200s0:  float = 15.0 # Air temp
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
            if self.tree200s0 <= 0:
                factor = min(factor, 0.80)
            elif self.tree200s0 <= 2:
                factor = min(factor, 0.92)

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
            tree200s0       = float(row.get("tree200s0", 15.0)),  # temp C
            fu3010z0      = float(row.get("fu3010z0", 0.0)),   # wind velocity km/h
            fkl010z1    = float(row.get("fkl010z1", 0.0)),    # gust peak
            rre150z0    = float(row.get("rre150z0", 0.0)),  # rain fall
            htoauts0       = float(row.get("htoauts0", 0.0)),   # snow height at the moment
            hto000d0    = float(row.get("hto000d0", 0.0))   # snow at 6am in Neuchatel
            
        )
 
    def __str__(self) -> str:
        parts = [f"T={self.tree200s0}°C", f"wind={self.fu3010z0}m/s",
                 f"precip={self.rre150z0}mm/h", f"snow={self.htoauts0}cm",
                ]
        return "  ".join(parts)