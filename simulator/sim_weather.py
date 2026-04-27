from dataclasses import dataclass, field
from sim_topology import Segment
import random

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
    temp_c:       float = 15.0
    wind_ms:      float = 0.0
    precip_mm:    float = 0.0
    snow_cm:      float = 0.0
    visibility_m: float = 10_000.0
 
    # ── derived speed factors ─────────────────────────────────────────────────
 
    def speed_factor(self, segment: Segment) -> float:
        """
        Return a multiplicative factor (0 < f ≤ 1.0) applied to the segment's
        line speed.  The most restrictive applicable rule wins.
        """
        factor = 1.0
 
        if not segment.tunnel:
            # Ice: frozen rails → speed restricted to 120 km/h max on all tracks
            if self.temp_c <= 0:
                factor = min(factor, 0.80)
 
            # Frost (just below freezing, not full ice)
            if 0 < self.temp_c <= 2:
                factor = min(factor, 0.90)
 
            # Wind on exposed (lakeside) segments
            if segment.exposed:
                if self.wind_ms >= 25:
                    factor = min(factor, 0.60)   # severe storm
                elif self.wind_ms >= 20:
                    factor = min(factor, 0.75)   # strong wind
                elif self.wind_ms >= 15:
                    factor = min(factor, 0.90)   # moderate wind
 
            # Heavy precipitation leads to extended braking distance
            if self.precip_mm >= 10:
                factor = min(factor, 0.85)
            elif self.precip_mm >= 3:
                factor = min(factor, 0.93)
 
            # Fog
            if self.visibility_m < 200:
                factor = min(factor, 0.70)
            elif self.visibility_m < 500:
                factor = min(factor, 0.85)
 
        return factor
 
    def switch_failure_prob(self) -> float:
        """
        Probability that a switch failure adds extra dwell time at a station.
        Driven by snow depth.
        """
        if self.snow_cm >= 20:
            return 0.15
        if self.snow_cm >= 10:
            return 0.08
        if self.snow_cm >= 5:
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
 
    def __str__(self) -> str:
        parts = [f"T={self.temp_c}°C", f"wind={self.wind_ms}m/s",
                 f"precip={self.precip_mm}mm/h", f"snow={self.snow_cm}cm",
                 f"vis={self.visibility_m}m"]
        return "  ".join(parts)