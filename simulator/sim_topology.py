from dataclasses import dataclass
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass(frozen=True)
class Segment:
    """A directed track segment between two nodes."""
    origin:       str
    destination:  str
    length_km:    float
    max_speed_kmh: float      # line speed limit
    double_track: bool        # False = single track (requires token/occupation)
    exposed:      bool = False  # True = lakeside, subject to wind restriction
    tunnel:       bool = False  # True = protected from weather
 
 
@dataclass(frozen=True)
class Station:
    name:          str
    abbr:          str
    km:            float        # distance from BI
    platforms:     int          # number of platform tracks
    crossing_loop: bool         # can trains cross here on single-track section?
    is_terminus:   bool = False
 
 
# ── Real topology (2025, before Ligerz Tunnel opens) ─────────────────────────
#
# Sources:
#   • Wikipedia – Jura Foot Railway (double/single track)
#   • Railway Gazette – Ligerz Tunnel article (10 km single section TWN–LIG)
#   • SBB timetable – speeds and journey times
#   • OpenStreetMap distances (approximate)
#
STATIONS: dict[str, Station] = {
    "BI":   Station("Biel/Bienne",       "BI",   0.0,  8, False, is_terminus=True),
    "TUE":  Station("Tüscherz",          "TUE",  9.5,  2, True),    # crossing loop
    "TWN":  Station("Twann      ",       "TWN", 14.2,  2, True),    # crossing loop
    "LIG":  Station("Ligerz",            "LIG", 16.8,  1, False),   # single-track station
    "NV":   Station("La Neuveville",     "NV",  20.3,  2, True),    # crossing loop
    "LD":   Station("Le Landeron",       "LD",  24.1,  2, False),
    "CRNE": Station("Cressier NE",       "CRNE",27.0,  2, False),
    "CORN": Station("Corneaux",          "CORN",29.8,  2, False),
    "SBL": Station("St-Blaise CFF",      "SBL", 33.2,  2, False),
    "NE":   Station("Neuchâtel",         "NE",  38.0,  8, False, is_terminus=True),
}
 
# Canonical order BI → NE
LINE_ORDER = ["BI", "TUE", "TWN", "LIG", "NV", "LD", "CRNE", "CORN", "SBL", "NE"]
 
# Track segments (both directions defined)
_SEGMENTS_BASE = [
    # (origin, dest,   km,   v_max, double, exposed, tunnel)
    ("BI",   "TUE",  9.5,  160, True,  False, False),
    ("TUE",  "TWN",  4.7,  100, True,  True,  False),  # lakeside, exposed
    ("TWN",  "LIG",  2.6,   80, False, True,  False),  # SINGLE TRACK, exposed, lakeside cliff
    ("LIG",  "NV",   3.5,   80, True, True,  False),  # exposed
    ("NV",   "LD",   3.8,  140, True,  False, False),
    ("LD",   "CRNE", 2.9,  140, True,  False, False),
    ("CRNE", "CORN", 2.8,  140, True,  False, False),
    ("CORN", "SBL", 3.4,  140, True,  False, False),
    ("SBL", "NE",   4.8,  140, True,  False, False),
]
 
SEGMENTS: dict[tuple[str, str], Segment] = {}
for _o, _d, _km, _v, _dt, _ex, _tn in _SEGMENTS_BASE:
    SEGMENTS[(_o, _d)] = Segment(_o, _d, _km, _v, _dt, _ex, _tn)
    SEGMENTS[(_d, _o)] = Segment(_d, _o, _km, _v, _dt, _ex, _tn)  # reverse
 

def build_planned_segment_times(
    df: pd.DataFrame,
) -> dict[tuple[str, str, str], int]:
    """
    Returns {(origin, dest, line): seconds} for all station pairs a train
    traverses on the corridor, including non-adjacent jumps (e.g. IC5 skips
    SBL and jumps CORN->NE directly).
 
    Segment time is always: departure at origin -> arrival at destination.
    This is correct because pass stops have identical arr/dep timestamps,
    so dep->dep or arr->arr gives wrong (0s or 60s) values.
 
    Non-adjacent pairs (skipped stations) are stored directly from the data
    rather than being reconstructed by summing physics estimates, which are
    systematically too short due to operational speed limits and braking.
 
    A "fallback" key per adjacent segment pair holds the cross-line median,
    used when a specific line has no data for that segment.
    """
    df = df.copy()
    df["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(df["OPERATION_PLANNED_TIMESTAMP"])
    df = df[df["OPERATING_POINT_ABBREVIATION"].isin(LINE_ORDER)]
 
    times: dict[tuple[str, str, str], list[float]] = {}
 
    for (operational_day, train_num), grp in df.groupby(["OPERATIONAL_DAY", "TRAIN_NUMBER"]):
        line = grp["COMMERCIAL_LINE_NUMBER_DESIGNATION"].iloc[0]
        grp  = grp.sort_values("OPERATION_TRAIN_RUN_SEQUENCE_NUMBER")
 
        # Index departures and arrivals by station for this train/day
        deps = (
            grp[grp["EVENT_TYPE"] == "departure"]
            .drop_duplicates("OPERATING_POINT_ABBREVIATION", keep="first")
            .set_index("OPERATING_POINT_ABBREVIATION")["OPERATION_PLANNED_TIMESTAMP"]
        )
        arrs = (
            grp[grp["EVENT_TYPE"] == "arrival"]
            .drop_duplicates("OPERATING_POINT_ABBREVIATION", keep="first")
            .set_index("OPERATING_POINT_ABBREVIATION")["OPERATION_PLANNED_TIMESTAMP"]
        )
 
        # Ordered list of stations this train departs from / arrives at
        dep_stns = [s for s in LINE_ORDER if s in deps.index]
        arr_stns = [s for s in LINE_ORDER if s in arrs.index]
 
        for origin in dep_stns:
            o_idx = LINE_ORDER.index(origin)
            for dest in arr_stns:
                d_idx = LINE_ORDER.index(dest)
                if origin == dest:
                    continue
                # Skip this pair if there is an intermediate departure station
                # between origin and dest — a finer-grained segment covers it.
                intermediate_deps = [
                    s for s in dep_stns
                    if min(o_idx, d_idx) < LINE_ORDER.index(s) < max(o_idx, d_idx)
                ]
                if intermediate_deps:
                    continue
 
                delta = (arrs[dest] - deps[origin]).total_seconds()
                if 30 < delta < 3600:
                    times.setdefault((origin, dest, line), []).append(delta)
 
    # Build result: per-line median + cross-line fallback (adjacent segments only)
    result: dict[tuple[str, str, str], int] = {}
    all_pairs = set((o, d) for o, d, _ in times.keys())
 
    for (o, d) in all_pairs:
        seg_vals = {
            line: vals
            for (oo, dd, line), vals in times.items()
            if oo == o and dd == d
        }
        for line, vals in seg_vals.items():
            result[(o, d, line)] = int(np.median(vals))
 
        # Only store fallback for adjacent pairs
        #if abs(LINE_ORDER.index(o) - LINE_ORDER.index(d)) == 1:
        all_vals = [v for vals in seg_vals.values() for v in vals]
        result[(o, d, "fallback")] = int(np.median(all_vals))
 
    return result
 

# Minimum dwell times (seconds)
MIN_DWELL = {"commercialStop": 30, "pass": 0}
 
# Punctuality threshold (Swiss standard: 3 minutes)
PUNCTUALITY_SEC = 180