from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION EVENTS (output records)
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class SimEvent:
    """One recorded event from the simulation."""
    train_number:    int
    line:            str
    station:         str
    event_type:      str          # "arrival" | "departure"
    stop_type:       str
    planned_ts:      datetime
    simulated_ts:    datetime
    simulated_delay: float        # seconds vs planned
    actual_delay:    float        # ground-truth (nan if unknown)
    causes:          list[str]    # delay cause annotations
    blocked_by:      Optional[int] = None   # train number that caused headway block
 
 
@dataclass
class ConflictEvent:
    """Recorded single-track conflict / headway wait."""
    sim_time:        float        # SimPy clock (seconds from midnight)
    train_number:    int
    segment:         tuple[str, str]
    waited_sec:      float
    blocked_by:      Optional[int]