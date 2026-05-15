import simpy
from sim_timetable import TrainSchedule, StopEntry
from sim_weather import WeatherConditions, WeatherTimeline
from sim_events import SimEvent, ConflictEvent
from datetime import datetime, timedelta
from sim_topology import SEGMENTS, Segment, LINE_ORDER, MIN_DWELL, STATIONS
import random
from typing import Optional
import json
import math

# ══════════════════════════════════════════════════════════════════════════════
# SIMPY PROCESSES
# ══════════════════════════════════════════════════════════════════════════════
 
class TrainProcess:
    """
    SimPy process representing one train running along its planned route.
 
    Delay propagation logic
    ───────────────────────
    1.  The train starts with any initial delay present at its first departure.
    2.  For each segment it traverses:
          a. Request the segment resource (single-track: capacity=1; double: capacity=∞)
          b. Travel time = planned_time / weather_speed_factor
          c. Extra time = travel_time - planned_time  -> logged as weather delay
    3.  At each station:
          a. Check for switch failure (probabilistic, driven by snow)
          b. Enforce minimum dwell
          c. Record arrival and departure SimEvents
    4.  Delay = (simulated_departure_time - planned_departure_time)
    """
 
    def __init__(
        self,
        PLANNED_SEGMENT_TIMES,
        env:         simpy.Environment,
        schedule:    TrainSchedule,
        resources:   dict[tuple[str, str], simpy.Resource],
        weather:     WeatherTimeline | WeatherConditions,
        sim_events:  list[SimEvent],
        conflict_log: list[ConflictEvent],
        day_start:   datetime,
        param_type : str, # Type of parameter to be loaded, either normal or learned
        inject_delay : Optional[tuple] = None, # Optionally give tuple (a,b, c) where a is the station where you want 
                                       # to inject delay, b is a line, train_number or None and 
                                       # c is amount of delay in seconds
        entry_delays_sec = None
    ):
        self.env          = env
        self.schedule     = schedule
        self.resources    = resources
        # Normalise: always store a WeatherTimeline internally
        if isinstance(weather, WeatherConditions):
            self.weather = WeatherTimeline.from_single(weather)
        else:
            self.weather = weather
        self.sim_events   = sim_events
        self.conflict_log = conflict_log
        self.day_start    = day_start
        self.current_delay = 0.0    # seconds, positive = late
        self.PLANNED_SEGMENT_TIMES = PLANNED_SEGMENT_TIMES
        self.category = schedule.category
        self.param_type = param_type 
        self.inject_delay = inject_delay
        self.entry_delays_sec = entry_delays_sec

        with open("simulator\weather_factors.json") as f:
            self.speed_factors = json.load(f)
 
    def _ts_to_sim(self, ts: datetime) -> float:
        """Convert a datetime to SimPy clock time (seconds from midnight)."""
        return (ts - self.day_start).total_seconds()
 
    def _sim_to_ts(self, t: float) -> datetime:
        return self.day_start + timedelta(seconds=t)
 
    def run(self):
        """Main SimPy generator for this train."""
        schedule = self.schedule
        stops    = schedule.stops

 
        # Group stops into station visits: list of (arrival_stop, departure_stop)
        # Some stations only have departure (origin) or only arrival (terminus)
        visits = self._group_visits(stops)
 
        prev_station = None
 
        for arr_stop, dep_stop in visits:
            station_abbr = (arr_stop or dep_stop).station
 
            # ── 1. TRAVEL to this station ──────────────────────────────────────
            if prev_station is not None:
                seg_key  = (prev_station, station_abbr)
                segment  = SEGMENTS.get(seg_key)
                #key = (prev_station, station_abbr, self.schedule.line)
                #planned_travel = self.PLANNED_SEGMENT_TIMES.get(key, 180)

 
                if segment is None:
                    # Non-adjacent jump — check dict first, then fall back to physics sum
                    direct_key = (prev_station, station_abbr, self.schedule.line)
                    fallback_key = (prev_station, station_abbr, "fallback")
                    if direct_key in self.PLANNED_SEGMENT_TIMES:
                        planned_travel = self.PLANNED_SEGMENT_TIMES[direct_key]
                    elif fallback_key in self.PLANNED_SEGMENT_TIMES:
                        planned_travel = self.PLANNED_SEGMENT_TIMES[fallback_key]
                    else:
                        planned_travel = self._sum_planned_times(prev_station, station_abbr)

                    segment = Segment(prev_station, station_abbr, 0, 140, True, False, False)
                else:
                    planned_travel = self._get_segment_time(prev_station, station_abbr)

                """ print(f"[{self.schedule.train_number}/{self.schedule.line}] "
                    f"{prev_station}→{station_abbr}: planned={planned_travel}s "
                    f"env.now={self.env.now:.0f} "
                    f"arr_planned={(self._ts_to_sim(arr_stop.planned_ts) if arr_stop else 'n/a')}") """
 
                # Weather-adjusted travel time — resolve conditions at the
                # moment the train departs this segment (env.now after any
                # headway wait), so rapidly-changing weather is captured.
                current_weather = self.weather.at(self.env.now)
                weather_travel = current_weather.travel_time(segment, planned_travel)
                weather_extra  = weather_travel - planned_travel
 
                # ── 2. ACQUIRE segment resource (blocks on single track) ────────
                resource   = self.resources.get(seg_key)
                blocked_by = None
                wait_start = self.env.now
 
                if resource is not None:
                    req = resource.request()
                    yield req   # wait until segment is free
                    waited = self.env.now - wait_start
                    if waited > 1:
                        blocked_by = self._find_occupant(seg_key)
                        self.conflict_log.append(ConflictEvent(
                            sim_time     = self.env.now,
                            train_number = schedule.train_number,
                            segment      = seg_key,
                            waited_sec   = waited,
                            blocked_by   = blocked_by,
                        ))
                        #self.current_delay += waited
 
                # ── 3. TRAVEL ──────────────────────────────────────────────────
                travel_causes = []
                if weather_extra > 1:
                    travel_causes.append(
                        f"weather(+{weather_extra:.0f}s on {seg_key[0]}→{seg_key[1]})"
                    )

                # add noise
                travel_factor = self.speed_factors[self.param_type]["sigma_travel"]
                travel_noise = random.gauss(0, weather_travel * travel_factor)  # 5% std
                yield self.env.timeout(weather_travel + travel_noise)
 
                # Release segment
                if resource is not None:
                    resource.release(req)
 
                #self.current_delay += weather_extra
 
            # ── 4. ARRIVAL ────────────────────────────────────────────────────
            if arr_stop is not None:
                arr_planned_sim = self._ts_to_sim(arr_stop.planned_ts)

                # If this is the very first event and no travel has happened yet,
                # the train is arriving from outside the corridor. Wait until the
                # planned arrival time before recording it.

                # TODO ask where to put
                """ first_dep = next((s for s in stops if s.event_type == "departure"), None)
                if first_dep and not np.isnan(first_dep.actual_delay):
                    self.current_delay = first_dep.actual_delay  # seed from ground truth """

                if prev_station is None:
                    wait = max(0, arr_planned_sim - self.env.now)
                    yield self.env.timeout(wait)
                
                arr_actual_sim  = self.env.now
                arr_delay       = arr_actual_sim - arr_planned_sim
                self.current_delay = arr_delay

                causes = []
                if arr_delay > self.current_delay + 1:
                    causes.append(f"propagated(+{arr_delay:.0f}s)")
 
                """ print(f"[{self.schedule.train_number}/{self.schedule.line}] "
                    f"ARRIVAL {station_abbr}: planned_sim={arr_planned_sim:.0f} "
                    f"actual_sim={arr_actual_sim:.0f} delay={arr_delay:.0f}s") """
                self.sim_events.append(SimEvent(
                    train_number    = schedule.train_number,
                    traffic_category= schedule.category,
                    line            = schedule.line,
                    station         = station_abbr,
                    event_type      = "arrival",
                    event_served    = schedule.event_served,
                    max_velocity    = schedule.max_speed_kmh,
                    period_id       = schedule.period_id,
                    stop_type       = arr_stop.stop_type,
                    planned_ts      = arr_stop.planned_ts,
                    actual_ts       = arr_stop.actual_ts,
                    simulated_ts    = self._sim_to_ts(arr_actual_sim),
                    simulated_delay = arr_delay,
                    actual_delay    = arr_stop.actual_delay,
                    causes          = causes,
                    blocked_by      = None,
                ))
 
            # ── 5. DWELL ──────────────────────────────────────────────────────
            if arr_stop is not None and dep_stop is not None:
                dep_planned_sim = self._ts_to_sim(dep_stop.planned_ts)
                
                if dep_stop.stop_type == "pass":
                    # Pass stops: no dwell, no noise — depart immediately
                    pass
                else:
                    # Commercial stops: wait until planned departure + small boarding noise
                    min_dwell    = MIN_DWELL.get(dep_stop.stop_type, 30)
                    earliest_dep = self.env.now + min_dwell
                    target_dep   = max(earliest_dep, dep_planned_sim, self.env.now)
                    dwell_factor = self.speed_factors[self.param_type]["sigma_dwell"]
                    dwell_noise  = abs(random.gauss(0, dwell_factor))
                    target_dep  += dwell_noise
                    dwell = target_dep - self.env.now
                    yield self.env.timeout(dwell)

            elif dep_stop is not None and arr_stop is None:
                dep_planned_sim = self._ts_to_sim(dep_stop.planned_ts)
                wait = max(0, dep_planned_sim - self.env.now)
                yield self.env.timeout(wait)

                # Seed with GCN prediction
                if self.entry_delays_sec is not None:
                    self.current_delay = self.entry_delays_sec
                    yield self.env.timeout(self.entry_delays_sec)
                    
                # Seed with the real initial delay if available
                elif not math.isnan(dep_stop.actual_delay):
                    self.current_delay = dep_stop.actual_delay
                    yield self.env.timeout(max(0, dep_stop.actual_delay))

            """ elif dep_stop is not None and arr_stop is None:
                dep_planned_sim = self._ts_to_sim(dep_stop.planned_ts)
                wait = max(0, dep_planned_sim - self.env.now)
                yield self.env.timeout(wait) """
            
            
            

 
            # ── 6. SWITCH FAILURE ─────────────────────────────────────────────
            switch_causes = []
            if dep_stop is not None:
                station = STATIONS.get(station_abbr)
                # Resolve weather at the current sim time for switch failure odds
                current_weather = self.weather.at(self.env.now)
                if station and random.random() < current_weather.switch_failure_prob():
                    extra = current_weather.switch_failure_delay_sec()
                    yield self.env.timeout(extra)
                    switch_causes.append(f"switch_failure(+{extra:.0f}s,snow={current_weather.htoauts0}cm)")
 
            # ── 7. DEPARTURE ──────────────────────────────────────────────────
            if dep_stop is not None:
                causes = switch_causes.copy()
                dep_planned_sim = self._ts_to_sim(dep_stop.planned_ts)

                if self.inject_delay is not None and str(self.inject_delay[0]) == str(station_abbr):
                    if self.inject_delay[1] is None:
                        yield self.env.timeout(self.inject_delay[2])
                        causes.append(f"injected {self.inject_delay[2]}s at station {self.inject_delay[0]}")
                    elif str(self.inject_delay[1]) == str(schedule.train_number):
                        yield self.env.timeout(self.inject_delay[2])
                        causes.append(f"injected {self.inject_delay[2]}s at train {self.inject_delay[1]}")
                    elif str(self.inject_delay[1]) == str(schedule.line):
                        yield self.env.timeout(self.inject_delay[2])
                        causes.append(f"injected {self.inject_delay[2]}s on line {self.inject_delay[1]}")

                dep_actual_sim = self.env.now   # now reflects the injected wait
                dep_delay      = dep_actual_sim - dep_planned_sim
                self.current_delay = dep_delay
               
                if dep_delay > 5:
                    causes.append(f"total_delay={dep_delay:.0f}s")
 
                self.sim_events.append(SimEvent(
                    train_number    = schedule.train_number,
                    traffic_category= schedule.category,
                    line            = schedule.line,
                    station         = station_abbr,
                    event_type      = "departure",
                    event_served    = True,
                    max_velocity    = schedule.max_speed_kmh,
                    period_id       = schedule.period_id,
                    stop_type       = dep_stop.stop_type,
                    planned_ts      = dep_stop.planned_ts,
                    actual_ts       = dep_stop.actual_ts,
                    simulated_ts    = self._sim_to_ts(dep_actual_sim),
                    simulated_delay = dep_delay,
                    actual_delay    = dep_stop.actual_delay,
                    causes          = causes,
                    blocked_by      = None,
                ))
 
            prev_station = station_abbr
 
    @staticmethod
    def _group_visits(
        stops: list[StopEntry],
        ) -> list[tuple[Optional[StopEntry], Optional[StopEntry]]]:
        """
        Pair arrival and departure stops at the same station into visit tuples.
        Returns list of (arrival_or_None, departure_or_None).
 
        Handles multiple visits to the same station (e.g. terminus reversal,
        layover, or duplicate data rows) by treating each consecutive
        arrival/departure pair as a separate visit, in chronological order.
        The old dict-based approach silently overwrote earlier stops when a
        station appeared more than once, causing massive phantom dwell waits.
        """
        # Sort by sequence number so order is guaranteed
        sorted_stops = sorted(stops, key=lambda s: (s.sequence, s.event_type))
 
        visits: list[tuple[Optional[StopEntry], Optional[StopEntry]]] = []
        i = 0
        while i < len(sorted_stops):
            s = sorted_stops[i]
            if s.event_type == "arrival":
                # Look ahead for a departure at the same station
                if (i + 1 < len(sorted_stops)
                        and sorted_stops[i + 1].event_type == "departure"
                        and sorted_stops[i + 1].station == s.station):
                    visits.append((s, sorted_stops[i + 1]))
                    i += 2
                else:
                    # Arrival only (terminus with no onward departure in corridor)
                    visits.append((s, None))
                    i += 1
            else:  # departure
                # Departure only (origin with no prior arrival in corridor)
                visits.append((None, s))
                i += 1
 
        return visits
    
    
    def _get_segment_time(self, origin: str, dest: str) -> int:
        key = (origin, dest, self.schedule.line)
        if key in self.PLANNED_SEGMENT_TIMES:
            return self.PLANNED_SEGMENT_TIMES[key]
        # Fallback: estimate from distance and max_speed
        seg = SEGMENTS.get((origin, dest))
        if seg:
            speed_ms = min(self.schedule.max_speed_kmh, seg.max_speed_kmh) / 3.6
            return int(seg.length_km * 1000 / speed_ms)
        return 180
 
    def _sum_planned_times(self, origin: str, dest: str) -> int:
        """Sum planned times for a multi-segment jump."""
        if origin not in LINE_ORDER or dest not in LINE_ORDER:
            return 300
        o_idx = LINE_ORDER.index(origin)
        d_idx = LINE_ORDER.index(dest)
        total = 0
        step  = 1 if d_idx > o_idx else -1
        for i in range(o_idx, d_idx, step):
            a, b = LINE_ORDER[i], LINE_ORDER[i + step]
            total += self._get_segment_time(a, b)
        return total
 
    def _find_occupant(self, seg_key: tuple[str, str]) -> Optional[int]:
        """Return the train number currently in the segment (best-effort)."""
        # Without a shared occupancy registry this is approximate;
        # the conflict log still records the wait time accurately.
        return None