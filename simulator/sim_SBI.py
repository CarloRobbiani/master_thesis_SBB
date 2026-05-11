"""
sim_sbi.py — Simulation-Based Inference calibration for the railway simulator
==============================================================================

Uses the `sbi` package (Neural Posterior Estimation) to infer the posterior
distribution over uncertain simulator parameters given observed delay statistics.

Install
-------
    pip install sbi torch

Usage
-----
    from sim_sbi import RailwayPrior, build_simulator_fn, run_sbi, posterior_summary

    # 1. Fit the posterior on your observed data
    posterior, summary = run_sbi(
        df_raw=df_raw,
        day="2025-01-15",
        PLANNED_SEGMENT_TIMES=PLANNED_SEGMENT_TIMES,
        num_simulations=2000,
    )

    # 2. Sample best parameters and inspect
    samples = posterior.sample((1000,), x=summary)
    posterior_summary(samples)

    # 3. Run the simulator with MAP estimate
    map_params = samples.mean(dim=0)
    result = run_with_params(map_params, df_raw, "2025-01-15", PLANNED_SEGMENT_TIMES)
    print(result.summary())
"""

from __future__ import annotations

import random
import warnings

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

# sbi imports — install with: pip install sbi
from sbi import analysis as analysis

from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform

from sim_weather import WeatherConditions, WeatherTimeline
from sim_timetable import Timetable
from sim_topology import build_planned_segment_times
from Simulator import RailwaySimulator


# ------------------------------------------------------------------------------
# 1. PARAMETER SPACE
# ------------------------------------------------------------------------------

class SBIParams:
    """
    The 5 uncertain simulator parameters we want to infer.

    All are dimensionless scalars or seconds — easy to put uniform priors on.

    sigma_travel  : std dev of segment travel noise as a fraction of planned time.
                    Currently hard-coded to 0.05 in TrainProcess.run().
    sigma_dwell   : std dev of the abs(gauss) boarding noise in seconds.
                    Currently hard-coded to 5 s.
    alpha_wind    : speed factor applied on exposed segments when fu3010z0 ≥ 25 m/s.
                    Currently hard-coded to 0.93.
    alpha_rain    : speed factor applied when rre150z0*6 ≥ 3 mm/h.
                    Currently hard-coded to 0.93.
    sf_scale      : multiplier on all switch-failure probabilities.
                    Currently implicit 1.0.
    """

    # ── names and bounds (same order as the tensor representation) ───────────
    NAMES  = ["sigma_travel", "sigma_dwell", "alpha_wind", "alpha_rain", "sf_scale"]
    LOWERS = [0.01,  1.0, 0.70, 0.80, 0.5]
    UPPERS = [0.15, 30.0, 1.00, 1.00, 3.0]

    def __init__(
        self,
        sigma_travel: float,
        sigma_dwell:  float,
        alpha_wind:   float,
        alpha_rain:   float,
        sf_scale:     float,
    ):
        self.sigma_travel = sigma_travel
        self.sigma_dwell  = sigma_dwell
        self.alpha_wind   = alpha_wind
        self.alpha_rain   = alpha_rain
        self.sf_scale     = sf_scale

    @classmethod
    def from_tensor(cls, t: Tensor) -> "SBIParams":
        # simulate_for_sbi may pass a batched tensor of shape [1, 5] or [5]
        v = t.squeeze().tolist()
        if isinstance(v, float):
            raise ValueError(f"Expected a 1-D parameter vector, got scalar. Shape was {t.shape}")
        return cls(*v)

    def to_tensor(self) -> Tensor:
        return torch.tensor(
            [self.sigma_travel, self.sigma_dwell,
             self.alpha_wind, self.alpha_rain, self.sf_scale],
            dtype=torch.float32,
        )

    @classmethod
    def default(cls) -> "SBIParams":
        """Current hard-coded values (the baseline before calibration)."""
        return cls(
            sigma_travel=0.05,
            sigma_dwell=5.0,
            alpha_wind=0.93,
            alpha_rain=0.93,
            sf_scale=1.0,
        )


def make_prior() -> BoxUniform:
    """Uniform prior over the 5-dimensional parameter space."""
    low  = torch.tensor(SBIParams.LOWERS, dtype=torch.float32)
    high = torch.tensor(SBIParams.UPPERS, dtype=torch.float32)
    return BoxUniform(low=low, high=high)


# ------------------------------------------------------------------------------
# 2. PATCHED SIMULATOR HELPERS
# ------------------------------------------------------------------------------

def _patch_weather_conditions(wc: WeatherConditions, p: SBIParams) -> WeatherConditions:
    """
    Return a WeatherConditions whose speed_factor() respects the SBI params.
    """

    class PatchedWeather(WeatherConditions):
        def speed_factor(self, segment) -> float:
            factor = 1.0
            if not segment.tunnel:
                if self.tre200s0 <= -5:
                    factor = min(factor, 0.9)
                if segment.exposed:
                    if self.fu3010z0 >= 40:
                        factor = min(factor, 0.70)
                    elif self.fu3010z0 >= 30:
                        factor = min(factor, 0.85)
                    elif self.fu3010z0 >= 25:
                        # ← tuned parameter: alpha_wind
                        factor = min(factor, p.alpha_wind)

                if self.rre150z0 * 6 >= 8:
                    factor = min(factor, 0.85)
                elif self.rre150z0 * 6 >= 3:
                    # ← tuned parameter: alpha_rain
                    factor = min(factor, p.alpha_rain)
                elif self.rre150z0 * 6 >= 1:
                    factor = min(factor, 0.97)

                if self.htoauts0 > 20:
                    factor = min(factor, 0.70)
                elif self.htoauts0 > 10:
                    factor = min(factor, 0.85)
            return factor

        def switch_failure_prob(self) -> float:
            base = super().switch_failure_prob()
            # ← tuned parameter: sf_scale
            return min(1.0, base * p.sf_scale)

    # Copy all field values into the patched subclass instance
    patched = PatchedWeather(
        tre200s0=wc.tre200s0, fkl010z1=wc.fkl010z1,
        fu3010z0=wc.fu3010z0, rre150z0=wc.rre150z0,
        htoauts0=wc.htoauts0, hto000d0=wc.hto000d0,
    )
    return patched


def _patch_timeline(timeline: WeatherTimeline, p: SBIParams) -> WeatherTimeline:
    """Rebuild a WeatherTimeline whose snapshots use the patched WeatherConditions."""
    patched_snapshots = [
        (t, _patch_weather_conditions(c, p))
        for t, c in zip(timeline._times, timeline._conds)
    ]
    return WeatherTimeline(patched_snapshots)


def _patch_train_process_noise(p: SBIParams):
    """
    Monkey-patch the `random.gauss` calls inside TrainProcess.run() by
    injecting a module-level override.
    We store the params in a thread-local-like global so the patched gauss
    function can read them.
    """
    import sim_processes as sp

    _orig_gauss = random.gauss

    def _patched_gauss(mu, sigma):
        # The simulator calls gauss in two places:
        #   travel_noise = gauss(0, weather_travel * 0.05)   → 5% of travel time
        #   dwell_noise  = abs(gauss(0, 5))                  → 5 s std dev
        # We intercept by checking the sigma magnitude:
        #   sigma ~ 0.05 * 60..300 s  → 3..15 s  → travel noise
        #   sigma == 5                → dwell noise
        if 1 < sigma < 50:            # travel noise call
            return _orig_gauss(mu, sigma * (p.sigma_travel / 0.05))
        elif abs(sigma - 5) < 0.1:   # dwell noise call
            return _orig_gauss(mu, p.sigma_dwell)
        return _orig_gauss(mu, sigma)

    sp.random.gauss = _patched_gauss   # type: ignore[attr-defined]
    return _orig_gauss, sp


def _restore_gauss(orig_gauss, sp_module):
    sp_module.random.gauss = orig_gauss


# ------------------------------------------------------------------------------
# 3. SUMMARY STATISTICS
# ------------------------------------------------------------------------------

def compute_summary(result) -> Tensor:
    """
    Compress a SimResult into a fixed-length summary statistic vector.

    Chosen statistics (12 total):
    ─────────────────────────────
    Delay distribution (departures only, vs plan):
      [0]  mean delay
      [1]  std delay
      [2]  median delay
      [3]  90th percentile delay
      [4]  fraction > 180 s (late by Swiss standard)
      [5]  fraction < -30 s (early departures)

    Conflict statistics:
      [6]  total number of conflicts
      [7]  mean conflict wait (s); 0 if no conflicts

    Error vs ground truth (where actual delay is known):
      [8]  MAE
      [9]  RMSE
      [10] bias (mean signed error)

    Cause mix:
      [11] fraction of events with a weather cause
    """
    df = result.to_dataframe()
    deps = df[df["EVENT_TYPE"] == "departure"]

    d = deps["SIMULATED_DELAY"].values.astype(float)
    if len(d) == 0:
        d = np.array([0.0])

    mean_d  = float(np.mean(d))
    std_d   = float(np.std(d))
    med_d   = float(np.median(d))
    p90_d   = float(np.percentile(d, 90))
    frac_late  = float(np.mean(d > 180))
    frac_early = float(np.mean(d < -30))

    cf = result.conflicts
    n_conflicts = float(len(cf))
    mean_wait   = float(np.mean([c.waited_sec for c in cf])) if cf else 0.0

    # Accuracy vs ground truth
    acc = result.accuracy()
    mae  = acc["mae"]  if not np.isnan(acc["mae"])  else 0.0
    rmse = acc["rmse"] if not np.isnan(acc["rmse"]) else 0.0

    valid = deps.dropna(subset=["DAILY_PLAN_OPERATIONAL_DELAY_SEC"])
    if not valid.empty:
        bias = float((valid["SIMULATED_DELAY"] - valid["DAILY_PLAN_OPERATIONAL_DELAY_SEC"]).mean())
    else:
        bias = 0.0

    # Weather cause fraction
    causes_col = df["causes"].fillna("")
    frac_weather = float((causes_col.str.contains("weather")).mean())

    stats = [
        mean_d, std_d, med_d, p90_d,
        frac_late, frac_early,
        n_conflicts, mean_wait,
        mae, rmse, bias,
        frac_weather,
    ]
    return torch.tensor(stats, dtype=torch.float32)


def observed_summary(df_raw: pd.DataFrame, day: str) -> Tensor:
    """
    Compute the same summary statistics directly from the *real* operational data
    (no simulation needed).  This is the x_o we condition the posterior on.
    """
    df = df_raw.copy()
    df["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(df["OPERATION_PLANNED_TIMESTAMP"])
    df["OPERATIONAL_DAY"] = pd.to_datetime(df["OPERATIONAL_DAY"]).dt.date.astype(str)
    day_df = df[df["OPERATIONAL_DAY"] == day].copy()

    deps = day_df[day_df["EVENT_TYPE"] == "departure"]
    d = deps["DAILY_PLAN_OPERATIONAL_DELAY_SEC"].dropna().values.astype(float)
    if len(d) == 0:
        d = np.array([0.0])

    mean_d  = float(np.mean(d))
    std_d   = float(np.std(d))
    med_d   = float(np.median(d))
    p90_d   = float(np.percentile(d, 90))
    frac_late  = float(np.mean(d > 180))
    frac_early = float(np.mean(d < -30))

    # No conflict / cause info in raw data — fill with zeros
    n_conflicts  = 0.0
    mean_wait    = 0.0
    mae          = 0.0
    rmse         = 0.0
    bias         = 0.0
    frac_weather = 0.0

    stats = [
        mean_d, std_d, med_d, p90_d,
        frac_late, frac_early,
        n_conflicts, mean_wait,
        mae, rmse, bias,
        frac_weather,
    ]
    return torch.tensor(stats, dtype=torch.float32)


# ------------------------------------------------------------------------------
# 4. SIMULATOR WRAPPER (theta → x)
# ------------------------------------------------------------------------------

def build_simulator_fn(
    df_raw: pd.DataFrame,
    day: str,
    PLANNED_SEGMENT_TIMES: dict,
    seed: Optional[int] = None,
):
    """
    Returns a callable  f(theta: Tensor) -> Tensor  compatible with sbi.

    Parameters
    ----------
    df_raw                : raw operational + weather DataFrame
    day                   : operational day string, e.g. "2025-01-15"
    PLANNED_SEGMENT_TIMES : pre-built segment time dict (pass the pickled one)
    seed                  : if set, fixes the random seed for reproducibility
    """
    # Build timetable once — shared across all simulations
    tt = Timetable.from_dataframe(df_raw, day)
    weather_timeline = WeatherTimeline.from_day_dataframe(df_raw, day)

    def simulator(theta: Tensor) -> Tensor:
        p = SBIParams.from_tensor(theta)

        # Patch noise
        orig_gauss, sp_mod = _patch_train_process_noise(p)

        try:
            patched_timeline = _patch_timeline(weather_timeline, p)

            sim = RailwaySimulator(
                PLANNED_SEGMENT_TIMES=PLANNED_SEGMENT_TIMES,
                timetable=tt,
                weather=patched_timeline,
                seed=seed,
            )
            result = sim.run()
            return compute_summary(result)

        except Exception as exc:
            warnings.warn(f"Simulation failed: {exc}; returning NaN summary.")
            return torch.full((12,), float("nan"))

        finally:
            _restore_gauss(orig_gauss, sp_mod)

    return simulator


# ------------------------------------------------------------------------------
# 5. MAIN SBI RUNNER
# ------------------------------------------------------------------------------

def run_sbi(
    df_raw: pd.DataFrame,
    day: str,
    PLANNED_SEGMENT_TIMES: dict,
    num_simulations: int = 2000,
    training_batch_size: int = 50,
    seed: int = 42,
):
    """
    Run the full SBI pipeline:
      prior → simulate → train NPE → return posterior.

    Parameters
    ----------
    df_raw               : raw operational + weather DataFrame
    day                  : operational day to calibrate on
    PLANNED_SEGMENT_TIMES: pre-built segment time lookup dict
    num_simulations      : number of (theta, x) pairs to generate.
                           ≥ 1000 recommended; 5000+ for publication quality.
    training_batch_size  : mini-batch size for NPE training
    seed                 : RNG seed for reproducibility

    Returns
    -------
    posterior   : sbi DirectPosterior — call .sample((N,), x=x_obs)
    x_obs       : the observed summary statistic tensor for `day`
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    prior     = make_prior() # Build priors based on LOWERS and UPPERS
    simulator = build_simulator_fn(df_raw, day, PLANNED_SEGMENT_TIMES, seed=None)
    x_obs     = observed_summary(df_raw, day)

    print(f"[SBI] Observed summary statistics for {day}:")
    for name, val in zip(
        ["mean_delay","std_delay","median_delay","p90_delay",
         "frac_late","frac_early","n_conflicts","mean_wait",
         "MAE","RMSE","bias","frac_weather"],
        x_obs.tolist()
    ):
        print(f"        {name:<20} {val:+.2f}")

    print(f"\n[SBI] Running {num_simulations} simulations …")

    theta, x = simulate_for_sbi(
        simulator=simulator,
        proposal=prior,
        num_simulations=num_simulations,
        num_workers=1,           
    )

    # simulate_for_sbi returns x with shape [num_simulations, summary_dim]
    # ensure it is 2D before filtering (older sbi versions may squeeze it)
    if x.dim() == 1:
        x = x.unsqueeze(0) if len(x) == 12 else x.reshape(num_simulations, -1)

    # Drop NaN simulations (failed runs)
    valid_mask = ~torch.isnan(x).any(dim=1)
    n_dropped  = (~valid_mask).sum().item()
    if n_dropped > 0:
        print(f"[SBI] Dropped {n_dropped} failed simulations.")
    theta, x = theta[valid_mask], x[valid_mask]
    print(f"[SBI] x shape after filtering: {x.shape}, theta shape: {theta.shape}")

    print(f"[SBI] Training NPE on {len(theta)} simulations …")
    inference = NPE(prior=prior)
    inference.append_simulations(theta, x)
    density_estimator = inference.train(training_batch_size=training_batch_size)

    posterior = inference.build_posterior(density_estimator)

    print("[SBI] Done. Use  posterior.sample((1000,), x=x_obs)  to draw samples.")
    return posterior, x_obs


# ------------------------------------------------------------------------------
# 6. MULTI-DAY SBI (pool simulations across several days)
# ------------------------------------------------------------------------------

def run_sbi_multiday(
    df_raw: pd.DataFrame,
    days: list[str],
    PLANNED_SEGMENT_TIMES: dict,
    num_simulations_per_day: int = 1000,
    seed: int = 42,
):
    """
    Pool simulations across multiple days for a more robust posterior.
 
    Each (theta, x) pair is associated with the summary statistics of one
    specific day, so the inference jointly calibrates over all days.
    The x_obs returned is a stacked tensor of all observed summaries.
 
    This works because NPE supports amortised inference: once trained, you can
    condition on the x_obs of any individual day without re-training.
 
    Returns
    -------
    posterior   : amortised posterior, conditioned on any single day's x_obs
    x_obs_dict  : {day: Tensor} — observed summaries per day
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    prior = make_prior()
    all_theta, all_x = [], []
 
    SUMMARY_DIM = 12  # length of the vector returned by compute_summary()
 
    x_obs_dict = {}
    for day in days:
        print(f"\n[SBI] Day {day}: building simulator …")
        simulator = build_simulator_fn(df_raw, day, PLANNED_SEGMENT_TIMES, seed=None)
        x_obs_dict[day] = observed_summary(df_raw, day)
 
        theta_day, x_day = simulate_for_sbi(
            simulator=simulator,
            proposal=prior,
            num_simulations=num_simulations_per_day,
            num_workers=1,
        )
 
        # Ensure 2-D: simulate_for_sbi may return a flat tensor on some sbi versions
        if x_day.dim() == 1:
            x_day = x_day.reshape(num_simulations_per_day, SUMMARY_DIM)
 
        valid = ~torch.isnan(x_day).any(dim=1)
        all_theta.append(theta_day[valid])
        all_x.append(x_day[valid])
        print(f"[SBI] Day {day}: {valid.sum().item()} / {num_simulations_per_day} valid simulations.")
 
    theta_all = torch.cat(all_theta, dim=0)
    x_all     = torch.cat(all_x, dim=0)
    print(f"\n[SBI] Total training set: {len(theta_all)} simulations, x shape {x_all.shape}")
 
    print(f"[SBI] Training NPE …")
    inference = NPE(prior=prior)
    inference.append_simulations(theta_all, x_all)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
 
    print("[SBI] Multi-day training done.")
    # Return both the per-day dict AND the mean summary so callers can do either:
    #   posterior.sample((N,), x=x_obs_dict["2025-01-01"])   # single day
    #   posterior.sample((N,), x=x_obs_mean)                 # average across days
    x_obs_mean = torch.stack(list(x_obs_dict.values())).mean(dim=0)
    return posterior, x_obs_dict, x_obs_mean



# ------------------------------------------------------------------------------
# 7. ANALYSIS UTILITIES
# ------------------------------------------------------------------------------

def posterior_summary(samples: Tensor, ci: float = 0.9):
    """
    Print a table of posterior mean ± credible interval for each parameter.

    Parameters
    ----------
    samples : (N, 5) tensor from posterior.sample()
    ci      : credible interval width (default 90 %)
    """
    lo = (1 - ci) / 2
    hi = 1 - lo
    print(f"\n{'Parameter':<18} {'Mean':>10} {'Std':>8}  {int(ci*100)}% CI")
    print("─" * 55)
    for i, name in enumerate(SBIParams.NAMES):
        col = samples[:, i].numpy()
        print(
            f"{name:<18} {col.mean():>10.4f} {col.std():>8.4f}"
            f"  [{np.quantile(col, lo):.4f}, {np.quantile(col, hi):.4f}]"
        )


def plot_posterior(samples: Tensor, prior: BoxUniform, save_path: str | None = None):
    """
    Pair-plot of the posterior samples using sbi's built-in pairplot.

    Parameters
    ----------
    samples   : (N, 5) tensor
    prior     : the BoxUniform prior (for reference lines)
    save_path : if given, save the figure to this path
    """
    import matplotlib.pyplot as plt
    fig, axes = analysis.pairplot(
        samples,
        labels=SBIParams.NAMES,
        limits=list(zip(SBIParams.LOWERS, SBIParams.UPPERS)),
        figsize=(10, 10),
    )
    fig.suptitle("Posterior distribution of simulator parameters", y=1.01)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved pairplot to {save_path}")
    plt.show()


def run_with_params(
    theta: Tensor,
    df_raw: pd.DataFrame,
    day: str,
    PLANNED_SEGMENT_TIMES: dict,
    seed: int = 42,
):
    """
    Run the simulator with a specific parameter vector (e.g. posterior MAP).

    Returns a SimResult so you can call .summary(), .plot(), .to_csv() etc.
    """
    p  = SBIParams.from_tensor(theta)
    tt = Timetable.from_dataframe(df_raw, day)
    timeline = WeatherTimeline.from_day_dataframe(df_raw, day)

    orig_gauss, sp_mod = _patch_train_process_noise(p)
    try:
        patched_timeline = _patch_timeline(timeline, p)
        sim = RailwaySimulator(
            PLANNED_SEGMENT_TIMES=PLANNED_SEGMENT_TIMES,
            timetable=tt,
            weather=patched_timeline,
            seed=seed,
        )
        return sim.run()
    finally:
        _restore_gauss(orig_gauss, sp_mod)


# ------------------------------------------------------------------------------
# 8. ENTRY POINT — quick demo
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, pickle, os
    from pathlib import Path

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_data_weather.parquet"
    day_arg   = sys.argv[2] if len(sys.argv) > 2 else None
    n_sims    = int(sys.argv[3]) if len(sys.argv) > 3 else 500   # reduce for quick test

    p = Path(data_path)
    df_raw = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df_raw["OPERATIONAL_DAY"] = pd.to_datetime(
        df_raw["OPERATIONAL_DAY"]
    ).dt.date.astype(str)

    available = sorted(df_raw["OPERATIONAL_DAY"].unique())
    day = day_arg or available[0]
    day = "2025-02-20"

    # Load or build segment times
    pkl = "simulator/timetable.pkl"
    if os.path.isfile(pkl):
        with open(pkl, "rb") as f:
            PLANNED_SEGMENT_TIMES = pickle.load(f)
    else:
        PLANNED_SEGMENT_TIMES = build_planned_segment_times(df_raw)
        with open(pkl, "wb") as fp:
            pickle.dump(PLANNED_SEGMENT_TIMES, fp)

    # Apply manual corrections
    PLANNED_SEGMENT_TIMES.update({
        ("TUE", "BI",  "IC5"): 240,
        ("NE",  "SBL", "IC5"): 180,
        ("LIG", "TWN", "IC5"): 120,
        ("TWN", "LIG", "IC5"): 120,
        ("LIG", "NV",  "IC5"): 120,
        ("NV",  "LIG", "IC5"): 120,
        ("SBL", "CORN","IC5"): 120,
        ("NV",  "LIG", "R13"): 180,
        ("NE",  "SBL", "R13"): 180,
        ("NV",  "LIG", "R16"): 180,
    })

    print(f"Running SBI on day {day} with {n_sims} simulations …\n")

    """ posterior, x_obs = run_sbi(
        df_raw=df_raw,
        day=day,
        PLANNED_SEGMENT_TIMES=PLANNED_SEGMENT_TIMES,
        num_simulations=n_sims,
        seed=42,
    ) """

    list_of_days = ["2025-01-01", "2025-01-20", "2025-02-10", "2025-03-20"]
    posterior, x_obs, x_obs_mean = run_sbi_multiday(
        df_raw=df_raw,
        days=list_of_days,
        PLANNED_SEGMENT_TIMES=PLANNED_SEGMENT_TIMES,
        num_simulations_per_day=n_sims,
        seed=42
    )

    # Sample from the posterior
    #samples = posterior.sample((2000,), x=x_obs)
    samples = posterior.sample((2000,), x=x_obs_mean)
    posterior_summary(samples)

    prior = make_prior()
    plot_posterior(samples, prior, save_path="simulator/posterior_pairplot.png")

    # Run the simulator with the posterior mean
    map_theta = samples.mean(dim=0)
    print(f"\nRunning simulator with posterior-mean parameters …")
    result = run_with_params(map_theta, df_raw, day, PLANNED_SEGMENT_TIMES)
    print(result.summary())
    result.to_csv(f"simulator/sbi_calibrated_{day}.csv")