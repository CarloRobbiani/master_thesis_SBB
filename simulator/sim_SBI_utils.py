import pandas as pd
import numpy as np

def select_calibration_days(
    parquet_path: str,
    n_days: int = 16,
    delay_col: str = "DAILY_PLAN_OPERATIONAL_DELAY_SEC",
    day_col:   str = "OPERATIONAL_DAY",
) -> list[str]:

    df = pd.read_parquet(parquet_path)
    df[day_col] = pd.to_datetime(df[day_col]).dt.date.astype(str)
    df["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(df["OPERATION_PLANNED_TIMESTAMP"])

    weather_cols = ["htoauts0", "rre150z0", "fu3010z0", "tre200s0", "fkl010z1"]
    present = [c for c in weather_cols if c in df.columns]

    deps = df[df["EVENT_TYPE"] == "departure"].copy()

    delay_stats = (
        deps.groupby(day_col)[delay_col]
        .agg(
            mean_delay = "mean",
            std_delay  = "std",
            p90_delay  = lambda x: x.quantile(0.90),
            frac_late  = lambda x: (x > 180).mean(),
            n_trains   = "count",
        )
        .reset_index()
    )

    weather_stats = (
        df.groupby(day_col)[present]
        .mean()
        .reset_index()
        .rename(columns={
            "htoauts0": "snow_cm",
            "rre150z0": "rain_mm10",
            "fu3010z0": "wind_ms",
            "tre200s0": "temp_c",
            "fkl010z1": "gust_ms",
        })
    )

    day_df = delay_stats.merge(weather_stats, on=day_col, how="left")
    day_df = day_df[day_df["n_trains"] >= 10].copy()

    # -- Adaptive regime boundaries from actual data percentiles ---------------
    low_thresh  = float(day_df["mean_delay"].quantile(0.33))
    high_thresh = float(day_df["mean_delay"].quantile(0.67))
    print(f"\nAdaptive delay thresholds:")
    print(f"  low  < {low_thresh:.1f}s  (bottom third)")
    print(f"  medium {low_thresh:.1f}s - {high_thresh:.1f}s  (middle third)")
    print(f"  high > {high_thresh:.1f}s  (top third)")

    day_df["delay_regime"] = pd.cut(
        day_df["mean_delay"],
        bins  = [-np.inf, low_thresh, high_thresh, np.inf],
        labels= ["low", "medium", "high"],
    )

    # Adaptive weather threshold — use the 75th percentile of rain and wind
    # since your corridor has no snow, rain and wind are the active signals
    rain_thresh = float(day_df["rain_mm10"].quantile(0.75)) * 6  # convert to mm/h
    wind_thresh = float(day_df["wind_ms"].quantile(0.75))
    print(f"\nAdaptive weather thresholds (based on your data):")
    print(f"  rain > {rain_thresh:.2f} mm/h  (top 25% of rainy days)")
    print(f"  wind > {wind_thresh:.1f} m/s   (top 25% of windy days)")

    day_df["has_rain"] = day_df["rain_mm10"] * 6 > rain_thresh
    day_df["has_wind"] = day_df["wind_ms"] > wind_thresh
    day_df["has_frost"]= day_df["temp_c"]  < -5
    day_df["has_snow"] = day_df["snow_cm"] > 5

    day_df["has_weather"] = (
        day_df["has_rain"] | day_df["has_wind"] |
        day_df["has_frost"] | day_df["has_snow"]
    )

    day_df["regime"] = (
        day_df["delay_regime"].astype(str) + "_" +
        day_df["has_weather"].map({True: "weather", False: "calm"})
    )

    # -- Print regime overview ------
    print(f"\n{'Regime':<22} {'Days':>6}  {'Mean delay':>12}  "
          f"{'Mean rain':>10}  {'Mean wind':>10}")
    print("  " + "-" * 68)
    for regime, grp in day_df.groupby("regime"):
        print(f"  {regime:<22} {len(grp):>6}  "
              f"{grp['mean_delay'].mean():>+11.1f}s  "
              f"{grp.get('rain_mm10', pd.Series([0])).mean()*6:>9.2f}mm/h  "
              f"{grp.get('wind_ms', pd.Series([0])).mean():>9.1f}m/s")

    # -- Stratified sampling: equal slots per regime, fill from available -----
    regimes = day_df["regime"].unique().tolist()
    slots_per_regime = max(2, n_days // len(regimes))

    selected = []
    print()
    for regime in sorted(regimes):
        candidates = (
            day_df[day_df["regime"] == regime]
            .sort_values("p90_delay", ascending=False)
        )
        # Take up to slots_per_regime, but take all if regime is small
        n_take = min(slots_per_regime, len(candidates))
        chosen = candidates.head(n_take)[day_col].tolist()
        selected.extend(chosen)

        print(f"  Regime '{regime}' ({len(candidates)} available) "
              f"→ {len(chosen)} selected:")
        for d in chosen:
            row = candidates[candidates[day_col] == d].iloc[0]
            print(f"    {d}  mean={row['mean_delay']:>+6.1f}s  "
                  f"p90={row['p90_delay']:>+7.1f}s  "
                  f"rain={row.get('rain_mm10', 0)*6:.2f}mm/h  "
                  f"wind={row.get('wind_ms', 0):.1f}m/s")

    # If still under n_days, fill remaining slots from the most informative
    # days not yet selected (highest p90 delay overall)
    seen = set(selected)
    remaining = (
        day_df[~day_df[day_col].isin(seen)]
        .sort_values("p90_delay", ascending=False)
    )
    extra_needed = n_days - len(selected)
    if extra_needed > 0 and not remaining.empty:
        extra = remaining.head(extra_needed)[day_col].tolist()
        selected.extend(extra)
        print(f"Filling {len(extra)} remaining slot(s) from highest p90 days:")
        for d in extra:
            row = remaining[remaining[day_col] == d].iloc[0]
            print(f"    {d}  mean={row['mean_delay']:>+6.1f}s  "
                  f"p90={row['p90_delay']:>+7.1f}s")

    # Deduplicate preserving order
    seen2 = set()
    final = [d for d in selected if not (d in seen2 or seen2.add(d))]

    print(f"Final selection: {len(final)} days")
    print(f" {final}")
    return final

if __name__ == "__main__":
    calibration_days = select_calibration_days(
    parquet_path = "data/train_data_weather.parquet",
    n_days       = 12,  
    )

    print(calibration_days)

    df = pd.read_csv("simulator/data/sim_with_sbi_long.csv")
    df["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(df["OPERATION_PLANNED_TIMESTAMP"])
    df["day"] = df["OPERATION_PLANNED_TIMESTAMP"].dt.date.astype(str)

    deps = df[df["EVENT_TYPE"] == "departure"].dropna(
        subset=["DAILY_PLAN_OPERATIONAL_DELAY_SEC"]
    )

    per_day = (
        deps.groupby("day")
        .apply(lambda g: pd.Series({
            "mae":        (g["SIMULATED_DELAY"] - g["DAILY_PLAN_OPERATIONAL_DELAY_SEC"]).abs().mean(),
            "bias":       (g["SIMULATED_DELAY"] - g["DAILY_PLAN_OPERATIONAL_DELAY_SEC"]).mean(),
            "actual_mean": g["DAILY_PLAN_OPERATIONAL_DELAY_SEC"].mean(),
            "sim_mean":    g["SIMULATED_DELAY"].mean(),
            "n":           len(g),
        }))
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    print(f"\n{'Day':<12} {'MAE':>8} {'Bias':>8} {'Act mean':>10} {'Sim mean':>10} {'N':>6}")
    print("-" * 58)
    for _, row in per_day.iterrows():
        flag = " ← outlier" if row["mae"] > 80 else ""
        print(f"  {row['day']:<12} {row['mae']:>7.1f}s {row['bias']:>+8.1f}s "
            f"{row['actual_mean']:>+9.1f}s {row['sim_mean']:>+9.1f}s "
            f"{int(row['n']):>6}{flag}")

    print(f"Overall MAE: {deps['SIMULATED_DELAY'].sub(deps['DAILY_PLAN_OPERATIONAL_DELAY_SEC']).abs().mean():.1f}s")

