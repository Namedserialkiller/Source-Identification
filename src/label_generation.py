"""
Weak supervision logic for air pollution source labels.

Assigns labels: stubble, traffic, industry, mixed based on scientifically
robust heuristics using month, fire_count, PM ratio, and chemical tracers
(nh3, no2, so2).
"""

import pandas as pd
import numpy as np


def ensure_date_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure the date column is datetime and extract month if needed.

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with date parsed; month in 1-12 if used later.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def generate_labels(
    df: pd.DataFrame,
    date_col: str = "date",
    fire_count_col: str = "fire_count",
    pm_ratio_col: str = "pm_ratio",
    nh3_col: str = "nh3",
    no2_col: str = "no2",
    so2_col: str = "so2",
    pm10_col: str = "PM10",
    label_noise_fraction: float = 0.10,
    threshold_noise_fraction: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    """
    Assign source labels using refined weak supervision rules.

    Rules (applied in order; later rules can override):
    - Stubble Burning: month in [10, 11] AND fire_count > 5 AND high nh3
      (ammonia is a signature of biomass burning).
    - Traffic: no2 > 75th percentile AND fire_count == 0 AND high pm_ratio.
    - Industry: so2 > 80th percentile OR (low pm_ratio AND high pm10).
    - Mixed/Other: Default when no rule applies.

    Args:
        df: DataFrame with date, fire_count, pm_ratio, nh3, no2, so2, pm10.
        date_col: Name of date column.
        fire_count_col: Name of fire count column.
        pm_ratio_col: Name of PM ratio column (PM2.5/PM10).
        nh3_col: Name of NH3 (ammonia) column.
        no2_col: Name of NO2 column.
        so2_col: Name of SO2 column.
        pm10_col: Name of PM10 column.

    Returns:
        Series of string labels: 'stubble', 'traffic', 'industry', 'mixed'.
    """
    rng = np.random.default_rng(random_state)
    df = ensure_date_column(df, date_col=date_col)
    month = df[date_col].dt.month
    fire_count = df[fire_count_col].astype(float)
    pm_ratio = df[pm_ratio_col].astype(float)

    # Compute percentiles (handle NaNs and edge cases)
    nh3 = df[nh3_col].astype(float) if nh3_col in df.columns else pd.Series(0.0, index=df.index)
    no2 = df[no2_col].astype(float) if no2_col in df.columns else pd.Series(0.0, index=df.index)
    so2 = df[so2_col].astype(float) if so2_col in df.columns else pd.Series(0.0, index=df.index)
    pm10 = df[pm10_col].astype(float) if pm10_col in df.columns else pd.Series(0.0, index=df.index)

    nh3_75 = nh3.quantile(0.75) if nh3.notna().any() else 0
    no2_75 = no2.quantile(0.75) if no2.notna().any() else 0
    so2_80 = so2.quantile(0.80) if so2.notna().any() else 0
    pm10_75 = pm10.quantile(0.75) if pm10.notna().any() else 0

    # Soft thresholds: add 5% Gaussian noise so boundaries aren't identical
    nh3_75 *= 1 + rng.normal(0, threshold_noise_fraction)
    no2_75 *= 1 + rng.normal(0, threshold_noise_fraction)
    so2_80 *= 1 + rng.normal(0, threshold_noise_fraction)
    pm10_75 *= 1 + rng.normal(0, threshold_noise_fraction)
    nh3_75 = max(nh3_75, 0)
    no2_75 = max(no2_75, 0)
    so2_80 = max(so2_80, 0)
    pm10_75 = max(pm10_75, 0)

    high_nh3 = nh3 >= nh3_75
    high_no2 = no2 >= no2_75
    high_so2 = so2 >= so2_80
    high_pm_ratio = pm_ratio > 0.5
    low_pm_ratio = pm_ratio < 0.4
    high_pm10 = pm10 >= pm10_75

    labels = np.full(len(df), "mixed", dtype=object)

    # Industry: SO2 > 80th percentile (industrial combustion) OR
    #           low pm_ratio AND high pm10 (coarse-dominated, e.g. dust/industry)
    industry_mask = high_so2 | (low_pm_ratio & high_pm10)
    labels[industry_mask] = "industry"

    # Traffic: NO2 > 75th percentile (vehicular exhaust) AND no fires AND high pm_ratio
    traffic_mask = high_no2 & (fire_count == 0) & high_pm_ratio
    labels[traffic_mask] = "traffic"

    # Stubble: Oct/Nov (harvest season) AND fire_count > 5 AND high NH3
    # (ammonia signature of biomass burning)
    stubble_mask = (
        (month.isin([10, 11]))
        & (fire_count > 5)
        & high_nh3
    )
    labels[stubble_mask] = "stubble"

    # Introduce label noise: randomly flip 10% to a different class
    # Prevents model from finding perfect mathematical thresholds
    labels = pd.Series(labels, index=df.index)
    classes = np.array(["stubble", "traffic", "industry", "mixed"])
    n_flip = int(len(labels) * label_noise_fraction)
    if n_flip > 0:
        flip_idx = rng.choice(len(labels), size=n_flip, replace=False)
        replacements = []
        for i in flip_idx:
            current = labels.iloc[i]
            others = [c for c in classes if c != current]
            replacements.append(rng.choice(others) if others else current)
        labels.iloc[flip_idx] = replacements

    return labels


def add_labels(
    df: pd.DataFrame,
    label_col: str = "source",
    date_col: str = "date",
    fire_count_col: str = "fire_count",
    pm_ratio_col: str = "pm_ratio",
    nh3_col: str = "nh3",
    no2_col: str = "no2",
    so2_col: str = "so2",
    pm10_col: str = "PM10",
    label_noise_fraction: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add a source label column to the DataFrame using weak supervision.

    Args:
        df: DataFrame with date, fire_count, pm_ratio, nh3, no2, so2, pm10.
        label_col: Name of the new label column.
        date_col: Name of date column.
        fire_count_col: Name of fire count column.
        pm_ratio_col: Name of PM ratio column.
        nh3_col: Name of NH3 column.
        no2_col: Name of NO2 column.
        so2_col: Name of SO2 column.
        pm10_col: Name of PM10 column.

    Returns:
        DataFrame with new column `label_col` containing source labels.
    """
    df = df.copy()
    df[label_col] = generate_labels(
        df,
        date_col=date_col,
        fire_count_col=fire_count_col,
        pm_ratio_col=pm_ratio_col,
        nh3_col=nh3_col,
        no2_col=no2_col,
        so2_col=so2_col,
        pm10_col=pm10_col,
        label_noise_fraction=label_noise_fraction,
        random_state=random_state,
    )
    return df
