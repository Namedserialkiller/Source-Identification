"""
Preprocessing module for air pollution source identification.

Provides feature engineering: cyclical time encoding, wind vector
decomposition, pollution ratios (PM2.5 / PM10), nitrogen ratio (NO2/NO),
and normalization of gaseous pollutants.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Gaseous pollutant columns for normalization
GASEOUS_COLS = ["co", "no", "no2", "o3", "so2", "nh3"]

# Canonical column name mapping (delhi_aqi uses pm2_5, pm10)
COLUMN_ALIASES = {
    "pm2_5": "PM25",
    "pm25": "PM25",
    "PM2.5": "PM25",
    "pm10": "PM10",
    "PM10": "PM10",
}


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical names (e.g., pm2_5 -> PM25).

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with standardized column names.
    """
    df = df.copy()
    for alias, canonical in COLUMN_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            df[canonical] = df[alias]
        elif alias in df.columns and canonical in df.columns:
            df[canonical] = df[canonical].fillna(df[alias])
    return df


def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wind and fire_count with default values if missing.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with wind_dir, wind_speed, fire_count present.
    """
    df = df.copy()
    if "wind_dir" not in df.columns:
        df["wind_dir"] = 0.0
    if "wind_speed" not in df.columns:
        df["wind_speed"] = 0.0
    if "fire_count" not in df.columns:
        df["fire_count"] = 0
    return df


def add_cyclical_month(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Encode month as cyclical features using sin/cos to preserve periodicity.

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column (parsed if string).

    Returns:
        DataFrame with added columns month_sin and month_cos.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    months = df[date_col].dt.month
    df["month_sin"] = np.sin(2 * np.pi * months / 12)
    df["month_cos"] = np.cos(2 * np.pi * months / 12)
    return df


def add_wind_components(
    df: pd.DataFrame,
    wind_speed_col: str = "wind_speed",
    wind_dir_col: str = "wind_dir",
) -> pd.DataFrame:
    """
    Decompose wind into u (east-west) and v (north-south) components.

    Convention: wind_dir is direction wind comes FROM (meteorological).
    u positive = wind from west; v positive = wind from south.

    Args:
        df: DataFrame with wind_speed and wind_dir (degrees).
        wind_speed_col: Name of wind speed column.
        wind_dir_col: Name of wind direction column (degrees, 0=N).

    Returns:
        DataFrame with added columns wind_u and wind_v.
    """
    df = df.copy()
    speed = df[wind_speed_col].astype(float)
    deg = np.deg2rad(df[wind_dir_col].astype(float))
    df["wind_u"] = -speed * np.sin(deg)
    df["wind_v"] = -speed * np.cos(deg)
    return df


def add_pm_ratio(
    df: pd.DataFrame,
    pm25_col: str = "PM25",
    pm10_col: str = "PM10",
    ratio_name: str = "pm_ratio",
) -> pd.DataFrame:
    """
    Add pollution ratio PM2.5 / PM10, with safe division (avoid inf/NaN).

    Args:
        df: DataFrame with PM25 and PM10 columns.
        pm25_col: Name of PM2.5 column.
        pm10_col: Name of PM10 column.
        ratio_name: Name of the output ratio column.

    Returns:
        DataFrame with added ratio column (NaN where PM10 is 0).
    """
    df = df.copy()
    pm25 = df[pm25_col].astype(float)
    pm10 = df[pm10_col].astype(float)
    df[ratio_name] = np.where(pm10 > 0, pm25 / pm10, np.nan)
    return df


def add_nitrogen_ratio(
    df: pd.DataFrame,
    no_col: str = "no",
    no2_col: str = "no2",
    ratio_name: str = "nitrogen_ratio",
) -> pd.DataFrame:
    """
    Add nitrogen ratio NO2 / NO (tracer for combustion/oxidation state).

    Args:
        df: DataFrame with NO and NO2 columns.
        no_col: Name of NO column.
        no2_col: Name of NO2 column.
        ratio_name: Name of the output ratio column.

    Returns:
        DataFrame with added ratio column (NaN where NO is 0).
    """
    df = df.copy()
    no = df[no_col].astype(float)
    no2 = df[no2_col].astype(float)
    df[ratio_name] = np.where(no > 0, no2 / no, np.nan)
    return df


def normalize_gaseous_pollutants(
    df: pd.DataFrame,
    columns: list = None,
    scaler: StandardScaler = None,
) -> tuple:
    """
    Normalize gaseous pollutants (co, no, no2, o3, so2, nh3) to account for
    different scales. Uses StandardScaler (z-score normalization).

    Args:
        df: DataFrame containing gaseous pollutant columns.
        columns: List of column names to normalize; default GASEOUS_COLS.
        scaler: Fitted StandardScaler; if None, fit on data and return.

    Returns:
        Tuple of (transformed DataFrame, fitted StandardScaler).
    """
    df = df.copy()
    cols = columns or GASEOUS_COLS
    available = [c for c in cols if c in df.columns]
    if not available:
        return df, scaler
    X = df[available].astype(float)
    X = X.fillna(X.median())
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        fit_cols = getattr(scaler, "feature_names_in_", cols)
        available = [c for c in fit_cols if c in df.columns]
        if not available:
            return df, scaler
        X = df[available].astype(float).fillna(0)
        X_scaled = scaler.transform(X)
    df[available] = X_scaled
    return df, scaler


def preprocess(
    df: pd.DataFrame,
    date_col: str = "date",
    wind_speed_col: str = "wind_speed",
    wind_dir_col: str = "wind_dir",
    pm25_col: str = "PM25",
    pm10_col: str = "PM10",
    scaler: StandardScaler = None,
) -> tuple:
    """
    Apply full preprocessing pipeline: cyclical month, wind u/v, PM ratio,
    nitrogen ratio, and gaseous normalization.

    Args:
        df: Raw DataFrame with date, PM, gaseous pollutants, optionally wind
            and fire_count.
        date_col: Name of date column.
        wind_speed_col: Name of wind speed column.
        wind_dir_col: Name of wind direction column.
        pm25_col: Name of PM2.5 column (or pm2_5).
        pm10_col: Name of PM10 column.
        scaler: Fitted StandardScaler for gaseous norm; None = fit on data.

    Returns:
        Tuple of (preprocessed DataFrame, fitted StandardScaler).
        Scaler is returned for persistence; use for prediction.
    """
    df = _normalize_column_names(df)
    df = _ensure_optional_columns(df)
    df = add_cyclical_month(df, date_col=date_col)
    df = add_wind_components(
        df, wind_speed_col=wind_speed_col, wind_dir_col=wind_dir_col
    )
    df = add_pm_ratio(df, pm25_col="PM25", pm10_col="PM10")
    # Add nitrogen ratio if no/no2 present
    if "no" in df.columns and "no2" in df.columns:
        df = add_nitrogen_ratio(df)
    else:
        df["nitrogen_ratio"] = np.nan
    # Normalize gaseous pollutants (add missing with 0 for consistency)
    for c in GASEOUS_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df, scaler = normalize_gaseous_pollutants(df, columns=GASEOUS_COLS, scaler=scaler)
    return df, scaler


def get_feature_columns(exclude: list = None) -> list:
    """
    Return the list of feature column names used by the model.

    Includes cyclical time, wind, ratios, gaseous pollutants (normalized),
    and PM metrics.

    Args:
        exclude: Optional list of column names to exclude (e.g. for ablation).

    Returns:
        List of feature names.
    """
    cols = [
        "month_sin",
        "month_cos",
        "wind_u",
        "wind_v",
        "pm_ratio",
        "nitrogen_ratio",
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "nh3",
        "PM25",
        "PM10",
        "wind_speed",
        "wind_dir",
        "fire_count",
    ]
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols
