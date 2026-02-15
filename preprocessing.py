"""
Preprocessing module for air pollution source identification.

Provides feature engineering: cyclical time encoding, wind vector
decomposition, and pollution ratios (PM2.5 / PM10).
"""

import pandas as pd
import numpy as np


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
    # Meteorological: direction FROM which wind blows. Convert to radians.
    # 0° = N, 90° = E. u = -speed*sin(dir), v = -speed*cos(dir) for "from" convention.
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


def preprocess(
    df: pd.DataFrame,
    date_col: str = "date",
    wind_speed_col: str = "wind_speed",
    wind_dir_col: str = "wind_dir",
    pm25_col: str = "PM25",
    pm10_col: str = "PM10",
) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline: cyclical month, wind u/v, PM ratio.

    Args:
        df: Raw DataFrame with date, PM25, PM10, wind_dir, wind_speed, fire_count.
        date_col: Name of date column.
        wind_speed_col: Name of wind speed column.
        wind_dir_col: Name of wind direction column.
        pm25_col: Name of PM2.5 column.
        pm10_col: Name of PM10 column.

    Returns:
        DataFrame with original columns plus month_sin, month_cos, wind_u,
        wind_v, and pm_ratio. Suitable for model training or prediction.
    """
    df = add_cyclical_month(df, date_col=date_col)
    df = add_wind_components(
        df, wind_speed_col=wind_speed_col, wind_dir_col=wind_dir_col
    )
    df = add_pm_ratio(df, pm25_col=pm25_col, pm10_col=pm10_col)
    return df


def get_feature_columns() -> list:
    """
    Return the list of feature column names used by the model.

    Returns:
        List of feature names: month_sin, month_cos, wind_u, wind_v,
        pm_ratio, plus PM25, PM10, wind_speed, wind_dir, fire_count.
    """
    return [
        "month_sin",
        "month_cos",
        "wind_u",
        "wind_v",
        "pm_ratio",
        "PM25",
        "PM10",
        "wind_speed",
        "wind_dir",
        "fire_count",
    ]
