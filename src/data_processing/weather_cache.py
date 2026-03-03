"""
Simple weather caching helpers.

Weather data is expected to be keyed by:
- datetime_hour
- lat_grid
- lon_grid

and stored in a parquet file for re-use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import logging
import pandas as pd

from config import WEATHER_CACHE_PATH  # type: ignore

logger = logging.getLogger(__name__)


def load_weather_from_cache(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load cached weather data if it exists.

    Returns None if no cache is present.
    """
    cache_path = Path(path or WEATHER_CACHE_PATH)
    if not cache_path.exists():
        logger.info("Weather cache not found at %s", cache_path)
        return None

    try:
        df = pd.read_parquet(cache_path)
        logger.info("Loaded weather cache from %s with %d rows.", cache_path, len(df))
        return df
    except Exception as exc:
        logger.warning("Failed to load weather cache from %s: %s", cache_path, exc)
        return None


def save_weather_to_cache(weather_df: pd.DataFrame, path: Optional[Path] = None) -> None:
    """
    Save weather data to cache for later reuse.
    """
    cache_path = Path(path or WEATHER_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    weather_df.to_parquet(cache_path, index=False)
    logger.info("Saved weather cache to %s with %d rows.", cache_path, len(weather_df))

