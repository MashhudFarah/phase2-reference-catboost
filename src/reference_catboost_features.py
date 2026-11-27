"""Reference-repository CatBoost feature contract for Phase2 data.

The reference implementation consumes the raw columns present in the Phase2
processed CSV and engineers features per ward. All lagged/rolling values use
observations strictly before the origin; current-week values are only included
where the reference uses contemporaneous weather or dengue.
"""
from __future__ import annotations

import pandas as pd

from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    SOURCE_COLUMNS as RAW_COLUMNS,
    engineer_features as _engineer_features,
    usable_feature_mask as _usable_feature_mask,
)


def feature_names() -> list[str]:
    return list(FEATURE_COLUMNS)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical feature-engineered rows used by every backend."""
    return _engineer_features(frame)


def usable_feature_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows with all canonical model inputs available."""
    return _usable_feature_mask(frame)
