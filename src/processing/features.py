"""
src/processing/features.py — Feature engineering, cleaning, and normalisation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# Pure audio features from ReccoBeats — used for clustering and UMAP
CORE_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

# Optional metadata (release_year only — popularity removed from clustering)
META_FEATURES = ["release_year"]

# All numeric columns for descriptive statistics
ALL_NUMERIC = CORE_FEATURES + ["popularity", "release_year"]


def engineer_features(
    df: pd.DataFrame,
    include_meta: bool = True,
    include_categorical: bool = False,   # key/mode not available from ReccoBeats
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, list[str]]:
    """
    Clean and standardise features for UMAP + clustering.
    Uses pure audio features only (no popularity/genre).
    """
    df = df.copy()

    audio_available = (
        "energy" in df.columns
        and df["energy"].notna().any()
        and df["danceability"].notna().any()
        and df["tempo"].notna().any()
    )

    if audio_available:
        log.info("Audio features detected — using pure audio feature set (no popularity).")
        before = len(df)
        df = df.dropna(subset=CORE_FEATURES).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            log.warning("Dropped %d tracks with missing audio features.", dropped)

        # Fill release_year for optional inclusion
        if include_meta and "release_year" in df.columns:
            df["release_year"] = df["release_year"].fillna(df["release_year"].median())

        numeric_cols = CORE_FEATURES + (META_FEATURES if include_meta else [])
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        feature_parts: list[pd.DataFrame] = [df[numeric_cols]]

    else:
        log.warning("Audio features unavailable — falling back to metadata only.")
        for col in ["release_year", "duration_min", "explicit"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        df = df.dropna(subset=["duration_min"]).reset_index(drop=True)
        numeric_cols = [c for c in ["release_year", "duration_min", "explicit"]
                        if c in df.columns]
        feature_parts = [df[numeric_cols]]

    feature_df = pd.concat(feature_parts, axis=1).astype(float)
    feature_names = list(feature_df.columns)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)

    log.info("Feature matrix: %d tracks × %d features.", X.shape[0], X.shape[1])
    return df, X, scaler, feature_names


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in ALL_NUMERIC if c in df.columns]
    return df[available].describe().T.round(3)


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in CORE_FEATURES if c in df.columns]
    return df[available].corr().round(3)


def get_feature_importance_for_clusters(
    X_raw: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    from scipy.stats import f_oneway
    importances: dict[str, float] = {}
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    for i, name in enumerate(feature_names):
        groups = [X_raw[labels == lbl, i] for lbl in unique_labels]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            f_stat, _ = f_oneway(*groups)
            importances[name] = float(f_stat) if not np.isnan(f_stat) else 0.0
        else:
            importances[name] = 0.0

    series = pd.Series(importances).sort_values(ascending=False)
    return series.rename("F_statistic").reset_index().rename(columns={"index": "feature"})
