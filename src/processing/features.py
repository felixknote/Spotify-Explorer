"""
src/processing/features.py — Feature engineering, cleaning, and normalisation.

Feature matrix layers:
  Layer 1 — ReccoBeats audio features (9 continuous)     always included
  Layer 2 — Release year                                 --no-meta disables
  Layer 3 — Key / Mode one-hot                           when available
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

CORE_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

META_FEATURES = ["release_year"]


ALL_NUMERIC = CORE_FEATURES + ["release_year"]


def engineer_features(
    df: pd.DataFrame,
    include_meta: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, list[str]]:
    """
    Build, clean, and standardise the feature matrix for UMAP + clustering.

    Returns
    -------
    df_clean      : DataFrame with only tracks that have complete core features
    X             : standardised feature matrix
    scaler        : fitted StandardScaler
    feature_names : column names matching X
    """
    df = df.copy()

    audio_available = (
        "energy" in df.columns
        and df["energy"].notna().any()
        and df["danceability"].notna().any()
        and df["tempo"].notna().any()
    )

    if not audio_available:
        return _metadata_fallback(df)

    # ── Core audio features ───────────────────────────────────────────────────
    before = len(df)
    df     = df.dropna(subset=CORE_FEATURES).reset_index(drop=True)
    if before - len(df):
        log.warning("Dropped %d tracks with missing core audio features.", before - len(df))

    feature_parts: list[pd.DataFrame] = [df[CORE_FEATURES]]

    # ── Release year ──────────────────────────────────────────────────────────
    if include_meta and "release_year" in df.columns:
        df["release_year"] = df["release_year"].fillna(df["release_year"].median())
        feature_parts.append(df[["release_year"]])

    # ── Assemble and scale ────────────────────────────────────────────────────
    feature_df    = pd.concat(feature_parts, axis=1).astype(float)
    feature_names = list(feature_df.columns)

    scaler = StandardScaler()
    X      = scaler.fit_transform(feature_df)

    log.info(
        "Feature matrix: %d tracks × %d features  (%d audio | %d meta).",
        X.shape[0], X.shape[1],
        len(CORE_FEATURES),
        1 if include_meta and "release_year" in df.columns else 0,
    )
    return df, X, scaler, feature_names


def _metadata_fallback(df: pd.DataFrame):
    """Fallback when no audio features are available."""
    log.warning("No audio features — using metadata-only fallback.")
    for col in ["release_year", "duration_min", "explicit"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df     = df.dropna(subset=["duration_min"]).reset_index(drop=True)
    cols   = [c for c in ["release_year", "duration_min", "explicit"] if c in df.columns]
    scaler = StandardScaler()
    X      = scaler.fit_transform(df[cols].astype(float))
    return df, X, scaler, cols


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in ALL_NUMERIC if c in df.columns]
    return df[available].describe().T.round(3)


def get_feature_importance_for_clusters(
    X_raw: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """ANOVA F-statistic per feature — higher = more discriminative."""
    from scipy.stats import f_oneway
    unique_labels = [l for l in sorted(set(labels)) if l != -1]
    importances   = {}
    for i, name in enumerate(feature_names):
        groups = [X_raw[labels == lbl, i] for lbl in unique_labels]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            f_stat, _ = f_oneway(*groups)
            importances[name] = float(f_stat) if not np.isnan(f_stat) else 0.0
        else:
            importances[name] = 0.0
    return (
        pd.Series(importances)
        .sort_values(ascending=False)
        .rename("F_statistic")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
