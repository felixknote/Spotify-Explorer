"""
src/processing/stats.py — Playlist-level statistics and PCA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def compute_playlist_stats(df: pd.DataFrame) -> dict:
    """High-level summary statistics for a playlist."""
    def _safe_mean(col: str, decimals: int = 3) -> float | str:
        if col not in df.columns or not df[col].notna().any():
            return "N/A"
        return round(float(df[col].mean()), decimals)

    def _safe_int(col: str, fn) -> int | str:
        if col not in df.columns or not df[col].notna().any():
            return "N/A"
        return int(fn(df[col].dropna()))

    return {
        "total_tracks":       int(len(df)),
        "unique_artists":     int(df["artist"].nunique()),
        "unique_albums":      int(df["album"].nunique()),
        "year_min":           _safe_int("release_year", min),
        "year_max":           _safe_int("release_year", max),
        "avg_bpm":            _safe_mean("tempo", 1),
        "avg_energy":         _safe_mean("energy"),
        "avg_danceability":   _safe_mean("danceability"),
        "avg_valence":        _safe_mean("valence"),
        "total_duration_min": round(float(df["duration_min"].sum()), 1)
                              if "duration_min" in df.columns else "N/A",
    }


def compute_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA coordinates and explained variance ratios."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_
