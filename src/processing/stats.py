"""
src/processing/stats.py — Playlist-level statistics and PCA comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def compute_playlist_stats(df: pd.DataFrame) -> dict:
    """High-level summary statistics for a playlist."""
    return {
        "total_tracks":       int(len(df)),
        "unique_artists":     int(df["artist"].nunique()),
        "unique_albums":      int(df["album"].nunique()),
        "year_min":           int(df["release_year"].min()) if df["release_year"].notna().any() else "?",
        "year_max":           int(df["release_year"].max()) if df["release_year"].notna().any() else "?",
        "avg_bpm":            round(float(df["tempo"].mean()), 1),
        "avg_energy":         round(float(df["energy"].mean()), 3),
        "avg_danceability":   round(float(df["danceability"].mean()), 3),
        "avg_valence":        round(float(df["valence"].mean()), 3),
        "avg_popularity":     round(float(df["popularity"].mean()), 1),
        "total_duration_min": round(float(df["duration_min"].sum()), 1),
    }


def compute_pca(
    X: np.ndarray, n_components: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA coordinates and explained variance ratios."""
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def compute_diversity_score(X: np.ndarray) -> float:
    """
    Playlist diversity score = mean pairwise cosine distance ∈ [0, 1].
    Higher means the songs are more varied.
    """
    from sklearn.metrics.pairwise import cosine_distances

    n = len(X)
    if n < 2:
        return 0.0
    dists = cosine_distances(X)
    # Upper triangle only (exclude diagonal)
    upper = dists[np.triu_indices(n, k=1)]
    return float(upper.mean())
