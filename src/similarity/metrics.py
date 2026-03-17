"""
src/similarity/metrics.py — Cosine similarity, nearest neighbours, diversity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


def nearest_neighbors(X: np.ndarray, idx: int, n: int = 5) -> list[int]:
    """Return indices of the n songs most similar to song at idx (self excluded)."""
    sims = cosine_similarity(X[idx : idx + 1], X)[0]
    sims[idx] = -2.0
    return list(np.argsort(sims)[::-1][:n])


def centroid_songs(df: pd.DataFrame, X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """For each cluster, find the track closest to the centroid."""
    rows = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask        = labels == cid
        cluster_X   = X[mask]
        cluster_idx = np.where(mask)[0]
        sims        = cosine_similarity(cluster_X.mean(axis=0, keepdims=True), cluster_X)[0]
        best        = int(cluster_idx[np.argmax(sims)])
        rows.append({
            "cluster":                 cid,
            "representative":          df.iloc[best]["name"],
            "artist":                  df.iloc[best]["artist"],
            "cluster_size":            int(mask.sum()),
            "similarity_to_centroid":  round(float(sims.max()), 3),
        })
    return pd.DataFrame(rows)


def playlist_diversity(X: np.ndarray) -> float:
    """Mean pairwise cosine distance ∈ [0,1]. Higher = more varied playlist."""
    n = len(X)
    if n < 2:
        return 0.0
    dists = cosine_distances(X)
    return round(float(dists[np.triu_indices(n, k=1)].mean()), 4)
