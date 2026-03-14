"""
src/similarity/metrics.py — Cosine similarity, nearest-neighbour search,
centroid representatives, and playlist diversity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


# ── Core functions ────────────────────────────────────────────────────────────

def build_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Full n × n cosine similarity matrix."""
    return cosine_similarity(X)


def nearest_neighbors(
    X: np.ndarray,
    idx: int,
    n: int = 5,
) -> list[int]:
    """
    Return indices of the n songs most similar to song at position idx.
    The source song itself is excluded.
    """
    sims = cosine_similarity(X[idx : idx + 1], X)[0]
    sims[idx] = -2.0            # exclude self
    return list(np.argsort(sims)[::-1][:n])


def find_similar_songs(
    df: pd.DataFrame,
    X: np.ndarray,
    song_name: str,
    n: int = 5,
) -> pd.DataFrame:
    """
    Search by (partial) name and return the n most similar songs.
    """
    hits = df[df["name"].str.lower().str.contains(song_name.lower(), na=False)]
    if hits.empty:
        return pd.DataFrame(columns=["name", "artist", "tempo", "energy", "valence"])

    source_idx = int(hits.index[0])
    neighbor_idxs = nearest_neighbors(X, source_idx, n)
    cols = ["name", "artist", "tempo", "energy", "valence", "danceability", "popularity"]
    available = [c for c in cols if c in df.columns]
    return df.iloc[neighbor_idxs][available].reset_index(drop=True)


def centroid_songs(
    df: pd.DataFrame,
    X: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    For each cluster, find the track whose feature vector is closest to the
    cluster centroid — the most 'typical' representative of that cluster.
    """
    rows = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:        # DBSCAN noise
            continue
        mask = labels == cluster_id
        cluster_X = X[mask]
        cluster_idx = np.where(mask)[0]

        centroid = cluster_X.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, cluster_X)[0]
        best_local = int(np.argmax(sims))
        best_global = int(cluster_idx[best_local])

        rows.append(
            {
                "cluster":            cluster_id,
                "representative":     df.iloc[best_global]["name"],
                "artist":             df.iloc[best_global]["artist"],
                "cluster_size":       int(mask.sum()),
                "similarity_to_centroid": round(float(sims[best_local]), 3),
            }
        )
    return pd.DataFrame(rows)


def playlist_diversity(X: np.ndarray) -> float:
    """
    Diversity score = mean cosine distance between all song pairs.
    Range [0, 1]: 0 = identical songs, 1 = maximally diverse.
    """
    n = len(X)
    if n < 2:
        return 0.0
    dists = cosine_distances(X)
    upper = dists[np.triu_indices(n, k=1)]
    return round(float(upper.mean()), 4)


def pairwise_similarities_for_song(
    df: pd.DataFrame,
    X: np.ndarray,
    idx: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return a ranked DataFrame of most-similar songs for display."""
    sims = cosine_similarity(X[idx : idx + 1], X)[0]
    sims[idx] = -2.0

    top_idx = np.argsort(sims)[::-1][:top_n]
    result = df.iloc[top_idx][["name", "artist", "tempo", "energy", "valence"]].copy()
    result["similarity"] = sims[top_idx].round(3)
    return result.reset_index(drop=True)
