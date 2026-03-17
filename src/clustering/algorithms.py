"""
src/clustering/algorithms.py — K-Means, DBSCAN, and Hierarchical clustering.

All functions accept a feature matrix X (UMAP embedding or original space)
and return integer cluster labels (DBSCAN uses -1 for noise).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

import config

log = logging.getLogger(__name__)

# ── Algorithm wrappers ────────────────────────────────────────────────────────

def apply_kmeans(
    X: np.ndarray,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    n = n_clusters or config.DEFAULT_N_CLUSTERS
    km = KMeans(n_clusters=n, random_state=random_state, n_init=12)
    labels = km.fit_predict(X)
    _log_quality(X, labels, f"K-Means (k={n})")
    return labels, km


def apply_dbscan(
    X: np.ndarray,
    eps: float | None = None,
    min_samples: int | None = None,
) -> np.ndarray:
    db = DBSCAN(
        eps=eps or config.DBSCAN_EPS,
        min_samples=min_samples or config.DBSCAN_MIN_SAMPLES,
    )
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    log.info("DBSCAN → %d clusters, %d noise points.", n_clusters, n_noise)
    return labels


def apply_hierarchical(
    X: np.ndarray,
    n_clusters: int | None = None,
    linkage: str = "ward",
) -> np.ndarray:
    n = n_clusters or config.DEFAULT_N_CLUSTERS
    hc = AgglomerativeClustering(n_clusters=n, linkage=linkage)
    labels = hc.fit_predict(X)
    _log_quality(X, labels, f"Hierarchical (k={n}, linkage={linkage})")
    return labels


# ── Evaluation helpers ────────────────────────────────────────────────────────

def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
) -> pd.DataFrame:
    """
    Evaluate K-Means for a range of k values.
    Returns a DataFrame with silhouette score, Davies-Bouldin index, and inertia.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=12)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        rows.append(
            {
                "k":               k,
                "silhouette":      round(silhouette_score(X, labels), 4),
                "davies_bouldin":  round(davies_bouldin_score(X, labels), 4),
                "inertia":         round(km.inertia_, 2),
            }
        )
    return pd.DataFrame(rows)


def get_cluster_stats(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Mean audio-feature profile and track count per cluster."""
    numeric_cols = [
        "danceability", "energy", "loudness", "tempo",
        "valence", "acousticness", "instrumentalness",
        "speechiness", "liveness",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    tmp = df[available].copy()
    tmp["cluster"] = labels
    agg = tmp.groupby("cluster").agg(["mean", "std"]).round(3)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg["count"] = tmp.groupby("cluster").size()
    return agg.reset_index()


# ── Internal ──────────────────────────────────────────────────────────────────

def _log_quality(X: np.ndarray, labels: np.ndarray, name: str) -> None:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        sil = silhouette_score(X, labels)
        log.info("%s → %d clusters | silhouette=%.3f", name, n_clusters, sil)
    else:
        log.info("%s → %d clusters (cannot compute silhouette).", name, n_clusters)
