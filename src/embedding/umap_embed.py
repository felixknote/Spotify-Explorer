"""
src/embedding/umap_embed.py — UMAP dimensionality reduction to 2D.
"""
from __future__ import annotations

import logging

import numpy as np
import umap

import config

log = logging.getLogger(__name__)


def compute_umap(
    X: np.ndarray,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    metric: str | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, umap.UMAP]:
    """Fit a 2D UMAP embedding. Returns (embedding, fitted reducer)."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors  = n_neighbors   or config.UMAP_N_NEIGHBORS,
        min_dist     = min_dist      or config.UMAP_MIN_DIST,
        metric       = metric        or config.UMAP_METRIC,
        random_state = random_state  or config.UMAP_RANDOM_STATE,
        low_memory   = False,
    )
    log.info(
        "Running UMAP (n_neighbors=%d, min_dist=%.2f, metric=%s) on %d × %d …",
        reducer.n_neighbors, reducer.min_dist, reducer.metric,
        X.shape[0], X.shape[1],
    )
    embedding = reducer.fit_transform(X)
    log.info("UMAP complete → shape %s", embedding.shape)
    return embedding, reducer
