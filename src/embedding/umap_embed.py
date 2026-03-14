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
    """
    Fit a 2-D UMAP embedding.

    Parameters
    ----------
    X           : standardised feature matrix
    n_neighbors : UMAP neighbourhood size (default from config)
    min_dist    : minimum distance in embedding space (default from config)
    metric      : distance metric, e.g. 'euclidean', 'cosine' (default from config)
    random_state: for reproducibility

    Returns
    -------
    embedding   : (n_samples, 2) array of 2-D coordinates
    reducer     : fitted UMAP object (for transform() on new data)
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors  or config.UMAP_N_NEIGHBORS,
        min_dist=min_dist        or config.UMAP_MIN_DIST,
        metric=metric            or config.UMAP_METRIC,
        random_state=random_state or config.UMAP_RANDOM_STATE,
        low_memory=False,
    )
    log.info(
        "Running UMAP (n_neighbors=%d, min_dist=%.2f, metric=%s) on %d × %d matrix …",
        reducer.n_neighbors, reducer.min_dist, reducer.metric,
        X.shape[0], X.shape[1],
    )
    embedding = reducer.fit_transform(X)
    log.info("UMAP complete. Embedding shape: %s", embedding.shape)
    return embedding, reducer


def parameter_sweep(
    X: np.ndarray,
    n_neighbors_list: list[int]  = (10, 15, 20, 30),
    min_dist_list: list[float]   = (0.05, 0.10, 0.25, 0.50),
) -> list[dict]:
    """
    Run UMAP for every (n_neighbors, min_dist) combination.
    Useful for exploring how parameters affect the layout.

    Returns a list of dicts with keys: n_neighbors, min_dist, embedding.
    """
    results = []
    for nn in n_neighbors_list:
        for md in min_dist_list:
            emb, _ = compute_umap(X, n_neighbors=nn, min_dist=md)
            results.append({"n_neighbors": nn, "min_dist": md, "embedding": emb})
            log.info("  → nn=%d, md=%.2f done.", nn, md)
    return results
