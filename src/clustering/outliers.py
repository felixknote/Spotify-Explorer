"""
src/clustering/outliers.py — Outlier detection using Isolation Forest.

Outliers are detected on pure audio features only (no release_year).
The IsolationForest model + raw scores are cached so the dashboard can
re-threshold interactively without re-running the model.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

log = logging.getLogger(__name__)

# Pure audio features used for outlier detection — no temporal/metadata bias
OUTLIER_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


def build_outlier_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract and normalise the pure-audio feature matrix for outlier detection."""
    from sklearn.preprocessing import StandardScaler
    available = [f for f in OUTLIER_FEATURES if f in df.columns and df[f].notna().any()]
    sub = df[available].fillna(df[available].median())
    scaler = StandardScaler()
    return scaler.fit_transform(sub)


def detect_outliers(
    X: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, IsolationForest]:
    """
    Fit Isolation Forest and return labels, scores, and the fitted model.

    Returns
    -------
    labels  : 1 = normal, -1 = outlier  (based on contamination threshold)
    scores  : raw anomaly scores — more negative = more anomalous
    model   : fitted IsolationForest (store in cache for re-thresholding)
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X)
    scores = iso.decision_function(X)

    n_out = int((labels == -1).sum())
    log.info("Outlier detection: %d outliers (%.1f%% of %d tracks).",
             n_out, 100 * n_out / len(X), len(X))
    return labels, scores, iso


def rethreshold(scores: np.ndarray, pct: float) -> np.ndarray:
    """
    Re-label outliers using a manual percentile threshold.
    pct = 5.0 means the bottom 5% of scores are flagged as outliers.
    """
    threshold = float(np.percentile(scores, pct))
    return np.where(scores < threshold, -1, 1)


def get_outlier_summary(
    df: pd.DataFrame,
    outlier_labels: np.ndarray,
    scores: np.ndarray,
    top_n: int = 30,
) -> pd.DataFrame:
    """Return the most anomalous tracks sorted by anomaly score."""
    tmp = df.copy()
    tmp["outlier"]       = outlier_labels == -1
    tmp["anomaly_score"] = scores
    outliers = tmp[tmp["outlier"]].sort_values("anomaly_score")
    cols = ["name", "artist", "anomaly_score",
            "energy", "danceability", "valence", "tempo",
            "acousticness", "instrumentalness", "liveness", "speechiness"]
    available = [c for c in cols if c in outliers.columns]
    return outliers[available].head(top_n).reset_index(drop=True)
