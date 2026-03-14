"""
src/export.py — Save analysis artefacts to disk.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


def _ensure_dir() -> Path:
    path = Path(config.EXPORT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def export_clustered_csv(df: pd.DataFrame, labels: np.ndarray, embedding: np.ndarray) -> str:
    """Export full enriched DataFrame with cluster labels and UMAP coordinates."""
    out = df.copy()
    out["cluster"]  = labels
    out["umap_x"]   = embedding[:, 0]
    out["umap_y"]   = embedding[:, 1]
    p = _ensure_dir() / f"playlist_clustered_{_timestamp()}.csv"
    out.to_csv(p, index=False)
    log.info("Exported clustered CSV → %s", p)
    return str(p)


def export_playlist_report(
    playlist_info: dict,
    stats: dict,
    cluster_stats: pd.DataFrame,
    diversity: float,
) -> str:
    """Write a plain-text playlist analysis report."""
    lines = [
        "=" * 60,
        f"  Spotify Playlist Explorer — Report",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        f"Playlist : {playlist_info.get('name', '?')}",
        f"Owner    : {playlist_info.get('owner', {}).get('display_name', '?')}",
        "",
        "── Summary ─────────────────────────────────────────────────",
        f"  Total tracks      : {stats['total_tracks']}",
        f"  Unique artists    : {stats['unique_artists']}",
        f"  Year range        : {stats['year_min']} – {stats['year_max']}",
        f"  Avg BPM           : {stats['avg_bpm']}",
        f"  Avg energy        : {stats['avg_energy']}",
        f"  Avg valence       : {stats['avg_valence']}",
        f"  Avg popularity    : {stats['avg_popularity']}",
        f"  Total duration    : {stats['total_duration_min']} min",
        f"  Diversity score   : {diversity:.4f}",
        "",
        "── Cluster profiles ─────────────────────────────────────────",
        cluster_stats.to_string(),
        "",
    ]
    p = _ensure_dir() / f"playlist_report_{_timestamp()}.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    log.info("Exported report → %s", p)
    return str(p)
