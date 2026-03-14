"""
main.py — End-to-end pipeline orchestrator.

Usage:
    python main.py "https://open.spotify.com/playlist/..."

Flags:
    --clusters N        K-Means clusters (default 6)
    --algo              kmeans | dbscan | hierarchical
    --nn N              UMAP n_neighbors (default 15)
    --dist F            UMAP min_dist (default 0.1)
    --metric            UMAP distance metric (default euclidean)
    --contamination F   Outlier detection initial threshold, e.g. 0.05 = 5%
    --no-meta           Exclude release_year from UMAP embedding
    --no-browser        Don't open browser automatically
    --export            Export CSV + report to exports/
"""
from __future__ import annotations

import argparse
import logging
import pickle
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.spotify.client      import SpotifyClient
from src.spotify.parser      import parse_tracks
from src.processing.features import engineer_features, get_descriptive_stats
from src.processing.stats    import compute_playlist_stats, compute_pca, compute_diversity_score
from src.embedding.umap_embed  import compute_umap
from src.clustering.algorithms import (
    apply_kmeans, apply_dbscan, apply_hierarchical,
    get_cluster_stats, find_optimal_k,
)
from src.clustering.outliers   import (
    build_outlier_matrix, detect_outliers, get_outlier_summary,
)
from src.similarity.metrics    import centroid_songs, playlist_diversity
from src.export import export_clustered_csv, export_playlist_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spotify Playlist Explorer")
    p.add_argument("playlist_url")
    p.add_argument("--clusters",      type=int,   default=config.DEFAULT_N_CLUSTERS)
    p.add_argument("--algo",          type=str,   default="kmeans",
                   choices=["kmeans", "dbscan", "hierarchical"])
    p.add_argument("--nn",            type=int,   default=config.UMAP_N_NEIGHBORS,  dest="n_neighbors")
    p.add_argument("--dist",          type=float, default=config.UMAP_MIN_DIST,     dest="min_dist")
    p.add_argument("--metric",        type=str,   default=config.UMAP_METRIC)
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--no-meta",       action="store_true")
    p.add_argument("--no-browser",    action="store_true")
    p.add_argument("--export",        action="store_true")
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:

    # ── 1. Data acquisition ───────────────────────────────────────────────────
    log.info("=== Step 1/5  Data Acquisition ===")
    client        = SpotifyClient()
    playlist_info = client.get_playlist_info(args.playlist_url)
    log.info("Playlist: '%s'", playlist_info.get("name", "?"))

    raw_tracks = client.get_playlist_tracks(args.playlist_url)
    if not raw_tracks:
        log.error("No tracks found.")
        sys.exit(1)

    track_ids  = [t["id"] for t in raw_tracks]
    artist_ids = [t["artists"][0]["id"] for t in raw_tracks if t.get("artists")]

    audio_features = client.get_audio_features(track_ids)
    artist_genres  = client.get_artist_genres(artist_ids)

    df_raw = parse_tracks(raw_tracks, audio_features, artist_genres)
    log.info("Parsed %d tracks.", len(df_raw))

    # ── 2. Feature engineering ────────────────────────────────────────────────
    log.info("=== Step 2/5  Feature Engineering ===")
    df, X, scaler, feature_names = engineer_features(
        df_raw,
        include_meta=not args.no_meta,
    )
    log.info("\n%s", get_descriptive_stats(df).to_string())

    # ── 3. UMAP embedding ─────────────────────────────────────────────────────
    log.info("=== Step 3/5  UMAP Embedding ===")
    embedding, reducer = compute_umap(
        X, n_neighbors=args.n_neighbors,
        min_dist=args.min_dist, metric=args.metric,
    )
    pca_coords, pca_var = compute_pca(X)
    log.info("PCA explained variance: %.1f%% + %.1f%%",
             pca_var[0]*100, pca_var[1]*100)

    # ── 4. Clustering + outliers ──────────────────────────────────────────────
    log.info("=== Step 4/5  Clustering + Outlier Detection ===")

    if args.algo == "kmeans":
        labels, _ = apply_kmeans(embedding, n_clusters=args.clusters)
    elif args.algo == "dbscan":
        labels = apply_dbscan(embedding)
    else:
        labels = apply_hierarchical(embedding, n_clusters=args.clusters)

    df["cluster"] = labels

    # Outlier detection on PURE AUDIO features only (no year, no popularity)
    X_audio = build_outlier_matrix(df)
    outlier_labels, outlier_scores, iso_model = detect_outliers(
        X_audio, contamination=args.contamination
    )
    df["is_outlier"]    = outlier_labels == -1
    df["anomaly_score"] = outlier_scores

    cluster_stats   = get_cluster_stats(df, labels)
    reps            = centroid_songs(df, X, labels)
    div             = playlist_diversity(X)
    outlier_summary = get_outlier_summary(df, outlier_labels, outlier_scores)

    log.info("Cluster sizes:\n%s", df.groupby("cluster").size().to_string())
    log.info("Diversity: %.4f | Outliers: %d (%.1f%%)",
             div, int(df["is_outlier"].sum()),
             100 * df["is_outlier"].mean())

    if args.algo == "kmeans":
        k_scores = find_optimal_k(embedding)
        best_k   = k_scores.loc[k_scores["silhouette"].idxmax(), "k"]
        log.info("Optimal k=%d (you used %d)", best_k, args.clusters)

    # ── 5. Stats ──────────────────────────────────────────────────────────────
    log.info("=== Step 5/5  Statistics ===")
    stats = compute_playlist_stats(df)
    log.info("Stats: %s", stats)

    if args.export:
        export_clustered_csv(df, labels, embedding)
        export_playlist_report(playlist_info, stats, cluster_stats, div)

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache = dict(
        df              = df,
        embedding       = embedding,
        X               = X,
        X_audio         = X_audio,          # pure audio matrix for re-thresholding
        feature_names   = feature_names,
        labels          = labels,
        outlier_labels  = outlier_labels,
        outlier_scores  = outlier_scores,
        iso_model       = iso_model,        # stored for re-thresholding
        outlier_summary = outlier_summary,
        playlist_info   = playlist_info,
        stats           = stats,
    )
    cache_path = ROOT / "data" / "pipeline_cache.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    log.info("Cache saved → %s", cache_path)
    return cache


def launch_bokeh(no_browser: bool) -> None:
    cmd = ["bokeh", "serve", "--show", "app.py", f"--port={config.BOKEH_PORT}"]
    if no_browser:
        cmd.remove("--show")
    log.info("Launching: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        log.info("Server stopped.")


if __name__ == "__main__":
    args = parse_args()
    run(args)
    launch_bokeh(args.no_browser)
