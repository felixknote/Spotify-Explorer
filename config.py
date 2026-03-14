"""
config.py — Central configuration for Spotify Explorer.
Copy .env.example to .env and fill in your credentials.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Spotify API ──────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# ── UMAP defaults ────────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.10
UMAP_METRIC       = "euclidean"
UMAP_RANDOM_STATE = 42

# ── Clustering defaults ──────────────────────────────────────────────────────
DEFAULT_N_CLUSTERS   = 6
DBSCAN_EPS           = 0.5
DBSCAN_MIN_SAMPLES   = 3
HIERARCHICAL_LINKAGE = "ward"

# ── Bokeh server ─────────────────────────────────────────────────────────────
BOKEH_PORT  = 5006
BOKEH_TITLE = "🎵 Spotify Playlist Explorer"

# ── Export ───────────────────────────────────────────────────────────────────
EXPORT_DIR = "exports"
