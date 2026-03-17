"""app.py — Bokeh server entry point."""
import logging
import pickle
import sys
from pathlib import Path

from bokeh.io import curdoc

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("app")

CACHE = ROOT / "data" / "pipeline_cache.pkl"

if not CACHE.exists():
    from bokeh.models import Div
    curdoc().add_root(Div(
        text="<h2 style='color:#f85149;font-family:monospace'>"
             "No data found — run main.py first.</h2>",
    ))
else:
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)

    from src.visualization.dashboard import SpotifyDashboard

    dash = SpotifyDashboard(
        df              = cache["df"],
        embedding       = cache["embedding"],
        X               = cache["X"],
        X_audio         = cache.get("X_audio"),
        playlist_info   = cache.get("playlist_info", {}),
        stats           = cache.get("stats", {}),
        outlier_labels  = cache.get("outlier_labels"),
        outlier_scores  = cache.get("outlier_scores"),
        outlier_summary = cache.get("outlier_summary"),
    )
    curdoc().title = "🎵 Spotify Playlist Explorer"
    curdoc().add_root(dash.get_layout())
    log.info("Dashboard loaded.")
