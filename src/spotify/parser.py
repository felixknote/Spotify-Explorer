"""
src/spotify/parser.py — Convert raw Spotify + ReccoBeats payloads into a DataFrame.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

AUDIO_FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature",
]



def parse_tracks(
    tracks: list[dict],
    audio_features: list[dict],
    artist_genres: dict[str, list[str]],
) -> pd.DataFrame:
    """Merge all data sources into one DataFrame — one row per track."""
    af_map = {f["id"]: f for f in audio_features if f}
    has_af = len(af_map) > 0

    if not has_af:
        log.warning("No audio features available — metadata-only mode.")

    rows: list[dict] = []

    for track in tracks:
        if not track or not track.get("id"):
            continue

        tid            = track["id"]
        af             = af_map.get(tid, {})
        primary_artist = track["artists"][0] if track.get("artists") else {}
        artist_id      = primary_artist.get("id", "")
        genres         = artist_genres.get(artist_id, [])

        try:
            release_year = int(track["album"].get("release_date", "")[:4])
        except (ValueError, TypeError):
            release_year = None

        rows.append({
            "id":            tid,
            "name":          track["name"],
            "artist":        primary_artist.get("name", ""),
            "all_artists":   ", ".join(a["name"] for a in track.get("artists", [])),
            "artist_id":     artist_id,
            "album":         track["album"].get("name", ""),
            "release_year":  release_year,
            "duration_ms":   track.get("duration_ms", 0),
            "explicit":      int(track.get("explicit", False)),
            "spotify_url":   track.get("external_urls", {}).get("spotify", ""),
            "added_at":      track.get("added_at", ""),
            "added_by":      track.get("added_by_id", ""),
            "primary_genre": genres[0] if genres else "Unknown",
            "genres":        " | ".join(genres[:3]) if genres else "Unknown",
            **{col: af.get(col) for col in AUDIO_FEATURE_COLS},
        })

    df = pd.DataFrame(rows)

    # Derived columns
    df["duration_min"] = (df["duration_ms"] / 60_000).round(2)
    df["bpm"] = df["tempo"].round(1) if "tempo" in df.columns and df["tempo"].notna().any() else 0.0

    # Added-at timestamps
    if "added_at" in df.columns and df["added_at"].ne("").any():
        dt = pd.to_datetime(df["added_at"], utc=True, errors="coerce")
        df["added_at_dt"] = dt
        df["added_year"]  = dt.dt.year
        df["added_month"] = dt.dt.month
        df["added_date"]  = dt.dt.strftime("%Y-%m-%d").fillna("?")
    else:
        df["added_date"]  = "?"
        df["added_year"]  = None
        df["added_month"] = None

    log.info(
        "Parsed %d tracks (audio features: %s).",
        len(df),
        "yes" if has_af else "no",
    )
    return df
