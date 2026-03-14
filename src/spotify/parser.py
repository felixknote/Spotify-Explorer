"""
src/spotify/parser.py — Convert raw Spotify API payloads into a tidy DataFrame.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# All audio feature columns returned by the Spotify API
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
    """
    Merge track metadata, audio features, and genre information into one DataFrame.
    Works gracefully if audio_features is empty (Spotify API restriction).
    """
    af_map: dict[str, dict] = {f["id"]: f for f in audio_features if f}
    has_audio_features = len(af_map) > 0

    if not has_audio_features:
        log.warning(
            "No audio features available — clustering will use metadata only "
            "(popularity, release year, duration, explicit)."
        )

    rows: list[dict] = []

    for track in tracks:
        if not track or not track.get("id"):
            continue

        track_id = track["id"]
        af = af_map.get(track_id, {})

        primary_artist = track["artists"][0] if track["artists"] else {}
        artist_id = primary_artist.get("id", "")
        all_artists = ", ".join(a["name"] for a in track["artists"])
        genres = artist_genres.get(artist_id, [])

        raw_date = track["album"].get("release_date", "")
        try:
            release_year = int(raw_date[:4])
        except (ValueError, TypeError):
            release_year = None

        row: dict = {
            # ── Identity ───────────────────────────────────────────────────
            "id":            track_id,
            "name":          track["name"],
            "artist":        primary_artist.get("name", ""),
            "all_artists":   all_artists,
            "artist_id":     artist_id,
            "album":         track["album"].get("name", ""),
            "release_year":  release_year,
            "popularity":    track.get("popularity", 0),
            "duration_ms":   track.get("duration_ms", 0),
            "explicit":      int(track.get("explicit", False)),
            "spotify_url":   track.get("external_urls", {}).get("spotify", ""),
            # ── Genre ──────────────────────────────────────────────────────
            "primary_genre": genres[0] if genres else "Unknown",
            "genres":        " | ".join(genres[:3]) if genres else "Unknown",
            # ── Audio features (may be empty) ──────────────────────────────
            **{col: af.get(col) for col in AUDIO_FEATURE_COLS},
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Derived convenience columns
    df["duration_min"] = (df["duration_ms"] / 60_000).round(2)
    df["bpm"] = df["tempo"].round(1) if "tempo" in df.columns and df["tempo"].notna().any() else 0.0

    # Human-readable key / mode labels
    key_labels = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F",
                  "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]
    if "key" in df.columns and df["key"].notna().any():
        df["key_label"] = df["key"].apply(
            lambda k: key_labels[int(k)] if pd.notna(k) and 0 <= int(k) <= 11 else "?"
        )
        df["mode_label"] = df["mode"].map({1: "Major", 0: "Minor"}).fillna("?")
    else:
        df["key_label"] = "?"
        df["mode_label"] = "?"

    log.info("Parsed %d tracks into DataFrame (audio features: %s).",
             len(df), "yes" if has_audio_features else "NO — metadata only")
    return df
