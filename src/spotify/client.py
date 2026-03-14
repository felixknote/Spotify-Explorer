"""
src/spotify/client.py — Spotify Web API wrapper + ReccoBeats audio features.
"""
from __future__ import annotations

import logging
import re
import time

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

import config

log = logging.getLogger(__name__)

RECCOBEATS_BASE  = "https://api.reccobeats.com"
RECCOBEATS_BATCH = 10       # small batches — more reliable
RECCOBEATS_DELAY = 0.2      # seconds between batches


class SpotifyClient:

    def __init__(self) -> None:
        if not config.SPOTIFY_CLIENT_ID or not config.SPOTIFY_CLIENT_SECRET:
            raise EnvironmentError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set.\n"
                "Copy .env.example to .env and fill in your credentials."
            )
        auth = SpotifyOAuth(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET,
            redirect_uri="http://127.0.0.1:8888/callback",
            scope="playlist-read-private playlist-read-collaborative",
            open_browser=True,
        )
        self.sp      = spotipy.Spotify(auth_manager=auth, retries=3)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        log.info("Spotify + ReccoBeats clients initialised.")

    # ── Spotify ───────────────────────────────────────────────────────────────

    def get_playlist_info(self, playlist_url: str) -> dict:
        pid = self._parse_playlist_id(playlist_url)
        return self.sp.playlist(pid, fields="name,owner,description,images,followers")

    def get_playlist_tracks(self, playlist_url: str) -> list[dict]:
        pid = self._parse_playlist_id(playlist_url)
        log.info("Fetching tracks for playlist %s ...", pid)

        tracks: list[dict] = []
        results = self.sp.playlist_items(pid, limit=100, additional_types=["track"])

        while results:
            for item in results.get("items", []):
                if not item:
                    continue
                track = item.get("item") or item.get("track")
                if not track:
                    continue
                if track.get("type") != "track":
                    continue
                if not track.get("id"):
                    continue
                tracks.append(track)
            results = self.sp.next(results) if results.get("next") else None

        log.info("Retrieved %d tracks.", len(tracks))
        return tracks

    def get_artist_genres(self, artist_ids: list[str]) -> dict[str, list[str]]:
        genres: dict[str, list[str]] = {}
        unique_ids = list(dict.fromkeys(artist_ids))
        for i in range(0, len(unique_ids), 50):
            batch = unique_ids[i : i + 50]
            try:
                artists = self.sp.artists(batch).get("artists", [])
                for artist in artists:
                    if artist:
                        genres[artist["id"]] = artist.get("genres", [])
            except Exception as e:
                if "403" in str(e):
                    log.warning("Spotify artist/genre blocked (403). Continuing without genres.")
                    return {}
                raise
            time.sleep(0.05)
        log.info("Retrieved genres for %d artists.", len(genres))
        return genres

    # ── ReccoBeats audio features ─────────────────────────────────────────────

    def get_audio_features(self, track_ids: list[str]) -> list[dict]:
        """
        Fetch audio features from ReccoBeats using Spotify track IDs.
        Uses repeated query params: ?ids=id1&ids=id2&... (not CSV).
        Falls back to single-track requests on batch failure.
        """
        all_features: list[dict] = []
        total = len(track_ids)
        log.info("Fetching audio features for %d tracks via ReccoBeats ...", total)

        for i in range(0, total, RECCOBEATS_BATCH):
            batch = track_ids[i : i + RECCOBEATS_BATCH]

            # Use repeated params: [("ids", id1), ("ids", id2), ...]
            params = [("ids", sid) for sid in batch]

            try:
                resp = self.session.get(
                    f"{RECCOBEATS_BASE}/v1/audio-features",
                    params=params,
                    timeout=20,
                )

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 10))
                    log.warning("Rate limited — waiting %ds ...", wait)
                    time.sleep(wait)
                    resp = self.session.get(
                        f"{RECCOBEATS_BASE}/v1/audio-features",
                        params=params,
                        timeout=20,
                    )

                if resp.status_code == 200:
                    items = _extract_items(resp.json())
                    id_lookup = _build_lookup(items)
                    for j, sid in enumerate(batch):
                        feat = id_lookup.get(sid) or (items[j] if j < len(items) else {})
                        all_features.append(_normalise(feat, sid))
                else:
                    # Batch failed — try one by one
                    log.debug("Batch %d-%d got HTTP %d, trying individually ...",
                              i, i + len(batch), resp.status_code)
                    for sid in batch:
                        all_features.append(self._get_single(sid))
                        time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                log.error("ReccoBeats request error: %s", e)
                all_features.extend([{}] * len(batch))

            time.sleep(RECCOBEATS_DELAY)

            done = min(i + RECCOBEATS_BATCH, total)
            if done % 200 == 0 or done == total:
                valid_so_far = sum(1 for f in all_features if f.get("energy") is not None)
                log.info("  ... %d / %d processed (%d with features)", done, total, valid_so_far)

        valid = sum(1 for f in all_features if f.get("energy") is not None)
        log.info("ReccoBeats: %d / %d tracks have audio features.", valid, total)
        return all_features

    def _get_single(self, spotify_id: str) -> dict:
        """Fetch audio features for a single track."""
        try:
            resp = self.session.get(
                f"{RECCOBEATS_BASE}/v1/track/{spotify_id}/audio-features",
                timeout=15,
            )
            if resp.status_code == 200:
                return _normalise(resp.json(), spotify_id)
        except Exception:
            pass
        return {}

    @staticmethod
    def _parse_playlist_id(url_or_id: str) -> str:
        if url_or_id.startswith("spotify:"):
            return url_or_id.split(":")[-1]
        match = re.search(r"playlist/([A-Za-z0-9]+)", url_or_id)
        if match:
            return match.group(1)
        return url_or_id.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_items(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return (data.get("content")
                or data.get("data")
                or data.get("audioFeatures")
                or [])
    return []


def _build_lookup(items: list) -> dict[str, dict]:
    """Build {spotify_id: feature_dict} using the href field."""
    lookup: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        href = item.get("href", "")
        m = re.search(r"track/([A-Za-z0-9]+)", href)
        if m:
            lookup[m.group(1)] = item
    return lookup


def _normalise(feat: dict, spotify_id: str) -> dict:
    if not feat:
        return {}
    return {
        "id":               spotify_id,
        "danceability":     feat.get("danceability"),
        "energy":           feat.get("energy"),
        "loudness":         feat.get("loudness"),
        "speechiness":      feat.get("speechiness"),
        "acousticness":     feat.get("acousticness"),
        "instrumentalness": feat.get("instrumentalness"),
        "liveness":         feat.get("liveness"),
        "valence":          feat.get("valence"),
        "tempo":            feat.get("tempo"),
        "key":              None,
        "mode":             None,
        "time_signature":   None,
    }