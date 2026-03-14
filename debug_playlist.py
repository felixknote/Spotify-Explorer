"""
debug_playlist.py — Inspect the raw Spotify API response for a playlist.
Run from the project root:
    python debug_playlist.py "https://open.spotify.com/playlist/3y0gVFqcH0fUXwDLTuWyAT"
"""
import sys
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="playlist-read-private playlist-read-collaborative",
))

url = sys.argv[1] if len(sys.argv) > 1 else ""
pid = url.split("playlist/")[-1].split("?")[0]

print(f"\n=== Playlist ID: {pid} ===\n")

results = sp.playlist_items(pid, limit=5)

print(f"Total items reported by API: {results.get('total')}")
print(f"Items in this page: {len(results.get('items', []))}")
print()

for i, item in enumerate(results.get("items", [])):
    print(f"--- Item {i} ---")
    print(f"  item keys:    {list(item.keys()) if item else 'None'}")
    track = item.get("track") if item else None
    if track:
        print(f"  track keys:   {list(track.keys())}")
        print(f"  track id:     {track.get('id')}")
        print(f"  track name:   {track.get('name')}")
        print(f"  track type:   {track.get('type')}")
        print(f"  is_local:     {track.get('is_local')}")
    else:
        print("  track: None")
    print()
