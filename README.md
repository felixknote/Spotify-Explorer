# 🎵 Spotify Playlist Explorer

A fully local, interactive music analysis tool. Fetches a Spotify playlist,
enriches it with audio features and musical key data, clusters songs by
similarity using UMAP, and displays everything in a live browser dashboard.

---

## Dashboard

| Tab | Description |
|-----|-------------|
| **🗺 Map** | UMAP scatter — hover for song info, click to find 5 nearest neighbours. 9 audio sliders, year/cluster filters, outlier toggle, colour by any feature. |
| **📊 Distributions** | Playlist DNA radar chart. Histograms with mean annotations for all 11 features. Key distribution + Major/Minor split. Playlist timeline. |
| **🔗 Correlations** | Pearson correlation heatmap between all 9 audio features. |
| **🧩 Clusters** | Mean feature profile per cluster with dominant key/camelot. Bar charts with value labels. |
| **⚠️ Outliers** | Isolation Forest. Live threshold slider rebuilds chart and table instantly. |
| **🔍 Similarity** | Find the N most similar songs to any track. Results include key, mode, camelot. |
| **📈 Explorer** | Custom X/Y scatter — pick any two features. Sort table to find fastest, most energetic, most acoustic songs, etc. |
| **🕰 Taste Timeline** | Feature trends, mood map, BPM evolution, heatmap by year. UMAP HTML export. |
| **🎙 Librosa** | MFCCs, chroma, local key detection — shows guide if no audio files available. |

---

## Installation

### 1 — Create conda environment

```bash
conda create -n spotify_explorer python=3.11 -y
conda activate spotify_explorer
```

### 2 — Install dependencies

```bash
cd "Spotify Explorer"
pip install -r requirements.txt
```

If `umap-learn` fails on Windows:
```bash
conda install -c conda-forge umap-learn -y
pip install spotipy pandas numpy scikit-learn bokeh python-dotenv scipy requests
```

### 3 — Spotify API credentials

1. Go to **https://developer.spotify.com/dashboard**
2. Click **Create app**
3. Add Redirect URI: `http://127.0.0.1:8888/callback`
4. Copy **Client ID** and **Client Secret**

### 4 — GetSongBPM API key *(free — for Key / Mode / Camelot)*

1. Go to **https://getsongkey.com/api**
2. Register with your email address
3. Copy your API key
4. Add a small link to `getsongbpm.com` somewhere in your project (their only requirement)

### 5 — Configure credentials

```bash
copy .env.example .env
```

Edit `.env`:
```
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
GETSONGBPM_API_KEY=your_getsongbpm_key
```

> No quotes, no spaces around `=`. Leave `GETSONGBPM_API_KEY` blank to skip key/mode lookup.

### 6 — Create folder structure *(first time only)*

```bash
mkdir data exports
mkdir src src\spotify src\processing src\embedding src\clustering src\similarity src\visualization src\audio

type nul > src\__init__.py
type nul > src\spotify\__init__.py
type nul > src\processing\__init__.py
type nul > src\embedding\__init__.py
type nul > src\clustering\__init__.py
type nul > src\similarity\__init__.py
type nul > src\visualization\__init__.py
type nul > src\audio\__init__.py
```

---

## Usage

### Run the pipeline

```bash
conda activate spotify_explorer
cd "Spotify Explorer"
python main.py "https://open.spotify.com/playlist/YOUR_PLAYLIST_ID"
```

On first run, a browser opens for Spotify login. After approving, paste the
redirect URL back into the terminal. This happens **once** — token is cached in `.cache`.

Dashboard opens at **http://localhost:5006/app**.

### Reopen dashboard without re-fetching

```bash
bokeh serve --show app.py
```

### Run on a different playlist

```bash
del data\pipeline_cache.pkl
python main.py "https://open.spotify.com/playlist/NEW_ID"
```

---

## Command-line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--clusters N` | `6` | K-Means / Hierarchical cluster count |
| `--algo` | `kmeans` | `kmeans` · `dbscan` · `hierarchical` |
| `--nn N` | `15` | UMAP `n_neighbors` (higher = more global structure) |
| `--dist F` | `0.1` | UMAP `min_dist` (lower = tighter clusters) |
| `--metric` | `euclidean` | UMAP distance: `euclidean`, `cosine`, `manhattan` |
| `--contamination F` | `0.05` | Outlier threshold — `0.05` flags bottom 5% |
| `--no-keys` | — | Skip GetSongBPM lookup (faster re-runs) |
| `--no-meta` | — | Exclude release year from UMAP |
| `--no-librosa` | — | Skip librosa analysis |
| `--no-browser` | — | Start server without opening browser |
| `--export` | — | Save clustered CSV + text report to `exports/` |
| `--dl-workers N` | `8` | Parallel preview download threads |
| `--analysis-workers N` | `4` | Parallel librosa analysis threads |

### Examples

```bash
# More clusters, cosine distance
python main.py "https://..." --clusters 10 --metric cosine

# DBSCAN, skip key lookup
python main.py "https://..." --algo dbscan --no-keys

# Quick re-analysis (skip all slow steps)
python main.py "https://..." --no-keys --no-librosa

# Export results to CSV
python main.py "https://..." --export
```

---

## Project Structure

```
Spotify Explorer/
│
├── main.py                         # Pipeline (6 steps)
├── app.py                          # Bokeh server entry point
├── config.py                       # Configuration + defaults
├── requirements.txt
├── .env.example                    # Credential template
├── debug_playlist.py               # API diagnostic tool
│
├── data/
│   ├── pipeline_cache.pkl          # Auto-generated after each run
│   └── playlists/
│       └── {playlist_id}/
│           ├── previews/           # MP3 files for librosa
│           └── librosa_cache.pkl   # Incremental librosa cache
│
├── exports/                        # CSV + reports (--export)
│
└── src/
    ├── audio/
    │   └── librosa_features.py     # Local audio analysis
    ├── spotify/
    │   ├── client.py               # Spotify + ReccoBeats + GetSongBPM
    │   └── parser.py               # JSON → DataFrame
    ├── processing/
    │   ├── features.py             # Feature matrix (5 layers)
    │   └── stats.py                # Playlist statistics
    ├── embedding/
    │   └── umap_embed.py           # UMAP projection
    ├── clustering/
    │   ├── algorithms.py           # K-Means, DBSCAN, Hierarchical
    │   └── outliers.py             # Isolation Forest
    ├── similarity/
    │   └── metrics.py              # Cosine similarity, diversity
    ├── visualization/
    │   └── dashboard.py            # 9-tab Bokeh dashboard
    └── export.py                   # CSV + report export
```

---

## Data Sources

| Source | Provides | Cost | Key required |
|--------|----------|------|-------------|
| **Spotify Web API** | Tracks, metadata, timestamps | Free | OAuth (one-time) |
| **ReccoBeats** | 9 audio features | Free | None |
| **GetSongBPM** | Key, Mode, Camelot | Free | Email signup |
| **Librosa** *(optional)* | MFCC, Chroma, local key | Free | Audio files needed |

### Audio features (ReccoBeats)

| Feature | Range | Meaning |
|---------|-------|---------|
| `energy` | 0–1 | Intensity and activity |
| `danceability` | 0–1 | Rhythmic suitability for dancing |
| `valence` | 0–1 | Positivity (0 = sad, 1 = happy) |
| `acousticness` | 0–1 | Confidence the track is acoustic |
| `instrumentalness` | 0–1 | Likelihood of no vocals |
| `liveness` | 0–1 | Live audience presence |
| `speechiness` | 0–1 | Spoken word content |
| `tempo` | BPM | Estimated beats per minute |
| `loudness` | dB | Overall loudness (−60 to 0) |

### GetSongBPM — Key / Mode / Camelot

Searches by song name + artist. Expect **70–85% coverage** on mainstream tracks.
At 4 requests/second, a 1,600-track playlist takes ~7–10 minutes.
Use `--no-keys` on re-runs after you've already analysed the playlist.

### Feature matrix dimensions

| Configuration | Dimensions |
|---|---|
| Audio only | 9 |
| + Release year | 10 |
| + Key/Mode one-hot | ~23 |
| + Librosa (MFCC + Chroma) | ~59 |

More dimensions = better UMAP cluster separation.

---

## Librosa (optional — MFCC + Chroma)

Improves clustering significantly by adding timbral features. Requires 30-second
audio clips. **Not available in Germany/EU** (Spotify removed preview URLs).

```bash
# Install
conda install -c conda-forge librosa ffmpeg -y
pip install soundfile

# Provide audio files manually:
# data\playlists\{playlist_id}\previews\{spotify_track_id}.mp3
# Formats: .mp3  .flac  .wav  .ogg  .m4a
```

Fully incremental — only new/changed tracks are processed on each run.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run the `type nul >` commands to create `__init__.py` files |
| `ModuleNotFoundError: No module named 'dotenv'` | `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'requests'` | `pip install requests` |
| `401 Unauthorized` | Delete `.cache` and re-run — OAuth token expired |
| `404 Not Found` | Spotify editorial playlists are blocked. Use a user-created playlist. |
| `403 Forbidden` on other user's playlist | Add their Spotify email to your app's User Management on the developer dashboard |
| `1839 have no preview URL` | Expected in Germany/EU. Use `--no-librosa` to skip. |
| `GetSongBPM: 0 found` | Check `.env` has `GETSONGBPM_API_KEY` set correctly |
| Dashboard is blank | Check terminal for Python errors — usually a missing import |
| Port 5006 in use | `bokeh serve --show app.py --port 5007` |
| KMeans/OpenMP warnings | Already suppressed — `OMP_NUM_THREADS=1` is set automatically |

---

## Credits

- [ReccoBeats](https://reccobeats.com) — free Spotify audio features replacement
- [GetSongBPM / GetSongKey](https://getsongbpm.com) — free key, mode, camelot lookup
- [UMAP-learn](https://umap-learn.readthedocs.io) — dimensionality reduction
- [Bokeh](https://bokeh.org) — interactive browser visualisation

Data usage subject to [Spotify's Terms](https://developer.spotify.com/terms),
[ReccoBeats' terms](https://reccobeats.com), and [GetSongBPM's terms](https://getsongbpm.com).
