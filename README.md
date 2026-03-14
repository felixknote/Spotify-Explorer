# 🎵 Spotify Playlist Explorer

A fully local, interactive music analysis tool that fetches a Spotify playlist, extracts audio features, clusters songs by musical similarity using UMAP, and displays everything in a live browser dashboard.

---

## What it does

1. **Fetches** all tracks from any public or private Spotify playlist via the Spotify Web API
2. **Retrieves audio features** (energy, danceability, valence, BPM, etc.) via the [ReccoBeats API](https://reccobeats.com) — free, no key required
3. **Embeds** songs into a 2D space using UMAP so that sonically similar songs cluster together visually
4. **Clusters** the embedding with K-Means, DBSCAN, or Hierarchical clustering
5. **Detects outliers** using Isolation Forest on pure audio features
6. **Launches** an interactive Bokeh dashboard in your browser at `http://localhost:5006`

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **🗺 Map** | Interactive UMAP scatter plot. Hover to inspect songs. Click a song to find its 5 nearest audio neighbours. 9 audio feature sliders + year/cluster filter + outlier toggle. Colour by cluster, energy, valence, BPM, artist, and more. |
| **📊 Distributions** | Histograms for all 11 audio features. Playlist summary card. Key/Mode charts when data is available. |
| **🔗 Correlations** | Pearson correlation heatmap between all 9 audio features. |
| **🧩 Clusters** | Per-cluster mean feature table + bar charts for every audio feature broken down by cluster. |
| **⚠️ Outliers** | Isolation Forest detection. Adjustable threshold slider (1–20%) updates the chart and table live. Pure audio features only — no year or popularity bias. |
| **🔍 Similarity** | Type any song name and find the N most similar tracks by cosine similarity in feature space. |
| **📈 Explorer** | Choose any two features as X/Y axes for a custom scatter. Sort the table by any feature to find the fastest, most energetic, most acoustic tracks, etc. |

---

## Installation

### Requirements
- Windows, macOS, or Linux
- [Anaconda](https://www.anaconda.com/download) or Python 3.11+

### 1 — Create a conda environment

```bash
conda create -n spotify_explorer python=3.11 -y
conda activate spotify_explorer
```

### 2 — Install dependencies

```bash
cd "Spotify Explorer"
pip install -r requirements.txt
```

If `umap-learn` fails:

```bash
conda install -c conda-forge umap-learn -y
pip install spotipy pandas numpy scikit-learn bokeh python-dotenv scipy requests
```

### 3 — Get Spotify API credentials

1. Go to **https://developer.spotify.com/dashboard**
2. Log in with your Spotify account (free)
3. Click **Create app**
4. Under **Redirect URIs** add exactly: `http://127.0.0.1:8888/callback`
5. Click **Save**
6. Copy your **Client ID** and **Client Secret**

### 4 — Configure credentials

```bash
copy .env.example .env
```

Open `.env` and fill in:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

> No quotes, no spaces around the `=`.

### 5 — Create required folders and init files (first time only)

```bash
mkdir data exports
mkdir src src\spotify src\processing src\embedding src\clustering src\similarity src\visualization

type nul > src\__init__.py
type nul > src\spotify\__init__.py
type nul > src\processing\__init__.py
type nul > src\embedding\__init__.py
type nul > src\clustering\__init__.py
type nul > src\similarity\__init__.py
type nul > src\visualization\__init__.py
```

---

## Usage

### Run the full pipeline

```bash
conda activate spotify_explorer
cd "Spotify Explorer"
python main.py "https://open.spotify.com/playlist/YOUR_PLAYLIST_ID"
```

On first run, a browser window opens asking you to log in to Spotify. After approving, paste the redirect URL back into the terminal. This only happens **once** — the token is cached in `.cache`.

The pipeline runs through 5 steps (~2–8 minutes depending on playlist size) and then opens the dashboard at **http://localhost:5006/app**.

### Re-open the dashboard without re-fetching

```bash
bokeh serve --show app.py
```

### Run on a different playlist

```bash
del data\pipeline_cache.pkl
python main.py "https://open.spotify.com/playlist/NEW_PLAYLIST_ID"
```

---

## Command-line Options

```
python main.py <playlist_url> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--clusters N` | `6` | Number of clusters for K-Means or Hierarchical |
| `--algo` | `kmeans` | Algorithm: `kmeans`, `dbscan`, `hierarchical` |
| `--nn N` | `15` | UMAP `n_neighbors` — higher = more global structure |
| `--dist F` | `0.1` | UMAP `min_dist` — lower = tighter clusters |
| `--metric` | `euclidean` | Distance metric: `euclidean`, `cosine`, `manhattan` |
| `--contamination F` | `0.05` | Outlier threshold, e.g. `0.05` = bottom 5% |
| `--no-meta` | — | Exclude release year from UMAP embedding |
| `--no-browser` | — | Start server without opening browser |
| `--export` | — | Export clustered CSV + text report to `exports/` |

### Examples

```bash
# More clusters, tighter layout
python main.py "https://..." --clusters 10 --dist 0.05

# Hierarchical clustering with export
python main.py "https://..." --algo hierarchical --clusters 8 --export

# DBSCAN (clusters auto-detected)
python main.py "https://..." --algo dbscan

# Cosine distance UMAP
python main.py "https://..." --metric cosine

# Flag only the most extreme 2% as outliers
python main.py "https://..." --contamination 0.02
```

---

## Project Structure

```
Spotify Explorer/
│
├── main.py                     # Pipeline: fetch → embed → cluster → launch
├── app.py                      # Bokeh server entry point
├── config.py                   # Central configuration and defaults
├── requirements.txt            # Python dependencies
├── .env.example                # Credential template — copy to .env
├── debug_playlist.py           # Diagnostic: inspect raw API responses
│
├── data/
│   └── pipeline_cache.pkl      # Cached pipeline output (auto-generated)
│
├── exports/                    # CSV and text reports (--export flag)
│
└── src/
    ├── spotify/
    │   ├── client.py           # Spotify OAuth + ReccoBeats audio features
    │   └── parser.py           # Raw JSON → clean DataFrame
    │
    ├── processing/
    │   ├── features.py         # Feature engineering, StandardScaler
    │   └── stats.py            # Playlist statistics, PCA, diversity score
    │
    ├── embedding/
    │   └── umap_embed.py       # UMAP 2D embedding
    │
    ├── clustering/
    │   ├── algorithms.py       # K-Means, DBSCAN, Hierarchical
    │   └── outliers.py         # Isolation Forest + live re-thresholding
    │
    ├── similarity/
    │   └── metrics.py          # Cosine similarity, nearest neighbours
    │
    ├── visualization/
    │   └── dashboard.py        # 7-tab Bokeh dashboard
    │
    └── export.py               # CSV + report export
```

---

## Audio Features

All features are provided by [ReccoBeats](https://reccobeats.com) using Spotify track IDs — free, no API key required.

| Feature | Range | Description |
|---------|-------|-------------|
| `energy` | 0–1 | Perceived intensity and activity |
| `danceability` | 0–1 | Suitability for dancing |
| `valence` | 0–1 | Musical positivity (0 = sad, 1 = happy) |
| `acousticness` | 0–1 | Confidence the track is acoustic |
| `instrumentalness` | 0–1 | Likelihood of no vocals |
| `liveness` | 0–1 | Presence of a live audience |
| `speechiness` | 0–1 | Presence of spoken words |
| `tempo` | BPM | Estimated beats per minute |
| `loudness` | dB | Overall loudness (typically −60 to 0 dB) |

> **Key / Mode** — ReccoBeats does not return musical key or mode. These fields show `?` throughout the dashboard. They populate automatically if Spotify Extended Quota access is granted (apply at developer.spotify.com/dashboard).

---

## API Notes

### Why ReccoBeats instead of Spotify?

Spotify restricted the `/v1/audio-features` endpoint for new apps in late 2024. ReccoBeats is a community-built replacement that accepts the same Spotify track IDs and returns the same field names — completely free with no API key.

### Coverage

ReccoBeats does not have every song. Expect 85–95% coverage for typical playlists. Tracks without features are excluded from UMAP and clustering but counted in the total.

### Rate limits

The client uses batches of 10 tracks with 200ms delays. A 1,800-track playlist takes roughly 5–7 minutes to fetch.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src.spotify'` | `__init__.py` files are missing — run the `type nul >` commands above |
| `ModuleNotFoundError: No module named 'dotenv'` | `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'requests'` | `pip install requests` |
| `401 Unauthorized` | OAuth flow incomplete — delete `.cache` and run again |
| `404 Not Found` | Spotify editorial playlists are blocked. Use a user-created playlist. |
| Dashboard blank / white | Check terminal for errors — usually a missing file or import error |
| Browser doesn't open | Navigate manually to `http://localhost:5006/app` |
| Port already in use | `bokeh serve --show app.py --port 5007` |
| KMeans OpenMP warnings | Add `import os; os.environ["OMP_NUM_THREADS"] = "1"` at the top of `main.py` |

---

## License

For personal use. Spotify data is subject to [Spotify's Terms of Service](https://developer.spotify.com/terms). ReccoBeats data is subject to [ReccoBeats' terms](https://reccobeats.com).
