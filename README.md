# 🎵 Spotify Playlist Explorer

An interactive local application that clusters and visualises Spotify playlists based
on their audio features using UMAP, K-Means / DBSCAN / Hierarchical clustering, and a
live Bokeh dashboard.

---

## Quick Start

### 1. Get Spotify API credentials

1. Go to <https://developer.spotify.com/dashboard>
2. Create an application
3. Copy your **Client ID** and **Client Secret**

### 2. Set up the environment

```bash
cd spotify_explorer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
```

### 3. Run the pipeline

```bash
python main.py "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
```

This will:
- Fetch all tracks and audio features from the playlist
- Engineer and normalise features
- Compute UMAP embedding
- Cluster songs (K-Means by default)
- Launch the Bokeh dashboard in your browser at **http://localhost:5006**

---

## Command-line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--clusters N` | 6 | Number of clusters for K-Means / Hierarchical |
| `--algo` | `kmeans` | `kmeans` · `dbscan` · `hierarchical` |
| `--nn N` | 15 | UMAP `n_neighbors` |
| `--dist F` | 0.1 | UMAP `min_dist` |
| `--metric` | `euclidean` | UMAP distance metric (`cosine`, `manhattan`, …) |
| `--no-meta` | — | Exclude popularity & release year from features |
| `--no-cat` | — | Exclude one-hot key/mode features |
| `--export` | — | Export clustered CSV + text report to `exports/` |
| `--no-browser` | — | Start server without opening browser |

**Examples:**

```bash
# 8 clusters, cosine UMAP, DBSCAN variant
python main.py "https://..." --clusters 8 --metric cosine

# Hierarchical clustering, export results
python main.py "https://..." --algo hierarchical --clusters 7 --export

# Tight clusters with DBSCAN
python main.py "https://..." --algo dbscan

# Re-open dashboard without re-fetching data
bokeh serve --show app.py
```

---

## Dashboard Tabs

| Tab | What you see |
|-----|-------------|
| **🗺 Map** | Interactive UMAP scatter — hover for song details, click to find nearest neighbours, filter by BPM / energy / year / cluster, recolour by any feature |
| **📊 Distributions** | Histograms for BPM, energy, valence, danceability, loudness, popularity, acousticness, liveness |
| **🔗 Correlations** | Interactive Pearson correlation heat-map for all audio features |
| **🧩 Clusters** | Cluster profile table + bar charts for mean energy and valence per cluster |
| **🔍 Similarity** | Type any song name → ranked list of most similar songs in the playlist |

---

## Project Structure

```
spotify_explorer/
├── main.py                         # Pipeline orchestrator (fetch → embed → cluster → launch)
├── app.py                          # Bokeh server entry point
├── config.py                       # Central configuration
├── requirements.txt
├── .env.example                    # Credential template
├── data/
│   └── pipeline_cache.pkl          # Cached pipeline output (auto-generated)
├── exports/                        # CSV + reports (when --export is used)
└── src/
    ├── spotify/
    │   ├── client.py               # Spotify Web API wrapper (pagination, batching)
    │   └── parser.py               # Raw JSON → tidy DataFrame
    ├── processing/
    │   ├── features.py             # Feature engineering, normalisation, correlation
    │   └── stats.py                # Playlist stats, PCA, diversity score
    ├── embedding/
    │   └── umap_embed.py           # UMAP 2D embedding + parameter sweep
    ├── clustering/
    │   └── algorithms.py           # K-Means, DBSCAN, Hierarchical + evaluation
    ├── similarity/
    │   └── metrics.py              # Cosine similarity, NN search, centroid representatives
    ├── visualization/
    │   └── dashboard.py            # Full 5-tab Bokeh dashboard
    └── export.py                   # CSV + text report export
```

---

## Audio Features Reference

| Feature | Range | Description |
|---------|-------|-------------|
| `danceability` | 0–1 | How suitable for dancing |
| `energy` | 0–1 | Perceived intensity and activity |
| `valence` | 0–1 | Musical positivity (sad → happy) |
| `acousticness` | 0–1 | Confidence the track is acoustic |
| `instrumentalness` | 0–1 | Predicts absence of vocals |
| `liveness` | 0–1 | Presence of a live audience |
| `speechiness` | 0–1 | Presence of spoken words |
| `tempo` | BPM | Estimated beats per minute |
| `loudness` | dB | Overall loudness in dB |
| `key` | 0–11 | Pitch class (C, C♯, D, …) |
| `mode` | 0/1 | Minor (0) or Major (1) |
| `time_signature` | int | Estimated beats per bar |

---

## Stretch-Goal Hooks

The codebase is structured so these are straightforward additions:

- **Multi-playlist comparison**: run `run()` for two playlist URLs, concatenate
  DataFrames with a `playlist` column, re-run UMAP on the merged feature matrix.
- **Outlier detection**: after DBSCAN, label `-1` points are already flagged as outliers.
- **Temporal taste evolution**: filter by `release_year` and trace centroid drift.
- **Recommendation**: use `find_similar_songs()` with an external track against the
  playlist's feature matrix.
