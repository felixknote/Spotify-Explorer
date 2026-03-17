# 🎵 Spotify Playlist Explorer

A fully local, interactive music analysis tool. Fetches any Spotify playlist,
enriches it with audio features via ReccoBeats, clusters songs by sonic
similarity using UMAP, and displays everything in a live browser dashboard.

No data leaves your machine. No subscriptions. No API fees.

---

## What it does

1. Fetches every track from a Spotify playlist (public or private)
2. Retrieves 9 audio features per track from the ReccoBeats API (free, no key required)
3. Projects songs into a 2D space with UMAP — similar-sounding songs end up close together
4. Clusters the embedding with K-Means, DBSCAN, or Hierarchical clustering
5. Detects sonic outliers with Isolation Forest
6. Launches an interactive 8-tab dashboard in your browser

---

## Dashboard

| Tab | What you get |
|-----|-------------|
| **🗺 Map** | Interactive UMAP scatter. Hover any dot to inspect a song. Click to instantly find its 5 nearest sonic neighbours. 9 audio sliders, year filter, cluster filter, outlier toggle, colour by any feature. |
| **📊 Distributions** | Playlist DNA radar. Histograms for all 9 audio features + release year + duration with mean annotations. Timeline of when tracks were added. |
| **🔗 Correlations** | Pearson correlation heatmap across all 9 audio features. |
| **🧩 Clusters** | Mean feature profiles per cluster with value-labelled bar charts. |
| **⚠️ Outliers** | Isolation Forest detection. Live threshold slider — move it and the chart and table rebuild instantly. |
| **🔍 Similarity** | Type any song name and find the most similar tracks by cosine distance in feature space. |
| **📈 Explorer** | Choose any two features as X and Y axes. Sort the table by any feature to find the fastest, most energetic, or most acoustic songs. |
| **🕰 Taste Timeline** | Feature trends over time with confidence bands. Mood map (energy vs valence). BPM evolution. Feature heatmap by year. Export the UMAP as a standalone HTML file. |

---

## Installation

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

If `umap-learn` fails on Windows:
```bash
conda install -c conda-forge umap-learn -y
pip install spotipy pandas numpy scikit-learn bokeh python-dotenv scipy requests
```

### 3 — Get Spotify API credentials

1. Go to **https://developer.spotify.com/dashboard**
2. Click **Create app** — name and description can be anything
3. Under **Redirect URIs** add exactly: `http://127.0.0.1:8888/callback`
4. Click **Save**, then copy your **Client ID** and **Client Secret**

### 4 — Configure credentials

```bash
copy .env.example .env
```

Open `.env` and fill in:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

### 5 — Create the folder structure (first time only)

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

## Running

```bash
conda activate spotify_explorer
cd "Spotify Explorer"
python main.py "https://open.spotify.com/playlist/YOUR_PLAYLIST_ID"
```

On first run, a browser window opens asking you to log in to Spotify. After
approving, paste the redirect URL back into the terminal. This happens **once**
— the token is cached in `.cache`.

The pipeline runs in 5 steps (~3–10 minutes depending on playlist size) and
then opens the dashboard automatically at **http://localhost:5006/app**.

### Reopen the dashboard without re-fetching

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
| `--clusters N` | `6` | Number of clusters for K-Means or Hierarchical |
| `--algo` | `kmeans` | Clustering algorithm: `kmeans` · `dbscan` · `hierarchical` |
| `--nn N` | `15` | UMAP `n_neighbors` — higher values capture more global structure |
| `--dist F` | `0.1` | UMAP `min_dist` — lower values produce tighter clusters |
| `--metric` | `euclidean` | UMAP distance metric: `euclidean` · `cosine` · `manhattan` |
| `--contamination F` | `0.05` | Outlier sensitivity — `0.05` flags the bottom 5% |
| `--no-meta` | — | Exclude release year from the feature matrix |
| `--no-browser` | — | Start the server without opening a browser tab |
| `--export` | — | Save a clustered CSV and text report to `exports/` |

### Examples

```bash
# Tighter layout, more clusters
python main.py "https://..." --clusters 10 --dist 0.05

# DBSCAN — number of clusters found automatically
python main.py "https://..." --algo dbscan

# Cosine distance, flag only the most extreme 2% as outliers
python main.py "https://..." --metric cosine --contamination 0.02

# Export results to CSV
python main.py "https://..." --export
```

---

## Project Structure

```
Spotify Explorer/
│
├── main.py                     # Pipeline (5 steps: fetch → embed → cluster → export)
├── app.py                      # Bokeh server entry point
├── config.py                   # Configuration and defaults
├── requirements.txt            # Python dependencies
├── .env.example                # Credential template — copy to .env
├── debug_playlist.py           # Diagnostic tool for API issues
│
├── data/
│   └── pipeline_cache.pkl      # Auto-generated after each run
│
├── exports/                    # Clustered CSV + text reports (--export)
│
└── src/
    ├── spotify/
    │   ├── client.py           # Spotify OAuth + ReccoBeats audio features
    │   └── parser.py           # Raw API JSON → clean DataFrame
    ├── processing/
    │   ├── features.py         # Feature engineering and normalisation
    │   └── stats.py            # Playlist statistics
    ├── embedding/
    │   └── umap_embed.py       # UMAP 2D projection
    ├── clustering/
    │   ├── algorithms.py       # K-Means, DBSCAN, Hierarchical
    │   └── outliers.py         # Isolation Forest
    ├── similarity/
    │   └── metrics.py          # Cosine similarity and diversity score
    ├── visualization/
    │   └── dashboard.py        # 8-tab Bokeh dashboard
    └── export.py               # CSV + report writer
```

---

## Data Sources

| Source | What it provides | Cost | Auth required |
|--------|-----------------|------|--------------|
| **Spotify Web API** | Track name, artist, album, release year, duration, explicit flag, added-to-playlist date | Free | OAuth (one-time login) |
| **ReccoBeats** | 9 audio features per track | Free | None |

### Audio Features (ReccoBeats)

| Feature | Range | What it captures |
|---------|-------|-----------------|
| `energy` | 0–1 | Perceived intensity and activity |
| `danceability` | 0–1 | Rhythmic suitability for dancing |
| `valence` | 0–1 | Musical positivity — 0 is sad, 1 is euphoric |
| `acousticness` | 0–1 | Confidence the track is acoustic |
| `instrumentalness` | 0–1 | Likelihood of no vocals |
| `liveness` | 0–1 | Presence of a live audience |
| `speechiness` | 0–1 | Spoken word content |
| `tempo` | BPM | Estimated beats per minute |
| `loudness` | dB | Overall loudness, typically −60 to 0 dB |

ReccoBeats does not require an API key and was built as a drop-in replacement
for Spotify's deprecated audio features endpoint. Expect **85–95% coverage**
for mainstream playlists. Tracks without features are excluded from UMAP and
clustering but do appear in the total track count.

---

## Known Limitations

**Audio features blocked by Spotify:** The `/v1/audio-features` endpoint is
restricted for new apps without Extended Quota Mode approval. ReccoBeats fills
this gap for most tracks.

**Popularity removed:** Spotify removed the `popularity` field from track
responses for Development Mode apps in February 2026.

**Private playlists from other users:** You can only access another user's
private playlist if their Spotify account email is added to your app's User
Management on the developer dashboard (up to 25 users in Development Mode).

**Spotify editorial playlists:** Playlists owned by the Spotify account
(e.g. Today's Top Hits) return 404 — use user-created playlists instead.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run the `type nul >` commands to create `__init__.py` files |
| `ModuleNotFoundError: No module named 'dotenv'` | `pip install -r requirements.txt` |
| `401 Unauthorized` | Delete `.cache` and re-run — OAuth token expired |
| `404 Not Found` | Spotify editorial playlists are blocked. Use a user-created playlist. |
| `403 Forbidden` on another user's playlist | Add their Spotify email to User Management on the developer dashboard |
| Dashboard is blank | Check the terminal for Python errors — usually a missing import |
| Port 5006 already in use | `bokeh serve --show app.py --port 5007` |

---

## Credits

- [ReccoBeats](https://reccobeats.com) — free audio features API
- [UMAP-learn](https://umap-learn.readthedocs.io) — dimensionality reduction
- [Bokeh](https://bokeh.org) — interactive browser visualisation
- [scikit-learn](https://scikit-learn.org) — clustering and outlier detection

Data usage is subject to [Spotify's Terms of Service](https://developer.spotify.com/terms)
and [ReccoBeats' terms](https://reccobeats.com).
