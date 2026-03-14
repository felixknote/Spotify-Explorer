"""
src/visualization/dashboard.py — Spotify Playlist Explorer Dashboard.

Visual design principles:
  • High-contrast text — no dark-on-dark
  • Cohesive indigo/teal accent palette
  • Cluster colours: vibrant categorical palette on dark background
  • All axis labels, titles, tooltips readable at a glance

Tabs:
  1. Map          — UMAP scatter, 9 audio sliders, outlier toggle
  2. Distributions — Histograms (no popularity), Key/Mode pie charts
  3. Correlations  — Heatmap
  4. Clusters      — Full per-cluster bar charts + table
  5. Outliers      — Score chart, manual threshold slider, table
  6. Similarity    — Cosine nearest-neighbour search
  7. Explorer      — Configurable X/Y axis scatter + sorted table
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from bokeh.layouts import column, row, gridplot
from bokeh.models import (
    ColumnDataSource, HoverTool, Select, Slider, RangeSlider,
    Div, DataTable, TableColumn, StringFormatter, NumberFormatter,
    LinearColorMapper, ColorBar, CustomJS, Button, Tabs, TabPanel,
    TextInput, MultiChoice, Toggle, NumeralTickFormatter,
    InlineStyleSheet,
)
from bokeh.plotting import figure
from bokeh.transform import cumsum

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

BG           = "#0d1117"    # page / plot background
PANEL        = "#161b22"    # sidebar, card background
CARD         = "#1c2128"    # inner card, table background
BORDER       = "#30363d"    # borders, grid lines
TEXT_HI      = "#e6edf3"    # primary text (high contrast)
TEXT_MID     = "#b1bac4"    # secondary text
TEXT_LO      = "#8b949e"    # muted text, axis ticks
ACCENT       = "#58a6ff"    # primary blue accent
ACCENT_GREEN = "#3fb950"    # positive / normal
ACCENT_WARN  = "#d29922"    # warning / threshold
ACCENT_RED   = "#f85149"    # outlier / error
ACCENT_PURP  = "#bc8cff"    # cluster / categorical
TITLE_COL    = "#79c0ff"    # chart titles

# Cluster palette — 12 vibrant colours that all read well on dark backgrounds
CLUSTER_PAL = [
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff",
    "#ffa657", "#79c0ff", "#56d364", "#ff7b72",
    "#cae8ff", "#7ee787", "#ffa198", "#e3b341",
    "#a5f3fc", "#86efac", "#fca5a5", "#c4b5fd",
    "#67e8f9", "#4ade80", "#f87171", "#a78bfa",
]

# Continuous palette (light → vivid for heatmap / colour-by)
from bokeh.palettes import Plasma256, RdYlGn11, Turbo256
CONT_PAL = Plasma256

# ── Global CSS: force white text in Bokeh widgets & DataTable ─────────────────
# Bokeh DataTable renders using browser defaults (black text).
# InlineStyleSheet injects CSS into the shadow DOM of each widget.
_TABLE_CSS = InlineStyleSheet(css="""
  :host .slick-cell { color: #e6edf3 !important; font-size: 12px; }
  :host .slick-header-column { color: #79c0ff !important; background: #1c2128 !important;
                                border-right: 1px solid #30363d !important; font-size: 11px; }
  :host .slick-row.odd  { background: #1c2128 !important; }
  :host .slick-row.even { background: #161b22 !important; }
  :host .slick-row:hover .slick-cell { background: #2d333b !important; }
  :host .grid-canvas { background: #1c2128 !important; }
""")

_WIDGET_CSS = InlineStyleSheet(css="""
  :host { --bk-color: #e6edf3 !important; color: #e6edf3 !important; }
  :host label, :host .bk-label, :host .bk-input-group label {
    color: #b1bac4 !important; font-size: 12px; }
  :host input, :host select, :host .bk-input {
    color: #e6edf3 !important; background: #1c2128 !important;
    border-color: #30363d !important; }
  :host .noUi-handle { background: #58a6ff !important; border-color: #58a6ff !important; }
  :host .noUi-connect { background: #388bfd !important; }
  :host .noUi-base, :host .noUi-target { background: #30363d !important; border: none !important; }
""")



def _cluster_pal(n: int) -> list[str]:
    return (CLUSTER_PAL * ((n // len(CLUSTER_PAL)) + 2))[:n]


COLORBY = [
    ("Cluster",          "cluster_str"),
    ("Energy",           "energy"),
    ("Danceability",     "danceability"),
    ("Valence",          "valence"),
    ("BPM",              "tempo"),
    ("Acousticness",     "acousticness"),
    ("Instrumentalness", "instrumentalness"),
    ("Liveness",         "liveness"),
    ("Speechiness",      "speechiness"),
    ("Loudness",         "loudness"),
    ("Release year",     "release_year"),
    ("Outlier",          "is_outlier_int"),
    ("Artist",           "artist"),
]

AUDIO_FEATS = [
    ("energy",           "Energy"),
    ("danceability",     "Danceability"),
    ("valence",          "Valence"),
    ("acousticness",     "Acousticness"),
    ("liveness",         "Liveness"),
    ("instrumentalness", "Instrumental."),
    ("speechiness",      "Speechiness"),
    ("tempo",            "Tempo (BPM)"),
    ("loudness",         "Loudness (dB)"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fig(title="", w=300, h=220, tools="", x_range=None) -> figure:
    kw = dict(title=title, width=w, height=h, toolbar_location=None,
              background_fill_color=BG, border_fill_color=PANEL,
              outline_line_color=BORDER)
    if x_range is not None:
        kw["x_range"] = x_range
    p = figure(**kw, tools=tools)
    p.title.text_color      = TITLE_COL
    p.title.text_font_size  = "11px"
    p.title.text_font_style = "bold"
    p.grid.grid_line_color  = BORDER
    p.grid.grid_line_alpha  = 0.6
    p.axis.axis_label_text_color   = TEXT_MID
    p.axis.major_label_text_color  = TEXT_MID
    p.axis.major_label_text_font_size = "10px"
    p.axis.minor_tick_line_color   = None
    p.axis.axis_line_color         = BORDER
    return p


def _div_title(text: str, color: str = ACCENT) -> Div:
    return Div(text=f"<b style='font-size:14px;color:{color}'>{text}</b>",
               styles={"margin": "8px 0 4px"})


def _div_sub(text: str) -> Div:
    return Div(text=f"<p style='font-size:12px;color:{TEXT_LO};margin:2px 0 8px'>{text}</p>")


def _card(content: str) -> Div:
    return Div(text=f"""
    <div style='background:{CARD};color:{TEXT_HI};padding:14px 20px;border-radius:8px;
                border:1px solid {BORDER};font-family:monospace;line-height:1.8em;
                font-size:12px'>{content}</div>""")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SpotifyDashboard:

    def __init__(
        self,
        df: pd.DataFrame,
        embedding: np.ndarray,
        X: np.ndarray,
        playlist_info: dict | None = None,
        stats: dict | None = None,
        outlier_labels: np.ndarray | None = None,
        outlier_scores: np.ndarray | None = None,
        outlier_summary: pd.DataFrame | None = None,
        X_audio: np.ndarray | None = None,
    ) -> None:
        self.df             = df.reset_index(drop=True)
        self.embedding      = embedding
        self.X              = X
        self.X_audio        = X_audio if X_audio is not None else X
        self.playlist_info  = playlist_info or {}
        self.stats          = stats or {}
        self.outlier_labels = outlier_labels
        self.outlier_scores = outlier_scores if outlier_scores is not None else np.zeros(len(df))
        self.outlier_summary= outlier_summary

        self._prepare_data()
        self._build_layout()

    # ── Data preparation ──────────────────────────────────────────────────────

    def _prepare_data(self) -> None:
        df  = self.df
        emb = self.embedding

        cluster_labels  = df["cluster"].astype(str).tolist()
        unique_clusters = sorted(set(cluster_labels),
                                 key=lambda x: int(x) if x.lstrip("-").isdigit() else x)
        pal             = _cluster_pal(len(unique_clusters))
        cluster_colors  = [pal[unique_clusters.index(c)] for c in cluster_labels]

        is_outlier     = df.get("is_outlier", pd.Series([False]*len(df))).fillna(False)
        is_outlier_int = is_outlier.astype(int).tolist()

        def _safe(col, default=0.0):
            if col in df.columns:
                return df[col].fillna(default).round(4).tolist()
            return [default] * len(df)

        key_labels = df["key_label"].tolist()  if "key_label"  in df.columns else ["?"]*len(df)
        mode_labels= df["mode_label"].tolist() if "mode_label" in df.columns else ["?"]*len(df)

        self.source = ColumnDataSource(dict(
            x                = emb[:, 0].tolist(),
            y                = emb[:, 1].tolist(),
            name             = df["name"].tolist(),
            artist           = df["artist"].tolist(),
            album            = df["album"].tolist(),
            release_year     = _safe("release_year", 0),
            popularity       = _safe("popularity", 0),
            tempo            = _safe("tempo", 0),
            energy           = _safe("energy", 0),
            danceability     = _safe("danceability", 0),
            valence          = _safe("valence", 0),
            acousticness     = _safe("acousticness", 0),
            liveness         = _safe("liveness", 0),
            instrumentalness = _safe("instrumentalness", 0),
            speechiness      = _safe("speechiness", 0),
            loudness         = _safe("loudness", -10),
            duration_min     = _safe("duration_min", 0),
            key_label        = key_labels,
            mode_label       = mode_labels,
            primary_genre    = df["primary_genre"].tolist() if "primary_genre" in df.columns else ["?"]*len(df),
            cluster          = df["cluster"].tolist(),
            cluster_str      = cluster_labels,
            anomaly_score    = self.outlier_scores.tolist(),
            is_outlier_int   = is_outlier_int,
            color            = cluster_colors,
            alpha            = [0.85] * len(df),
            size             = [8]    * len(df),
        ))

        self._full_data       = dict(self.source.data)
        self._unique_clusters = unique_clusters
        self._cluster_pal     = pal

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        tabs = Tabs(tabs=[
            TabPanel(child=self._build_scatter_tab(),       title="🗺  Map"),
            TabPanel(child=self._build_distributions_tab(), title="📊 Distributions"),
            TabPanel(child=self._build_correlation_tab(),   title="🔗 Correlations"),
            TabPanel(child=self._build_cluster_tab(),       title="🧩 Clusters"),
            TabPanel(child=self._build_outlier_tab(),       title="⚠️  Outliers"),
            TabPanel(child=self._build_similarity_tab(),    title="🔍 Similarity"),
            TabPanel(child=self._build_explorer_tab(),         title="📈 Explorer"),
        ])
        self.layout = column(self._build_header(), tabs)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — UMAP MAP
    # ══════════════════════════════════════════════════════════════════════════

    def _build_scatter_tab(self) -> Any:
        p = figure(
            title="UMAP Musical Space — Songs clustered by audio similarity",
            width=820, height=580,
            tools="pan,wheel_zoom,box_zoom,reset,save,tap",
            toolbar_location="above",
            output_backend="webgl",
            background_fill_color=BG, border_fill_color=PANEL,
        )
        p.title.text_color      = TITLE_COL
        p.title.text_font_size  = "13px"
        p.title.text_font_style = "bold"
        p.grid.grid_line_color  = BORDER
        p.grid.grid_line_alpha  = 0.5
        p.axis.major_label_text_color  = TEXT_MID
        p.axis.axis_line_color         = BORDER
        p.axis.minor_tick_line_color   = None
        p.outline_line_color           = BORDER
        p.toolbar.logo                 = None

        renderer = p.scatter(
            x="x", y="y",
            color="color", alpha="alpha", size="size",
            source=self.source,
            nonselection_alpha=0.12,
            nonselection_color="color",
            line_color=None,
        )

        hover = HoverTool(renderers=[renderer], tooltips=[
            ("", "<b style='color:#79c0ff;font-size:13px'>@name</b>"),
            ("Artist",       "@artist"),
            ("BPM",          "@tempo{0.0}"),
            ("Energy",       "@energy{0.00}"),
            ("Valence",      "@valence{0.00}"),
            ("Dance.",       "@danceability{0.00}"),
            ("Acoustic.",    "@acousticness{0.00}"),
            ("Instrument.",  "@instrumentalness{0.00}"),
            ("Key / Mode",   "@key_label / @mode_label"),
            ("Cluster",      "@cluster_str"),
            ("Outlier",      "@is_outlier_int"),
            ("Year",         "@release_year{0}"),
        ])
        p.add_tools(hover)

        # NN result table
        nn_src  = ColumnDataSource(dict(name=[], artist=[], tempo=[], energy=[], valence=[], danceability=[]))
        nn_cols = [
            TableColumn(field="name",         title="Track",       formatter=StringFormatter(), width=230),
            TableColumn(field="artist",        title="Artist",      formatter=StringFormatter(), width=150),
            TableColumn(field="tempo",         title="BPM",         formatter=NumberFormatter(format="0.0"),  width=60),
            TableColumn(field="energy",        title="Energy",      formatter=NumberFormatter(format="0.00"), width=68),
            TableColumn(field="valence",       title="Valence",     formatter=NumberFormatter(format="0.00"), width=68),
            TableColumn(field="danceability",  title="Dance.",      formatter=NumberFormatter(format="0.00"), width=68),
        ]
        nn_table = DataTable(source=nn_src, columns=nn_cols, width=820, height=155,
                             background=CARD, index_position=None,
                             stylesheets=[_TABLE_CSS])
        nn_label = Div(
            text=f"<b style='color:{TEXT_LO}'>Click a song on the map to find its 5 nearest neighbours.</b>",
            styles={"margin": "6px 0 3px"},
        )

        tap_cb = CustomJS(
            args=dict(
                source=self.source, nn_source=nn_src, nn_label=nn_label,
                X=self.X.tolist(),
                df_names        = self.df["name"].tolist(),
                df_artists      = self.df["artist"].tolist(),
                df_tempos       = self.df["tempo"].fillna(0).round(1).tolist(),
                df_energies     = self.df["energy"].fillna(0).round(2).tolist(),
                df_valences     = self.df["valence"].fillna(0).round(2).tolist(),
                df_danceability = self.df["danceability"].fillna(0).round(2).tolist(),
                ACCENT=ACCENT,
            ),
            code="""
const sel=source.selected.indices;
if(sel.length===0)return;
const idx=sel[0];
const xi=X[idx];const n=X.length;
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
function norm(a){return Math.sqrt(dot(a,a));}
let sims=[];const ni=norm(xi);
for(let j=0;j<n;j++){
  if(j===idx){sims.push(-2);continue;}
  const nj=norm(X[j]);
  sims.push((ni>0&&nj>0)?dot(xi,X[j])/(ni*nj):-2);
}
const top=Array.from({length:n},(_,i)=>i).sort((a,b)=>sims[b]-sims[a]).slice(0,5);
nn_source.data={
  name:top.map(i=>df_names[i]),artist:top.map(i=>df_artists[i]),
  tempo:top.map(i=>df_tempos[i]),energy:top.map(i=>df_energies[i]),
  valence:top.map(i=>df_valences[i]),danceability:top.map(i=>df_danceability[i])
};
nn_label.text=`<b style='color:${ACCENT}'>5 nearest neighbours of: ${df_names[idx]}</b>`;
""")
        self.source.selected.js_on_change("indices", tap_cb)

        controls = self._build_map_controls()
        return column(row(p, controls), nn_label, nn_table)

    def _build_map_controls(self) -> Any:
        src = self.source
        fd  = self._full_data

        # Colour-by selector
        colorby_sel = Select(
            title="Colour by:", value="Cluster",
            options=[lbl for lbl, _ in COLORBY], width=195,
            stylesheets=[_WIDGET_CSS],
        )

        def update_color(attr, old, new):
            col = dict(COLORBY).get(new, "cluster_str")
            data = self._full_data
            if col == "cluster_str":
                cats = self._unique_clusters
                colors = [self._cluster_pal[cats.index(c)] for c in data["cluster_str"]]
            elif col == "artist":
                arts = list(dict.fromkeys(data["artist"]))
                pal  = _cluster_pal(len(arts))
                colors = [pal[arts.index(a)] for a in data["artist"]]
            elif col == "is_outlier_int":
                colors = [ACCENT_RED if v == 1 else ACCENT for v in data["is_outlier_int"]]
            else:
                vals = np.array(data[col], dtype=float)
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
                nv = (vals - lo) / (hi - lo + 1e-9)
                colors = [CONT_PAL[int(v * (len(CONT_PAL)-1))] for v in nv]
            src.patch({"color": [(slice(None), colors)]})

        colorby_sel.on_change("value", update_color)

        # Build all RangeSliders
        def rslider(col, title, lo=0.0, hi=1.0, step=0.01):
            has = col in self.df.columns and self.df[col].notna().any()
            rlo = round(float(self.df[col].dropna().min()), 2) if has else lo
            rhi = round(float(self.df[col].dropna().max()), 2) if has else hi
            s = RangeSlider(
                title=title if has else f"{title} (N/A)",
                start=rlo, end=rhi, value=(rlo, rhi), step=step, width=195,
                disabled=not has, stylesheets=[_WIDGET_CSS],
            )
            return s, rlo, rhi

        sl_bpm,  bpm_lo,  bpm_hi  = rslider("tempo",            "BPM",            40,  220, 1.0)
        sl_nrg,  nrg_lo,  nrg_hi  = rslider("energy",           "Energy")
        sl_val,  val_lo,  val_hi  = rslider("valence",          "Valence")
        sl_dnc,  dnc_lo,  dnc_hi  = rslider("danceability",     "Danceability")
        sl_acs,  acs_lo,  acs_hi  = rslider("acousticness",     "Acousticness")
        sl_liv,  liv_lo,  liv_hi  = rslider("liveness",         "Liveness")
        sl_sph,  sph_lo,  sph_hi  = rslider("speechiness",      "Speechiness")
        sl_ins,  ins_lo,  ins_hi  = rslider("instrumentalness", "Instrumental.")
        yr_vals = self.df["release_year"].dropna()
        yr_lo = int(yr_vals.min()) if len(yr_vals) > 0 else 1900
        yr_hi = int(yr_vals.max()) if len(yr_vals) > 0 else 2025
        sl_yr = RangeSlider(title="Release year", start=yr_lo, end=yr_hi,
                            value=(yr_lo, yr_hi), step=1, width=195,
                            stylesheets=[_WIDGET_CSS])

        cl_multi = MultiChoice(title="Clusters:", value=self._unique_clusters,
                               options=self._unique_clusters, width=195,
                               stylesheets=[_WIDGET_CSS])
        out_tog  = Toggle(label="Show outliers only", button_type="warning", width=195)

        all_filters = [sl_bpm, sl_nrg, sl_val, sl_dnc, sl_acs,
                       sl_liv, sl_sph, sl_ins, sl_yr, cl_multi]

        def apply_filters(attr, old, new):
            blo, bhi = sl_bpm.value
            elo, ehi = sl_nrg.value
            vlo, vhi = sl_val.value
            dlo, dhi = sl_dnc.value
            alo, ahi = sl_acs.value
            llo, lhi = sl_liv.value
            slo, shi = sl_sph.value
            ilo, ihi = sl_ins.value
            ylo, yhi = sl_yr.value
            sel_cl   = set(cl_multi.value)
            only_out = out_tog.active

            def ok(v, lo, hi): return lo <= v <= hi if v == v else True

            alphas, sizes = [], []
            for i in range(len(fd["x"])):
                vis = (
                    ok(fd["tempo"][i], blo, bhi)
                    and ok(fd["energy"][i], elo, ehi)
                    and ok(fd["valence"][i], vlo, vhi)
                    and ok(fd["danceability"][i], dlo, dhi)
                    and ok(fd["acousticness"][i], alo, ahi)
                    and ok(fd["liveness"][i], llo, lhi)
                    and ok(fd["speechiness"][i], slo, shi)
                    and ok(fd["instrumentalness"][i], ilo, ihi)
                    and ok(fd["release_year"][i], ylo, yhi)
                    and fd["cluster_str"][i] in sel_cl
                )
                if only_out and fd["is_outlier_int"][i] != 1:
                    vis = False
                alphas.append(0.88 if vis else 0.04)
                sizes.append(9 if vis else 3)
            src.patch({"alpha": [(slice(None), alphas)], "size": [(slice(None), sizes)]})

        for w in all_filters:
            w.on_change("value", apply_filters)
        out_tog.on_change("active", lambda attr, old, new: apply_filters(attr, old, new))

        reset_btn = Button(label="↺  Reset all filters", button_type="default", width=195)
        def reset_all():
            sl_bpm.value = (bpm_lo, bpm_hi); sl_nrg.value = (nrg_lo, nrg_hi)
            sl_val.value = (val_lo, val_hi); sl_dnc.value = (dnc_lo, dnc_hi)
            sl_acs.value = (acs_lo, acs_hi); sl_liv.value = (liv_lo, liv_hi)
            sl_sph.value = (sph_lo, sph_hi); sl_ins.value = (ins_lo, ins_hi)
            sl_yr.value  = (yr_lo, yr_hi)
            cl_multi.value = self._unique_clusters
            if out_tog.active: out_tog.active = False
            src.patch({"alpha": [(slice(None), [0.88]*len(self.df))],
                       "size":  [(slice(None), [9]*len(self.df))]})
        reset_btn.on_click(lambda: reset_all())

        sep = lambda t: Div(text=f"<p style='color:{TEXT_LO};font-size:10px;"
                            f"letter-spacing:1px;margin:6px 0 2px'>── {t} ──</p>")
        panel_style = {"padding": "12px", "background": PANEL,
                       "border-radius": "8px", "border": f"1px solid {BORDER}"}
        return column(
            _div_title("Controls"),
            colorby_sel,
            sep("Audio filters"),
            sl_bpm, sl_nrg, sl_val, sl_dnc,
            sl_acs, sl_liv, sl_sph, sl_ins,
            sep("Metadata"),
            sl_yr, cl_multi, out_tog, reset_btn,
            styles=panel_style,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — DISTRIBUTIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_distributions_tab(self) -> Any:
        df = self.df

        # ── Histograms (popularity excluded, key/mode shown separately) ───────
        hist_specs = [
            ("tempo",            "BPM",           (40,  220), 28),
            ("energy",           "Energy",        (0,   1),   25),
            ("danceability",     "Danceability",  (0,   1),   25),
            ("valence",          "Valence",       (0,   1),   25),
            ("acousticness",     "Acousticness",  (0,   1),   25),
            ("liveness",         "Liveness",      (0,   1),   25),
            ("speechiness",      "Speechiness",   (0,   1),   25),
            ("instrumentalness", "Instrumental.", (0,   1),   25),
            ("loudness",         "Loudness (dB)", (-60, 0),   28),
            ("release_year",     "Release Year",  (1950,2025),30),
            ("duration_min",     "Duration (min)",(0.5, 10),  25),
        ]

        hist_plots = []
        bar_colors = [ACCENT, ACCENT_GREEN, ACCENT_PURP, "#ffa657",
                      "#79c0ff", "#56d364", "#ff7b72", "#cae8ff",
                      "#e3b341", "#7ee787", "#f87171"]

        for j, (col, label, (lo, hi), bins) in enumerate(hist_specs):
            if col not in df.columns or not df[col].notna().any():
                continue
            vals = df[col].dropna().clip(lo, hi)
            hist, edges = np.histogram(vals, bins=bins, range=(lo, hi))
            p = _fig(label, w=270, h=200)
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                   fill_color=bar_colors[j % len(bar_colors)],
                   line_color=BG, alpha=0.9)
            # Mean line
            mean_v = float(vals.mean())
            p.line([mean_v, mean_v], [0, hist.max()],
                   line_color=TEXT_HI, line_dash="dashed", line_width=1.5, alpha=0.7)
            hist_plots.append(p)

        hist_grid = gridplot(hist_plots, ncols=4, merge_tools=False)

        # ── Key distribution (bar chart) ──────────────────────────────────────
        key_available = ("key_label" in df.columns
                         and df["key_label"].ne("?").any()
                         and df["key_label"].notna().any())

        if key_available:
            key_order = ["C","C♯/D♭","D","D♯/E♭","E","F",
                         "F♯/G♭","G","G♯/A♭","A","A♯/B♭","B"]
            key_counts = df["key_label"].value_counts()
            keys_present = [k for k in key_order if k in key_counts.index]
            key_vals     = [int(key_counts.get(k, 0)) for k in keys_present]
            pk = _fig("Key Distribution", w=380, h=230, x_range=keys_present)
            pk.vbar(x=keys_present, top=key_vals, width=0.75,
                    color=ACCENT_PURP, line_color=BG, alpha=0.9)
            pk.xaxis.major_label_text_color = TEXT_HI
            pk.xaxis.major_label_text_font_size = "10px"

            mode_counts = df["mode_label"].value_counts() if "mode_label" in df.columns else {}
            major = int(mode_counts.get("Major", 0))
            minor = int(mode_counts.get("Minor", 0))
            pm = _fig("Major vs Minor", w=240, h=230, x_range=["Major", "Minor"])
            pm.vbar(x=["Major", "Minor"], top=[major, minor], width=0.6,
                    color=[ACCENT_GREEN, ACCENT_RED], line_color=BG, alpha=0.9)
            pm.xaxis.major_label_text_color = TEXT_HI
            pm.xaxis.major_label_text_font_size = "12px"
            key_row = row(pk, pm)
            key_note = _div_sub("Key and Mode data available from Spotify API (when extended access granted).")
        else:
            key_note = _card(
                f"<b style='color:{ACCENT_WARN}'>Key / Mode data not available</b><br>"
                "ReccoBeats does not provide musical key or mode. "
                "These fields require Spotify Extended Quota access.<br>"
                "Apply at <a href='https://developer.spotify.com/dashboard' "
                f"style='color:{ACCENT}'>developer.spotify.com/dashboard</a>"
            )
            key_row = key_note

        s = self.stats
        summary = _card(
            f"<b style='font-size:13px;color:{ACCENT}'>📋 Playlist Summary</b><br><br>"
            f"Tracks: <b style='color:{TEXT_HI}'>{s.get('total_tracks','?')}</b>"
            f" &nbsp;|&nbsp; Artists: <b style='color:{TEXT_HI}'>{s.get('unique_artists','?')}</b>"
            f" &nbsp;|&nbsp; Years: <b style='color:{TEXT_HI}'>{s.get('year_min','?')} – {s.get('year_max','?')}</b><br>"
            f"Avg BPM: <b style='color:{ACCENT_GREEN}'>{s.get('avg_bpm','?')}</b>"
            f" &nbsp;|&nbsp; Avg Energy: <b style='color:{ACCENT_GREEN}'>{s.get('avg_energy','?')}</b>"
            f" &nbsp;|&nbsp; Avg Valence: <b style='color:{ACCENT_GREEN}'>{s.get('avg_valence','?')}</b><br>"
            f"Total duration: <b style='color:{TEXT_HI}'>{s.get('total_duration_min','?')} min</b>"
            f" &nbsp;|&nbsp; Avg danceability: <b style='color:{ACCENT_GREEN}'>{s.get('avg_danceability','?')}</b>"
        )

        return column(summary, _div_title("Audio Feature Distributions"),
                      _div_sub("Dashed line = playlist mean."),
                      hist_grid,
                      _div_title("Key & Mode"),
                      key_row)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — CORRELATIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_correlation_tab(self) -> Any:
        features = [f for f in
                    ["danceability","energy","loudness","speechiness","acousticness",
                     "instrumentalness","liveness","valence","tempo"]
                    if f in self.df.columns and self.df[f].notna().any()]

        corr = self.df[features].corr().round(2)
        xs, ys, vs, ts = [], [], [], []
        for r in features:
            for c in features:
                xs.append(c); ys.append(r)
                v = corr.loc[r, c]; vs.append(v); ts.append(str(v))

        src    = ColumnDataSource(dict(x=xs, y=ys, vals=vs, texts=ts))
        mapper = LinearColorMapper(palette=RdYlGn11[::-1], low=-1, high=1)

        p = figure(title="Audio Feature Correlation Matrix",
                   x_range=features, y_range=list(reversed(features)),
                   width=640, height=620, toolbar_location=None,
                   x_axis_location="above",
                   background_fill_color=BG, border_fill_color=PANEL,
                   outline_line_color=BORDER)
        p.rect(x="x", y="y", width=1, height=1, source=src,
               fill_color={"field": "vals", "transform": mapper},
               line_color=BG, line_width=2)
        p.text(x="x", y="y", text="texts", source=src,
               text_align="center", text_baseline="middle",
               text_font_size="10px", text_color="#111111",
               text_font_style="bold")
        cb = ColorBar(color_mapper=mapper, location=(0, 0), width=14,
                      major_label_text_color=TEXT_MID,
                      major_label_text_font_size="10px",
                      background_fill_color=PANEL,
                      bar_line_color=BORDER)
        p.add_layout(cb, "right")
        p.title.text_color = TITLE_COL
        p.title.text_font_size  = "13px"
        p.grid.grid_line_color  = None
        p.axis.major_label_text_color     = TEXT_HI
        p.axis.major_label_text_font_size = "11px"
        p.xaxis.major_label_orientation   = 0.9
        p.axis.axis_line_color            = BORDER
        p.outline_line_color              = BORDER
        p.toolbar.logo                    = None

        return column(
            _div_title("Feature Correlation Matrix"),
            _div_sub("+1 = perfectly correlated · −1 = inversely correlated · 0 = no linear relationship"),
            p,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — CLUSTERS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_cluster_tab(self) -> Any:
        df  = self.df
        pal = self._cluster_pal

        feat_cols = [f for f in
                     ["energy","danceability","valence","acousticness","liveness",
                      "instrumentalness","speechiness","tempo","loudness"]
                     if f in df.columns and df[f].notna().any()]

        rows_data = []
        for cid in sorted(df["cluster"].unique()):
            if cid == -1: continue
            sub = df[df["cluster"] == cid]
            rd  = {"cluster": str(cid), "count": len(sub)}
            for c in feat_cols:
                rd[c] = round(sub[c].mean(), 3)
            idx = sub.index[0]
            rd["representative"] = f"{sub.loc[idx,'name']} — {sub.loc[idx,'artist']}"
            rows_data.append(rd)

        stats_df = pd.DataFrame(rows_data)
        tbl_src  = ColumnDataSource(stats_df)

        cols = [
            TableColumn(field="cluster",        title="Cluster", width=65),
            TableColumn(field="count",           title="Tracks",  width=60,
                        formatter=NumberFormatter()),
        ]
        for f in feat_cols:
            fmt = NumberFormatter(format="0.0") if f in ("tempo","loudness") \
                  else NumberFormatter(format="0.00")
            cols.append(TableColumn(field=f, title=f.capitalize()[:9], width=80, formatter=fmt))
        cols.append(TableColumn(field="representative", title="Representative track", width=310))

        table = DataTable(source=tbl_src, columns=cols, width=980, height=240,
                          background=CARD, index_position=None,
                          row_height=28, stylesheets=[_TABLE_CSS])

        # Bar charts — one per feature
        cids = [str(r["cluster"]) for r in rows_data]
        colors = pal[:len(cids)]

        bar_plots = []
        for feat, label in AUDIO_FEATS:
            if feat not in stats_df.columns:
                continue
            means = stats_df[feat].tolist()
            p = _fig(label, w=210, h=195, x_range=cids)
            p.vbar(x=cids, top=means, width=0.72,
                   color=colors, line_color=BG, alpha=0.92)
            p.xaxis.major_label_text_color     = TEXT_HI
            p.xaxis.major_label_text_font_size = "10px"
            p.y_range.start = 0
            bar_plots.append(p)

        bar_grid = gridplot(bar_plots, ncols=5, merge_tools=False)

        return column(
            _div_title("Cluster Profiles"),
            _div_sub("Mean audio feature values per cluster. Click column headers to sort."),
            table,
            _div_title("Feature Breakdown by Cluster"),
            bar_grid,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — OUTLIERS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_outlier_tab(self) -> Any:
        df     = self.df
        scores = self.outlier_scores

        initial_pct  = 5.0
        n_total      = len(df)
        n_out_init   = int(df.get("is_outlier", pd.Series([False]*n_total)).sum())

        info = _card(
            f"<b style='font-size:13px;color:{ACCENT_RED}'>⚠️  Isolation Forest Outlier Detection</b><br><br>"
            f"Detected on 9 pure audio features only — no release year, no popularity.<br>"
            f"Initial threshold: <b style='color:{ACCENT_WARN}'>{initial_pct:.0f}%</b> → "
            f"<b style='color:{ACCENT_RED}'>{n_out_init}</b> outliers out of "
            f"<b style='color:{TEXT_HI}'>{n_total}</b> tracks.<br><br>"
            f"<span style='color:{TEXT_LO};font-size:11px'>"
            f"Use the slider below to change the threshold. "
            f"Lower % = only the most extreme songs flagged. "
            f"Tip: Use <b style='color:{TEXT_HI}'>\"Show outliers only\"</b> toggle on the Map tab.</span>"
        )

        # ── Threshold slider ──────────────────────────────────────────────────
        thresh_slider = Slider(
            title="Outlier threshold (%)",
            start=1, end=20, value=initial_pct, step=0.5, width=400,
            stylesheets=[_WIDGET_CSS],
        )
        thresh_label = Div(
            text=f"<b style='color:{ACCENT_RED}'>{n_out_init} outliers</b>"
                 f" <span style='color:{TEXT_LO}'>({initial_pct:.1f}%)</span>",
            styles={"margin": "6px 0"},
        )

        # Score scatter — x = track index, y = anomaly score
        outlier_labels_init = df.get("is_outlier", pd.Series([False]*n_total)).fillna(False)
        colors_init = [ACCENT_RED if o else ACCENT for o in outlier_labels_init]
        alphas_init = [0.95 if o else 0.35 for o in outlier_labels_init]

        scatter_src = ColumnDataSource(dict(
            x      = list(range(n_total)),
            y      = scores.tolist(),
            name   = df["name"].tolist(),
            artist = df["artist"].tolist(),
            color  = colors_init,
            alpha  = alphas_init,
        ))

        p_score = figure(
            title="Anomaly Score per Track  (lower score = more anomalous)",
            width=960, height=270,
            tools="pan,wheel_zoom,reset,hover,save",
            toolbar_location="above",
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p_score.scatter(x="x", y="y", color="color", alpha="alpha",
                        size=5, source=scatter_src, line_color=None)
        p_score.add_tools(HoverTool(tooltips=[
            ("", "<b style='color:#79c0ff'>@name</b>"),
            ("Artist", "@artist"),
            ("Score",  "@y{0.000}"),
        ]))
        p_score.title.text_color     = TITLE_COL
        p_score.title.text_font_size = "12px"
        p_score.grid.grid_line_color = BORDER
        p_score.grid.grid_line_alpha = 0.5
        p_score.axis.major_label_text_color = TEXT_MID
        p_score.axis.axis_line_color        = BORDER
        p_score.toolbar.logo                = None

        # Dynamic threshold line
        threshold_init = float(np.percentile(scores, initial_pct))
        thresh_src = ColumnDataSource(dict(
            x=[0, n_total], y=[threshold_init, threshold_init]
        ))
        p_score.line(x="x", y="y", source=thresh_src,
                     line_color=ACCENT_WARN, line_dash="dashed",
                     line_width=2, alpha=0.9)

        # Outlier table
        n_show = min(30, n_out_init)
        outlier_df = df.copy()
        outlier_df["anomaly_score"] = scores
        outlier_df["is_out_now"]    = outlier_labels_init.values

        sorted_df = outlier_df.sort_values("anomaly_score")
        init_out  = sorted_df[sorted_df["is_out_now"]].head(n_show)

        tbl_cols_spec = [
            ("name",             "Track",       StringFormatter(),                   220),
            ("artist",           "Artist",      StringFormatter(),                   150),
            ("anomaly_score",    "Score",        NumberFormatter(format="0.000"),     70),
            ("energy",           "Energy",       NumberFormatter(format="0.00"),      70),
            ("danceability",     "Dance.",       NumberFormatter(format="0.00"),      70),
            ("valence",          "Valence",      NumberFormatter(format="0.00"),      70),
            ("tempo",            "BPM",          NumberFormatter(format="0.0"),       65),
            ("acousticness",     "Acoustic.",    NumberFormatter(format="0.00"),      75),
            ("instrumentalness", "Instrument.",  NumberFormatter(format="0.00"),      80),
            ("speechiness",      "Speech.",      NumberFormatter(format="0.00"),      70),
        ]
        avail_cols = [c for c, _, _, _ in tbl_cols_spec if c in init_out.columns]
        otbl_src = ColumnDataSource(
            {c: init_out[c].tolist() if c in init_out.columns else []
             for c, _, _, _ in tbl_cols_spec}
        )
        otbl_cols = [
            TableColumn(field=c, title=t, formatter=fmt, width=w)
            for c, t, fmt, w in tbl_cols_spec if c in avail_cols
        ]
        otable = DataTable(source=otbl_src, columns=otbl_cols, width=960, height=340,
                           background=CARD, index_position=None, row_height=28,
                           stylesheets=[_TABLE_CSS])

        # ── Python callback for threshold slider ──────────────────────────────
        def update_threshold(attr, old, new):
            pct       = float(thresh_slider.value)
            threshold = float(np.percentile(scores, pct))
            new_labels= scores < threshold   # True = outlier

            n_new = int(new_labels.sum())
            thresh_label.text = (
                f"<b style='color:{ACCENT_RED}'>{n_new} outliers</b>"
                f" <span style='color:{TEXT_LO}'>({pct:.1f}%)</span>"
            )

            # Update scatter colours
            new_colors = [ACCENT_RED if o else ACCENT for o in new_labels]
            new_alphas = [0.95 if o else 0.35 for o in new_labels]
            scatter_src.patch({"color": [(slice(None), new_colors)],
                               "alpha": [(slice(None), new_alphas)]})

            # Update threshold line
            thresh_src.data = dict(x=[0, n_total], y=[threshold, threshold])

            # Update table
            n_show_new = min(30, n_new)
            tmp = outlier_df.copy()
            tmp["is_out_now"] = new_labels
            new_out = tmp[tmp["is_out_now"]].sort_values("anomaly_score").head(n_show_new)
            new_data = {}
            for c, _, _, _ in tbl_cols_spec:
                if c in new_out.columns:
                    new_data[c] = new_out[c].tolist()
                else:
                    new_data[c] = []
            otbl_src.data = new_data

        thresh_slider.on_change("value", update_threshold)

        return column(
            info,
            _div_title("Threshold Control"),
            row(thresh_slider, thresh_label),
            _div_title("Anomaly Score Distribution"),
            _div_sub(f"Each dot = one track. <span style='color:{ACCENT_RED}'>Red</span> = outlier. "
                     f"<span style='color:{ACCENT_WARN}'>Dashed line</span> = current threshold."),
            p_score,
            _div_title("Most Anomalous Tracks"),
            otable,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — SIMILARITY
    # ══════════════════════════════════════════════════════════════════════════

    def _build_similarity_tab(self) -> Any:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        res_src = ColumnDataSource(dict(
            name=[], artist=[], similarity=[], tempo=[],
            energy=[], valence=[], danceability=[], acousticness=[],
            key_label=[], mode_label=[],
        ))
        res_cols = [
            TableColumn(field="name",        title="Track",      formatter=StringFormatter(), width=230),
            TableColumn(field="artist",       title="Artist",     formatter=StringFormatter(), width=155),
            TableColumn(field="similarity",   title="Similarity", formatter=NumberFormatter(format="0.000"), width=85),
            TableColumn(field="tempo",        title="BPM",        formatter=NumberFormatter(format="0.0"),   width=62),
            TableColumn(field="energy",       title="Energy",     formatter=NumberFormatter(format="0.00"),  width=70),
            TableColumn(field="valence",      title="Valence",    formatter=NumberFormatter(format="0.00"),  width=70),
            TableColumn(field="danceability", title="Dance.",     formatter=NumberFormatter(format="0.00"),  width=70),
            TableColumn(field="acousticness", title="Acoustic.",  formatter=NumberFormatter(format="0.00"),  width=75),
            TableColumn(field="key_label",    title="Key",        formatter=StringFormatter(),                width=60),
            TableColumn(field="mode_label",   title="Mode",       formatter=StringFormatter(),                width=60),
        ]
        res_table  = DataTable(source=res_src, columns=res_cols, width=960, height=300,
                               background=CARD, index_position=None, row_height=28,
                               stylesheets=[_TABLE_CSS])

        search_inp = TextInput(placeholder="Type a song name …", title="Find similar songs:", width=400)
        n_slider   = Slider(title="Number of results", start=3, end=25, value=10, step=1, width=200,
                        stylesheets=[_WIDGET_CSS])
        status_div = Div(text="", styles={"margin": "4px 0 8px"})

        df = self.df
        X  = self.X

        def do_search():
            q = search_inp.value.strip().lower()
            if not q:
                return
            hits = df[df["name"].str.lower().str.contains(q, na=False)]
            if hits.empty:
                status_div.text = (f"<span style='color:{ACCENT_RED}'>"
                                   f"No track matching '{q}' found.</span>")
                return
            idx  = int(hits.index[0])
            sims = cos_sim(X[idx:idx+1], X)[0]
            sims[idx] = -2.0
            top  = int(n_slider.value)
            tidx = np.argsort(sims)[::-1][:top]

            def _c(col, default=0.0, rnd=2):
                return df.iloc[tidx][col].fillna(default).round(rnd).tolist() \
                       if col in df.columns else [default]*top
            def _s(col):
                return df.iloc[tidx][col].fillna("?").tolist() \
                       if col in df.columns else ["?"]*top

            res_src.data = dict(
                name         = df.iloc[tidx]["name"].tolist(),
                artist       = df.iloc[tidx]["artist"].tolist(),
                similarity   = [round(float(sims[i]), 3) for i in tidx],
                tempo        = _c("tempo",   0, 1),
                energy       = _c("energy"),
                valence      = _c("valence"),
                danceability = _c("danceability"),
                acousticness = _c("acousticness"),
                key_label    = _s("key_label"),
                mode_label   = _s("mode_label"),
            )
            src_name   = df.iloc[idx]["name"]
            src_artist = df.iloc[idx]["artist"]
            k = df.iloc[idx].get("key_label", "?")
            m = df.iloc[idx].get("mode_label", "?")
            status_div.text = (
                f"<span style='color:{TEXT_MID}'>Showing {top} songs most similar to </span>"
                f"<b style='color:{ACCENT}'>{src_name}</b>"
                f"<span style='color:{TEXT_MID}'> by {src_artist}"
                f" &nbsp;·&nbsp; Key: <b style='color:{ACCENT_PURP}'>{k} {m}</b></span>"
            )

        search_btn = Button(label="Search", button_type="primary", width=110)
        search_btn.on_click(lambda: do_search())
        search_inp.on_change("value", lambda a, o, n: do_search())

        key_note = _div_sub(
            "Key and Mode columns show '?' when audio feature data is unavailable "
            "(requires Spotify Extended Quota access)."
        )

        return column(
            _div_title("Song Similarity Finder"),
            _div_sub("Ranks all songs by cosine similarity in the 9-dimensional audio feature space."),
            row(search_inp, search_btn, n_slider),
            status_div,
            res_table,
            key_note,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════

    def _build_header(self) -> Div:
        pl     = self.playlist_info
        name   = pl.get("name", "Playlist")
        owner  = pl.get("owner", {}).get("display_name", "")
        s      = self.stats
        n_out  = int(self.df.get("is_outlier", pd.Series([False]*len(self.df))).sum())
        pct    = round(100 * n_out / max(len(self.df), 1), 1)
        return Div(text=f"""
        <div style='background:linear-gradient(135deg,{PANEL},{BG});
                    padding:18px 26px;border-radius:10px;margin-bottom:10px;
                    border-left:4px solid {ACCENT};'>
          <h2 style='margin:0 0 4px;color:{ACCENT};font-size:21px;font-weight:700'>
            🎵 {name}
          </h2>
          <p style='margin:0;color:{TEXT_MID};font-size:12px;line-height:1.8em'>
            by <b style='color:{TEXT_HI}'>{owner}</b>
            &nbsp;·&nbsp; <b style='color:{TEXT_HI}'>{s.get('total_tracks','?')}</b> tracks
            &nbsp;·&nbsp; <b style='color:{TEXT_HI}'>{s.get('unique_artists','?')}</b> artists
            &nbsp;·&nbsp; <b style='color:{TEXT_HI}'>{s.get('total_duration_min','?')}</b> min
            &nbsp;·&nbsp; Avg BPM <b style='color:{ACCENT_GREEN}'>{s.get('avg_bpm','?')}</b>
            &nbsp;·&nbsp; Avg Energy <b style='color:{ACCENT_GREEN}'>{s.get('avg_energy','?')}</b>
            &nbsp;·&nbsp; <span style='color:{ACCENT_RED}'>{n_out} outliers ({pct}%)</span>
          </p>
        </div>""")

    def get_layout(self):
        return self.layout

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — EXPLORER  (user-configurable X / Y axes)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_explorer_tab(self) -> Any:
        df = self.df

        AXIS_OPTIONS = [
            ("Tempo (BPM)",       "tempo"),
            ("Energy",            "energy"),
            ("Danceability",      "danceability"),
            ("Valence",           "valence"),
            ("Acousticness",      "acousticness"),
            ("Instrumentalness",  "instrumentalness"),
            ("Liveness",          "liveness"),
            ("Speechiness",       "speechiness"),
            ("Loudness (dB)",     "loudness"),
            ("Release Year",      "release_year"),
            ("Duration (min)",    "duration_min"),
        ]
        axis_labels  = [lbl for lbl, _ in AXIS_OPTIONS]
        axis_map     = {lbl: col for lbl, col in AXIS_OPTIONS}
        available    = [lbl for lbl, col in AXIS_OPTIONS
                        if col in df.columns and df[col].notna().any()]

        default_x = "Tempo (BPM)"
        default_y = "Energy"

        def _safe(col, default=0.0):
            if col in df.columns:
                return df[col].fillna(default).round(4).tolist()
            return [default] * len(df)

        cluster_labels  = df["cluster"].astype(str).tolist()
        unique_clusters = sorted(set(cluster_labels),
                                 key=lambda x: int(x) if x.lstrip("-").isdigit() else x)
        pal    = _cluster_pal(len(unique_clusters))
        colors = [pal[unique_clusters.index(c)] for c in cluster_labels]

        is_outlier = df.get("is_outlier", pd.Series([False]*len(df))).fillna(False)

        src = ColumnDataSource(dict(
            x            = _safe(axis_map[default_x]),
            y            = _safe(axis_map[default_y]),
            name         = df["name"].tolist(),
            artist       = df["artist"].tolist(),
            tempo        = _safe("tempo", 0),
            energy       = _safe("energy"),
            danceability = _safe("danceability"),
            valence      = _safe("valence"),
            acousticness = _safe("acousticness"),
            liveness     = _safe("liveness"),
            speechiness  = _safe("speechiness"),
            instrumentalness = _safe("instrumentalness"),
            loudness     = _safe("loudness", -10),
            release_year = _safe("release_year", 0),
            duration_min = _safe("duration_min", 0),
            cluster_str  = cluster_labels,
            is_outlier   = is_outlier.tolist(),
            color        = colors,
        ))

        # ── Scatter ───────────────────────────────────────────────────────────
        p = figure(
            title=f"{default_y} vs {default_x}",
            width=820, height=520,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            toolbar_location="above",
            output_backend="webgl",
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p.title.text_color      = TITLE_COL
        p.title.text_font_size  = "13px"
        p.title.text_font_style = "bold"
        p.grid.grid_line_color  = BORDER
        p.grid.grid_line_alpha  = 0.5
        p.axis.major_label_text_color  = TEXT_MID
        p.axis.axis_label_text_color   = TEXT_HI
        p.axis.axis_label_text_font_size = "12px"
        p.axis.axis_line_color         = BORDER
        p.axis.minor_tick_line_color   = None
        p.outline_line_color           = BORDER
        p.toolbar.logo                 = None
        p.xaxis.axis_label = default_x
        p.yaxis.axis_label = default_y

        renderer = p.scatter(
            x="x", y="y",
            color="color", alpha=0.8, size=8,
            source=src, line_color=None,
            nonselection_alpha=0.15,
        )

        hover = p.select(dict(type=HoverTool))
        hover.tooltips = [
            ("",         "<b style='color:#79c0ff'>@name</b>"),
            ("Artist",   "@artist"),
            ("Cluster",  "@cluster_str"),
            ("X value",  "@x{0.000}"),
            ("Y value",  "@y{0.000}"),
        ]

        # ── Axis selectors ────────────────────────────────────────────────────
        x_sel = Select(title="X axis:", value=default_x, options=available, width=230,
                       stylesheets=[_WIDGET_CSS])
        y_sel = Select(title="Y axis:", value=default_y, options=available, width=230,
                       stylesheets=[_WIDGET_CSS])

        sort_sel = Select(
            title="Sort table by:", value=default_x, options=available, width=230,
            stylesheets=[_WIDGET_CSS],
        )
        sort_dir = Select(
            title="Order:", value="Descending",
            options=["Descending", "Ascending"], width=140,
            stylesheets=[_WIDGET_CSS],
        )
        top_n_sl = Slider(title="Show top N tracks", start=5, end=50,
                          value=20, step=5, width=230,
                          stylesheets=[_WIDGET_CSS])

        # ── Sorted table ──────────────────────────────────────────────────────
        tbl_cols_spec = [
            ("name",             "Track",         StringFormatter(), 230),
            ("artist",           "Artist",        StringFormatter(), 160),
            ("tempo",            "BPM",           NumberFormatter(format="0.0"),   65),
            ("energy",           "Energy",        NumberFormatter(format="0.00"),  70),
            ("danceability",     "Dance.",         NumberFormatter(format="0.00"),  70),
            ("valence",          "Valence",       NumberFormatter(format="0.00"),  70),
            ("acousticness",     "Acoustic.",     NumberFormatter(format="0.00"),  75),
            ("loudness",         "Loudness",      NumberFormatter(format="0.0"),   72),
            ("release_year",     "Year",          NumberFormatter(format="0"),     58),
            ("cluster_str",      "Cluster",       StringFormatter(),               65),
        ]

        def _sorted_data(sort_col, descending, n):
            tmp = df.copy()
            if sort_col not in tmp.columns:
                return tmp.head(n)
            return tmp.nlargest(n, sort_col) if descending else tmp.nsmallest(n, sort_col)

        init_sorted = _sorted_data(axis_map[default_x], True, 20)
        tbl_src = ColumnDataSource({
            c: init_sorted[c].fillna(0).round(3).tolist()
               if init_sorted[c].dtype != object
               else init_sorted[c].fillna("?").tolist()
            for c, _, _, _ in tbl_cols_spec if c in init_sorted.columns
        })
        tbl_cols_bokeh = [
            TableColumn(field=c, title=t, formatter=fmt, width=w)
            for c, t, fmt, w in tbl_cols_spec if c in init_sorted.columns
        ]
        tbl = DataTable(source=tbl_src, columns=tbl_cols_bokeh,
                        width=960, height=320, background=CARD,
                        index_position=None, row_height=28,
                        stylesheets=[_TABLE_CSS])

        status = Div(text="", styles={"margin": "4px 0 8px", "color": TEXT_MID,
                                      "font-size": "12px"})

        # ── Callbacks ─────────────────────────────────────────────────────────
        def refresh_scatter(attr, old, new):
            x_col = axis_map.get(x_sel.value, "tempo")
            y_col = axis_map.get(y_sel.value, "energy")
            src.patch({
                "x": [(slice(None), _safe(x_col))],
                "y": [(slice(None), _safe(y_col))],
            })
            p.title.text    = f"{y_sel.value} vs {x_sel.value}"
            p.xaxis.axis_label = x_sel.value
            p.yaxis.axis_label = y_sel.value

        def refresh_table(attr, old, new):
            sort_col  = axis_map.get(sort_sel.value, "tempo")
            desc      = sort_dir.value == "Descending"
            n         = int(top_n_sl.value)
            sorted_df = _sorted_data(sort_col, desc, n)
            new_data  = {}
            for c, _, _, _ in tbl_cols_spec:
                if c not in sorted_df.columns:
                    continue
                if sorted_df[c].dtype == object:
                    new_data[c] = sorted_df[c].fillna("?").tolist()
                else:
                    new_data[c] = sorted_df[c].fillna(0).round(3).tolist()
            tbl_src.data = new_data
            direction = "highest" if desc else "lowest"
            status.text = (f"Showing top {n} tracks by "
                           f"<b style='color:{ACCENT}'>{sort_sel.value}</b>"
                           f" ({direction} first)")

        x_sel.on_change("value", refresh_scatter)
        y_sel.on_change("value", refresh_scatter)
        sort_sel.on_change("value", refresh_table)
        sort_dir.on_change("value", refresh_table)
        top_n_sl.on_change("value", refresh_table)

        # Initial table status
        status.text = (f"Showing top 20 tracks by "
                       f"<b style='color:{ACCENT}'>{default_x}</b> (highest first)")

        ctrl_style = {"padding": "12px", "background": PANEL,
                      "border-radius": "8px", "border": f"1px solid {BORDER}",
                      "min-width": "250px"}
        controls = column(
            _div_title("Scatter Axes"),
            x_sel, y_sel,
            Div(text=f"<p style='color:{TEXT_LO};font-size:11px;"
                     "margin:8px 0 4px'>Each dot = one track. "
                     "Coloured by cluster.</p>"),
            _div_title("Table Sort"),
            sort_sel, sort_dir, top_n_sl,
            styles=ctrl_style,
        )

        return column(
            _div_title("Feature Explorer"),
            _div_sub("Choose any two features as axes. Sort the table independently "
                     "to find e.g. fastest, most energetic, or most acoustic tracks."),
            row(p, controls),
            status,
            _div_title("Ranked Track Table"),
            tbl,
        )
