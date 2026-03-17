"""
src/visualization/dashboard.py — Spotify Playlist Explorer Dashboard.

Visual design principles:
  • High-contrast text — no dark-on-dark
  • Cohesive indigo/teal accent palette
  • Cluster colours: vibrant categorical palette on dark background
  • All axis labels, titles, tooltips readable at a glance

Tabs:
  1. Map           — UMAP scatter, 9 audio sliders, outlier toggle
  2. Distributions — Histograms + Playlist DNA radar + Key/Mode + Timeline
  3. Correlations  — Pearson heatmap
  4. Clusters      — Feature profiles + bar charts with value labels
  5. Outliers      — Isolation Forest, live threshold slider
  6. Similarity    — Cosine nearest-neighbour search
  7. Explorer      — Configurable X/Y scatter + sorted table
  8. Taste Timeline — Temporal evolution + UMAP HTML export
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
from bokeh.resources import CDN
from bokeh.embed import file_html
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


        self.source = ColumnDataSource(dict(
            x                = emb[:, 0].tolist(),
            y                = emb[:, 1].tolist(),
            name             = df["name"].tolist(),
            artist           = df["artist"].tolist(),
            album            = df["album"].tolist(),
            release_year     = _safe("release_year", 0),
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
            primary_genre    = df["primary_genre"].tolist() if "primary_genre" in df.columns else ["?"]*len(df),
            added_date       = df["added_date"].tolist() if "added_date" in df.columns else ["?"]*len(df),
            added_year       = df["added_year"].fillna(0).astype(int).tolist() if "added_year" in df.columns else [0]*len(df),
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
            TabPanel(child=self._build_temporal_tab(),          title="🕰  Taste Timeline"),
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
            ("Cluster",      "@cluster_str"),
            ("Outlier",      "@is_outlier_int"),
            ("Year",         "@release_year{0}"),
            ("Added",         "@added_date"),
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

        # ── Histograms ───────────────────────────────────────────────────────────
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
            # Mean line + label
            mean_v   = float(vals.mean())
            mean_fmt = f"{mean_v:.0f}" if lo >= 40 else f"{mean_v:.2f}"
            p.line([mean_v, mean_v], [0, hist.max()],
                   line_color=TEXT_HI, line_dash="dashed", line_width=1.5, alpha=0.8)
            p.text(x=[mean_v], y=[hist.max() * 0.97],
                   text=[f"μ {mean_fmt}"],
                   text_color=TEXT_HI, text_font_size="9px",
                   text_align="left", x_offset=4, alpha=0.9)
            hist_plots.append(p)

        hist_grid = gridplot(hist_plots, ncols=4, merge_tools=False)

        key_row = Div(text="")

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

        # ── Tracks added over time ───────────────────────────────────────────
        added_plot = None
        if "added_year" in df.columns and df["added_year"].notna().any():
            year_counts = df["added_year"].dropna().astype(int).value_counts().sort_index()
            if len(year_counts) > 1:
                p_add = _fig("Tracks Added per Year", w=580, h=220)
                p_add.vbar(x=year_counts.index.tolist(), top=year_counts.values.tolist(),
                           width=0.7, color=ACCENT_GREEN, line_color=BG, alpha=0.9)
                p_add.xaxis.major_label_text_color = TEXT_HI
                p_add.xaxis.major_label_text_font_size = "10px"
                p_add.xaxis.major_label_orientation = 0.8
                p_add.y_range.start = 0
                added_plot = column(
                    _div_title("Playlist Timeline"),
                    _div_sub("When songs were added to this playlist."),
                    p_add,
                )
            elif len(year_counts) == 1:
                added_plot = _card(
                    f"All tracks were added in <b style='color:{TEXT_HI}'>{year_counts.index[0]}</b>."
                )

        # ── Monthly breakdown (if all additions in same year) ─────────────────
        if added_plot is None and "added_month" in df.columns and df["added_month"].notna().any():
            month_counts = df["added_month"].dropna().astype(int).value_counts().sort_index()
            months = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
            xlabels = [months[m-1] for m in month_counts.index]
            p_mon = _fig("Tracks Added by Month", w=580, h=220, x_range=xlabels)
            p_mon.vbar(x=xlabels, top=month_counts.values.tolist(),
                       width=0.7, color=ACCENT_GREEN, line_color=BG, alpha=0.9)
            p_mon.xaxis.major_label_text_color = TEXT_HI
            added_plot = column(
                _div_title("Playlist Timeline"),
                _div_sub("When songs were added (by month)."),
                p_mon,
            )

        timeline_section = added_plot if added_plot else _card(
            f"<span style='color:{TEXT_LO}'>Added date data not available for this playlist.</span>"
        )

        # ── Playlist DNA radar (pentagon of key audio features) ──────────────
        radar_feats = [
            ("Energy",       "energy"),
            ("Danceability", "danceability"),
            ("Valence",      "valence"),
            ("Acousticness", "acousticness"),
            ("Liveness",     "liveness"),
        ]
        radar_available = [
            (lbl, col) for lbl, col in radar_feats
            if col in df.columns and df[col].notna().any()
        ]
        if len(radar_available) >= 3:
            import math
            n_sides = len(radar_available)
            angles  = [2 * math.pi * i / n_sides - math.pi / 2 for i in range(n_sides)]
            values  = [float(df[col].mean()) for _, col in radar_available]
            labels  = [lbl for lbl, _ in radar_available]

            # Polygon points (scaled to 0–1)
            poly_x = [v * math.cos(a) for v, a in zip(values, angles)]
            poly_y = [v * math.sin(a) for v, a in zip(values, angles)]
            # Close the polygon
            poly_x.append(poly_x[0]); poly_y.append(poly_y[0])

            # Grid circles
            p_radar = figure(
                title="Playlist DNA", width=280, height=280,
                toolbar_location=None,
                x_range=(-1.3, 1.3), y_range=(-1.3, 1.3),
                background_fill_color=BG, border_fill_color=PANEL,
                outline_line_color=BORDER,
            )
            p_radar.title.text_color    = TITLE_COL
            p_radar.title.text_font_size= "12px"
            p_radar.title.text_font_style = "bold"
            p_radar.grid.visible        = False
            p_radar.axis.visible        = False
            p_radar.outline_line_color  = BORDER

            # Draw grid rings
            for r in [0.25, 0.5, 0.75, 1.0]:
                ring_x = [r * math.cos(2 * math.pi * t / 60) for t in range(61)]
                ring_y = [r * math.sin(2 * math.pi * t / 60) for t in range(61)]
                p_radar.line(ring_x, ring_y,
                             line_color=BORDER, line_width=1, alpha=0.6)

            # Draw spokes
            for a in angles:
                p_radar.line([0, math.cos(a)], [0, math.sin(a)],
                             line_color=BORDER, line_width=1, alpha=0.5)

            # Fill polygon
            p_radar.patch(poly_x, poly_y,
                          fill_color=ACCENT, fill_alpha=0.25,
                          line_color=ACCENT, line_width=2)
            p_radar.scatter(poly_x[:-1], poly_y[:-1],
                            size=8, color=ACCENT, alpha=0.9, line_color=BG)

            # Labels
            for i, (lbl, a, v) in enumerate(zip(labels, angles, values)):
                lx = 1.18 * math.cos(a)
                ly = 1.18 * math.sin(a)
                p_radar.text(
                    [lx], [ly],
                    text=[f"{lbl}\n{v:.2f}"],
                    text_align="center", text_baseline="middle",
                    text_font_size="10px", text_color=TEXT_HI,
                )

            radar_section = column(
                _div_title("Playlist DNA"),
                _div_sub("Mean values of key audio features — the sonic fingerprint of this playlist."),
                p_radar,
            )
        else:
            radar_section = Div(text="")

        return column(
            row(summary, radar_section) if len(radar_available) >= 3 else summary,
            _div_title("Audio Feature Distributions"),
            _div_sub("Dashed line = playlist mean. Value shown in top-left of each chart."),
            hist_grid,
            _div_title("Key & Mode"),
            key_row,
            timeline_section,
        )

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
        cols.append(TableColumn(field="representative", title="Representative track", width=270))

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
            # Value labels on bars
            fmt = "0.0" if label in ("Tempo (BPM)", "Loudness (dB)") else "0.00"
            p.text(
                x=cids,
                y=[m + (max(means) * 0.03) for m in means],
                text=[f"{m:.0f}" if label in ("Tempo (BPM)","Loudness (dB)") else f"{m:.2f}" for m in means],
                text_align="center", text_baseline="bottom",
                text_font_size="9px", text_color=TEXT_MID,
            )
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
            f"Detected on 9 pure audio features only — no release year.<br>"
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
            added_date=[],
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
            TableColumn(field="added_date",    title="Date Added", formatter=StringFormatter(),                width=100),
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
                added_date   = _s("added_date"),
            )
            src_name   = df.iloc[idx]["name"]
            src_artist = df.iloc[idx]["artist"]
            status_div.text = (
                f"<span style='color:{TEXT_MID}'>Showing {top} songs most similar to </span>"
                f"<b style='color:{ACCENT}'>{src_name}</b>"
                f"<span style='color:{TEXT_MID}'> by {src_artist}</span>"
            )

        search_btn = Button(label="Search", button_type="primary", width=110)
        search_btn.on_click(lambda: do_search())
        search_inp.on_change("value", lambda a, o, n: do_search())

        key_note = _div_sub(
            "Key, Mode and Camelot show '?' until a GetSongBPM API key is added to .env."
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
        n_out = int(self.df.get("is_outlier", pd.Series([False]*len(self.df))).sum())
        pct   = round(100 * n_out / max(len(self.df), 1), 1)
        # Data source badges
        rb_badge = f"<span style='background:#1c2128;color:{ACCENT_GREEN};border:1px solid {ACCENT_GREEN};border-radius:4px;padding:1px 7px;font-size:10px;margin-right:4px'>ReccoBeats ✓</span>"
        return Div(text=f"""
        <div style='background:linear-gradient(135deg,{PANEL},{BG});
                    padding:18px 26px;border-radius:10px;margin-bottom:10px;
                    border-left:4px solid {ACCENT};'>
          <h2 style='margin:0 0 6px;color:{ACCENT};font-size:21px;font-weight:700'>
            🎵 {name}
          </h2>
          <p style='margin:0 0 8px;color:{TEXT_MID};font-size:12px;line-height:1.8em'>
            by <b style='color:{TEXT_HI}'>{owner}</b>
            &nbsp;·&nbsp; <b style='color:{TEXT_HI}'>{s.get('total_tracks','?')}</b> tracks
            &nbsp;·&nbsp; <b style='color:{TEXT_HI}'>{s.get('unique_artists','?')}</b> artists
            &nbsp;·&nbsp; <b style='color:{TEXT_HI}'>{s.get('total_duration_min','?')}</b> min
            &nbsp;·&nbsp; Avg BPM <b style='color:{ACCENT_GREEN}'>{s.get('avg_bpm','?')}</b>
            &nbsp;·&nbsp; Avg Energy <b style='color:{ACCENT_GREEN}'>{s.get('avg_energy','?')}</b>
            &nbsp;·&nbsp; Avg Valence <b style='color:{ACCENT_GREEN}'>{s.get('avg_valence','?')}</b>
            &nbsp;·&nbsp; <span style='color:{ACCENT_RED}'>{n_out} outliers ({pct}%)</span>
          </p>
          <div style='margin-top:4px'>{rb_badge}</div>
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
            ("Year Added",         "added_year"),
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
            added_year   = _safe("added_year", 0),
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
            ("added_date",        "Date Added",    StringFormatter(),               100),
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

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 — TASTE TIMELINE  (temporal evolution of audio features)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_temporal_tab(self) -> Any:
        df  = self.df.copy()

        # ── Determine temporal axis ───────────────────────────────────────────
        # Prefer added_year (when you added songs) over release_year
        has_added  = "added_year"   in df.columns and df["added_year"].notna().any()  and df["added_year"].ne(0).any()
        has_release= "release_year" in df.columns and df["release_year"].notna().any()

        if not has_added and not has_release:
            return column(
                _div_title("Taste Timeline"),
                _card(f"<span style='color:{TEXT_LO}'>No temporal data available. "
                      "Re-run the pipeline with the updated client.py to capture "
                      "added_at timestamps.</span>")
            )

        # Use added_year when present, release_year as fallback
        df["_year"] = df["added_year"].where(
            has_added and df["added_year"].ne(0), df.get("release_year", pd.Series([None]*len(df)))
        ) if has_added else df["release_year"]

        df = df.dropna(subset=["_year"])
        df["_year"] = df["_year"].astype(int)

        years      = sorted(df["_year"].unique())
        year_min   = int(df["_year"].min())
        year_max   = int(df["_year"].max())
        n_years    = year_max - year_min + 1

        FEATS = [
            ("energy",           "Energy",       ACCENT),
            ("danceability",     "Danceability",  ACCENT_GREEN),
            ("valence",          "Valence",       ACCENT_PURP),
            ("acousticness",     "Acousticness",  "#ffa657"),
            ("speechiness",      "Speechiness",   "#79c0ff"),
            ("liveness",         "Liveness",      "#56d364"),
            ("instrumentalness", "Instrumental.", "#ff7b72"),
        ]
        FEATS = [(c, l, col) for c, l, col in FEATS
                 if c in df.columns and df[c].notna().any()]

        # ── Aggregate by year ─────────────────────────────────────────────────
        grp = df.groupby("_year")

        def year_agg(col):
            return grp[col].agg(["mean","median","std","count"]).reset_index()

        # ── Plot 1: Multi-feature trend lines (mean) ──────────────────────────
        p1 = figure(
            title="Audio Feature Trends Over Time  (mean per year)",
            width=980, height=320,
            tools="pan,wheel_zoom,reset,hover,save",
            toolbar_location="above",
            x_range=(year_min - 0.5, year_max + 0.5),
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p1.title.text_color    = TITLE_COL
        p1.title.text_font_size= "12px"
        p1.grid.grid_line_color= BORDER
        p1.grid.grid_line_alpha= 0.5
        p1.axis.major_label_text_color = TEXT_MID
        p1.axis.axis_line_color        = BORDER
        p1.toolbar.logo                = None
        p1.yaxis.axis_label            = "Feature value (0–1)"
        p1.yaxis.axis_label_text_color = TEXT_HI
        p1.xaxis.axis_label            = "Added year" if has_added else "Release year"
        p1.xaxis.axis_label_text_color = TEXT_HI

        legend_items = []
        for col, label, color in FEATS:
            agg = year_agg(col)
            agg = agg[agg["count"] >= 2]
            if len(agg) < 2:
                continue
            xs = agg["_year"].tolist()
            ys = agg["mean"].round(3).tolist()
            ys_std = agg["std"].fillna(0).tolist()

            # Band (mean ± std)
            band_x  = xs + xs[::-1]
            band_hi = [min(1.0, m + s) for m, s in zip(ys, ys_std)]
            band_lo = [max(0.0, m - s) for m, s in zip(ys, ys_std)]
            band_y  = band_hi + band_lo[::-1]

            band_src = ColumnDataSource(dict(x=band_x, y=band_y))
            p1.patch("x", "y", source=band_src,
                     fill_color=color, fill_alpha=0.08, line_color=None)

            src1 = ColumnDataSource(dict(x=xs, y=ys, label=[label]*len(xs),
                                         count=agg["count"].tolist()))
            r = p1.line("x", "y", source=src1, line_color=color,
                        line_width=2.5, alpha=0.95)
            p1.scatter("x", "y", source=src1, color=color,
                       size=7, alpha=0.9, line_color=BG, line_width=1)
            legend_items.append((label, [r]))

        from bokeh.models import Legend
        legend = Legend(items=legend_items, location="top_right",
                        background_fill_color=PANEL,
                        background_fill_alpha=0.85,
                        border_line_color=BORDER,
                        label_text_color=TEXT_HI,
                        label_text_font_size="10px")
        p1.add_layout(legend, "right")
        p1.add_tools(HoverTool(tooltips=[
            ("Year",    "@x"),
            ("Value",   "@y{0.000}"),
            ("Feature", "@label"),
            ("Tracks",  "@count"),
        ]))

        # ── Plot 2: Energy × Valence mood map (scatter, coloured by year) ─────
        mood_src = ColumnDataSource(dict(
            energy   = df["energy"].fillna(0).tolist(),
            valence  = df["valence"].fillna(0).tolist(),
            year     = df["_year"].tolist(),
            name     = df["name"].tolist(),
            artist   = df["artist"].tolist(),
        ))
        mapper2 = LinearColorMapper(
            palette=CONT_PAL,
            low=year_min, high=year_max,
        )
        p2 = figure(
            title="Mood Map: Energy vs Valence  (coloured by year)",
            width=480, height=350,
            tools="pan,wheel_zoom,reset,hover,save",
            toolbar_location="above",
            x_range=(-0.02, 1.02), y_range=(-0.02, 1.02),
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p2.title.text_color     = TITLE_COL
        p2.title.text_font_size = "11px"
        p2.grid.grid_line_color = BORDER
        p2.grid.grid_line_alpha = 0.4
        p2.axis.major_label_text_color = TEXT_MID
        p2.axis.axis_line_color        = BORDER
        p2.xaxis.axis_label            = "Valence  →  happier"
        p2.yaxis.axis_label            = "Energy  →  more intense"
        p2.xaxis.axis_label_text_color = TEXT_HI
        p2.yaxis.axis_label_text_color = TEXT_HI
        p2.toolbar.logo = None

        p2.scatter("valence", "energy", source=mood_src, size=5, alpha=0.7,
                   fill_color={"field": "year", "transform": mapper2},
                   line_color=None)
        p2.add_tools(HoverTool(tooltips=[
            ("",       "<b style='color:#79c0ff'>@name</b>"),
            ("Artist", "@artist"),
            ("Year",   "@year"),
        ]))

        # Quadrant labels
        for tx, ty, label in [
            (0.75, 0.85, "Happy & Intense"),
            (0.15, 0.85, "Dark & Intense"),
            (0.75, 0.10, "Happy & Calm"),
            (0.15, 0.10, "Dark & Calm"),
        ]:
            p2.text(x=[tx], y=[ty], text=[label],
                    text_color=TEXT_LO, text_font_size="9px",
                    text_align="center", alpha=0.6)

        # Quadrant lines
        p2.line([0.5, 0.5], [0, 1], line_color=BORDER, line_dash="dashed",
                line_width=1, alpha=0.5)
        p2.line([0, 1], [0.5, 0.5], line_color=BORDER, line_dash="dashed",
                line_width=1, alpha=0.5)

        cb2 = ColorBar(color_mapper=mapper2, location=(0,0), width=12,
                       title="Year",
                       title_text_color=TEXT_MID,
                       title_text_font_size="10px",
                       major_label_text_color=TEXT_MID,
                       background_fill_color=PANEL)
        p2.add_layout(cb2, "right")

        # ── Plot 3: BPM trend over time ───────────────────────────────────────
        if "tempo" in df.columns and df["tempo"].notna().any():
            tempo_agg = year_agg("tempo")
            tempo_agg = tempo_agg[tempo_agg["count"] >= 2]

            p3 = figure(
                title="Average BPM Over Time",
                width=480, height=250,
                tools="pan,wheel_zoom,reset,save",
                toolbar_location="above",
                x_range=(year_min - 0.5, year_max + 0.5),
                background_fill_color=BG, border_fill_color=PANEL,
                outline_line_color=BORDER,
            )
            p3.title.text_color     = TITLE_COL
            p3.title.text_font_size = "11px"
            p3.grid.grid_line_color = BORDER
            p3.axis.major_label_text_color = TEXT_MID
            p3.axis.axis_line_color        = BORDER
            p3.yaxis.axis_label            = "BPM"
            p3.yaxis.axis_label_text_color = TEXT_HI
            p3.toolbar.logo = None

            bpm_src = ColumnDataSource(dict(
                x    = tempo_agg["_year"].tolist(),
                y    = tempo_agg["mean"].round(1).tolist(),
                hi   = (tempo_agg["mean"] + tempo_agg["std"].fillna(0)).round(1).tolist(),
                lo   = (tempo_agg["mean"] - tempo_agg["std"].fillna(0)).round(1).tolist(),
                count= tempo_agg["count"].tolist(),
            ))
            band_x = tempo_agg["_year"].tolist() + tempo_agg["_year"].tolist()[::-1]
            band_y = (tempo_agg["mean"] + tempo_agg["std"].fillna(0)).tolist() + \
                     (tempo_agg["mean"] - tempo_agg["std"].fillna(0)).tolist()[::-1]
            p3.patch(band_x, band_y, fill_color=ACCENT_WARN, fill_alpha=0.12,
                     line_color=None)
            p3.line("x", "y", source=bpm_src, line_color=ACCENT_WARN,
                    line_width=2.5, alpha=0.95)
            p3.scatter("x", "y", source=bpm_src, color=ACCENT_WARN,
                       size=7, alpha=0.9, line_color=BG)
            p3.add_tools(HoverTool(tooltips=[
                ("Year",   "@x"),
                ("Avg BPM","@y{0.0}"),
                ("Tracks", "@count"),
            ]))
        else:
            p3 = None

        # ── Plot 4: Track count per year (how active you were adding music) ────
        count_by_year = df.groupby("_year").size().reset_index(name="count")
        p4 = figure(
            title="Tracks Added per Year",
            width=480, height=250,
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
            x_range=(year_min - 0.5, year_max + 0.5),
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p4.title.text_color     = TITLE_COL
        p4.title.text_font_size = "11px"
        p4.grid.grid_line_color = BORDER
        p4.axis.major_label_text_color = TEXT_MID
        p4.axis.axis_line_color        = BORDER
        p4.yaxis.axis_label            = "Tracks"
        p4.yaxis.axis_label_text_color = TEXT_HI
        p4.y_range.start               = 0
        p4.toolbar.logo = None

        p4.vbar(x=count_by_year["_year"].tolist(),
                top=count_by_year["count"].tolist(),
                width=0.7, color=ACCENT_GREEN, line_color=BG, alpha=0.9)
        p4.add_tools(HoverTool(tooltips=[("Year","@x"), ("Tracks","@top")]))

        # ── Plot 5: Feature heatmap — year × feature ──────────────────────────
        feat_cols_hm = [c for c, _, _ in FEATS]
        feat_labels  = [l for _, l, _ in FEATS]
        hm_years     = [str(y) for y in sorted(df["_year"].unique())]

        xs_hm, ys_hm, vs_hm, ts_hm = [], [], [], []
        for yr in sorted(df["_year"].unique()):
            sub = df[df["_year"] == yr]
            for col, label in zip(feat_cols_hm, feat_labels):
                mean_v = float(sub[col].mean()) if col in sub.columns else 0.0
                xs_hm.append(str(yr))
                ys_hm.append(label)
                vs_hm.append(round(mean_v, 3))
                ts_hm.append(f"{mean_v:.2f}")

        hm_src    = ColumnDataSource(dict(x=xs_hm, y=ys_hm, vals=vs_hm, texts=ts_hm))
        hm_mapper = LinearColorMapper(palette="Viridis256", low=0, high=1)

        hm_width  = max(500, min(1000, len(hm_years) * 28 + 160))
        p5 = figure(
            title="Feature Heatmap by Year  (mean intensity)",
            x_range=hm_years,
            y_range=list(reversed(feat_labels)),
            width=hm_width, height=280,
            tools="hover,save",
            toolbar_location="above",
            x_axis_location="above",
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p5.title.text_color            = TITLE_COL
        p5.title.text_font_size        = "11px"
        p5.grid.grid_line_color        = None
        p5.axis.major_label_text_color = TEXT_HI
        p5.axis.axis_line_color        = BORDER
        p5.xaxis.major_label_orientation = 0.9
        p5.xaxis.major_label_text_font_size = "9px"
        p5.toolbar.logo = None

        p5.rect(x="x", y="y", width=1, height=1, source=hm_src,
                fill_color={"field":"vals","transform":hm_mapper},
                line_color=BG, line_width=1)

        # Only show text if not too many years
        if len(hm_years) <= 20:
            p5.text(x="x", y="y", text="texts", source=hm_src,
                    text_align="center", text_baseline="middle",
                    text_font_size="8px", text_color="#111111")

        cb5 = ColorBar(color_mapper=hm_mapper, location=(0,0), width=12,
                       major_label_text_color=TEXT_MID,
                       background_fill_color=PANEL)
        p5.add_layout(cb5, "right")
        p5.add_tools(HoverTool(tooltips=[
            ("Year",    "@x"),
            ("Feature", "@y"),
            ("Mean",    "@vals{0.000}"),
        ]))

        # ── UMAP HTML export button ───────────────────────────────────────────
        # Build a standalone UMAP scatter for export
        export_btn = Button(
            label="💾  Download UMAP as HTML",
            button_type="success", width=240,
        )
        export_status = Div(
            text=f"<span style='color:{TEXT_LO};font-size:11px'>"
                 "Saves a self-contained interactive HTML file to your Desktop.</span>",
            styles={"margin": "4px 0 10px"},
        )

        def do_export_umap():
            try:
                import os
                from pathlib import Path
                from bokeh.plotting import figure as bk_fig
                from bokeh.embed import file_html
                from bokeh.resources import CDN
                from bokeh.models import HoverTool as HT, ColumnDataSource as CDS

                ex_src = CDS(dict(
                    x    = self.embedding[:, 0].tolist(),
                    y    = self.embedding[:, 1].tolist(),
                    name = self.df["name"].tolist(),
                    artist = self.df["artist"].tolist(),
                    cluster = self.df["cluster"].astype(str).tolist(),
                    energy  = self.df["energy"].fillna(0).round(2).tolist()
                              if "energy" in self.df.columns else [0]*len(self.df),
                    tempo   = self.df["tempo"].fillna(0).round(1).tolist()
                              if "tempo" in self.df.columns else [0]*len(self.df),
                    valence = self.df["valence"].fillna(0).round(2).tolist()
                              if "valence" in self.df.columns else [0]*len(self.df),
                    color   = self._full_data["color"],
                ))
                ep = bk_fig(
                    title=f"UMAP — {self.playlist_info.get('name','Playlist')}",
                    width=1100, height=750,
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    toolbar_location="above",
                    output_backend="webgl",
                    background_fill_color=BG, border_fill_color=PANEL,
                    outline_line_color=BORDER,
                )
                ep.title.text_color     = TITLE_COL
                ep.title.text_font_size = "14px"
                ep.grid.grid_line_color = BORDER
                ep.axis.major_label_text_color = TEXT_MID
                ep.axis.axis_line_color        = BORDER
                ep.toolbar.logo = None

                ep.scatter("x","y", source=ex_src, color="color",
                           alpha=0.85, size=8, line_color=None,
                           nonselection_alpha=0.15)
                ep.add_tools(HT(tooltips=[
                    ("", "<b style='color:#79c0ff'>@name</b>"),
                    ("Artist",  "@artist"),
                    ("Cluster", "@cluster"),
                    ("Energy",  "@energy{0.00}"),
                    ("BPM",     "@tempo{0.0}"),
                    ("Valence", "@valence{0.00}"),
                ]))

                html = file_html(ep, CDN,
                                 title=f"UMAP — {self.playlist_info.get('name','Playlist')}")

                desktop = Path.home() / "Desktop"
                pl_name = self.playlist_info.get("name", "playlist")
                safe    = "".join(c for c in pl_name if c.isalnum() or c in " _-")[:40].strip()
                out     = desktop / f"UMAP_{safe}.html"
                out.write_text(html, encoding="utf-8")
                export_status.text = (
                    f"<span style='color:{ACCENT_GREEN}'>✓ Saved to: {out}</span>"
                )
            except Exception as ex:
                export_status.text = (
                    f"<span style='color:{ACCENT_RED}'>Export failed: {ex}</span>"
                )

        export_btn.on_click(lambda: do_export_umap())

        # ── Key stats summary ─────────────────────────────────────────────────
        timeline_axis = "added" if has_added else "release year"
        most_active_year = int(count_by_year.loc[count_by_year["count"].idxmax(), "_year"])
        most_active_n    = int(count_by_year["count"].max())

        recent = df[df["_year"] >= year_max - 2]
        older  = df[df["_year"] <= year_min + 2]

        def _feat_diff(col):
            if col not in df.columns or not df[col].notna().any():
                return "N/A"
            r = recent[col].mean(); o = older[col].mean()
            if pd.isna(r) or pd.isna(o): return "N/A"
            diff = r - o
            arrow = "↑" if diff > 0.02 else ("↓" if diff < -0.02 else "→")
            return f"{arrow} {abs(diff):.2f}"

        summary_html = (
            f"<b style='font-size:13px;color:{ACCENT}'>Timeline Insights</b><br><br>"
            f"Temporal axis: <b style='color:{TEXT_HI}'>{timeline_axis}</b>"
            f" &nbsp;·&nbsp; Range: <b style='color:{TEXT_HI}'>{year_min} – {year_max}</b>"
            f" &nbsp;·&nbsp; Most active: <b style='color:{ACCENT_GREEN}'>{most_active_year}</b>"
            f" ({most_active_n} tracks)<br>"
            f"Energy trend (recent vs early): <b style='color:{TEXT_HI}'>{_feat_diff('energy')}</b>"
            f" &nbsp;·&nbsp; Valence: <b style='color:{TEXT_HI}'>{_feat_diff('valence')}</b>"
            f" &nbsp;·&nbsp; Danceability: <b style='color:{TEXT_HI}'>{_feat_diff('danceability')}</b>"
        )

        # ── Assemble ──────────────────────────────────────────────────────────
        row2 = row(p2, column(p4, p3 if p3 else Div(text="")))

        return column(
            _card(summary_html),
            _div_title("Feature Trends", ACCENT),
            _div_sub("Shaded band = ±1 standard deviation. "
                     f"X-axis = {timeline_axis}."),
            p1,
            _div_title("Mood Map  ·  Feature Heatmap"),
            row2,
            _div_sub("Heatmap shows mean feature intensity per year — darker = higher."),
            p5,
            Div(text=f"<hr style='border-color:{BORDER};margin:18px 0'>"),
            _div_title("Export UMAP", ACCENT_GREEN),
            _div_sub("Downloads a fully interactive standalone HTML file — "
                     "zoom, pan, hover. No server needed to open it."),
            row(export_btn, export_status),
        )


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
            ("Year Added",         "added_year"),
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
            added_year   = _safe("added_year", 0),
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
            ("added_date",        "Date Added",    StringFormatter(),               100),
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

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 — TASTE TIMELINE  (temporal evolution of audio features)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_temporal_tab(self) -> Any:
        df  = self.df.copy()

        # ── Determine temporal axis ───────────────────────────────────────────
        # Prefer added_year (when you added songs) over release_year
        has_added  = "added_year"   in df.columns and df["added_year"].notna().any()  and df["added_year"].ne(0).any()
        has_release= "release_year" in df.columns and df["release_year"].notna().any()

        if not has_added and not has_release:
            return column(
                _div_title("Taste Timeline"),
                _card(f"<span style='color:{TEXT_LO}'>No temporal data available. "
                      "Re-run the pipeline with the updated client.py to capture "
                      "added_at timestamps.</span>")
            )

        # Use added_year when present, release_year as fallback
        df["_year"] = df["added_year"].where(
            has_added and df["added_year"].ne(0), df.get("release_year", pd.Series([None]*len(df)))
        ) if has_added else df["release_year"]

        df = df.dropna(subset=["_year"])
        df["_year"] = df["_year"].astype(int)

        years      = sorted(df["_year"].unique())
        year_min   = int(df["_year"].min())
        year_max   = int(df["_year"].max())
        n_years    = year_max - year_min + 1

        FEATS = [
            ("energy",           "Energy",       ACCENT),
            ("danceability",     "Danceability",  ACCENT_GREEN),
            ("valence",          "Valence",       ACCENT_PURP),
            ("acousticness",     "Acousticness",  "#ffa657"),
            ("speechiness",      "Speechiness",   "#79c0ff"),
            ("liveness",         "Liveness",      "#56d364"),
            ("instrumentalness", "Instrumental.", "#ff7b72"),
        ]
        FEATS = [(c, l, col) for c, l, col in FEATS
                 if c in df.columns and df[c].notna().any()]

        # ── Aggregate by year ─────────────────────────────────────────────────
        grp = df.groupby("_year")

        def year_agg(col):
            return grp[col].agg(["mean","median","std","count"]).reset_index()

        # ── Plot 1: Multi-feature trend lines (mean) ──────────────────────────
        p1 = figure(
            title="Audio Feature Trends Over Time  (mean per year)",
            width=980, height=320,
            tools="pan,wheel_zoom,reset,hover,save",
            toolbar_location="above",
            x_range=(year_min - 0.5, year_max + 0.5),
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p1.title.text_color    = TITLE_COL
        p1.title.text_font_size= "12px"
        p1.grid.grid_line_color= BORDER
        p1.grid.grid_line_alpha= 0.5
        p1.axis.major_label_text_color = TEXT_MID
        p1.axis.axis_line_color        = BORDER
        p1.toolbar.logo                = None
        p1.yaxis.axis_label            = "Feature value (0–1)"
        p1.yaxis.axis_label_text_color = TEXT_HI
        p1.xaxis.axis_label            = "Added year" if has_added else "Release year"
        p1.xaxis.axis_label_text_color = TEXT_HI

        legend_items = []
        for col, label, color in FEATS:
            agg = year_agg(col)
            agg = agg[agg["count"] >= 2]
            if len(agg) < 2:
                continue
            xs = agg["_year"].tolist()
            ys = agg["mean"].round(3).tolist()
            ys_std = agg["std"].fillna(0).tolist()

            # Band (mean ± std)
            band_x  = xs + xs[::-1]
            band_hi = [min(1.0, m + s) for m, s in zip(ys, ys_std)]
            band_lo = [max(0.0, m - s) for m, s in zip(ys, ys_std)]
            band_y  = band_hi + band_lo[::-1]

            band_src = ColumnDataSource(dict(x=band_x, y=band_y))
            p1.patch("x", "y", source=band_src,
                     fill_color=color, fill_alpha=0.08, line_color=None)

            src1 = ColumnDataSource(dict(x=xs, y=ys, label=[label]*len(xs),
                                         count=agg["count"].tolist()))
            r = p1.line("x", "y", source=src1, line_color=color,
                        line_width=2.5, alpha=0.95)
            p1.scatter("x", "y", source=src1, color=color,
                       size=7, alpha=0.9, line_color=BG, line_width=1)
            legend_items.append((label, [r]))

        from bokeh.models import Legend
        legend = Legend(items=legend_items, location="top_right",
                        background_fill_color=PANEL,
                        background_fill_alpha=0.85,
                        border_line_color=BORDER,
                        label_text_color=TEXT_HI,
                        label_text_font_size="10px")
        p1.add_layout(legend, "right")
        p1.add_tools(HoverTool(tooltips=[
            ("Year",    "@x"),
            ("Value",   "@y{0.000}"),
            ("Feature", "@label"),
            ("Tracks",  "@count"),
        ]))

        # ── Plot 2: Energy × Valence mood map (scatter, coloured by year) ─────
        mood_src = ColumnDataSource(dict(
            energy   = df["energy"].fillna(0).tolist(),
            valence  = df["valence"].fillna(0).tolist(),
            year     = df["_year"].tolist(),
            name     = df["name"].tolist(),
            artist   = df["artist"].tolist(),
        ))
        mapper2 = LinearColorMapper(
            palette=CONT_PAL,
            low=year_min, high=year_max,
        )
        p2 = figure(
            title="Mood Map: Energy vs Valence  (coloured by year)",
            width=480, height=350,
            tools="pan,wheel_zoom,reset,hover,save",
            toolbar_location="above",
            x_range=(-0.02, 1.02), y_range=(-0.02, 1.02),
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p2.title.text_color     = TITLE_COL
        p2.title.text_font_size = "11px"
        p2.grid.grid_line_color = BORDER
        p2.grid.grid_line_alpha = 0.4
        p2.axis.major_label_text_color = TEXT_MID
        p2.axis.axis_line_color        = BORDER
        p2.xaxis.axis_label            = "Valence  →  happier"
        p2.yaxis.axis_label            = "Energy  →  more intense"
        p2.xaxis.axis_label_text_color = TEXT_HI
        p2.yaxis.axis_label_text_color = TEXT_HI
        p2.toolbar.logo = None

        p2.scatter("valence", "energy", source=mood_src, size=5, alpha=0.7,
                   fill_color={"field": "year", "transform": mapper2},
                   line_color=None)
        p2.add_tools(HoverTool(tooltips=[
            ("",       "<b style='color:#79c0ff'>@name</b>"),
            ("Artist", "@artist"),
            ("Year",   "@year"),
        ]))

        # Quadrant labels
        for tx, ty, label in [
            (0.75, 0.85, "Happy & Intense"),
            (0.15, 0.85, "Dark & Intense"),
            (0.75, 0.10, "Happy & Calm"),
            (0.15, 0.10, "Dark & Calm"),
        ]:
            p2.text(x=[tx], y=[ty], text=[label],
                    text_color=TEXT_LO, text_font_size="9px",
                    text_align="center", alpha=0.6)

        # Quadrant lines
        p2.line([0.5, 0.5], [0, 1], line_color=BORDER, line_dash="dashed",
                line_width=1, alpha=0.5)
        p2.line([0, 1], [0.5, 0.5], line_color=BORDER, line_dash="dashed",
                line_width=1, alpha=0.5)

        cb2 = ColorBar(color_mapper=mapper2, location=(0,0), width=12,
                       title="Year",
                       title_text_color=TEXT_MID,
                       title_text_font_size="10px",
                       major_label_text_color=TEXT_MID,
                       background_fill_color=PANEL)
        p2.add_layout(cb2, "right")

        # ── Plot 3: BPM trend over time ───────────────────────────────────────
        if "tempo" in df.columns and df["tempo"].notna().any():
            tempo_agg = year_agg("tempo")
            tempo_agg = tempo_agg[tempo_agg["count"] >= 2]

            p3 = figure(
                title="Average BPM Over Time",
                width=480, height=250,
                tools="pan,wheel_zoom,reset,save",
                toolbar_location="above",
                x_range=(year_min - 0.5, year_max + 0.5),
                background_fill_color=BG, border_fill_color=PANEL,
                outline_line_color=BORDER,
            )
            p3.title.text_color     = TITLE_COL
            p3.title.text_font_size = "11px"
            p3.grid.grid_line_color = BORDER
            p3.axis.major_label_text_color = TEXT_MID
            p3.axis.axis_line_color        = BORDER
            p3.yaxis.axis_label            = "BPM"
            p3.yaxis.axis_label_text_color = TEXT_HI
            p3.toolbar.logo = None

            bpm_src = ColumnDataSource(dict(
                x    = tempo_agg["_year"].tolist(),
                y    = tempo_agg["mean"].round(1).tolist(),
                hi   = (tempo_agg["mean"] + tempo_agg["std"].fillna(0)).round(1).tolist(),
                lo   = (tempo_agg["mean"] - tempo_agg["std"].fillna(0)).round(1).tolist(),
                count= tempo_agg["count"].tolist(),
            ))
            band_x = tempo_agg["_year"].tolist() + tempo_agg["_year"].tolist()[::-1]
            band_y = (tempo_agg["mean"] + tempo_agg["std"].fillna(0)).tolist() + \
                     (tempo_agg["mean"] - tempo_agg["std"].fillna(0)).tolist()[::-1]
            p3.patch(band_x, band_y, fill_color=ACCENT_WARN, fill_alpha=0.12,
                     line_color=None)
            p3.line("x", "y", source=bpm_src, line_color=ACCENT_WARN,
                    line_width=2.5, alpha=0.95)
            p3.scatter("x", "y", source=bpm_src, color=ACCENT_WARN,
                       size=7, alpha=0.9, line_color=BG)
            p3.add_tools(HoverTool(tooltips=[
                ("Year",   "@x"),
                ("Avg BPM","@y{0.0}"),
                ("Tracks", "@count"),
            ]))
        else:
            p3 = None

        # ── Plot 4: Track count per year (how active you were adding music) ────
        count_by_year = df.groupby("_year").size().reset_index(name="count")
        p4 = figure(
            title="Tracks Added per Year",
            width=480, height=250,
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
            x_range=(year_min - 0.5, year_max + 0.5),
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p4.title.text_color     = TITLE_COL
        p4.title.text_font_size = "11px"
        p4.grid.grid_line_color = BORDER
        p4.axis.major_label_text_color = TEXT_MID
        p4.axis.axis_line_color        = BORDER
        p4.yaxis.axis_label            = "Tracks"
        p4.yaxis.axis_label_text_color = TEXT_HI
        p4.y_range.start               = 0
        p4.toolbar.logo = None

        p4.vbar(x=count_by_year["_year"].tolist(),
                top=count_by_year["count"].tolist(),
                width=0.7, color=ACCENT_GREEN, line_color=BG, alpha=0.9)
        p4.add_tools(HoverTool(tooltips=[("Year","@x"), ("Tracks","@top")]))

        # ── Plot 5: Feature heatmap — year × feature ──────────────────────────
        feat_cols_hm = [c for c, _, _ in FEATS]
        feat_labels  = [l for _, l, _ in FEATS]
        hm_years     = [str(y) for y in sorted(df["_year"].unique())]

        xs_hm, ys_hm, vs_hm, ts_hm = [], [], [], []
        for yr in sorted(df["_year"].unique()):
            sub = df[df["_year"] == yr]
            for col, label in zip(feat_cols_hm, feat_labels):
                mean_v = float(sub[col].mean()) if col in sub.columns else 0.0
                xs_hm.append(str(yr))
                ys_hm.append(label)
                vs_hm.append(round(mean_v, 3))
                ts_hm.append(f"{mean_v:.2f}")

        hm_src    = ColumnDataSource(dict(x=xs_hm, y=ys_hm, vals=vs_hm, texts=ts_hm))
        hm_mapper = LinearColorMapper(palette="Viridis256", low=0, high=1)

        hm_width  = max(500, min(1000, len(hm_years) * 28 + 160))
        p5 = figure(
            title="Feature Heatmap by Year  (mean intensity)",
            x_range=hm_years,
            y_range=list(reversed(feat_labels)),
            width=hm_width, height=280,
            tools="hover,save",
            toolbar_location="above",
            x_axis_location="above",
            background_fill_color=BG, border_fill_color=PANEL,
            outline_line_color=BORDER,
        )
        p5.title.text_color            = TITLE_COL
        p5.title.text_font_size        = "11px"
        p5.grid.grid_line_color        = None
        p5.axis.major_label_text_color = TEXT_HI
        p5.axis.axis_line_color        = BORDER
        p5.xaxis.major_label_orientation = 0.9
        p5.xaxis.major_label_text_font_size = "9px"
        p5.toolbar.logo = None

        p5.rect(x="x", y="y", width=1, height=1, source=hm_src,
                fill_color={"field":"vals","transform":hm_mapper},
                line_color=BG, line_width=1)

        # Only show text if not too many years
        if len(hm_years) <= 20:
            p5.text(x="x", y="y", text="texts", source=hm_src,
                    text_align="center", text_baseline="middle",
                    text_font_size="8px", text_color="#111111")

        cb5 = ColorBar(color_mapper=hm_mapper, location=(0,0), width=12,
                       major_label_text_color=TEXT_MID,
                       background_fill_color=PANEL)
        p5.add_layout(cb5, "right")
        p5.add_tools(HoverTool(tooltips=[
            ("Year",    "@x"),
            ("Feature", "@y"),
            ("Mean",    "@vals{0.000}"),
        ]))

        # ── UMAP HTML export button ───────────────────────────────────────────
        # Build a standalone UMAP scatter for export
        export_btn = Button(
            label="💾  Download UMAP as HTML",
            button_type="success", width=240,
        )
        export_status = Div(
            text=f"<span style='color:{TEXT_LO};font-size:11px'>"
                 "Saves a self-contained interactive HTML file to your Desktop.</span>",
            styles={"margin": "4px 0 10px"},
        )

        def do_export_umap():
            try:
                import os
                from pathlib import Path
                from bokeh.plotting import figure as bk_fig
                from bokeh.embed import file_html
                from bokeh.resources import CDN
                from bokeh.models import HoverTool as HT, ColumnDataSource as CDS

                ex_src = CDS(dict(
                    x    = self.embedding[:, 0].tolist(),
                    y    = self.embedding[:, 1].tolist(),
                    name = self.df["name"].tolist(),
                    artist = self.df["artist"].tolist(),
                    cluster = self.df["cluster"].astype(str).tolist(),
                    energy  = self.df["energy"].fillna(0).round(2).tolist()
                              if "energy" in self.df.columns else [0]*len(self.df),
                    tempo   = self.df["tempo"].fillna(0).round(1).tolist()
                              if "tempo" in self.df.columns else [0]*len(self.df),
                    valence = self.df["valence"].fillna(0).round(2).tolist()
                              if "valence" in self.df.columns else [0]*len(self.df),
                    color   = self._full_data["color"],
                ))
                ep = bk_fig(
                    title=f"UMAP — {self.playlist_info.get('name','Playlist')}",
                    width=1100, height=750,
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    toolbar_location="above",
                    output_backend="webgl",
                    background_fill_color=BG, border_fill_color=PANEL,
                    outline_line_color=BORDER,
                )
                ep.title.text_color     = TITLE_COL
                ep.title.text_font_size = "14px"
                ep.grid.grid_line_color = BORDER
                ep.axis.major_label_text_color = TEXT_MID
                ep.axis.axis_line_color        = BORDER
                ep.toolbar.logo = None

                ep.scatter("x","y", source=ex_src, color="color",
                           alpha=0.85, size=8, line_color=None,
                           nonselection_alpha=0.15)
                ep.add_tools(HT(tooltips=[
                    ("", "<b style='color:#79c0ff'>@name</b>"),
                    ("Artist",  "@artist"),
                    ("Cluster", "@cluster"),
                    ("Energy",  "@energy{0.00}"),
                    ("BPM",     "@tempo{0.0}"),
                    ("Valence", "@valence{0.00}"),
                ]))

                html = file_html(ep, CDN,
                                 title=f"UMAP — {self.playlist_info.get('name','Playlist')}")

                desktop = Path.home() / "Desktop"
                pl_name = self.playlist_info.get("name", "playlist")
                safe    = "".join(c for c in pl_name if c.isalnum() or c in " _-")[:40].strip()
                out     = desktop / f"UMAP_{safe}.html"
                out.write_text(html, encoding="utf-8")
                export_status.text = (
                    f"<span style='color:{ACCENT_GREEN}'>✓ Saved to: {out}</span>"
                )
            except Exception as ex:
                export_status.text = (
                    f"<span style='color:{ACCENT_RED}'>Export failed: {ex}</span>"
                )

        export_btn.on_click(lambda: do_export_umap())

        # ── Key stats summary ─────────────────────────────────────────────────
        timeline_axis = "added" if has_added else "release year"
        most_active_year = int(count_by_year.loc[count_by_year["count"].idxmax(), "_year"])
        most_active_n    = int(count_by_year["count"].max())

        recent = df[df["_year"] >= year_max - 2]
        older  = df[df["_year"] <= year_min + 2]

        def _feat_diff(col):
            if col not in df.columns or not df[col].notna().any():
                return "N/A"
            r = recent[col].mean(); o = older[col].mean()
            if pd.isna(r) or pd.isna(o): return "N/A"
            diff = r - o
            arrow = "↑" if diff > 0.02 else ("↓" if diff < -0.02 else "→")
            return f"{arrow} {abs(diff):.2f}"

        summary_html = (
            f"<b style='font-size:13px;color:{ACCENT}'>Timeline Insights</b><br><br>"
            f"Temporal axis: <b style='color:{TEXT_HI}'>{timeline_axis}</b>"
            f" &nbsp;·&nbsp; Range: <b style='color:{TEXT_HI}'>{year_min} – {year_max}</b>"
            f" &nbsp;·&nbsp; Most active: <b style='color:{ACCENT_GREEN}'>{most_active_year}</b>"
            f" ({most_active_n} tracks)<br>"
            f"Energy trend (recent vs early): <b style='color:{TEXT_HI}'>{_feat_diff('energy')}</b>"
            f" &nbsp;·&nbsp; Valence: <b style='color:{TEXT_HI}'>{_feat_diff('valence')}</b>"
            f" &nbsp;·&nbsp; Danceability: <b style='color:{TEXT_HI}'>{_feat_diff('danceability')}</b>"
        )

        # ── Assemble ──────────────────────────────────────────────────────────
        row2 = row(p2, column(p4, p3 if p3 else Div(text="")))

        return column(
            _card(summary_html),
            _div_title("Feature Trends", ACCENT),
            _div_sub("Shaded band = ±1 standard deviation. "
                     f"X-axis = {timeline_axis}."),
            p1,
            _div_title("Mood Map  ·  Feature Heatmap"),
            row2,
            _div_sub("Heatmap shows mean feature intensity per year — darker = higher."),
            p5,
            Div(text=f"<hr style='border-color:{BORDER};margin:18px 0'>"),
            _div_title("Export UMAP", ACCENT_GREEN),
            _div_sub("Downloads a fully interactive standalone HTML file — "
                     "zoom, pan, hover. No server needed to open it."),
            row(export_btn, export_status),
        )

    def get_layout(self):
        return self.layout
