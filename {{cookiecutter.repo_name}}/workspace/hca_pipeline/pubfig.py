"""pubfig - publication-quality, reader-friendly matplotlib figures for notebooks.

One small, dependency-light module (matplotlib + numpy; pandas and scipy for a few helpers).
Copy it next to your notebook (or into your package) and ``import pubfig as pf``.

What it gives you
-----------------
* ``use_style``      one call sets fonts, sizes, line widths, colours and export settings.
* ``new_figure``     a figure sized in millimetres for a journal column (single / 1.5 / double).
* ``explain_text``   a plain-language headline + "how to read this" box ABOVE the plot.
                     It exists in the notebook ("explain" mode) and is left out of the
                     publication file ("publish" mode), where the caption does this job.
* ``export``         builds a figure twice (explained PNG for the notebook, clean
                     PDF/SVG/PNG for the paper) from ONE function, so they never drift apart.
* helpers            ``strip_box``, ``superplot``, ``plate_heatmap``, ``show_image``,
                     ``composite``, ``add_scalebar``, ``label_line_ends``, ``panel_labels``,
                     ``plain_log_axis``, ``color_map_for``,
                     ``pretty_feature`` (CellProfiler column name -> plain words).
* ``audit_figure``   checks a finished figure against the skill's rules (fonts too small,
                     missing axis labels, rainbow colour maps, dual y-axes, bars that do not
                     start at zero, clipped text, ...). Use it in notebooks and in tests.

Design rules are explained in ../SKILL.md. Run ``python pubfig.py`` for a self-test.
"""
from __future__ import annotations

import contextlib
import re
import textwrap
from collections import namedtuple
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_rgb

MM = 1 / 25.4  # inches per millimetre
PT_MM = 25.4 / 72  # millimetres per point

# --------------------------------------------------------------------------- journals
# Widths in mm. Sources and dates: references/journals.md. ALWAYS re-check the target
# journal's current author guidelines before submitting.
JOURNALS = {
    "generic": dict(single=85, one_half=120, double=175, max_height=230, fonts=(6, 8), panel="lower"),
    "nature": dict(single=89, one_half=120, double=183, max_height=170, fonts=(5, 7), panel="lower"),
    "cell": dict(single=85, one_half=114, double=174, max_height=230, fonts=(6, 8), panel="upper"),
    # PLOS gives a pixel range (789-2250 px at 300 dpi = 67-190.5 mm); "single" is our choice.
    "plos": dict(single=95, one_half=140, double=190.5, max_height=222, fonts=(8, 12), panel="upper"),
}

# --------------------------------------------------------------------------- colours
# Okabe-Ito subset in a validated order (adjacent colours stay distinguishable for the
# common colour-vision deficiencies). Six is the maximum for one panel: more -> facet.
# The light colours (orange, pink, sky blue) have low contrast on white, so ALWAYS pair
# colour with direct labels or marker shapes.
CATEGORICAL = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
GREY = "#8C8C8C"  # "everything else" / control context
INK = "#1A1A1A"  # text and key marks
CONTROL_COLOR = "#4D4D4D"  # negative / vehicle control: always dark grey, in every figure
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Additive image colours that stay readable with colour-vision deficiency (magenta/green,
# not red/green). Never merge all five Cell Painting channels in one picture.
CHANNEL_COLORS = {
    "DNA": "#3D8BFF", "ER": "#00D26A", "RNA": "#FFD400", "AGP": "#FF8A00", "Mito": "#FF2DA0",
    "AO_Green": "#00D26A", "AO_Red": "#FF2DA0", "BF": "#FFFFFF", "Phase": "#FFFFFF",
}

_STATE = {"journal": "generic", "explain": False, "base": 7.0}


# --------------------------------------------------------------------------- style
def _available_font(candidates):
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


def use_style(journal: str = "generic", base_size: float | None = None, explain: bool = True,
              notebook_dpi: int = 200) -> dict:
    """Set matplotlib defaults for publication figures. Call once in the first notebook cell.

    journal      key of JOURNALS (sets column widths, font range, panel-label case).
    base_size    body font size in pt AT PRINT SIZE (default: middle of the journal's range).
    explain      True = notebook mode (headline + guide drawn); False = publication mode.
    notebook_dpi on-screen resolution. A 85 mm figure at 200 dpi is ~670 px wide: readable
                 in a notebook without zooming, while fonts stay at true print size.
    """
    spec = JOURNALS[journal]
    lo, hi = spec["fonts"]
    base = float(base_size or min(max(7.0, lo), hi))
    font = _available_font(["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"])
    _STATE.update(journal=journal, explain=explain, base=base)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font, "DejaVu Sans"],
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base + 1,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.labelsize": base - 0.5, "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5, "legend.frameon": False,
        "figure.titlesize": base + 1, "figure.titleweight": "bold",
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.visible": False,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "patch.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": "#E3E3E3", "grid.linewidth": 0.5,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "image.cmap": "viridis", "image.interpolation": "nearest",
        "figure.dpi": notebook_dpi, "savefig.dpi": 300, "savefig.facecolor": "white",
        "figure.facecolor": "white",
        # Editable text in vector files; reproducible bytes.
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "pubfig",
        "mathtext.default": "regular", "text.usetex": False,
    })
    return {"journal": journal, "font": font, "base_size": base, **spec}


@contextlib.contextmanager
def explain_mode(on: bool):
    """Temporarily switch between notebook (explained) and publication figures."""
    old = _STATE["explain"]
    _STATE["explain"] = on
    try:
        yield
    finally:
        _STATE["explain"] = old


def width_mm(width="single", journal: str | None = None) -> float:
    """'single' | 'one_half' | 'double' | a number in mm."""
    if isinstance(width, (int, float)):
        return float(width)
    return float(JOURNALS[journal or _STATE["journal"]][width])


def new_figure(width="single", height: float | None = None, nrows: int = 1, ncols: int = 1,
               aspect: float = 0.72, **subplot_kw):
    """Figure sized in mm. ``height`` is the PLOT height in mm (the explain header is extra).

    Returns (fig, ax) for one axes, (fig, array_of_axes) otherwise.
    Uses constrained layout; never save with bbox_inches='tight' (it changes the size you set).
    """
    w = width_mm(width)
    h = float(height) if height else w * aspect
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * MM, h * MM), layout="constrained", **subplot_kw)
    fig._pf_explain = _STATE["explain"]
    fig._pf_plot_h_mm = h
    fig._pf_width_mm = w
    return fig, axes


def explain_text(fig, claim: str, guide: str | None = None) -> None:
    """Add a headline (the finding, as a sentence) and a how-to-read guide above the plot.

    Only drawn in explain mode (notebooks). In publish mode this does nothing: the journal
    wants the claim in the caption, and the figure keeps its exact column size.
    """
    if not getattr(fig, "_pf_explain", False):
        return
    base = _STATE["base"]
    w = fig._pf_width_mm
    claim_lines = textwrap.wrap(claim, max(18, int(0.93 * w / (0.56 * (base + 1) * PT_MM))))
    guide_lines = textwrap.wrap(guide, max(20, int(0.93 * w / (0.50 * base * PT_MM)))) if guide else []
    lh_claim, lh_guide = 1.3 * (base + 1) * PT_MM, 1.3 * base * PT_MM
    header = 1.5 + len(claim_lines) * lh_claim + len(guide_lines) * lh_guide + 1.5
    total = fig._pf_plot_h_mm + header
    fig.set_size_inches(w * MM, total * MM)
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - header / total))
    y = 1 - 1.5 / total
    fig.text(0.01, y, "\n".join(claim_lines), ha="left", va="top", fontsize=base + 1,
             fontweight="bold", color=INK, gid="explain")
    y -= len(claim_lines) * lh_claim / total
    if guide_lines:
        fig.text(0.01, y, "\n".join(guide_lines), ha="left", va="top", fontsize=base,
                 color="#4A4A4A", gid="explain")


def panel_labels(axes, style: str | None = None, size: float | None = None, dx: float = -22, dy: float = 4):
    """Bold panel letters (a, b, c ... or A, B, C ...) at the top-left of each axes."""
    style = style or JOURNALS[_STATE["journal"]]["panel"]
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        letter = chr(ord("A" if style == "upper" else "a") + i)
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx * 0.5, dy),
                    textcoords="offset points", fontsize=size or _STATE["base"] + 2,
                    fontweight="bold", ha="left", va="bottom", annotation_clip=False)


# --------------------------------------------------------------------------- colour helpers
def color_map_for(levels, control: str | None = None, palette=None) -> dict:
    """Stable level -> colour map. Build it ONCE per notebook and reuse in every figure, so
    'compound A' is the same colour everywhere. The control is always dark grey."""
    palette = list(palette or CATEGORICAL)
    levels = [l for l in levels if l != control]
    if len(levels) > len(palette):
        raise ValueError(f"{len(levels)} categories > {len(palette)} colours: facet, group or "
                         "highlight a few and grey the rest (see references/palettes.md).")
    cmap = {l: palette[i] for i, l in enumerate(levels)}
    if control is not None:
        cmap[control] = CONTROL_COLOR
    return cmap


def label_line_ends(ax, series, min_gap_pt: float = 11, pad_frac: float = 0.22, size=None):
    """Label lines directly at their right end (no legend lookup needed).

    series: list of (text, x_end, y_end, colour). Spreads labels apart so they never overlap.
    Call after plotting; it widens the x-range to make room and draws a short leader line.
    """
    fig = ax.figure
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + (x1 - x0) * pad_frac)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    items = sorted(series, key=lambda s: s[2])
    disp = [ax.transData.transform((x, y)) for _, x, y, _ in items]
    gap = min_gap_pt * fig.dpi / 72
    bb = ax.get_window_extent()
    y_lo, y_hi = bb.y0 + gap / 2, bb.y1 - gap / 2
    adj = [min(max(d[1], y_lo), y_hi) for d in disp]  # keep labels inside the axes
    for i in range(1, len(adj)):  # push apart going up ...
        adj[i] = max(adj[i], adj[i - 1] + gap)
    over = adj[-1] - y_hi
    if over > 0:  # ... then back down if the top ran out of room
        adj = [a - over for a in adj]
        for i in range(len(adj) - 2, -1, -1):
            adj[i] = min(adj[i], adj[i + 1] - gap)
    for (text, x, y, color), d, ya in zip(items, disp, adj):
        xt, yt = inv.transform((d[0] + 5 * fig.dpi / 72, ya))
        ax.annotate(text, xy=(x, y), xytext=(xt, yt), textcoords="data", ha="left", va="center",
                    fontsize=size or _STATE["base"] - 0.5, color=INK, gid="direct_label",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=color, shrinkA=0, shrinkB=1))


def plain_log_axis(ax, axis: str = "x", minor: bool = False):
    """Log axis with plain numbers (0.01, 0.1, 1, 10) instead of 10^-2. Matplotlib draws the
    exponents of 10^n labels at ~70% size (about 4.5 pt), below most journals' 5 pt minimum,
    and general readers read '0.01' faster than '10^-2'."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    if not minor:
        a.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------------- distributions
def _tick_labels_with_n(ax, order, counts, show_n):
    labels = [f"{o}\n(n={counts[o]:,})" if show_n else str(o) for o in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)


def strip_box(ax, df, x, y, order=None, colors=None, jitter=0.18, size=4, alpha=0.45,
              show_n=True, seed=0, max_points=2000):
    """Every data point (jittered) + median and interquartile range. Shows the data, not just
    a bar. Large groups are randomly thinned to ``max_points`` for drawing only (the median and
    IQR always use all values). Returns a summary DataFrame."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    order = list(order or pd.unique(df[x]))
    colors = colors or color_map_for(order)
    counts, rows = {}, []
    for i, lev in enumerate(order):
        v = df.loc[df[x] == lev, y].dropna().to_numpy()
        counts[lev] = len(v)
        if len(v) == 0:
            continue
        vv = v if len(v) <= max_points else rng.choice(v, max_points, replace=False)
        ax.scatter(i + rng.uniform(-jitter, jitter, len(vv)), vv, s=size, color=colors[lev],
                   alpha=alpha, linewidths=0, rasterized=True, zorder=1)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=INK, lw=1.4, solid_capstyle="butt", zorder=3)
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.6, zorder=4)
        rows.append({x: lev, "n": len(v), "median": med, "q1": q1, "q3": q3})
    _tick_labels_with_n(ax, order, counts, show_n)
    ax.set_xlim(-0.6, len(order) - 0.4)
    return pd.DataFrame(rows)


def superplot(ax, df, x, y, rep, order=None, colors=None, jitter=0.2, size=3, seed=0, show_n=True,
              max_points=1500):
    """Cell-level points (small, faint) + one large marker per biological replicate + the mean of
    replicate medians with a 95% t-interval. Statistics are over REPLICATES (plates/days), never
    over cells: cells in one well are not independent (Lord et al. 2020, 'SuperPlots').
    Returns a summary DataFrame (group, mean, ci_low, ci_high, n_reps)."""
    import pandas as pd
    from scipy import stats

    rng = np.random.default_rng(seed)
    order = list(order or pd.unique(df[x]))
    colors = colors or color_map_for(order)
    reps = list(pd.unique(df[rep]))
    counts, rows = {}, []
    for i, lev in enumerate(order):
        sub = df[df[x] == lev]
        counts[lev] = sub[rep].nunique()
        pts = sub[y].dropna().to_numpy()
        pts = pts if len(pts) <= max_points else rng.choice(pts, max_points, replace=False)
        ax.scatter(i + rng.uniform(-jitter, jitter, len(pts)), pts, s=size, color=colors[lev],
                   alpha=0.22, linewidths=0, rasterized=True, zorder=1)
        med = sub.groupby(rep)[y].median()
        for j, (r, m) in enumerate(med.items()):
            ax.scatter(i + (j - (len(med) - 1) / 2) * 0.12, m, s=26, color=colors[lev],
                       marker=MARKERS[reps.index(r) % len(MARKERS)], edgecolor=INK, linewidth=0.7, zorder=4)
        mean, n = med.mean(), len(med)
        half = stats.t.ppf(0.975, n - 1) * med.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        ax.plot([i - 0.3, i + 0.3], [mean, mean], color=INK, lw=1.6, zorder=5)
        if n > 1:
            ax.plot([i, i], [mean - half, mean + half], color=INK, lw=1.0, zorder=5)
        rows.append({x: lev, "mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n_reps": n})
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{o}\n({counts[o]} reps)" if show_n else str(o) for o in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plates and images
_WELL = re.compile(r"^([A-Za-z]{1,2})0*(\d{1,2})$")


def plate_heatmap(ax, df, well: str, value: str, rows: int = 16, cols: int = 24, cmap="viridis",
                  vmin=None, vmax=None, label: str | None = None, colorbar: bool = True):
    """Plate layout heatmap (default 384-well: 16 x 24; use rows=8, cols=12 for 96-well).
    Missing wells are light grey. Returns the image artist."""
    grid = np.full((rows, cols), np.nan)
    for w, v in zip(df[well], df[value]):
        m = _WELL.match(str(w))
        if not m:
            continue
        r = ord(m.group(1).upper()[-1]) - 65 + (26 if len(m.group(1)) == 2 else 0)
        c = int(m.group(2)) - 1
        if r < rows and c < cols:
            grid[r, c] = v
    cm = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(bad="#E6E6E6")
    im = ax.imshow(grid, cmap=cm, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(0, cols, 2)); ax.set_xticklabels(np.arange(1, cols + 1, 2))
    ax.set_yticks(np.arange(rows)); ax.set_yticklabels([chr(65 + i) for i in range(rows)])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Plate column"); ax.set_ylabel("Plate row")
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(label or value); cb.outline.set_visible(False)
    return im


def add_scalebar(ax, image_width_px: int, pixel_size_um: float, length_um: float = 20,
                 color="white", lw: float = 1.8, loc: str = "lower right", label: bool = True):
    """Scale bar in a corner of an image axes. Needs the pixel size from the microscope metadata."""
    px = length_um / pixel_size_um
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xspan, yspan = abs(x1 - x0), abs(y1 - y0)
    pad_x, pad_y = 0.05 * xspan, 0.06 * yspan
    xr = max(x0, x1) - pad_x if "right" in loc else min(x0, x1) + pad_x + px
    xl = xr - px
    ybar = max(y0, y1) - pad_y if "lower" in loc else min(y0, y1) + pad_y
    ax.plot([xl, xr], [ybar, ybar], color=color, lw=lw, solid_capstyle="butt", clip_on=False)
    if label:
        dy = -0.035 * yspan * (1 if y0 > y1 else -1) * -1
        ax.text((xl + xr) / 2, ybar - 0.03 * yspan * (1 if y0 > y1 else -1), f"{length_um:g} µm",
                color=color, ha="center", va="bottom" if y0 > y1 else "top", fontsize=_STATE["base"] - 0.5)


def show_image(ax, img, vmin=None, vmax=None, cmap="gray", pixel_size_um=None, scalebar_um=20,
               title=None, pct=(0.5, 99.8)):
    """Show one image with intensity limits you can reuse. For a fair comparison between panels,
    compute vmin/vmax ONCE (e.g. from the control images) and pass the same values to every call;
    otherwise each panel is stretched differently and brightness cannot be compared."""
    img = np.asarray(img, dtype=float)
    if vmin is None or vmax is None:
        lo, hi = np.percentile(img, pct)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=_STATE["base"], fontweight="bold", loc="left")
    if pixel_size_um:
        add_scalebar(ax, img.shape[1], pixel_size_um, scalebar_um)
    return vmin, vmax


def composite(channels: dict, colors: dict | None = None, limits: dict | None = None, pct=(1, 99.8)):
    """Additive colour overlay of 2D arrays -> RGB array in [0, 1]. Keep to 2-3 channels."""
    colors = {**CHANNEL_COLORS, **(colors or {})}
    limits = limits or {}
    out = None
    for name, im in channels.items():
        im = np.asarray(im, float)
        lo, hi = limits.get(name, tuple(np.percentile(im, pct)))
        norm = np.clip((im - lo) / max(hi - lo, 1e-12), 0, 1)
        layer = norm[..., None] * np.array(to_rgb(colors[name]))
        out = layer if out is None else out + layer
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------------- plain-language names
_COMPARTMENTS = {"Cells": "whole cell", "Nuclei": "nucleus", "Cytoplasm": "cytoplasm", "Image": "per image"}
_UNITS = {"AreaShape_Area": "pixels²", "AreaShape_Perimeter": "pixels", "AreaShape_MajorAxisLength": "pixels",
          "AreaShape_MinorAxisLength": "pixels", "AreaShape_EquivalentDiameter": "pixels"}


def _words(camel: str) -> str:
    """'MeanIntensity' -> 'Mean intensity'; 'StdIntensity' -> 'SD of intensity'; keeps 'DNA'."""
    toks = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", camel).split()
    toks = [t if (t.isupper() and len(t) > 1) else t.lower() for t in toks]
    s = " ".join(toks)
    s = "SD of " + s[4:] if s.startswith("std ") else s
    return s[:1].upper() + s[1:]


def pretty_feature(col: str, with_units: bool = True) -> str:
    """CellProfiler column name -> words a general reader understands.

    'Cells_Intensity_MeanIntensity_DNA'      -> 'Mean intensity, DNA channel (whole cell)'
    'Nuclei_AreaShape_Area'                  -> 'Area (nucleus, pixels²)'
    'Cytoplasm_Texture_Contrast_AGP_3_01_256' -> 'Contrast of AGP texture, scale 3 (cytoplasm)'
    Best effort: unknown patterns fall back to spaced words. CellProfiler reports AreaShape in
    PIXELS unless you convert with the pixel size; say so in the axis label or caption."""
    parts = col.split("_")
    if parts[0] == "Metadata":
        return " ".join(_words(p) for p in parts[1:])
    if parts[0] not in _COMPARTMENTS or len(parts) < 3:
        return " ".join(_words(p) for p in parts)
    comp = _COMPARTMENTS[parts[0]]
    cat, feat, rest = parts[1], parts[2], parts[3:]
    name = _words(feat)
    unit = _UNITS.get(f"{cat}_{feat}") if with_units else None
    if cat == "Texture" and rest:
        name = f"{name} of {rest[0]} texture" + (f", scale {rest[1]}" if len(rest) > 1 else "")
    elif cat == "Correlation" and rest:
        name = f"{name} correlation, {' vs '.join(rest[:2])}"
    elif cat == "Granularity":
        name = f"Granularity {feat}" + (f", {rest[0]} channel" if rest else "")
    elif rest:
        name += f", {rest[0]} channel"
    return f"{name} ({comp}{', ' + unit if unit else ''})"


# --------------------------------------------------------------------------- saving
def save_figure(fig, stem, formats=("pdf", "png"), dpi: int | None = None, overwrite: bool = True) -> list[Path]:
    """Save with reproducible metadata (no timestamps). Never uses bbox_inches='tight'.
    overwrite=False leaves files that already exist untouched (pipelines with an overwrite policy)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    meta = {"pdf": {"CreationDate": None, "Creator": "pubfig"}, "png": {"Software": None},
            "svg": {"Date": None}}
    out = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        if p.exists() and not overwrite:
            out.append(p)
            continue
        fig.savefig(p, format=fmt, dpi=dpi or mpl.rcParams["savefig.dpi"], metadata=meta.get(fmt))
        if fmt == "png":  # matplotlib writes RGBA; several journals reject alpha channels
            try:
                from PIL import Image
                with Image.open(p) as im:
                    rgb, d = im.convert("RGB"), im.info.get("dpi")
                rgb.save(p, dpi=d)
            except ImportError:
                pass
        out.append(p)
    return out


def export(make_fig, stem: str, *args, outdir="figures", formats=("pdf", "svg", "png"), dpi=300,
           audit: bool = True, return_fig: bool = False, overwrite: bool = True, **kwargs):
    """Build ``make_fig`` twice from one function: clean files for the paper, explained PNG for the
    notebook. The explained figure is left open, so the notebook displays it; the function returns
    None unless return_fig=True (a returned Figure would also print its repr in the cell).

    overwrite=False keeps files that already exist (the figure is still built, shown and audited).

    make_fig(*args, **kwargs) must create and return a figure using new_figure() and call
    explain_text(...) for the headline and how-to-read guide."""
    outdir = Path(outdir)
    with explain_mode(False):
        fig = make_fig(*args, **kwargs)
        files = save_figure(fig, outdir / stem, formats, dpi, overwrite)
        issues = audit_figure(fig) if audit else []
        plt.close(fig)
    with explain_mode(True):
        fig = make_fig(*args, **kwargs)
        save_figure(fig, outdir / f"{stem}_explained", ("png",), 200, overwrite)
    if issues:
        print(f"[pubfig] {stem}: {len(issues)} issue(s)")
        for i in issues:
            print(f"  {i.level.upper():5s} {i.where}: {i.msg}")
    else:
        print(f"[pubfig] {stem}: audit clean -> {', '.join(p.name for p in files)}")
    return fig if return_fig else None


# --------------------------------------------------------------------------- audit
Issue = namedtuple("Issue", "level where msg")
_BAD_CMAPS = {"jet", "rainbow", "hsv", "gist_rainbow", "nipy_spectral", "gist_ncar", "gist_stern", "turbo"}


def audit_figure(fig, journal: str | None = None, min_font: float | None = None) -> list[Issue]:
    """Check a finished figure against the rules in SKILL.md. Returns a list of Issue(level, where,
    msg); level is 'error' (fix before submitting) or 'warn' (look at it and decide)."""
    from matplotlib import patches, text as mtext

    spec = JOURNALS[journal or _STATE["journal"]]
    min_font = min_font or spec["fonts"][0]
    issues: list[Issue] = []
    add = lambda lvl, where, msg: issues.append(Issue(lvl, where, msg))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fbox = fig.bbox

    # Matplotlib creates tick labels outside the view limits and on hidden axes; they are never
    # drawn, so they must not be judged.
    skip = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not ax.axison or not (lo - 1e-9 <= tick.get_loc() <= hi + 1e-9):
                    skip.update({id(tick.label1), id(tick.label2)})
        if not ax.axison:
            skip.update(id(t) for t in ax.findobj(mtext.Text))

    # text: size and clipping
    for t in fig.findobj(mtext.Text):
        if id(t) in skip or not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < min_font - 1e-6:
            add("error", repr(t.get_text()[:30]), f"font {t.get_fontsize():.1f} pt < {min_font} pt minimum")
        if re.search(r"\^\{|_\{", t.get_text()):
            add("warn", repr(t.get_text()[:30]), "mathtext super/subscript draws at ~70% size (may fall under the journal minimum): use plain_log_axis() or plain text")
        bb = t.get_window_extent(renderer)
        if bb.width and (bb.x0 < fbox.x0 - 2 or bb.x1 > fbox.x1 + 2 or bb.y0 < fbox.y0 - 2 or bb.y1 > fbox.y1 + 2):
            add("error", repr(t.get_text()[:30]), "text runs outside the figure (clipped)")

    # size
    w, h = fig.get_size_inches() * 25.4
    if not getattr(fig, "_pf_explain", False):
        if h > spec["max_height"] + 0.5:
            add("error", "figure", f"height {h:.0f} mm > {spec['max_height']} mm maximum for this journal")
        cols = [spec["single"], spec["one_half"], spec["double"]]
        if min(abs(w - c) for c in cols) > 1.5:
            add("warn", "figure", f"width {w:.0f} mm is not a standard column ({', '.join(f'{c:g}' for c in cols)} mm)")

    axes = [a for a in fig.axes]
    for i, ax in enumerate(axes):
        where = f"axes[{i}]"
        has_data = bool(ax.lines or ax.collections or ax.patches or ax.images)
        if not has_data or not ax.axison:
            continue
        is_cbar = ax.get_label() == "<colorbar>"
        if not is_cbar:
            if not ax.get_xlabel() and not ax.images and not getattr(ax, "_pf_no_xlabel", False):
                add("warn", where, "x axis has no label (say what it is and its unit)")
            if not ax.get_ylabel() and not ax.images:
                add("warn", where, "y axis has no label (say what it is and its unit)")
        for im in list(ax.images) + [c for c in ax.collections if getattr(c, "get_array", lambda: None)() is not None]:
            try:
                name = im.get_cmap().name.replace("_r", "")
            except Exception:
                continue
            if name in _BAD_CMAPS:
                add("error", where, f"rainbow-type colour map '{name}': use viridis/cividis (sequential) or a diverging map")
        # bars must start at zero
        bars = [p for p in ax.patches if isinstance(p, patches.Rectangle) and p.get_width() > 0 and p.get_height() != 0]
        if bars and ax.get_yscale() == "linear" and min(ax.get_ylim()) > 1e-9:
            add("error", where, "bar chart y-axis does not start at 0 (bar length encodes the value)")
        if any(isinstance(p, patches.Wedge) for p in ax.patches):
            add("warn", where, "pie/donut chart: angles are hard to compare; use sorted bars or dots")
        # legend or direct labels
        labelled = [a for a in ax.lines + ax.collections if a.get_label() and not a.get_label().startswith("_")]
        direct = [c for c in ax.texts if c.get_gid() == "direct_label"]
        if len(labelled) >= 2 and ax.get_legend() is None and len(direct) < len(labelled):
            add("warn", where, "several labelled series but no legend or direct labels")
        if ax.get_legend() is not None and len(ax.get_legend().get_texts()) > 8:
            add("warn", where, "more than 8 legend entries: facet, group, or highlight a few")
    # dual y-axes
    for a in range(len(axes)):
        for b in range(a + 1, len(axes)):
            A, B = axes[a], axes[b]
            if (A.get_position().bounds == B.get_position().bounds
                    and A.get_shared_x_axes().joined(A, B) and not A.get_shared_y_axes().joined(A, B)):
                add("error", f"axes[{a}]/axes[{b}]", "dual y-axis (twinx): split into two panels or index to a common baseline")
    return issues


def assert_figure_ok(fig, **kw) -> None:
    """For tests: raise AssertionError listing every 'error' issue."""
    bad = [i for i in audit_figure(fig, **kw) if i.level == "error"]
    assert not bad, "figure audit failed:\n" + "\n".join(f"  {i.where}: {i.msg}" for i in bad)


if __name__ == "__main__":  # self-test
    import tempfile
    import pandas as pd

    use_style("generic", explain=True)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.repeat(["DMSO", "A", "B"], 200), "v": np.r_[rng.normal(1, .2, 200), rng.normal(1.4, .3, 200), rng.normal(.7, .2, 200)]})
    fig, ax = new_figure("single")
    cm = color_map_for(["DMSO", "A", "B"], control="DMSO")
    strip_box(ax, df, "g", "v", order=["DMSO", "A", "B"], colors=cm)
    ax.set_ylabel("Cell area (relative to DMSO)")
    ax.set_xlabel("Treatment"); ax._pf_no_xlabel = True
    explain_text(fig, "Compound A makes cells larger; B makes them smaller.", "Each dot is one cell. Line = median; bar = middle 50%.")
    print("issues:", audit_figure(fig))
    with tempfile.TemporaryDirectory() as d:
        print(save_figure(fig, Path(d) / "selftest", ("png", "pdf")))
    print(pretty_feature("Cells_Intensity_MeanIntensity_DNA"), "|", pretty_feature("Nuclei_AreaShape_Area"),
          "|", pretty_feature("Cytoplasm_Texture_Contrast_AGP_3_01_256"))
    print("pubfig self-test OK")
