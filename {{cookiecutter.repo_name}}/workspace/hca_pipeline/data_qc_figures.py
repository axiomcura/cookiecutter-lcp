"""Figures for the NB00 data-QC notebook, written to the ``publication-figures`` rules.

One function per figure: data in, figure out, no globals. Each function builds its figure with
:func:`hca_pipeline.pubfig.new_figure` and ends with :func:`hca_pipeline.pubfig.explain_text`.
The notebook calls them through ``pubfig.export``, which draws the headline and how-to-read
line in the notebook and leaves them out of the exported PDF/PNG (the caption does that job).

Headlines are built from the data passed in, so the numbers in them cannot go stale.
"""
from __future__ import annotations

import textwrap
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from . import pubfig as pf

TILE_PX_FALLBACK_UM = 10.0
CLIP_FRACTION = 0.99  # a pixel at >= 99 % of the detector ceiling counts as clipped
CLIP_COLOR = pf.CHANNEL_COLORS["AO_Red"]  # magenta: stays visible to red/green colour-blind readers
LIMIT_COLOR = pf.CATEGORICAL[1]  # vermillion: QC limits in every figure


def readable(name: str) -> str:
    """``under_segmented_nuclei`` -> ``under segmented nuclei`` (axis and legend text)."""
    return name.replace("_", " ")


def _nice_length_um(width_um: float, fraction: float = 0.2) -> float:
    """Scale-bar length (1, 2 or 5 x 10^k µm) at most ``fraction`` of the image width."""
    target = max(width_um * fraction, 1e-9)
    magnitude = 10 ** np.floor(np.log10(target))
    return float(max(m * magnitude for m in (1, 2, 5) if m * magnitude <= target))


def shared_limits(images: Sequence[np.ndarray], pct: tuple[float, float] = (0.5, 99.8)) -> tuple[float, float]:
    """One brightness window for several images (percentiles of all their pixels pooled)."""
    pooled = np.concatenate([np.asarray(im, float).ravel() for im in images if im is not None])
    low, high = np.percentile(pooled, pct)
    return float(low), float(high if high > low else low + 1.0)


def _clip_mask(image: np.ndarray) -> np.ndarray:
    """Pixels at the detector ceiling: 8-bit, 16-bit or 0-1 float images."""
    ceiling = 65535 if image.max() > 255 else (255 if image.max() > 1 else 1.0)
    return image >= CLIP_FRACTION * ceiling


def _tile(ax, image, title, limits, pixel_size_um=None, mark_clipped=False, centre_marker=False):
    """One image panel on a shared brightness scale; ``None`` draws a 'not found' note."""
    if image is None:
        ax.set_axis_off()
        ax.set_title(title, fontsize=pf._STATE["base"], loc="left")
        ax.text(0.5, 0.5, "image not found", ha="center", va="center", transform=ax.transAxes,
                fontsize=pf._STATE["base"] - 0.5, color=pf.GREY)
        return
    image = np.asarray(image, float)
    scalebar = _nice_length_um(image.shape[1] * pixel_size_um) if pixel_size_um else None
    pf.show_image(ax, image, vmin=limits[0], vmax=limits[1], title=title,
                  pixel_size_um=pixel_size_um, scalebar_um=scalebar)
    if mark_clipped and (mask := _clip_mask(image)).any():
        ax.imshow(np.ma.masked_where(~mask, np.ones(image.shape)), cmap=ListedColormap([CLIP_COLOR]),
                  vmin=0, vmax=1, interpolation="nearest")
    if centre_marker:
        ax.plot(image.shape[1] / 2, image.shape[0] / 2, "+", color=CLIP_COLOR, ms=5, mew=0.8)


# --------------------------------------------------------------------------- image QC
def fig_image_qc_distributions(site_flags: pd.DataFrame, channels: Sequence[str],
                               blur_z_limit: float, saturation_pct_limit: float):
    """Blur and clipped-pixel distributions per channel, with the QC limits.

    ``site_flags`` is the output of :func:`hca_pipeline.data_qc.flag_sites`.
    """
    n_sites = len(site_flags)
    n_fail = int(site_flags["Metadata_image_qc"].eq("fail").sum())
    n_blur = int(site_flags[[f"blurred_{c}" for c in channels]].any(axis=1).sum())
    n_sat = int(site_flags[[f"saturated_{c}" for c in channels]].any(axis=1).sum())

    n = len(channels)
    fig, axes = pf.new_figure("double" if n >= 3 else "one_half", height=72, nrows=2, ncols=n, squeeze=False)
    for col, channel in enumerate(channels):
        top, bottom = axes[0, col], axes[1, col]
        top.hist(site_flags[f"blur_z_{channel}"].dropna(), bins=40, color=pf.GREY)
        top.axvline(blur_z_limit, color=LIMIT_COLOR, ls="--")
        top.set_title(channel)
        top.set_xlabel("Sharpness vs plate median\n(robust z-score; lower = blurrier)")
        top.set_ylabel("Sites")
        bottom.hist(site_flags[f"pct_saturated_{channel}"].dropna(), bins=40, color=pf.GREY)
        bottom.axvline(saturation_pct_limit, color=LIMIT_COLOR, ls="--")
        bottom.set_xlim(left=0)  # a percentage cannot be negative
        bottom.set_yscale("log")
        pf.plain_log_axis(bottom, "y")
        bottom.set_xlabel("Pixels at detector ceiling\n(% of image)")
        bottom.set_ylabel("Sites (log scale)")
    pf.explain_text(
        fig,
        f"{n_fail} of {n_sites} sites fail image QC: {n_blur} blurred, {n_sat} saturated.",
        "One panel per channel; each bar counts sites (fields of view). Top: sharpness relative to "
        "the other sites on the same plate. Bottom: share of pixels at the detector ceiling. "
        "Dashed line = QC limit; sites beyond it fail.",
    )
    return fig


def fig_image_qc_plate_maps(site_flags: pd.DataFrame, plate_format: int = 96):
    """One plate map per plate: number of failed sites in each well (grey = well not imaged)."""
    n_rows, n_cols = (8, 12) if plate_format == 96 else (16, 24)
    flagged = site_flags.assign(failed_sites=site_flags["Metadata_image_qc"].eq("fail"))
    per_well = flagged.groupby(["Metadata_Plate", "Metadata_Well"], as_index=False)["failed_sites"].sum()
    plates = sorted(per_well["Metadata_Plate"].unique())
    vmax = max(1, int(per_well["failed_sites"].max()))

    ncols = min(len(plates), 3 if plate_format == 96 else 2)
    nrows = int(np.ceil(len(plates) / ncols))
    size = "one_half" if ncols == 1 else "double"
    panel_h = (pf.width_mm(size) / ncols) * (n_rows / n_cols) + 14
    fig, axes = pf.new_figure(size, height=nrows * panel_h, nrows=nrows, ncols=ncols, squeeze=False)
    for ax, plate in zip(axes.ravel(), plates):
        image = pf.plate_heatmap(ax, per_well[per_well["Metadata_Plate"] == plate], "Metadata_Well",
                                 "failed_sites", rows=n_rows, cols=n_cols, vmin=0, vmax=vmax,
                                 label="Failed sites per well")
        if vmax <= 8:  # counts are whole numbers
            image.colorbar.set_ticks(np.arange(vmax + 1))
        ax.set_title(str(plate))
    for ax in axes.ravel()[len(plates):]:
        ax.set_axis_off()
    n_fail = int(flagged["failed_sites"].sum())
    wells_failing = int(per_well["failed_sites"].gt(0).sum())
    pf.explain_text(
        fig,
        f"{n_fail} of {len(site_flags)} sites fail image QC, in {wells_failing} of {len(per_well)} wells.",
        "Each square is one well; colour = how many of its sites fail image QC. Grey = well not imaged.",
    )
    return fig


def fig_image_montage(channel: str, failed: Sequence[tuple[str, np.ndarray | None]],
                      passed: Sequence[tuple[str, np.ndarray | None]], limits: tuple[float, float],
                      pixel_size_um: float | None = None):
    """Failing sites (top row) next to passing sites of the same plate (bottom row), one brightness scale.

    ``failed`` and ``passed`` are lists of ``(title, image)``; ``image`` may be ``None`` when the
    file was not found. Pixels at the detector ceiling are drawn magenta.
    """
    if not failed:
        raise ValueError("fig_image_montage needs at least one failing site.")
    # At least 3 columns, so a single flagged site does not become one huge tile.
    n = max(len(failed), len(passed), 3)
    size = "double" if n >= 5 else "one_half"
    fig, axes = pf.new_figure(size, height=2 * (pf.width_mm(size) / n + 10), nrows=2, ncols=n, squeeze=False)
    for row, tiles in enumerate((failed, passed)):
        for col in range(n):
            if col < len(tiles):
                title, image = tiles[col]
                _tile(axes[row, col], image, title, limits, pixel_size_um, mark_clipped=True)
            else:
                axes[row, col].set_axis_off()
    guide = (f"All panels use one brightness scale (white = {limits[1]:.3g}). Magenta pixels are at the "
             "detector ceiling. Sites were picked at random.")
    if pixel_size_um:
        guide += " Scale bar in µm."
    pf.explain_text(
        fig,
        f"{channel}: sites that fail image QC (top row) next to passing sites from the same plate (bottom row).",
        guide,
    )
    return fig


# --------------------------------------------------------------------------- profile QC
def fig_profile_qc_rates(rates: pd.DataFrame, fail_pct: float, active: Sequence[str]):
    """Share of cells flagged by each coSMicQC rule, per plate (``rates``: plates x rules, in %)."""
    colors = pf.color_map_for(list(rates.columns))
    n_rules, n_plates = rates.shape[1], len(rates)
    bar_w = 0.8 / n_rules
    fig, ax = pf.new_figure("one_half", height=66)
    for i, rule in enumerate(rates.columns):
        used = rule in active
        bars = ax.bar(np.arange(n_plates) + (i - (n_rules - 1) / 2) * bar_w, rates[rule], bar_w,
               color=colors[rule], alpha=1.0 if used else 0.4,
               label=readable(rule) + ("" if used else " (not applied)"))
        ax.bar_label(bars, fmt="%.1f", fontsize=pf._STATE["base"] - 1, padding=1.5)
    ax.set_ylim(top=1.12 * float(rates.to_numpy().max()))  # room for the value labels
    ax.set_xticks(np.arange(n_plates))
    ax.set_xticklabels([str(p) for p in rates.index])
    ax.set_xlabel("Plate")
    ax.set_ylabel("Cells flagged (%)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=min(n_rules, 2))
    top = rates[list(active)].mean().idxmax() if len(active) else None
    pf.explain_text(
        fig,
        f"{fail_pct:.1f} % of cells fail profile QC" + (f", most often by rule “{readable(top)}”." if top else "."),
        "Each bar is the share of a plate's cells flagged by one coSMicQC rule; a cell can trip several rules. "
        "Faded bars are rules that are switched off and remove no cells.",
    )
    return fig


def fig_feature_z(z_by_feature: Mapping[str, tuple[np.ndarray, float]]):
    """Per-plate z-score of each QC feature with the coSMicQC limit (``{feature: (z, limit)}``)."""
    shares = []
    for z, limit in z_by_feature.values():
        z = np.asarray(z, float)
        z = z[np.isfinite(z)]
        shares.append(100 * (np.mean(z > limit) if limit > 0 else np.mean(z < limit)))
    n = len(z_by_feature)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = pf.new_figure("double", height=nrows * 52, nrows=nrows, ncols=ncols, squeeze=False)
    for ax, (feature, (z, limit)) in zip(axes.ravel(), z_by_feature.items()):
        z = np.asarray(z, float)
        ax.hist(np.clip(z[np.isfinite(z)], -8, 8), bins=60, color=pf.GREY)
        ax.axvline(limit, color=LIMIT_COLOR, ls="--")
        ax.set_yscale("log")
        pf.plain_log_axis(ax, "y")
        ax.set_title("\n".join(textwrap.wrap(pf.pretty_feature(feature), 24)))
        ax.set_xlabel("z-score within plate\n(capped at ±8)")
        ax.set_ylabel("Cells (log scale)")
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    pf.explain_text(
        fig,
        f"Between {min(shares):.1f}% and {max(shares):.1f}% of cells lie beyond a single feature's limit.",
        "One panel per CellProfiler feature, as a z-score within its plate (0 = plate average, 1 = one "
        "standard deviation). Dashed line = coSMicQC limit. A rule flags a cell only if it passes every "
        "limit of that rule, so some rules need two features.",
    )
    return fig


def fig_cell_crops(blocks: Sequence[tuple[str, Sequence[np.ndarray | None]]], n_cols: int,
                   pixel_size_um: float | None = None):
    """Rows of nucleus-centred crops: ``blocks`` is ``[(row label, [crop or None, ...]), ...]``.

    Crops are already scaled to 0-1 per field of view.
    """
    n_rows = len(blocks)
    size = "double" if n_cols >= 6 else "one_half"
    fig, axes = pf.new_figure(size, height=n_rows * (pf.width_mm(size) / n_cols + 6),
                              nrows=n_rows, ncols=n_cols, squeeze=False)
    shown = 0
    for r, (label, crops) in enumerate(blocks):
        for c in range(n_cols):
            ax = axes[r, c]
            crop = crops[c] if c < len(crops) else None
            if crop is None:
                ax.set_axis_off()
            else:
                shown += 1
                _tile(ax, crop, "", (0.0, 1.0), pixel_size_um if c == 0 else None, centre_marker=True)
            if c == 0:
                ax.set_title(label, fontsize=pf._STATE["base"], loc="left")
    side = next((c.shape[0] for _, crops in blocks for c in crops if c is not None), None)
    pf.explain_text(
        fig,
        f"Cells that fail each rule (FAIL rows) next to random passing cells (pass rows); {shown} crops shown.",
        (f"Each tile is a {side} × {side}-pixel crop centred on one nucleus (+). " if side else "")
        + "Brightness is scaled per field of view, not per cell. Cells were picked at random.",
    )
    return fig


# --------------------------------------------------------------------------- DNA content
def fig_dna_content(dna_peaks: pd.DataFrame, values_by_plate: Mapping[str, np.ndarray], dna_channel: str):
    """Nuclear DNA-stain histogram per plate with the G1 peak, 2 x G1 and the G2/M window."""
    peaks = dna_peaks.reset_index(drop=True)
    n = len(peaks)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = pf.new_figure("double", height=nrows * 55, nrows=nrows, ncols=ncols, squeeze=False)
    for i, (ax, (_, peak)) in enumerate(zip(axes.ravel(), peaks.iterrows())):
        x = np.asarray(values_by_plate[str(peak["Metadata_Plate"])], float)
        x = x[np.isfinite(x) & (x > 0)]
        ax.hist(x, bins=np.geomspace(x.min(), x.max(), 100), color=pf.GREY)
        ax.set_xscale("log", base=2)
        pf.plain_log_axis(ax, "x")
        if np.isfinite(peak["g1_peak"]):
            g1 = float(peak["g1_peak"])
            ax.axvline(g1, color=pf.CATEGORICAL[0], label="G1 peak")
            ax.axvline(2 * g1, color=pf.CATEGORICAL[0], ls=":", label="2 × G1")
            ax.axvspan(1.6 * g1, 2.5 * g1, color=pf.CATEGORICAL[3], alpha=0.15, label="G2/M search window")
        if np.isfinite(peak["g2_peak"]):
            ax.axvline(float(peak["g2_peak"]), color=pf.CATEGORICAL[3], label=f"G2/M peak ({peak['g2_g1_ratio']:.2f}×)")
        ax.set_title(str(peak["Metadata_Plate"]))
        ax.set_xlabel(f"Total {dna_channel} intensity per nucleus\n(log scale)")
        ax.set_ylabel("Nuclei")
        if i == 0:
            ax.legend(loc="upper right")
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    k = int(peaks["has_g2_peak"].sum())
    ratios = peaks.loc[peaks["has_g2_peak"], "g2_g1_ratio"]
    pf.explain_text(
        fig,
        f"{k} of {n} {'plate shows' if n == 1 else 'plates show'} a G2/M peak at 1.6–2.5× the G1 intensity"
        + (f" (median {ratios.median():.2f}×)." if k else "."),
        "Each bar counts nuclei at one stain intensity. Blue line = G1 peak (cells that have not yet copied "
        "their DNA); dotted = twice that, where G2/M cells (DNA copied, not yet divided) should sit; shaded = "
        "search window."
        + (" Acridine orange green also stains RNA, so the stain only approximates DNA content." if dna_channel == "AOGFP" else ""),
    )
    return fig
