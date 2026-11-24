"""Data-quality checks for NB00: image level, single-cell level, and DNA content.

Three independent checks, each run per plate, each producing flags (never
deleting data):

* **Image QC** - one row per site. A site fails when any channel is blurred
  (robust z of the blur metric below a limit) or saturated (too many pixels
  clipped at the detector ceiling). The blur/saturation metrics come from
  CellProfiler ``MeasureImageQuality`` when present, otherwise from the raw
  pixels (``hca_pipeline.image_qc``). Rules follow the Panc1-hPSC pilot QC.
* **Profile QC** - one row per cell. coSMicQC (Tomkinson, Bunten & Way 2025,
  doi:10.1101/2025.10.14.682427) flags likely segmentation errors with
  z-score thresholds on raw CellProfiler features.
* **DNA content** - one label per cell (sub-G1, G1, G2/M, >G2) relative to the
  G1 peak of the nuclear-stain histogram. Used as a biological sanity check:
  a stoichiometric DNA stain gives a second peak at ~2x the first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SITE_KEYS = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
ROBUST_SD_FACTOR = 1.4826  # MAD -> SD for normally distributed data

# Nuclear stain used for DNA content, in order of preference. Hoechst binds
# DNA stoichiometrically; acridine orange green (AOGFP) also stains RNA
# (nucleoli), so DNA content from AOGFP is a proxy only.
DNA_CHANNEL_PREFERENCE = ("Hoechst", "DNA", "DAPI", "AOGFP")

# CellProfiler MeasureImageQuality columns (image table, "Image_" prefix).
_CP_SLOPE = "Image_ImageQuality_PowerLogLogSlope_{}"
_CP_PERCENT_MAXIMAL = "Image_ImageQuality_PercentMaximal_{}"
_CP_MAX_INTENSITY = "Image_ImageQuality_MaxIntensity_{}"
# MaxIntensity is on CellProfiler's 0-1 scale; >= 0.99 means the brightest
# pixels sit at the detector ceiling, so PercentMaximal measures clipping.
CLIPPED_MAX_INTENSITY = 0.99

# coSMicQC conditions (thresholds are z-scores; > 0 flags high, < 0 flags low;
# all features in one condition must exceed their threshold). Values are those
# of the coSMicQC paper and the Panc1-hPSC pilot QC notebook.
PROFILE_QC_DESCRIPTIONS = {
    "under_segmented_nuclei": "two or more nuclei merged into one mask: off-centre mass AND too much DNA",
    "poorly_segmented_nuclei": "irregular, grooved nuclear mask: low solidity",
    "under_segmented_cells": "cell mask holds extra nuclear signal from a neighbour: too much DNA in the cell",
}

# DNA content groups, as multiples of the G1 peak (Panc1-hPSC pilot QC).
DNA_GROUP_LIMITS = {"sub_g1_max": 0.7, "g1": (0.8, 1.25), "g2": (1.6, 2.5), "above_g2_min": 2.8}


def robust_z(values: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """Return the robust z-score ``(x - median) / (1.4826 * MAD)``, per group.

    Parameters
    ----------
    values : pd.Series
        Numeric values (non-numeric entries become NaN).
    groups : pd.Series, optional
        Group label per value, aligned on the index (e.g. plate). Scored
        together when omitted.

    Returns
    -------
    pd.Series
        Robust z-scores on the same index. NaN where the group MAD is 0.
    """
    values = pd.to_numeric(values, errors="coerce").astype(float)
    groups = pd.Series(0, index=values.index) if groups is None else groups
    median = values.groupby(groups).transform("median")
    mad = (values - median).abs().groupby(groups).transform("median")
    return (values - median) / (ROBUST_SD_FACTOR * mad).replace(0.0, np.nan)


def resolve_dna_channel(
    columns: Iterable[str], preference: Sequence[str] = DNA_CHANNEL_PREFERENCE
) -> str:
    """Return the first preferred channel with a nuclear integrated-intensity column.

    Raises
    ------
    ValueError
        If no ``Nuclei_Intensity_IntegratedIntensity_<channel>`` column exists.
    """
    columns = set(columns)
    for channel in preference:
        if f"Nuclei_Intensity_IntegratedIntensity_{channel}" in columns:
            return channel
    raise ValueError(
        "No nuclear integrated-intensity column found for any of "
        f"{list(preference)} (expected Nuclei_Intensity_IntegratedIntensity_<channel>)."
    )


# ---------------------------------------------------------------------------
# Image (site) level
# ---------------------------------------------------------------------------


def cellprofiler_quality_channels(columns: Iterable[str]) -> list[str]:
    """Return channels that have all three MeasureImageQuality columns we use."""
    columns = set(columns)
    prefix = _CP_SLOPE.format("")
    candidates = sorted(c[len(prefix):] for c in columns if c.startswith(prefix))
    return [
        ch for ch in candidates
        if _CP_PERCENT_MAXIMAL.format(ch) in columns and _CP_MAX_INTENSITY.format(ch) in columns
    ]


def _one_row_per_site(table: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    table = table.drop_duplicates()
    if table.duplicated(list(key_cols)).any():
        raise ValueError(
            f"Different image-quality values share the same {list(key_cols)} key. "
            "If each site was imaged at several time points, add the time column to key_cols."
        )
    return table.sort_values(list(key_cols)).reset_index(drop=True)


def site_quality_from_cellprofiler(
    df: pd.DataFrame, channels: Sequence[str], key_cols: Sequence[str] = SITE_KEYS
) -> pd.DataFrame:
    """Build the per-site blur/saturation table from CellProfiler image columns.

    ``blur_metric_<ch>`` is ``PowerLogLogSlope`` (lower = blurrier).
    ``pct_saturated_<ch>`` is ``PercentMaximal`` when the image maximum is at the
    detector ceiling, else 0 (pixels at a low maximum are not clipped).
    Works on a per-cell table too: rows are reduced to one per site.
    """
    out = df[list(key_cols)].copy()
    for ch in channels:
        clipped = pd.to_numeric(df[_CP_MAX_INTENSITY.format(ch)], errors="coerce") >= CLIPPED_MAX_INTENSITY
        out[f"blur_metric_{ch}"] = pd.to_numeric(df[_CP_SLOPE.format(ch)], errors="coerce")
        out[f"pct_saturated_{ch}"] = pd.to_numeric(
            df[_CP_PERCENT_MAXIMAL.format(ch)], errors="coerce"
        ).where(clipped, 0.0)
    return _one_row_per_site(out, key_cols)


def site_quality_from_pixels(
    image_metrics: pd.DataFrame, channels: Sequence[str], key_cols: Sequence[str] = SITE_KEYS
) -> pd.DataFrame:
    """Build the same per-site table from ``image_qc.scan_images`` output (long format).

    ``blur_metric_<ch>`` is ``log10(focus_score)`` (Laplacian variance; logged
    because it is strongly right-skewed). ``pct_saturated_<ch>`` is
    ``100 * saturated_fraction`` (pixels at the 16-bit ceiling).
    """
    subset = image_metrics[image_metrics["Metadata_Channel"].isin(channels)]
    wide = subset.pivot_table(
        index=list(key_cols), columns="Metadata_Channel",
        values=["focus_score", "saturated_fraction"], aggfunc="median",
    )
    out = wide.index.to_frame(index=False)
    for ch in channels:
        focus = wide[("focus_score", ch)].to_numpy(dtype=float)
        out[f"blur_metric_{ch}"] = np.log10(np.where(focus > 0, focus, np.nan))
        out[f"pct_saturated_{ch}"] = 100.0 * wide[("saturated_fraction", ch)].to_numpy(dtype=float)
    return _one_row_per_site(out, key_cols)


def flag_sites(
    quality: pd.DataFrame,
    channels: Sequence[str],
    *,
    group_cols: Sequence[str] = ("Metadata_Plate",),
    blur_z_limit: float = -5.0,
    saturation_pct_limit: float = 1.0,
) -> pd.DataFrame:
    """Flag blurred and saturated sites.

    A channel is *blurred* when its blur metric has robust z < ``blur_z_limit``
    within its group, and *saturated* when more than ``saturation_pct_limit``
    percent of its pixels are clipped. A site fails if any channel does.

    Returns
    -------
    pd.DataFrame
        ``quality`` plus ``blur_z_<ch>``, ``blurred_<ch>``, ``saturated_<ch>``,
        ``Metadata_image_qc`` (``"pass"``/``"fail"``) and
        ``Metadata_image_qc_reason`` (e.g. ``"blurred (AOGFP); saturated (AOPI)"``).
    """
    out = quality.copy()
    groups = out[list(group_cols)].astype(str).agg("|".join, axis=1)
    for ch in channels:
        out[f"blur_z_{ch}"] = robust_z(out[f"blur_metric_{ch}"], groups)
        out[f"blurred_{ch}"] = out[f"blur_z_{ch}"] < blur_z_limit
        out[f"saturated_{ch}"] = out[f"pct_saturated_{ch}"] > saturation_pct_limit

    def reason(row: pd.Series) -> str:
        parts = []
        for problem in ("blurred", "saturated"):
            hit = [ch for ch in channels if row[f"{problem}_{ch}"]]
            if hit:
                parts.append(f"{problem} ({', '.join(hit)})")
        return "; ".join(parts)

    out["Metadata_image_qc_reason"] = out.apply(reason, axis=1) if len(out) else pd.Series(dtype=str)
    out["Metadata_image_qc"] = np.where(out["Metadata_image_qc_reason"].eq(""), "pass", "fail")
    return out


# ---------------------------------------------------------------------------
# Single-cell (profile) level - coSMicQC
# ---------------------------------------------------------------------------


def profile_qc_conditions(dna_channel: str) -> dict[str, dict[str, float]]:
    """Return the three coSMicQC conditions for this experiment's nuclear stain."""
    return {
        "under_segmented_nuclei": {
            f"Nuclei_Intensity_MassDisplacement_{dna_channel}": 0.05,
            f"Nuclei_Intensity_IntegratedIntensity_{dna_channel}": 1.5,
        },
        "poorly_segmented_nuclei": {"Nuclei_AreaShape_Solidity": -1.6},
        "under_segmented_cells": {f"Cells_Intensity_IntegratedIntensity_{dna_channel}": 0.5},
    }


def label_cell_outliers(
    df: pd.DataFrame,
    conditions: Mapping[str, Mapping[str, float]],
    *,
    group_col: str = "Metadata_Plate",
) -> pd.DataFrame:
    """Run ``cosmicqc.analyze.label_outliers`` per plate.

    coSMicQC z-scores every row it is given, so it is called once per plate
    (plate-wide intensity shifts do not create outliers) and only on rows
    where every QC feature is finite (one NaN would otherwise make the whole
    plate's z-scores NaN and flag nothing).

    Returns
    -------
    pd.DataFrame
        Same index as ``df``. One boolean ``Metadata_cqc_<condition>_is_outlier``
        column per condition, plus ``Metadata_cqc_features_missing``.
    """
    from cosmicqc import analyze

    features = sorted({feature for thresholds in conditions.values() for feature in thresholds})
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"coSMicQC features missing from the single-cell table: {missing}")

    values = df[features].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    columns = [f"Metadata_cqc_{name}_is_outlier" for name in conditions]
    flags = pd.DataFrame(False, index=df.index, columns=columns)

    plates = df[group_col] if group_col in df.columns else pd.Series("all", index=df.index)
    for _, index in values[finite].groupby(plates[finite]).groups.items():
        if len(index) < 2:
            continue
        labeled = analyze.label_outliers(
            df=values.loc[index].reset_index(drop=True),
            feature_thresholds={name: dict(t) for name, t in conditions.items()},
        )
        for column in columns:
            flags.loc[index, column] = labeled[column].to_numpy(dtype=bool)
    flags["Metadata_cqc_features_missing"] = ~finite
    return flags


def profile_qc_status(flags: pd.DataFrame, active_conditions: Sequence[str]) -> pd.DataFrame:
    """Combine per-condition flags into ``Metadata_profile_qc`` and its reason.

    A cell fails when it trips any *active* condition, or when a QC feature is
    missing (it could not be checked).
    """
    reasons = pd.Series("", index=flags.index, dtype=object)
    checks = [(name, flags[f"Metadata_cqc_{name}_is_outlier"]) for name in active_conditions]
    checks.append(("missing QC feature", flags["Metadata_cqc_features_missing"]))
    for name, hit in checks:
        hit = hit.to_numpy(dtype=bool)
        reasons[hit] = np.where(reasons[hit].eq(""), name, reasons[hit] + "; " + name)
    return pd.DataFrame(
        {
            "Metadata_profile_qc": np.where(reasons.eq(""), "pass", "fail"),
            "Metadata_profile_qc_reason": reasons,
        },
        index=flags.index,
    )


# ---------------------------------------------------------------------------
# DNA content (biological QC)
# ---------------------------------------------------------------------------


def _g1_and_g2_peaks(log2_values: np.ndarray, bins: int, smoothing_log2: float) -> tuple[float, float]:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    low, high = np.percentile(log2_values, [0.5, 99.5])
    counts, edges = np.histogram(log2_values, bins=bins, range=(low - 0.5, high + 0.5))
    # Smoothing is set in log2 units, so it does not depend on the data range.
    smooth = gaussian_filter1d(counts.astype(float), smoothing_log2 / (edges[1] - edges[0]))
    centres = (edges[:-1] + edges[1:]) / 2
    g1_index = int(np.argmax(smooth))
    g1 = centres[g1_index]

    # G2/M: tallest clear local maximum at 1.6-2.5x the G1 position.
    low_g2, high_g2 = (g1 + np.log2(limit) for limit in DNA_GROUP_LIMITS["g2"])
    peaks, props = find_peaks(smooth, prominence=0.05 * smooth[g1_index])
    in_window = (centres[peaks] >= low_g2) & (centres[peaks] <= high_g2)
    if not in_window.any():
        return float(2**g1), float("nan")
    best = peaks[in_window][np.argmax(smooth[peaks[in_window]])]
    return float(2**g1), float(2 ** centres[best])


def dna_content_groups(
    dna: pd.Series,
    groups: pd.Series,
    *,
    reference_mask: pd.Series | None = None,
    bins: int = 120,
    smoothing_log2: float = 0.05,
    min_cells: int = 200,
) -> tuple[pd.Series, pd.DataFrame]:
    """Label each cell's DNA content relative to the G1 peak of its group.

    The G1 peak is the tallest peak of the log2 histogram of integrated
    nuclear intensity, smoothed with a Gaussian of ``smoothing_log2`` (0.05
    log2 units = about 3.5 % intensity; narrower than a G1 peak, wide enough
    to steady the peak position with a few hundred cells). With ``reference_mask`` (e.g. negative-control
    cells) only those cells locate the peaks, so a treatment that shifts the
    cell cycle cannot move the G1 anchor.

    Labels (ratio to the G1 peak): ``sub-G1`` < 0.7, ``G1`` 0.8-1.25,
    ``G2/M`` 1.6-2.5, ``>G2`` > 2.8, ``between`` otherwise; ``not assessed``
    for non-positive values or groups with fewer than ``min_cells`` cells.

    Returns
    -------
    labels : pd.Series
        One label per cell, same index as ``dna``.
    peaks : pd.DataFrame
        One row per group: ``group``, ``n_cells``, ``n_reference``,
        ``g1_peak``, ``g2_peak``, ``g2_g1_ratio``, ``has_g2_peak``.
    """
    dna = pd.to_numeric(dna, errors="coerce").astype(float)
    usable = np.isfinite(dna) & (dna > 0)
    reference = usable if reference_mask is None else usable & reference_mask.astype(bool)
    labels = pd.Series("not assessed", index=dna.index, dtype=object)
    rows = []
    for group, index in dna[usable].groupby(groups[usable]).groups.items():
        ref_index = reference[reference].index.intersection(index)
        if len(ref_index) < min_cells:
            ref_index = index  # too few reference cells: use the whole group
        row = {"group": group, "n_cells": len(index), "n_reference": len(ref_index),
               "g1_peak": np.nan, "g2_peak": np.nan}
        if len(index) >= min_cells:
            row["g1_peak"], row["g2_peak"] = _g1_and_g2_peaks(
                np.log2(dna.loc[ref_index].to_numpy()), bins, smoothing_log2
            )
            ratio = dna.loc[index] / row["g1_peak"]
            limits = DNA_GROUP_LIMITS
            labels.loc[index] = np.select(
                [
                    ratio < limits["sub_g1_max"],
                    ratio.between(*limits["g1"]),
                    ratio.between(*limits["g2"]),
                    ratio > limits["above_g2_min"],
                ],
                ["sub-G1", "G1", "G2/M", ">G2"],
                default="between",
            )
        rows.append(row)
    peaks = pd.DataFrame(rows, columns=["group", "n_cells", "n_reference", "g1_peak", "g2_peak"])
    peaks["g2_g1_ratio"] = peaks["g2_peak"] / peaks["g1_peak"]
    peaks["has_g2_peak"] = peaks["g2_g1_ratio"].notna()
    return labels, peaks


def lookup_qc_status(
    df: pd.DataFrame,
    table: pd.DataFrame,
    key_cols: Sequence[str],
    status_col: str,
    missing: str = "not assessed",
) -> pd.Series:
    """Look up a QC status (``"pass"``/``"fail"``) for every row of ``df``.

    Rows of ``df`` are matched to ``table`` on ``key_cols``. Keys are compared
    as text (integer keys often come back from CSV as text or floats; ``2.0``
    is treated as ``2``). A row with no match in ``table`` was never assessed
    and gets ``missing``, never ``"pass"``.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to label (cells), with ``key_cols``.
    table : pd.DataFrame
        QC table with ``key_cols`` and ``status_col`` (one status per key).
    key_cols : sequence of str
        Columns that identify a site or a cell in both tables.
    status_col : str
        Status column in ``table``, e.g. ``"Metadata_profile_qc"``.
    missing : str
        Value for rows not found in ``table``.

    Returns
    -------
    pd.Series
        One status per row of ``df``, aligned to ``df.index``.

    Raises
    ------
    ValueError
        If ``table`` gives one key two different statuses.
    """
    def as_text(frame: pd.DataFrame) -> pd.DataFrame:
        text = frame[list(key_cols)].astype(str)
        return text.apply(lambda col: col.str.strip().str.replace(r"\.0$", "", regex=True))

    lookup = as_text(table).assign(**{status_col: table[status_col].to_numpy()}).drop_duplicates()
    if lookup.duplicated(list(key_cols)).any():
        raise ValueError(f"{status_col}: some {list(key_cols)} keys have more than one status")
    status = as_text(df).merge(lookup, on=list(key_cols), how="left")[status_col]
    return pd.Series(status.fillna(missing).to_numpy(), index=df.index, name=status_col)


# ---------------------------------------------------------------------------
# Image display helpers
# ---------------------------------------------------------------------------


def index_images(image_root: Path | None) -> dict[str, Path]:
    """Map file name -> path for every file under ``image_root`` (empty if unset)."""
    if image_root is None or not Path(image_root).is_dir():
        return {}
    return {p.name: p for p in sorted(Path(image_root).rglob("*")) if p.is_file()}


def resolve_image_path(path_name, file_name, image_index: Mapping[str, Path]) -> Path | None:
    """Find an image from CellProfiler's PathName/FileName, else by name under the image root.

    CellProfiler stores the path on the acquisition machine, which often does
    not exist where the analysis runs; the file name usually still matches.
    """
    if not isinstance(file_name, str) or not file_name:
        return None
    if isinstance(path_name, str) and path_name:
        candidate = Path(path_name) / file_name
        if candidate.is_file():
            return candidate
    return image_index.get(file_name)


def crop_centered(image: np.ndarray, x: float, y: float, half_size: int) -> np.ndarray:
    """Return a ``2*half_size`` square crop centred on (x, y), zero-padded at edges."""
    padded = np.pad(image, half_size, mode="constant")
    col, row = int(round(x)) + half_size, int(round(y)) + half_size
    return padded[row - half_size: row + half_size, col - half_size: col + half_size]
