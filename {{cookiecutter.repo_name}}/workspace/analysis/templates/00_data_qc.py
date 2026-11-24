import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    def print(*values, sep=" ", end="\n"):
        """Display console-style progress in both notebook and App mode."""
        message = sep.join(str(value) for value in values) + end
        mo.output.append(mo.plain_text(message))

    return mo, print


@app.cell
def _(mo):
    mo.md(r"""
    # 00 — Data Quality Control (images and single cells)

    **Pipeline step:** 0 of 6 (runs before NB01)
    **Position in pipeline:** CellProfiler output → **NB00 (data QC)** → NB01 (samples retrieval)
    **Purpose:** find bad images and badly segmented cells in the raw
    CellProfiler output, show them, and write exclusion lists that NB01 applies.

    > This notebook **flags only**. It never edits or deletes CellProfiler
    > output. NB01 adds two labels to every cell, **`Metadata_image_qc`** and
    > **`Metadata_profile_qc`** (`pass` / `fail`), and removes the `fail` cells.

    **Input:** CellProfiler SQLite database(s) or `Per_*` CSV tables in
    `workspace/backend/<EXPERIMENT_ID>/`; raw images (optional, for display and
    for the pixel-based fallback)
    **Outputs** (in `analysis/<EXPERIMENT_ID>/results/data_qc/`):
    `site_qc.csv`, `excluded_sites.csv`, `cell_qc.parquet`,
    `excluded_cells.parquet`, `dna_content_peaks.csv`,
    `dna_content_per_well.csv`, `provenance_nb00_latest.json`; figures in
    `figures/data_qc/` (`<name>.pdf` and `.png` for papers, `<name>_explained.png` with headline)

    ## Three checks

    | Check | Unit | Question | Fails when |
    |---|---|---|---|
    | **Image QC** | site (field of view) | Is any channel out of focus or clipped? | blur robust z < −5, or > 1 % of pixels at the detector ceiling |
    | **Profile QC** ([coSMicQC](https://github.com/cytomining/coSMicQC)) | cell | Is the segmentation of this cell likely wrong? | a cell trips an active coSMicQC condition |
    | **DNA content** | plate, well, cell | Does the nuclear stain show G1 and G2/M populations? | reported, not used to exclude cells |

    Image QC and profile QC are independent: a cell can fail either one. NB01
    keeps a cell only if its site passes image QC **and** the cell passes
    profile QC (the same rule as the Panc1-hPSC pilot QC).

    **Figures** follow the `publication-figures` skill: each figure has a plain-language
    headline and a how-to-read line in the notebook, one colour per meaning, one brightness
    scale for compared images, and a note below it on what to notice and what not to
    conclude. The exported PDF/PNG files leave the headline out; the caption carries it.

    ## Execution logic

    1. Choose the experiment and (optionally) adjust thresholds.
    2. Load the CellProfiler tables and keep only the columns QC needs.
    3. Image QC: blur and saturation per site and channel; montages of flagged
       and passing sites.
    4. Profile QC: coSMicQC conditions per plate; crops of flagged and passing cells.
    5. DNA content: G1/G2 peaks per plate, cell-cycle fractions per well, and
       how often each coSMicQC condition flags G2/M cells.
    6. Compare flag rates of treatments and negative controls.
    7. Write flag tables and exclusion lists, provenance, and integrity checks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 0 — Imports and shared utilities
    """)
    return


@app.cell
def _():
    import json
    import platform
    from datetime import datetime, timezone
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", 80)
    return Path, datetime, np, pd, platform, plt, timezone


@app.cell
def _(Path, mo):
    # Locate the repo root before hca_pipeline can be imported (bootstrap:
    # can't import find_repo_root from the package until sys.path includes
    # workspace). __file__ is used instead of cwd so this notebook
    # behaves identically under `marimo edit`, a plain `python` run, or an
    # automated headless run from any working directory.
    import sys

    _notebook_path = Path(__file__).resolve()
    _repo_root_candidate = None
    for _p in (_notebook_path, *_notebook_path.parents):
        if (_p / "pixi.toml").exists() or (_p / ".git").exists() or (_p / "pixi.lock").exists():
            _repo_root_candidate = _p
            break
    if _repo_root_candidate is None:
        raise FileNotFoundError(
            "Could not find repo root (pixi.toml / .git / pixi.lock) starting "
            f"from {_notebook_path}. Ensure this notebook lives inside the project."
        )

    REPO_ROOT = _repo_root_candidate
    _pipelines_dir = REPO_ROOT / "workspace"
    if not (_pipelines_dir / "hca_pipeline").exists():
        raise FileNotFoundError(f"hca_pipeline package directory not found: {_pipelines_dir / 'hca_pipeline'}")
    sys.path.insert(0, str(_pipelines_dir))

    from hca_pipeline.config import ExperimentConfig
    from hca_pipeline.data_qc import (
        PROFILE_QC_DESCRIPTIONS,
        SITE_KEYS,
        cellprofiler_quality_channels,
        crop_centered,
        dna_content_groups,
        flag_sites,
        index_images,
        label_cell_outliers,
        profile_qc_conditions,
        profile_qc_status,
        resolve_dna_channel,
        resolve_image_path,
        site_quality_from_cellprofiler,
        site_quality_from_pixels,
    )
    from hca_pipeline import pubfig as pf
    from hca_pipeline.data_qc_figures import (
        fig_cell_crops,
        fig_dna_content,
        fig_feature_z,
        fig_image_montage,
        fig_image_qc_distributions,
        fig_image_qc_plate_maps,
        fig_profile_qc_rates,
        readable,
        shared_limits,
    )
    from hca_pipeline.image_qc import read_image, robust_rescale_image, scan_images
    from hca_pipeline.io import (
        discover_single_cell_inputs,
        read_cellprofiler_csvs,
        read_cellprofiler_sqlite,
        write_parquet_protected,
        write_summary_table_protected,
    )
    from hca_pipeline.metadata import (
        annotate_per_plate,
        dedupe_meta,
        ensure_core_metadata,
        read_barcode_platemap,
    )

    # One figure style for the whole notebook (fonts, sizes, colours, export settings).
    pf.use_style("generic")

    mo.callout(mo.md(f"✅ Loaded `hca_pipeline` from `{_pipelines_dir}`."), kind="success")
    return (
        ExperimentConfig,
        PROFILE_QC_DESCRIPTIONS,
        REPO_ROOT,
        SITE_KEYS,
        annotate_per_plate,
        cellprofiler_quality_channels,
        crop_centered,
        dedupe_meta,
        discover_single_cell_inputs,
        dna_content_groups,
        ensure_core_metadata,
        fig_cell_crops,
        fig_dna_content,
        fig_feature_z,
        fig_image_montage,
        fig_image_qc_distributions,
        fig_image_qc_plate_maps,
        fig_profile_qc_rates,
        flag_sites,
        index_images,
        label_cell_outliers,
        profile_qc_conditions,
        pf,
        profile_qc_status,
        read_barcode_platemap,
        read_cellprofiler_csvs,
        read_cellprofiler_sqlite,
        read_image,
        readable,
        resolve_dna_channel,
        resolve_image_path,
        robust_rescale_image,
        scan_images,
        shared_limits,
        site_quality_from_cellprofiler,
        site_quality_from_pixels,
        write_parquet_protected,
        write_summary_table_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Pick the experiment folder. Settings saved by NB01's configuration wizard
    (`experiment_config.json`: control vocabulary, image folder) are used when
    they exist; on a first run, before NB01, defaults are used.

    The thresholds below are the published starting points, **not validated
    cut-offs for your assay**. Run this notebook interactively once per
    experiment, look at the montages and crops, and change a threshold only
    when the images show it is wrong. Thresholds are under **Advanced settings**.
    """)
    return


@app.cell
def _(REPO_ROOT, mo):
    _backend_dir = REPO_ROOT / "workspace" / "backend"
    available_experiment_ids = (
        sorted(p.name for p in _backend_dir.iterdir() if p.is_dir()) if _backend_dir.is_dir() else []
    )
    _default_experiment_id = available_experiment_ids[0] if available_experiment_ids else "SET_EXPERIMENT_ID_HERE"
    experiment_id_input = mo.ui.dropdown(
        options=available_experiment_ids or [_default_experiment_id],
        value=_default_experiment_id,
        label="Experiment ID",
    )
    experiment_id_input
    return (experiment_id_input,)


@app.cell
def _(ExperimentConfig, REPO_ROOT, experiment_id_input):
    loaded_config = ExperimentConfig.load(REPO_ROOT, experiment_id_input.value)
    return (loaded_config,)


@app.cell
def _(loaded_config, mo):
    overwrite_input = mo.ui.checkbox(
        value=loaded_config.overwrite_existing_outputs, label="Overwrite existing outputs"
    )
    image_root_input = mo.ui.text(
        value=loaded_config.image_root or "",
        label="Raw image folder (relative to the repo root; empty = use CellProfiler paths only)",
        full_width=True,
    )
    blur_z_input = mo.ui.number(start=-20, stop=-1, step=0.5, value=-5.0, label="Blur limit (robust z of blur metric)")
    saturation_input = mo.ui.number(start=0.01, stop=20, step=0.1, value=1.0, label="Saturation limit (% clipped pixels)")
    condition_inputs = mo.ui.dictionary(
        {
            "under_segmented_nuclei": mo.ui.checkbox(value=True, label="under_segmented_nuclei"),
            "poorly_segmented_nuclei": mo.ui.checkbox(value=True, label="poorly_segmented_nuclei"),
            "under_segmented_cells": mo.ui.checkbox(value=True, label="under_segmented_cells"),
        }
    )
    pixel_size_input = mo.ui.number(
        start=0, stop=5, step=0.01, value=0.0,
        label="Pixel size (µm per pixel; 0 = unknown, no scale bars)",
    )
    n_examples_input = mo.ui.slider(start=2, stop=12, step=1, value=6, label="Examples per montage")
    mo.vstack(
        [
            overwrite_input,
            image_root_input,
            mo.accordion(
                {
                    "Advanced settings: QC thresholds (change only after looking at the images)": mo.vstack(
                        [
                            blur_z_input,
                            saturation_input,
                            mo.md("**Active coSMicQC conditions** (a cell fails if it trips any active one):"),
                            condition_inputs,
                            pixel_size_input,
                            n_examples_input,
                        ]
                    )
                }
            ),
        ]
    )
    return (
        blur_z_input,
        condition_inputs,
        image_root_input,
        n_examples_input,
        overwrite_input,
        pixel_size_input,
        saturation_input,
    )


@app.cell
def _(
    REPO_ROOT,
    blur_z_input,
    condition_inputs,
    experiment_id_input,
    image_root_input,
    loaded_config,
    mo,
    n_examples_input,
    overwrite_input,
    pixel_size_input,
    saturation_input,
):
    EXPERIMENT_ID = experiment_id_input.value
    CONFIG = loaded_config
    OVERWRITE = bool(overwrite_input.value)
    BLUR_Z_LIMIT = float(blur_z_input.value)
    SATURATION_PCT_LIMIT = float(saturation_input.value)
    ACTIVE_CONDITIONS = [name for name, on in condition_inputs.value.items() if on]
    N_EXAMPLES = int(n_examples_input.value)
    RANDOM_STATE = 0
    PIXEL_SIZE_UM = float(pixel_size_input.value) or None
    FIG_FORMATS = ("pdf", "png")  # vector first; the explained PNG is added by pubfig.export
    IMAGE_ROOT = (REPO_ROOT / image_root_input.value.strip()) if image_root_input.value.strip() else None

    WORKSPACE_DIR = REPO_ROOT / "workspace"
    DB_ROOT = WORKSPACE_DIR / "backend" / EXPERIMENT_ID
    METADATA_DIR = WORKSPACE_DIR / "metadata" / EXPERIMENT_ID
    ANALYSIS_DIR = WORKSPACE_DIR / "analysis" / EXPERIMENT_ID
    RESULTS_DIR = ANALYSIS_DIR / "results" / "data_qc"
    FIGS_DIR = ANALYSIS_DIR / "figures" / "data_qc"
    for _directory in (RESULTS_DIR, FIGS_DIR):
        _directory.mkdir(parents=True, exist_ok=True)

    mo.callout(
        mo.md(
            f"""
            ### Effective settings

            | Setting | Value |
            |---|---|
            | Experiment | `{EXPERIMENT_ID}` |
            | CellProfiler input | `{DB_ROOT}` |
            | Pixel size | {f'{PIXEL_SIZE_UM} µm' if PIXEL_SIZE_UM else 'unknown (no scale bars)'} |
            | Raw image folder | `{IMAGE_ROOT or 'not set (CellProfiler paths only)'}` |
            | Blur limit | robust z < {BLUR_Z_LIMIT} |
            | Saturation limit | > {SATURATION_PCT_LIMIT} % clipped pixels |
            | Active coSMicQC conditions | {', '.join(ACTIVE_CONDITIONS) or 'none'} |
            | Negative-control values | {', '.join(CONFIG.negcon_values)} |
            | Overwrite existing outputs | {OVERWRITE} |
            | Results | `{RESULTS_DIR}` |
            """
        ),
        kind="info",
    )
    return (
        ACTIVE_CONDITIONS,
        BLUR_Z_LIMIT,
        CONFIG,
        DB_ROOT,
        EXPERIMENT_ID,
        FIGS_DIR,
        FIG_FORMATS,
        IMAGE_ROOT,
        METADATA_DIR,
        N_EXAMPLES,
        OVERWRITE,
        PIXEL_SIZE_UM,
        RANDOM_STATE,
        RESULTS_DIR,
        SATURATION_PCT_LIMIT,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Load the CellProfiler output

    The same readers as NB01 merge the Image, Cells, Cytoplasm and Nuclei
    tables. Only the columns QC needs are kept (metadata, object numbers,
    `ImageQuality`, file names, nuclear/cell DNA intensity, solidity and
    nucleus position), so memory use stays small.
    """)
    return


@app.cell
def _(
    DB_ROOT,
    dedupe_meta,
    discover_single_cell_inputs,
    ensure_core_metadata,
    mo,
    pd,
    print,
    read_cellprofiler_csvs,
    read_cellprofiler_sqlite,
):
    _inputs = discover_single_cell_inputs(DB_ROOT)
    INPUT_PATHS = _inputs["sqlite"] or _inputs["csv"]
    mo.stop(
        not INPUT_PATHS,
        mo.callout(
            mo.md(
                f"**NB00 stopped:** no CellProfiler SQLite database or `Per_*` CSV tables in `{DB_ROOT}`. "
                "Copy the CellProfiler output there and rerun this cell."
            ),
            kind="warn",
        ),
    )
    if _inputs["sqlite"]:
        _raw = pd.concat([read_cellprofiler_sqlite(p, progress=print) for p in INPUT_PATHS], ignore_index=True)
    else:
        _raw = read_cellprofiler_csvs(INPUT_PATHS, progress=print)
    _raw = dedupe_meta(ensure_core_metadata(_raw))

    _missing = [c for c in ("Metadata_Plate", "Metadata_Well", "Metadata_Site") if c not in _raw.columns]
    if _missing:
        raise ValueError(
            f"Required metadata missing: {_missing}. Image QC works per site, so the CellProfiler "
            "export must include Plate, Well and Site metadata."
        )

    _keep_prefixes = (
        "Metadata_",
        "Image_ImageQuality_",
        "Image_FileName_",
        "Image_PathName_",
        "Image_Count_",
        "Nuclei_Intensity_IntegratedIntensity_",
        "Nuclei_Intensity_MassDisplacement_",
        "Cells_Intensity_IntegratedIntensity_",
    )
    _keep_exact = {
        "ImageNumber",
        "Cells_Number_Object_Number",
        "Nuclei_Number_Object_Number",
        "Nuclei_AreaShape_Solidity",
        "Nuclei_AreaShape_Area",
        "Nuclei_Location_Center_X",
        "Nuclei_Location_Center_Y",
        "Nuclei_AreaShape_Center_X",
        "Nuclei_AreaShape_Center_Y",
    }
    qc_cells = _raw[[c for c in _raw.columns if c.startswith(_keep_prefixes) or c in _keep_exact]].copy()
    del _raw
    qc_cells["Metadata_Plate"] = qc_cells["Metadata_Plate"].astype("string").str.strip()
    # One text form for site IDs, so site keys match across CSV/SQLite/image names.
    qc_cells["Metadata_Site"] = qc_cells["Metadata_Site"].astype(str).str.replace(r"\.0$", "", regex=True)

    print(f"✓ Loaded {len(qc_cells):,} cells × {qc_cells.shape[1]} QC columns from {len(INPUT_PATHS)} file(s)")
    print(f"  Plates: {qc_cells['Metadata_Plate'].nunique()}  ·  "
          f"Sites: {len(qc_cells[['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']].drop_duplicates()):,}")
    return INPUT_PATHS, qc_cells


@app.cell
def _(
    CONFIG,
    METADATA_DIR,
    annotate_per_plate,
    dedupe_meta,
    print,
    qc_cells,
    read_barcode_platemap,
):
    # Optional: treatment and control labels from the platemap. NB00 works
    # without them; they add the negative-control G1 anchor (section 5) and
    # the treatment-vs-control comparison (section 6).
    _barcode_csv = METADATA_DIR / "barcode_platemap.csv"
    cells_annotated = qc_cells
    HAS_CONTROLS = False
    if not _barcode_csv.is_file():
        print(f"ℹ️  No platemap ({_barcode_csv}) — sections 5–6 run without control labels.")
    else:
        try:
            cells_annotated = dedupe_meta(
                annotate_per_plate(qc_cells, read_barcode_platemap(_barcode_csv), METADATA_DIR / "platemap")
            )
            HAS_CONTROLS = CONFIG.control_type_col in cells_annotated.columns
            print(f"✓ Platemap annotation added ({cells_annotated[CONFIG.control_type_col].notna().mean():.0%} of cells labelled)")
        except (KeyError, FileNotFoundError, ValueError) as _error:
            print(f"⚠️  Platemap annotation skipped: {_error}")
    return HAS_CONTROLS, cells_annotated


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Image-level QC (blur and saturation)

    **Question:** is any channel of a site out of focus or clipped?

    **Method.** Rules of the Panc1-hPSC pilot QC, per plate and channel:

    - **Blurred:** the blur metric has a robust z-score
      (`(x − median) / (1.4826 × MAD)`) below the blur limit (default −5).
      The metric is CellProfiler's `ImageQuality_PowerLogLogSlope` (slope of
      the log-log power spectrum; a blurred image has less high-frequency
      power, so a lower slope). Without `MeasureImageQuality` in the
      CellProfiler pipeline, the fallback is `log10` of the Laplacian-variance
      focus score computed from the raw images (`hca_pipeline.image_qc`).
    - **Saturated:** more than the saturation limit (default 1 %) of pixels
      are at the detector ceiling (`ImageQuality_PercentMaximal`, counted only
      when `ImageQuality_MaxIntensity` ≥ 0.99).

    A site fails if any channel is blurred or saturated: `Metadata_image_qc`
    is `fail` for the site and for all its cells. Otherwise it is `pass`.

    **Assumption:** most sites on a plate are in focus. The robust z is
    relative to the plate median, so a plate that is blurred everywhere will
    pass. Look at the montages, not only the counts.
    """)
    return


@app.cell
def _(
    BLUR_Z_LIMIT,
    IMAGE_ROOT,
    SATURATION_PCT_LIMIT,
    SITE_KEYS,
    cellprofiler_quality_channels,
    flag_sites,
    mo,
    print,
    qc_cells,
    scan_images,
    site_quality_from_cellprofiler,
    site_quality_from_pixels,
):
    IMAGE_CHANNELS = cellprofiler_quality_channels(qc_cells.columns)
    pixel_metrics = None
    if IMAGE_CHANNELS:
        IMAGE_QC_SOURCE = "CellProfiler MeasureImageQuality"
        _quality = site_quality_from_cellprofiler(qc_cells, IMAGE_CHANNELS)
    elif IMAGE_ROOT is not None and IMAGE_ROOT.is_dir():
        IMAGE_QC_SOURCE = "raw pixels (fallback)"
        print(f"No ImageQuality columns — scanning raw images under {IMAGE_ROOT} (slow) …")
        pixel_metrics, _failed = scan_images(IMAGE_ROOT)
        if _failed:
            print(f"  ⚠️  {len(_failed)} image(s) could not be read")
        pixel_metrics["Metadata_Site"] = pixel_metrics["Metadata_Site"].astype(str).str.replace(r"\.0$", "", regex=True)
        IMAGE_CHANNELS = sorted(c for c in pixel_metrics["Metadata_Channel"].unique() if c != "Unknown")
        _quality = site_quality_from_pixels(pixel_metrics, IMAGE_CHANNELS)
    else:
        IMAGE_QC_SOURCE = "not assessed"
        _quality = qc_cells[SITE_KEYS].drop_duplicates().reset_index(drop=True)

    if IMAGE_CHANNELS:
        site_flags = flag_sites(
            _quality, IMAGE_CHANNELS, group_cols=["Metadata_Plate"],
            blur_z_limit=BLUR_Z_LIMIT, saturation_pct_limit=SATURATION_PCT_LIMIT,
        )
    else:
        site_flags = _quality.assign(Metadata_image_qc="not assessed", Metadata_image_qc_reason="")

    _n_fail = int(site_flags["Metadata_image_qc"].eq("fail").sum())
    mo.callout(
        mo.md(
            f"**Source:** {IMAGE_QC_SOURCE} · **Channels:** {', '.join(IMAGE_CHANNELS) or 'none'} · "
            f"**Sites failing:** {_n_fail:,} of {len(site_flags):,}"
            + (
                "\n\nNo `ImageQuality` columns and no raw image folder: image QC was skipped. "
                "Add `MeasureImageQuality` to the CellProfiler pipeline (see `PIPELINE.md`) "
                "or set the raw image folder above."
                if IMAGE_QC_SOURCE == "not assessed" else ""
            )
        ),
        kind="warn" if IMAGE_QC_SOURCE == "not assessed" else "info",
    )
    return IMAGE_CHANNELS, IMAGE_QC_SOURCE, pixel_metrics, site_flags


@app.cell
def _(IMAGE_CHANNELS, pd, site_flags):
    def _image_summary():
        rows = []
        for plate, sub in site_flags.groupby("Metadata_Plate"):
            row = {"Metadata_Plate": plate, "sites": len(sub), "failed": int(sub["Metadata_image_qc"].eq("fail").sum())}
            for ch in IMAGE_CHANNELS:
                row[f"blurred_{ch}"] = int(sub[f"blurred_{ch}"].sum())
                row[f"saturated_{ch}"] = int(sub[f"saturated_{ch}"].sum())
            rows.append(row)
        table = pd.DataFrame(rows)
        if not table.empty:
            table["failed_%"] = (100 * table["failed"] / table["sites"]).round(1)
        return table

    image_qc_summary = _image_summary()
    image_qc_summary
    return


@app.cell
def _(
    BLUR_Z_LIMIT,
    FIGS_DIR,
    FIG_FORMATS,
    IMAGE_CHANNELS,
    OVERWRITE,
    SATURATION_PCT_LIMIT,
    fig_image_qc_distributions,
    mo,
    pf,
    site_flags,
):
    mo.stop(not IMAGE_CHANNELS, mo.md("_No image-QC metrics to plot._"))
    pf.export(
        fig_image_qc_distributions, "image_qc_distributions", site_flags, IMAGE_CHANNELS,
        BLUR_Z_LIMIT, SATURATION_PCT_LIMIT,
        outdir=FIGS_DIR, formats=FIG_FORMATS, overwrite=OVERWRITE, return_fig=True,
    )
    return



@app.cell
def _(mo):
    mo.md(r"""
    **What to notice:** sites beyond a dashed line are the ones NB01 removes. Most sites
    should sit in one main group, with the flagged sites clearly apart from it.

    **Do not conclude:** that every flagged site is bad. The blur score is relative to the
    other sites on the same plate, so a plate that is blurred everywhere has no outliers.
    The montages below show what the flagged images look like.
    """)
    return



@app.cell
def _(CONFIG, FIGS_DIR, FIG_FORMATS, OVERWRITE, fig_image_qc_plate_maps, pf, site_flags):
    pf.export(
        fig_image_qc_plate_maps, "image_qc_plate_maps", site_flags, CONFIG.plate_format,
        outdir=FIGS_DIR, formats=FIG_FORMATS, overwrite=OVERWRITE, return_fig=True,
    )
    return



@app.cell
def _(mo):
    mo.md(r"""
    **What to notice:** failed sites bunched in a corner, along an edge or in whole rows or
    columns point to a plate or microscope problem, not to random bad fields.

    **Do not conclude:** a cause. A well with every site failed may also hold few cells.
    This map shows where the failures are, not why.
    """)
    return



@app.cell
def _(mo):
    mo.md(r"""
    ### Montages: flagged vs. passing sites

    Each failing site (top row) is shown next to a passing site from the same plate
    (bottom row), same channel. **All panels in one montage share one brightness scale**,
    taken from the passing sites, so brightness and sharpness can be compared by eye.
    **Magenta pixels are at the detector ceiling.** Check that the flagged images really are
    blurred or clipped. If the passing images look as bad, the limit is too loose; if the
    flagged images look fine, it is too strict. Sites are picked at random; set the pixel
    size under **Advanced settings** to get a scale bar.
    """)
    return



@app.cell
def _(IMAGE_ROOT, index_images, pixel_metrics):
    # One file-name index serves every montage and crop below. CellProfiler
    # stores the acquisition-machine path, which usually does not exist here.
    IMAGE_INDEX = index_images(IMAGE_ROOT)
    if pixel_metrics is not None:
        for _path in pixel_metrics["filepath"]:
            IMAGE_INDEX.setdefault(_path.rsplit("/", 1)[-1], _path)
    return (IMAGE_INDEX,)


@app.cell
def _(
    FIGS_DIR,
    FIG_FORMATS,
    IMAGE_CHANNELS,
    IMAGE_INDEX,
    N_EXAMPLES,
    OVERWRITE,
    PIXEL_SIZE_UM,
    Path,
    RANDOM_STATE,
    SITE_KEYS,
    fig_image_montage,
    mo,
    np,
    pf,
    pixel_metrics,
    qc_cells,
    read_image,
    resolve_image_path,
    shared_limits,
    site_flags,
):
    def _site_image(site: dict, channel: str):
        """Image of one site and channel as an array, or None when the file is not found."""
        path = None
        if pixel_metrics is not None:
            match = pixel_metrics
            for key in SITE_KEYS:
                match = match[match[key].astype(str) == str(site[key])]
            match = match[match["Metadata_Channel"] == channel]
            path = Path(match["filepath"].iloc[0]) if len(match) else None
        else:
            rows = qc_cells
            for key in SITE_KEYS:
                rows = rows[rows[key].astype(str) == str(site[key])]
            if not rows.empty and f"Image_FileName_{channel}" in rows:
                path = resolve_image_path(
                    rows[f"Image_PathName_{channel}"].iloc[0] if f"Image_PathName_{channel}" in rows else None,
                    rows[f"Image_FileName_{channel}"].iloc[0],
                    IMAGE_INDEX,
                )
        return None if path is None else read_image(Path(path))

    def _label(site: dict, note: str) -> str:
        return f"{site['Metadata_Well']} site {site['Metadata_Site']}\n{note}"

    def _montage(channel: str):
        failed = site_flags[site_flags.get(f"blurred_{channel}", False) | site_flags.get(f"saturated_{channel}", False)]
        if failed.empty:
            return None, 0
        rng = np.random.default_rng(RANDOM_STATE)
        failed = failed.iloc[rng.permutation(len(failed))[:N_EXAMPLES]]
        passing = site_flags[site_flags["Metadata_image_qc"].eq("pass")]
        failed_tiles, passed_tiles = [], []
        for _, site in failed.iterrows():
            blurred, saturated = bool(site[f"blurred_{channel}"]), bool(site[f"saturated_{channel}"])
            note = " and ".join(n for n, hit in (("blurred", blurred), ("saturated", saturated)) if hit)
            failed_tiles.append((_label(site, note), _site_image(site, channel)))
            same_plate = passing[passing["Metadata_Plate"] == site["Metadata_Plate"]]
            if len(same_plate):
                partner = same_plate.iloc[rng.integers(len(same_plate))]
                passed_tiles.append((_label(partner, "pass"), _site_image(partner, channel)))
        images = [im for _, im in passed_tiles if im is not None] or [im for _, im in failed_tiles if im is not None]
        n_found = len(images)
        if not images:
            return None, 0
        fig = pf.export(
            fig_image_montage, f"image_qc_montage_{channel}", channel, failed_tiles, passed_tiles,
            shared_limits(images), PIXEL_SIZE_UM,
            outdir=FIGS_DIR, formats=FIG_FORMATS, overwrite=OVERWRITE, return_fig=True,
        )
        return fig, n_found

    _panels = []
    for _ch in IMAGE_CHANNELS:
        _fig, _n_found = _montage(_ch)
        if _fig is None:
            _panels.append(mo.md(f"**{_ch}:** no flagged sites, or images not found. "
                                 "Set the raw image folder in section 1 so file names can be matched."))
            continue
        _panels.append(_fig)
    mo.vstack(_panels) if _panels else mo.md("_No image-QC channels — no montages._")
    return



@app.cell
def _(mo):
    mo.md(r"""
    **What to notice:** blurred sites lack sharp edges and fine texture compared with the
    passing sites below them; saturated sites show large magenta areas.

    **Do not conclude:** from a few examples. They are random, not the worst cases. If you
    doubt a limit, change it under **Advanced settings** and rerun.
    """)
    return



@app.cell
def _(PROFILE_QC_DESCRIPTIONS, mo, profile_qc_conditions):
    _example = profile_qc_conditions("<DNA>")
    _rows = "\n".join(
        f"| `{name}` | {PROFILE_QC_DESCRIPTIONS[name]} | "
        + "<br>".join(f"`{feature}` {'>' if z > 0 else '<'} {z:+g}" for feature, z in thresholds.items())
        + " |"
        for name, thresholds in _example.items()
    )
    mo.md(
        rf"""
    ## 4 — Single-cell (profile) QC with coSMicQC

    **Question:** which cells were probably segmented wrongly?

    **Method.** coSMicQC z-scores each raw CellProfiler feature and flags a
    cell when **every** feature of a condition passes its threshold
    (z > threshold when positive, z < threshold when negative). z-scores
    are computed **per plate**, so a plate-wide brightness change does not
    create outliers. Thresholds are those of the coSMicQC paper and the
    Panc1-hPSC pilot QC. `<DNA>` is the nuclear stain chosen below.

    | Condition | What it catches | Rule (z-score) |
    |---|---|---|
    {_rows}

    **Assumptions:**

    - Most cells on a plate are segmented correctly, so the plate mean and SD
      describe good cells.
    - Treatments do not change these features strongly. If a treatment makes
      nuclei large or irregular, coSMicQC flags real biology. Section 6
      checks this.
    - `under_segmented_cells` (+0.5 SD of cell DNA) also flags **healthy G2/M
      cells**, which have twice the DNA. Section 5 measures how often; the
      Panc1 pilot turned this condition off for one cell line for that reason.
    """
    )
    return


@app.cell
def _(
    ACTIVE_CONDITIONS,
    label_cell_outliers,
    mo,
    print,
    profile_qc_conditions,
    profile_qc_status,
    qc_cells,
    resolve_dna_channel,
):
    try:
        DNA_CHANNEL = resolve_dna_channel(qc_cells.columns)
    except ValueError as _error:
        mo.stop(True, mo.callout(mo.md(f"**Profile QC stopped:** {_error}"), kind="warn"))
    PROFILE_CONDITIONS = profile_qc_conditions(DNA_CHANNEL)
    _missing = sorted({f for t in PROFILE_CONDITIONS.values() for f in t} - set(qc_cells.columns))
    mo.stop(
        bool(_missing),
        mo.callout(mo.md(f"**Profile QC stopped:** features missing from the CellProfiler output: `{_missing}`"), kind="warn"),
    )

    print(f"Nuclear stain for DNA features: {DNA_CHANNEL}"
          + ("  (acridine orange green also stains RNA — DNA content is a proxy)" if DNA_CHANNEL == "AOGFP" else ""))
    print("Running coSMicQC per plate …")
    cell_flags = label_cell_outliers(qc_cells, PROFILE_CONDITIONS, group_col="Metadata_Plate")
    cell_status = profile_qc_status(cell_flags, ACTIVE_CONDITIONS)
    print(f"✓ Profile QC: {int(cell_status['Metadata_profile_qc'].eq('fail').sum()):,} of {len(cell_status):,} cells fail "
          f"({cell_status['Metadata_profile_qc'].eq('fail').mean():.1%})")
    return DNA_CHANNEL, PROFILE_CONDITIONS, cell_flags, cell_status


@app.cell
def _(
    ACTIVE_CONDITIONS,
    FIGS_DIR,
    FIG_FORMATS,
    OVERWRITE,
    PROFILE_CONDITIONS,
    cell_flags,
    cell_status,
    fig_profile_qc_rates,
    mo,
    pf,
    qc_cells,
):
    _flags = cell_flags.assign(Metadata_Plate=qc_cells["Metadata_Plate"].to_numpy())
    _condition_cols = [f"Metadata_cqc_{n}_is_outlier" for n in PROFILE_CONDITIONS]
    profile_qc_summary = (
        _flags.groupby("Metadata_Plate")[_condition_cols + ["Metadata_cqc_features_missing"]].mean().mul(100).round(2)
    )
    profile_qc_summary.columns = [c.replace("Metadata_cqc_", "").replace("_is_outlier", "") + " %" for c in profile_qc_summary.columns]
    profile_qc_summary["profile_qc_fail %"] = (
        cell_status.assign(Metadata_Plate=qc_cells["Metadata_Plate"].to_numpy())
        .groupby("Metadata_Plate")["Metadata_profile_qc"].apply(lambda s: round(100 * s.eq("fail").mean(), 2))
    )
    _overlap = cell_flags[[f"Metadata_cqc_{n}_is_outlier" for n in ACTIVE_CONDITIONS]].sum(axis=1)
    condition_overlap = _overlap.value_counts().sort_index().rename_axis("active conditions tripped").to_frame("cells")

    _rates = profile_qc_summary[[f"{n} %" for n in PROFILE_CONDITIONS]].set_axis(list(PROFILE_CONDITIONS), axis=1)
    _fig = pf.export(
        fig_profile_qc_rates, "profile_qc_rates", _rates,
        100 * cell_status["Metadata_profile_qc"].eq("fail").mean(), ACTIVE_CONDITIONS,
        outdir=FIGS_DIR, formats=FIG_FORMATS, overwrite=OVERWRITE, return_fig=True,
    )
    mo.vstack([profile_qc_summary, condition_overlap, _fig])
    return



@app.cell
def _(mo):
    mo.md(r"""
    **What to notice:** rates should be similar across plates of the same experiment. One
    plate far above the others suggests a plate-level problem (focus, density, staining)
    rather than random segmentation errors.

    **Do not conclude:** that a high rate means bad cells. A rule can flag real biology
    (large or dividing cells). Sections 5 and 6 and the crops below test that.
    """)
    return



@app.cell
def _(FIGS_DIR, FIG_FORMATS, OVERWRITE, PROFILE_CONDITIONS, fig_feature_z, pf, qc_cells):
    # Per-plate z-scores (population SD, as coSMicQC uses), one entry per feature with its limit.
    _z_by_feature = {}
    for _thresholds in PROFILE_CONDITIONS.values():
        for _feature, _limit in _thresholds.items():
            if _feature in _z_by_feature:
                continue
            _grouped = qc_cells.groupby("Metadata_Plate")[_feature]
            _z = (qc_cells[_feature] - _grouped.transform("mean")) / _grouped.transform("std", ddof=0)
            _z_by_feature[_feature] = (_z.to_numpy(dtype=float), _limit)
    pf.export(
        fig_feature_z, "profile_qc_feature_z", _z_by_feature,
        outdir=FIGS_DIR, formats=FIG_FORMATS, overwrite=OVERWRITE, return_fig=True,
    )
    return



@app.cell
def _(mo):
    mo.md(r"""
    **What to notice:** where each limit sits relative to the main group of cells. A limit
    that cuts through the main group flags ordinary cells; the headline gives the share of
    cells beyond the limits.

    **Do not conclude:** that the limits are right for this assay. They are the published
    starting points. The crops below show whether the cells beyond them look wrong.
    """)
    return



@app.cell
def _(mo):
    mo.md(r"""
    ### Crops: flagged vs. passing cells

    For each rule, one row of random cells that trip it (FAIL) and one row of random passing
    cells from the same plates (pass), cropped around the nucleus centre (+) on the
    nuclear-stain image. Merged nuclei, irregular masks and doublets should be visible in the
    FAIL rows. If the FAIL and pass rows look the same, the rule is flagging normal cells.
    Brightness is scaled per field of view, not per crop.
    """)
    return



@app.cell
def _(
    DNA_CHANNEL,
    FIGS_DIR,
    FIG_FORMATS,
    IMAGE_INDEX,
    N_EXAMPLES,
    OVERWRITE,
    PIXEL_SIZE_UM,
    PROFILE_CONDITIONS,
    RANDOM_STATE,
    cell_flags,
    cell_status,
    crop_centered,
    fig_cell_crops,
    mo,
    np,
    pf,
    qc_cells,
    read_image,
    readable,
    resolve_image_path,
    robust_rescale_image,
):
    _x_col = next((c for c in ("Nuclei_Location_Center_X", "Nuclei_AreaShape_Center_X") if c in qc_cells), None)
    _y_col = _x_col.replace("_X", "_Y") if _x_col else None
    _file_col = f"Image_FileName_{DNA_CHANNEL}"
    _path_col = f"Image_PathName_{DNA_CHANNEL}"
    mo.stop(
        _x_col is None or _file_col not in qc_cells,
        mo.callout(mo.md(f"No nucleus position or `{_file_col}` column — cell crops are not available."), kind="warn"),
    )
    _cache: dict = {}

    def _crop(row_index, half_size=40):
        row = qc_cells.loc[row_index]
        path = resolve_image_path(row.get(_path_col), row[_file_col], IMAGE_INDEX)
        if path is None:
            return None
        if path not in _cache:
            _cache[path] = robust_rescale_image(read_image(path))
        return crop_centered(_cache[path], row[_x_col], row[_y_col], half_size)

    _rng = np.random.default_rng(RANDOM_STATE)
    _passed = cell_status.index[cell_status["Metadata_profile_qc"].eq("pass")]
    _blocks = []
    for _name in PROFILE_CONDITIONS:
        _hits = cell_flags.index[cell_flags[f"Metadata_cqc_{_name}_is_outlier"]]
        for _label, _pool in ((f"FAIL: {readable(_name)}", _hits), ("pass: random cells for comparison", _passed)):
            _chosen = _rng.choice(_pool, size=min(N_EXAMPLES, len(_pool)), replace=False) if len(_pool) else []
            _blocks.append((_label, [_crop(_i) for _i in _chosen]))
    _shown = sum(c is not None for _, crops in _blocks for c in crops)
    mo.stop(
        _shown == 0,
        mo.callout(mo.md("Images not found. Set the raw image folder in section 1."), kind="warn"),
    )
    pf.export(
        fig_cell_crops, "profile_qc_cell_crops", _blocks, N_EXAMPLES, PIXEL_SIZE_UM,
        outdir=FIGS_DIR, formats=FIG_FORMATS, overwrite=OVERWRITE, return_fig=True,
    )
    return



@app.cell
def _(mo):
    mo.md(r"""
    **What to notice:** FAIL rows should show merged or touching nuclei, ragged masks or
    doublets that the pass rows do not.

    **Do not conclude:** that a rule is wrong because a few FAIL crops look normal; no
    rule is perfect. If most of a FAIL row looks like the pass row, switch that rule off.
    Crops were picked at random, so they show the typical case, not the worst.
    """)
    return



@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — DNA content: are G1 and G2/M populations visible?

    **Question:** does the nuclear stain behave like a DNA-content stain on
    this plate, with a G1 peak and a G2/M peak at about twice its intensity?

    - **H0:** nuclear integrated intensity (`{DNA_CHANNEL}`) is not
      proportional to DNA content (no second peak at 1.6–2.5× G1).
    - **H1:** it is proportional: a G2/M peak at 1.6–2.5× the G1 peak.
    - **Supports H1:** a clear second peak in that window. **Fails to reject
      H0:** no second peak. Possible reasons: a non-stoichiometric stain
      (acridine orange green also stains RNA), cells that are not cycling
      (confluent or arrested), or merged nuclei / bad segmentation.

    **Method.** Histogram of `log2(Nuclei_Intensity_IntegratedIntensity_{DNA_CHANNEL})`
    per plate, smoothed (Gaussian, σ = 0.05 log2 units ≈ 3.5 % intensity).
    The tallest peak is G1. Use at least a few hundred reference cells per
    plate: with ~250 G1 cells the G1 position is uncertain by about ±4 %.
    When the platemap marks negative controls, only they locate the peaks, so
    a treatment that arrests cells cannot move the G1 anchor. Each cell is
    then labelled by its ratio to G1: `sub-G1` < 0.7 (fragments, debris,
    apoptotic), `G1` 0.8–1.25, `G2/M` 1.6–2.5, `>G2` > 2.8 (polyploid or
    merged nuclei), `between` otherwise (mostly S phase).

    **Positive control:** synthetic G1/G2 mixtures in `tests/test_data_qc.py`
    recover a ratio of 2.0 ± 0.15. **Negative control:** a single-peak
    distribution gives no G2 peak. These labels are **descriptive**: NB00 does
    not exclude cells on DNA content, because sub-G1 and >G2 cells can be
    treatment biology.
    """)
    return


@app.cell
def _(
    CONFIG,
    DNA_CHANNEL,
    HAS_CONTROLS,
    cells_annotated,
    dna_content_groups,
    mo,
    pd,
    qc_cells,
):
    _dna = qc_cells[f"Nuclei_Intensity_IntegratedIntensity_{DNA_CHANNEL}"]
    _reference = None
    if HAS_CONTROLS:
        _control = cells_annotated[CONFIG.control_type_col].astype(str).str.lower()
        _reference = pd.Series(
            _control.isin([v.lower() for v in CONFIG.negcon_values]).to_numpy(), index=qc_cells.index
        )
    dna_labels, dna_peaks = dna_content_groups(_dna, qc_cells["Metadata_Plate"], reference_mask=_reference)
    dna_peaks = dna_peaks.rename(columns={"group": "Metadata_Plate"})
    dna_peaks["peak_reference"] = [
        "negative controls" if (_reference is not None and n_ref < n) else "all cells"
        for n, n_ref in zip(dna_peaks["n_cells"], dna_peaks["n_reference"])
    ]
    dna_peaks["verdict"] = [
        "supports H1: G2/M peak at ~2× G1" if ok else "H0 not rejected: no G2/M peak at 1.6–2.5× G1"
        for ok in dna_peaks["has_g2_peak"]
    ]
    mo.vstack([mo.md("**Peaks per plate**"), dna_peaks.round(3)])
    return dna_labels, dna_peaks


@app.cell
def _(DNA_CHANNEL, FIGS_DIR, FIG_FORMATS, OVERWRITE, dna_peaks, fig_dna_content, pf, qc_cells):
    _column = f"Nuclei_Intensity_IntegratedIntensity_{DNA_CHANNEL}"
    _plate = qc_cells["Metadata_Plate"].astype(str)
    _values = {str(p): qc_cells.loc[_plate == str(p), _column].to_numpy(dtype=float) for p in dna_peaks["Metadata_Plate"]}
    pf.export(
        fig_dna_content, "dna_content_histograms", dna_peaks, _values, DNA_CHANNEL,
        outdir=FIGS_DIR, formats=FIG_FORMATS, overwrite=OVERWRITE, return_fig=True,
    )
    return



@app.cell
def _(mo):
    mo.md(r"""
    **What to notice:** a second peak inside the shaded window means the stain follows DNA
    content, which supports H1. Its height relative to G1 is the share of dividing cells.

    **Do not conclude:** biology from a missing peak alone. Confluent or arrested cells,
    a stain that is not proportional to DNA, or merged nuclei give the same picture.
    """)
    return



@app.cell
def _(dna_labels, pd, qc_cells):
    _groups = ["sub-G1", "G1", "between", "G2/M", ">G2", "not assessed"]
    dna_per_well = (
        pd.crosstab([qc_cells["Metadata_Plate"], qc_cells["Metadata_Well"]], dna_labels, normalize="index")
        .reindex(columns=_groups, fill_value=0.0)
        .mul(100)
        .round(1)
        .add_suffix(" %")
        .reset_index()
    )
    dna_per_well
    return (dna_per_well,)


@app.cell
def _(PROFILE_CONDITIONS, cell_flags, dna_labels, mo, pd):
    # Positive control for "is coSMicQC flagging biology?": G2/M cells are
    # healthy, so a condition that flags many of them is not only catching
    # segmentation errors.
    _rates = pd.DataFrame(
        {
            name: cell_flags[f"Metadata_cqc_{name}_is_outlier"].groupby(dna_labels).mean().mul(100)
            for name in PROFILE_CONDITIONS
        }
    ).round(1)
    _rates.index.name = "DNA content group"
    flags_by_dna_group = _rates.reindex([g for g in ["sub-G1", "G1", "between", "G2/M", ">G2"] if g in _rates.index])
    _g2_rate = flags_by_dna_group.loc["G2/M", "under_segmented_cells"] if "G2/M" in flags_by_dna_group.index else float("nan")
    mo.vstack(
        [
            mo.md("**% of cells flagged by each condition, per DNA-content group**"),
            flags_by_dna_group,
            mo.callout(
                mo.md(
                    f"`under_segmented_cells` flags **{_g2_rate:.0f} %** of G2/M cells. Above ~25 %, "
                    "consider turning it off under Advanced settings (it removes healthy dividing cells), "
                    "after checking the crops above."
                ),
                kind="warn",
            ) if _g2_rate > 25 else mo.md(""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 — Is QC removing treatment biology?

    **Question:** do treated wells lose many more cells to QC than negative
    controls? QC should remove technical errors, which hit all conditions
    alike. A treatment with a much higher fail rate than controls may have a
    real phenotype (large, irregular or multinucleated nuclei) that coSMicQC
    reads as a segmentation error. Look at its crops before trusting the flags.

    **Reading the table.** Rates are per cell, pooled over wells. Cells in a
    well are not independent, so the table is descriptive (no p-values).
    `ratio` = fail rate / negative-control fail rate on the same plate.
    Rows with ratio ≥ 2 and ≥ 500 cells are marked `check`.
    """)
    return


@app.cell
def _(
    CONFIG,
    HAS_CONTROLS,
    cell_status,
    cells_annotated,
    mo,
    np,
    pd,
    qc_cells,
    site_flags,
):
    mo.stop(not HAS_CONTROLS, mo.md("_No platemap control labels — comparison skipped._"))
    _image_fail = (
        qc_cells[["Metadata_Plate", "Metadata_Well", "Metadata_Site"]]
        .merge(site_flags[["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_image_qc"]], how="left")
        ["Metadata_image_qc"].eq("fail").to_numpy()
    )
    _table = pd.DataFrame(
        {
            "Metadata_Plate": qc_cells["Metadata_Plate"].to_numpy(),
            "treatment": cells_annotated[CONFIG.treatment_col].astype(str).to_numpy()
            if CONFIG.treatment_col in cells_annotated else "unknown",
            "is_negcon": cells_annotated[CONFIG.control_type_col].astype(str).str.lower()
            .isin([v.lower() for v in CONFIG.negcon_values]).to_numpy(),
            "profile_fail": cell_status["Metadata_profile_qc"].eq("fail").to_numpy(),
            "image_fail": _image_fail,
        }
    )
    _rates = (
        _table.groupby(["Metadata_Plate", "treatment", "is_negcon"])
        .agg(cells=("profile_fail", "size"), profile_fail_pct=("profile_fail", "mean"), image_fail_pct=("image_fail", "mean"))
        .reset_index()
    )
    _rates[["profile_fail_pct", "image_fail_pct"]] *= 100
    _negcon = (
        _table[_table["is_negcon"]].groupby("Metadata_Plate")["profile_fail"].mean().mul(100).rename("negcon_profile_fail_pct")
    )
    treatment_flag_rates = _rates.merge(_negcon, on="Metadata_Plate", how="left")
    treatment_flag_rates["ratio"] = treatment_flag_rates["profile_fail_pct"] / treatment_flag_rates["negcon_profile_fail_pct"].replace(0, np.nan)
    treatment_flag_rates["check"] = (treatment_flag_rates["ratio"] >= 2) & (treatment_flag_rates["cells"] >= 500)
    treatment_flag_rates = treatment_flag_rates.round(2).sort_values(["check", "ratio"], ascending=False)
    treatment_flag_rates
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 — Write flag tables and exclusion lists

    | File | Rows | Used by |
    |---|---|---|
    | `site_qc.csv` | every site: metrics, flags, reason | review |
    | `excluded_sites.csv` | sites failing image QC (Plate/Well/Site + reason) | NB01 |
    | `cell_qc.parquet` | every cell: keys, coSMicQC flags, profile/image status, DNA group | review, later notebooks |
    | `excluded_cells.parquet` | cells failing profile QC (keys + reason) | NB01 |
    | `dna_content_peaks.csv`, `dna_content_per_well.csv` | per plate / per well | review |

    Existing files are protected: a file whose content differs is not
    replaced unless **Overwrite existing outputs** is on.
    """)
    return


@app.cell
def _(
    OVERWRITE,
    RESULTS_DIR,
    SITE_KEYS,
    cell_flags,
    cell_status,
    dna_labels,
    dna_peaks,
    dna_per_well,
    pd,
    print,
    qc_cells,
    site_flags,
    write_parquet_protected,
    write_summary_table_protected,
):
    CELL_KEYS = [c for c in SITE_KEYS + ["ImageNumber", "Cells_Number_Object_Number"] if c in qc_cells.columns]
    cell_qc = pd.concat(
        [qc_cells[CELL_KEYS], cell_flags, cell_status, dna_labels.rename("Metadata_dna_content_group")], axis=1
    ).merge(site_flags[SITE_KEYS + ["Metadata_image_qc", "Metadata_image_qc_reason"]], on=SITE_KEYS, how="left")
    excluded_sites = site_flags.loc[site_flags["Metadata_image_qc"].eq("fail"), SITE_KEYS + ["Metadata_image_qc_reason"]]
    excluded_cells = cell_qc.loc[cell_qc["Metadata_profile_qc"].eq("fail"), CELL_KEYS + ["Metadata_profile_qc_reason"]]

    OUTPUT_PATHS = {
        "site_qc": RESULTS_DIR / "site_qc.csv",
        "excluded_sites": RESULTS_DIR / "excluded_sites.csv",
        "cell_qc": RESULTS_DIR / "cell_qc.parquet",
        "excluded_cells": RESULTS_DIR / "excluded_cells.parquet",
        "dna_content_peaks": RESULTS_DIR / "dna_content_peaks.csv",
        "dna_content_per_well": RESULTS_DIR / "dna_content_per_well.csv",
    }
    _status = {
        "site_qc": write_summary_table_protected(site_flags, OUTPUT_PATHS["site_qc"], overwrite=OVERWRITE),
        "excluded_sites": write_summary_table_protected(excluded_sites, OUTPUT_PATHS["excluded_sites"], overwrite=OVERWRITE),
        "cell_qc": write_parquet_protected(cell_qc, OUTPUT_PATHS["cell_qc"], overwrite=OVERWRITE),
        "excluded_cells": write_parquet_protected(excluded_cells, OUTPUT_PATHS["excluded_cells"], overwrite=OVERWRITE),
        "dna_content_peaks": write_summary_table_protected(dna_peaks, OUTPUT_PATHS["dna_content_peaks"], overwrite=OVERWRITE),
        "dna_content_per_well": write_summary_table_protected(dna_per_well, OUTPUT_PATHS["dna_content_per_well"], overwrite=OVERWRITE),
    }
    for _name, _state in _status.items():
        print(f"  {_state:>9}  {OUTPUT_PATHS[_name]}")
    if "Cells_Number_Object_Number" not in CELL_KEYS:
        print("⚠️  No Cells_Number_Object_Number column: NB01 cannot match single cells, only sites.")

    _image_fail = cell_qc["Metadata_image_qc"].eq("fail")
    _profile_fail = cell_qc["Metadata_profile_qc"].eq("fail")
    removal_summary = pd.DataFrame(
        {
            "cells": [len(cell_qc)],
            "image QC fail": [int(_image_fail.sum())],
            "profile QC fail": [int(_profile_fail.sum())],
            "either (removed by NB01)": [int((_image_fail | _profile_fail).sum())],
            "kept %": [round(100 * (1 - (_image_fail | _profile_fail).mean()), 1)],
        }
    )
    removal_summary
    return (
        CELL_KEYS,
        OUTPUT_PATHS,
        cell_qc,
        excluded_cells,
        excluded_sites,
        removal_summary,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 — Provenance

    Schema-v2 provenance record: inputs and outputs with SHA-256, settings,
    software versions and git state. Read-only git commands only.
    """)
    return


@app.cell
def _(
    ACTIVE_CONDITIONS,
    BLUR_Z_LIMIT,
    DNA_CHANNEL,
    EXPERIMENT_ID,
    IMAGE_CHANNELS,
    IMAGE_QC_SOURCE,
    INPUT_PATHS,
    OUTPUT_PATHS,
    OVERWRITE,
    PIXEL_SIZE_UM,
    PROFILE_CONDITIONS,
    RANDOM_STATE,
    REPO_ROOT,
    RESULTS_DIR,
    SATURATION_PCT_LIMIT,
    datetime,
    np,
    pd,
    platform,
    print,
    removal_summary,
    timezone,
):
    import cosmicqc
    from hca_pipeline.provenance import canonicalize_provenance, provenance_json

    _record = canonicalize_provenance(
        {
            "pipeline": {"executed_at_utc": datetime.now(timezone.utc).isoformat()},
            "configuration": {
                "image_qc_source": IMAGE_QC_SOURCE,
                "image_channels": IMAGE_CHANNELS,
                "blur_robust_z_limit": BLUR_Z_LIMIT,
                "saturation_pct_limit": SATURATION_PCT_LIMIT,
                "dna_channel": DNA_CHANNEL,
                "cosmicqc_conditions": PROFILE_CONDITIONS,
                "active_conditions": ACTIVE_CONDITIONS,
                "random_state": RANDOM_STATE,
                "figure_style": "publication-figures (hca_pipeline.pubfig, generic)",
                "pixel_size_um": PIXEL_SIZE_UM,
            },
            "dataset": removal_summary.iloc[0].to_dict(),
            "environment": {
                "python_version": platform.python_version(),
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "cosmicqc_version": getattr(cosmicqc, "__version__", "unknown"),
            },
        },
        notebook="00_data_qc.py",
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
        dependencies=INPUT_PATHS,
        outputs=list(OUTPUT_PATHS.values()),
    )
    provenance_path = RESULTS_DIR / "provenance_nb00_latest.json"
    if provenance_path.exists() and not OVERWRITE:
        print(f"  preserved  {provenance_path} (enable overwrite to replace)")
    else:
        provenance_path.write_text(provenance_json(_record), encoding="utf-8")
        print(f"✓ Provenance written: {provenance_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 — Final integrity checks
    """)
    return


@app.cell
def _(
    CELL_KEYS,
    SITE_KEYS,
    cell_qc,
    excluded_cells,
    excluded_sites,
    print,
    qc_cells,
    site_flags,
):
    _checks = [
        ("one QC row per loaded cell", len(cell_qc) == len(qc_cells)),
        ("one row per site in site_qc", not site_flags.duplicated(SITE_KEYS).any()),
        ("cell keys are unique", not cell_qc.duplicated(CELL_KEYS).any()),
        ("every cell has an image-QC status", cell_qc["Metadata_image_qc"].notna().all()),
        ("excluded sites ⊆ site table", len(excluded_sites.merge(site_flags[SITE_KEYS])) == len(excluded_sites)),
        ("excluded cells all fail profile QC", len(excluded_cells) == int(cell_qc["Metadata_profile_qc"].eq("fail").sum())),
    ]
    for _name, _ok in _checks:
        print(f"  {'PASS' if _ok else 'FAIL'}  {_name}")
    if not all(ok for _, ok in _checks):
        raise AssertionError("NB00 integrity checks failed — see the list above.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notebook complete

    This notebook flagged, without changing any data:

    1. **Sites** with a blurred or saturated channel (image QC).
    2. **Cells** with likely segmentation errors (coSMicQC, per plate).
    3. **DNA content** per cell, plate and well, and whether the nuclear stain
       shows a G2/M peak at ~2× G1.

    NB01 removes the sites in `excluded_sites.csv` and the cells in
    `excluded_cells.parquet`. DNA-content groups are descriptive only.

    ### Key outputs
    - `results/data_qc/excluded_sites.csv`, `excluded_cells.parquet` — read by NB01
    - `results/data_qc/site_qc.csv`, `cell_qc.parquet` — full flag tables
    - `results/data_qc/dna_content_peaks.csv`, `dna_content_per_well.csv` — cell-cycle check
    - `figures/data_qc/*.pdf`, `*.png` — distributions, plate maps, montages, crops, DNA histograms
      (paper versions); `*_explained.png` — the same with headline and how-to-read line
    - `results/data_qc/provenance_nb00_latest.json` — run provenance
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Save the analysis record

    Save the notebook's **current session** without rerunning cells:

    - **HTML** preserves code and the complete rich-output record;
    - **PDF** provides a clean reading copy without code inputs.

    Chromium can write both files directly to the experiment's `reports/`
    folder after authorization. Safari shows separate HTML and PDF download
    buttons because it cannot write directly to a selected folder.
    """)
    return


@app.cell
def _(EXPERIMENT_ID, mo):
    from hca_pipeline.report_export import SessionReportSaver

    _report_saver = SessionReportSaver(
        basename=f"{EXPERIMENT_ID}_00_data_qc",
        suggested_directory=f"workspace/analysis/{EXPERIMENT_ID}/reports",
    )
    mo.ui.anywidget(_report_saver)
    return


if __name__ == "__main__":
    app.run()
