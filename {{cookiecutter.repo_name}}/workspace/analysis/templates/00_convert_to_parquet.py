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
    # 00 — Convert CellProfiler output to Parquet (CytoTable)

    **Pipeline step:** 0 (first step)
    **Position in pipeline:** CellProfiler output → **NB00 (convert to Parquet)** → NB01 (data quality control)
    **Purpose:** turn the raw CellProfiler output into Parquet files with one row per cell, using
    [CytoTable](https://github.com/cytomining/CytoTable). NB01 reads these files.

    > This notebook **converts only**. It does no quality control (QC) and no normalisation,
    > and it does not change the raw files.

    **Input:** raw CellProfiler output in `workspace/backend/<EXPERIMENT_ID>/`. The notebook also
    searches subfolders, so each plate can have its own folder.

    | CellProfiler module | Files | CytoTable preset | Output |
    |---|---|---|---|
    | ExportToDatabase | `<name>.sqlite` (tables `Per_Image`, `Per_Cells`, `Per_Cytoplasm`, `Per_Nuclei`) | `cellprofiler_sqlite` | `<name>.parquet` |
    | ExportToSpreadsheet | a folder with `Image.csv`, `Cells.csv`, `Cytoplasm.csv`, `Nuclei.csv` | `cellprofiler_csv` | `<folder>.parquet` |

    Each SQLite file or CSV folder is one CellProfiler *run* (usually one plate) and becomes one
    Parquet file. If the folder holds both SQLite and CSV files, only the SQLite files are converted.

    **Output:** `analysis/<EXPERIMENT_ID>/outputs/parquet/<name>.parquet`

    ## Execution logic

    1. Choose the experiment.
    2. Convert with one call to `convert_to_parquet` (CytoTable).
    3. Write provenance.
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
    from datetime import datetime, timezone
    from pathlib import Path
    import platform

    import pandas as pd
    import pyarrow.parquet as pq

    return Path, datetime, pd, platform, pq, timezone


@app.cell
def _(Path, mo):
    # Find the repo root from this file (not the working directory), then make hca_pipeline importable.
    import sys

    _notebook_path = Path(__file__).resolve()
    REPO_ROOT = next(
        (p for p in _notebook_path.parents if (p / "pixi.toml").exists() or (p / ".git").exists()), None
    )
    if REPO_ROOT is None:
        raise FileNotFoundError(f"Could not find the repo root (pixi.toml or .git) above {_notebook_path}.")
    sys.path.insert(0, str(REPO_ROOT / "workspace"))

    from hca_pipeline.config import ExperimentConfig
    from hca_pipeline.cytotable_convert import convert_to_parquet, parquet_dir

    mo.callout(mo.md(f"✅ Loaded `hca_pipeline` from `{REPO_ROOT / 'workspace'}`."), kind="success")
    return ExperimentConfig, REPO_ROOT, convert_to_parquet, parquet_dir


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Pick the experiment folder under `workspace/backend/`. An existing Parquet file is kept
    unless **Overwrite existing outputs** is on. Turn it on after you replace the raw files.
    """)
    return


@app.cell
def _(REPO_ROOT, mo):
    _backend_dir = REPO_ROOT / "workspace" / "backend"
    _ids = sorted(p.name for p in _backend_dir.iterdir() if p.is_dir()) if _backend_dir.is_dir() else []
    experiment_id_input = mo.ui.dropdown(
        options=_ids or ["SET_EXPERIMENT_ID_HERE"],
        value=_ids[0] if _ids else "SET_EXPERIMENT_ID_HERE",
        label="Experiment ID",
    )
    experiment_id_input
    return (experiment_id_input,)


@app.cell
def _(ExperimentConfig, REPO_ROOT, experiment_id_input, mo):
    _config = ExperimentConfig.load(REPO_ROOT, experiment_id_input.value)
    overwrite_input = mo.ui.checkbox(value=_config.overwrite_existing_outputs, label="Overwrite existing outputs")
    overwrite_input
    return (overwrite_input,)


@app.cell
def _(REPO_ROOT, experiment_id_input, mo, overwrite_input, parquet_dir):
    EXPERIMENT_ID = experiment_id_input.value
    OVERWRITE = bool(overwrite_input.value)
    DB_ROOT = REPO_ROOT / "workspace" / "backend" / EXPERIMENT_ID
    ANALYSIS_DIR = REPO_ROOT / "workspace" / "analysis" / EXPERIMENT_ID
    PARQUET_DIR = parquet_dir(ANALYSIS_DIR)  # the folder NB01 reads
    RESULTS_DIR = ANALYSIS_DIR / "results" / "parquet_conversion"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mo.callout(
        mo.md(
            f"""
            | Setting | Value |
            |---|---|
            | Raw CellProfiler input | `{DB_ROOT}` |
            | Parquet output (read by NB01) | `{PARQUET_DIR}` |
            | Overwrite existing outputs | {OVERWRITE} |
            """
        ),
        kind="info",
    )
    return DB_ROOT, EXPERIMENT_ID, OVERWRITE, PARQUET_DIR, RESULTS_DIR


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Convert with CytoTable

    CytoTable uses a *preset*: a saved set of settings for one data layout. There is one preset
    for each CellProfiler export (see the table above). CytoTable reads the Image, Cells, Cytoplasm
    and Nuclei tables in chunks of 10,000 rows. It then *joins* them: it combines the four tables
    into one row per cell. A cell is kept when its cytoplasm object has a parent cell, a parent
    nucleus and an image.

    This notebook changes one preset setting: **the join**. The preset join keeps only the image
    file names from the Image table (the SQLite preset also keeps well and plate). Data QC also
    needs plate, well and site, `Image_ImageQuality_*` (blur and saturation) and `Image_PathName_*`.
    The join used here therefore keeps every image column.

    Column names are the ones CytoTable writes. NB01 renames the plate, well and site columns when
    it loads the files.
    """)
    return


@app.cell
def _(DB_ROOT, OVERWRITE, PARQUET_DIR, convert_to_parquet, mo, print):
    PARQUET_FILES = convert_to_parquet(DB_ROOT, PARQUET_DIR, overwrite=OVERWRITE, progress=print)

    mo.stop(
        not PARQUET_FILES,
        mo.callout(
            mo.md(f"**Nothing to convert:** no `.sqlite` file or `Image.csv` from CellProfiler under `{DB_ROOT}`."),
            kind="warn",
        ),
    )
    return (PARQUET_FILES,)


@app.cell
def _(PARQUET_FILES, pd, pq):
    pd.DataFrame(
        [
            {
                "file": _p.name,
                "cells": pq.ParquetFile(_p).metadata.num_rows,
                "columns": pq.ParquetFile(_p).metadata.num_columns,
                "size_mb": round(_p.stat().st_size / 1e6, 1),
            }
            for _p in PARQUET_FILES
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Provenance

    Raw inputs and Parquet outputs with SHA-256, software versions and git state.
    """)
    return


@app.cell
def _(
    DB_ROOT,
    EXPERIMENT_ID,
    OVERWRITE,
    PARQUET_FILES,
    REPO_ROOT,
    RESULTS_DIR,
    datetime,
    platform,
    print,
    timezone,
):
    import cytotable
    import duckdb
    from hca_pipeline.provenance import canonicalize_provenance, provenance_json

    _raw = sorted(p for p in DB_ROOT.rglob("*") if p.suffix.lower() in {".sqlite", ".csv"})
    _record = canonicalize_provenance(
        {
            "pipeline": {"executed_at_utc": datetime.now(timezone.utc).isoformat()},
            "configuration": {"converter": "CytoTable", "overwrite_existing_outputs": OVERWRITE},
            "environment": {
                "python_version": platform.python_version(),
                "cytotable_version": getattr(cytotable, "__version__", "unknown"),
                "duckdb_version": duckdb.__version__,
            },
        },
        notebook="00_convert_to_parquet.py",
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
        dependencies=_raw,
        outputs=PARQUET_FILES,
    )
    _path = RESULTS_DIR / "provenance_nb00_latest.json"
    _path.write_text(provenance_json(_record), encoding="utf-8")
    print(f"✓ Provenance written: {_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notebook complete

    The raw CellProfiler output is now Parquet, one row per cell, in
    `analysis/<EXPERIMENT_ID>/outputs/parquet/`. Next: open NB01 (data QC).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Save the analysis record

    Save the notebook's **current session** as HTML (code and outputs) and PDF (reading copy)
    to the experiment's `reports/` folder.
    """)
    return


@app.cell
def _(EXPERIMENT_ID, mo):
    from hca_pipeline.report_export import SessionReportSaver

    _report_saver = SessionReportSaver(
        basename=f"{EXPERIMENT_ID}_00_convert_to_parquet",
        suggested_directory=f"workspace/analysis/{EXPERIMENT_ID}/reports",
    )
    mo.ui.anywidget(_report_saver)
    return


if __name__ == "__main__":
    app.run()
