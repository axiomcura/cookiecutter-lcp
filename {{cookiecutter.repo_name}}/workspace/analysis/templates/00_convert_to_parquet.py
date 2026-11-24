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
    **Position in pipeline:** CellProfiler output → **NB00 (convert to Parquet)** → NB01 (data QC)
    **Purpose:** turn the raw CellProfiler output into one Parquet file per plate (or per
    input) with one row per cell, using [CytoTable](https://github.com/cytomining/CytoTable).

    > This notebook **converts only**. It never edits the raw files and does no QC,
    > normalisation or renaming of columns. Data QC is NB01.

    **Input:** `workspace/backend/<EXPERIMENT_ID>/`, one of:

    | What is in the folder | Output |
    |---|---|
    | CellProfiler SQLite database(s), one per plate | one Parquet per database, named after it |
    | The four tables `Per_Image.csv`, `Per_Cells.csv`, `Per_Cytoplasm.csv`, `Per_Nuclei.csv` | one Parquet, named after the experiment |
    | **One** merged single-cell CSV | one Parquet, named after the CSV |

    **Outputs:**

    - `analysis/<EXPERIMENT_ID>/outputs/parquet/<name>.parquet` — read by NB01
    - `analysis/<EXPERIMENT_ID>/results/parquet_conversion/conversion_summary.csv`,
      `input_checks.csv`, `provenance_nb00_latest.json`

    ## Execution logic

    1. Choose the experiment; set overwrite, threads and chunk size if needed.
    2. Find the raw input and list what will be converted.
    3. Convert with CytoTable (existing Parquet is kept unless overwrite is on).
    4. Check the Parquet files: what NB01 will find, and whether cells are unique.
    5. Write the summary tables and provenance; run integrity checks.
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

    pd.set_option("display.max_columns", 80)
    return Path, datetime, pd, platform, pq, timezone


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
    from hca_pipeline.cytotable_convert import (
        convert_raw_input,
        count_source_cells,
        detect_raw_inputs,
        inventory,
        is_stale,
        parquet_dir,
        read_qc_columns,
    )
    from hca_pipeline.data_qc import (
        SITE_KEYS,
        cellprofiler_quality_channels,
        profile_qc_conditions,
        resolve_dna_channel,
    )
    from hca_pipeline.io import write_summary_table_protected
    from hca_pipeline.metadata import ensure_core_metadata

    mo.callout(mo.md(f"✅ Loaded `hca_pipeline` from `{_pipelines_dir}`."), kind="success")
    return (
        ExperimentConfig,
        REPO_ROOT,
        SITE_KEYS,
        cellprofiler_quality_channels,
        convert_raw_input,
        count_source_cells,
        detect_raw_inputs,
        ensure_core_metadata,
        inventory,
        is_stale,
        parquet_dir,
        profile_qc_conditions,
        read_qc_columns,
        resolve_dna_channel,
        write_summary_table_protected,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — Experiment configuration

    Pick the experiment folder under `workspace/backend/`. Overwrite, threads and
    chunk size are the only settings.

    - **Overwrite:** an existing Parquet file is kept unless this is on. If the raw file
      is newer than the Parquet file, section 3 warns you to turn it on.
    - **Threads, chunk size** (under *Advanced*): CytoTable reads each table in chunks of
      this many rows. The defaults suit most plates; raise the chunk size for very large
      tables if memory allows.
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
    threads_input = mo.ui.number(start=1, stop=16, step=1, value=4, label="CytoTable threads")
    chunk_input = mo.ui.number(
        start=1000, stop=1_000_000, step=1000, value=10_000, label="Rows per chunk (CytoTable)"
    )
    mo.vstack(
        [
            overwrite_input,
            mo.accordion(
                {"Advanced settings: CytoTable performance (the defaults are fine for most plates)": mo.vstack(
                    [threads_input, chunk_input]
                )}
            ),
        ]
    )
    return chunk_input, overwrite_input, threads_input


@app.cell
def _(
    REPO_ROOT,
    chunk_input,
    experiment_id_input,
    mo,
    overwrite_input,
    parquet_dir,
    threads_input,
):
    EXPERIMENT_ID = experiment_id_input.value
    OVERWRITE = bool(overwrite_input.value)
    MAX_THREADS = int(threads_input.value)
    CHUNK_SIZE = int(chunk_input.value)

    WORKSPACE_DIR = REPO_ROOT / "workspace"
    DB_ROOT = WORKSPACE_DIR / "backend" / EXPERIMENT_ID
    ANALYSIS_DIR = WORKSPACE_DIR / "analysis" / EXPERIMENT_ID
    PARQUET_DIR = parquet_dir(ANALYSIS_DIR)  # the folder NB01 reads
    RESULTS_DIR = ANALYSIS_DIR / "results" / "parquet_conversion"
    for _directory in (PARQUET_DIR, RESULTS_DIR):
        _directory.mkdir(parents=True, exist_ok=True)

    mo.callout(
        mo.md(
            f"""
            ### Effective settings

            | Setting | Value |
            |---|---|
            | Experiment | `{EXPERIMENT_ID}` |
            | Raw CellProfiler input | `{DB_ROOT}` |
            | Parquet output (read by NB01) | `{PARQUET_DIR}` |
            | Summary tables | `{RESULTS_DIR}` |
            | Threads · rows per chunk | {MAX_THREADS} · {CHUNK_SIZE:,} |
            | Overwrite existing outputs | {OVERWRITE} |
            """
        ),
        kind="info",
    )
    return (
        ANALYSIS_DIR,
        CHUNK_SIZE,
        DB_ROOT,
        EXPERIMENT_ID,
        MAX_THREADS,
        OVERWRITE,
        PARQUET_DIR,
        RESULTS_DIR,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 — Find the raw input

    SQLite databases win over CSV files. CSV files must be the four `Per_*` tables or
    exactly one merged CSV; any other mix is refused, because converting unrelated CSV
    files together would mix their columns. A merged CSV needs a numeric `ImageNumber`
    column (CellProfiler writes it) so it can be read in chunks.
    """)
    return


@app.cell
def _(DB_ROOT, detect_raw_inputs, mo, pd):
    try:
        RAW_INPUTS = detect_raw_inputs(DB_ROOT)
    except ValueError as _error:
        mo.stop(True, mo.callout(mo.md(f"**NB00 stopped:** {_error}"), kind="danger"))
    mo.stop(
        not RAW_INPUTS,
        mo.callout(
            mo.md(
                f"**NB00 stopped:** no CellProfiler SQLite database or CSV file in `{DB_ROOT}`. "
                "Copy the CellProfiler output there and rerun this cell."
            ),
            kind="warn",
        ),
    )
    raw_inputs_table = pd.DataFrame(
        {
            "input": [r.name for r in RAW_INPUTS],
            "kind": [r.kind for r in RAW_INPUTS],
            "files": [", ".join(p.name for p in r.paths) for r in RAW_INPUTS],
            "size_mb": [round(sum(p.stat().st_size for p in r.paths) / 1e6, 2) for r in RAW_INPUTS],
        }
    )
    raw_inputs_table
    return RAW_INPUTS, raw_inputs_table


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 — Convert with CytoTable

    **What:** each input becomes one Parquet file with one row per cell. For SQLite and
    `Per_*` tables, CytoTable reads Image, Cells, Cytoplasm and Nuclei in chunks and joins
    them. A cell is kept when its cytoplasm object has a parent cell, a parent nucleus
    and an image (the same rule as the readers NB04 uses). A merged CSV is converted
    as it is.

    **Reason:** Parquet is typed and columnar, so NB01 can read only the few hundred
    columns QC needs instead of the whole table, and it does so in seconds. CytoTable
    does the chunked reading and the join for CellProfiler's table layout.

    **Choices made here:**

    - CytoTable's built-in join keeps only `Image_FileName_*` from the image table.
      Data QC also needs `Image_ImageQuality_*` (blur, saturation), `Image_PathName_*` and
      `Image_Count_*`, so the join here keeps **every** image column.
    - Column names are **not** changed (`ImageNumber`, `Image_Metadata_Plate`, ...);
      NB01 applies the usual Plate/Well/Site renaming when it loads.
    - CytoTable runs in threads, not subprocesses (works the same on Windows).

    **Assumptions:** object numbers in the Cells, Nuclei and Cytoplasm tables are
    unique within an image, and every database or table set holds whole plates (cells of
    one plate are not split across inputs).
    """)
    return


@app.cell
def _(
    CHUNK_SIZE,
    MAX_THREADS,
    OVERWRITE,
    PARQUET_DIR,
    RAW_INPUTS,
    convert_raw_input,
    count_source_cells,
    is_stale,
    pd,
    print,
):
    _rows = []
    for _raw in RAW_INPUTS:
        print(f"[{_raw.kind}] {_raw.name} …")
        _result = convert_raw_input(
            _raw, PARQUET_DIR, overwrite=OVERWRITE, chunk_size=CHUNK_SIZE, max_threads=MAX_THREADS
        )
        _stale = _result.status == "preserved" and is_stale(_result.path, _raw.paths)
        print(f"  {_result.status:>9}  {_result.path.name}  ({_result.rows:,} cells, {_result.seconds:.1f} s)")
        if _stale:
            print("  ⚠️  The raw file is newer than this Parquet file. Turn on 'Overwrite existing outputs' "
                  "and rerun so NB01 does not read old data.")
        _rows.append(
            {
                "input": _raw.name,
                "kind": _raw.kind,
                "source_files": ", ".join(p.name for p in _raw.paths),
                "source_cells": count_source_cells(_raw),
                "parquet_file": _result.path.name,
                "cells": _result.rows,
                "status": _result.status,
                "stale": _stale,
            }
        )
    conversion_log = pd.DataFrame(_rows)
    PARQUET_FILES = [PARQUET_DIR / name for name in conversion_log["parquet_file"]]
    conversion_log
    return PARQUET_FILES, conversion_log


@app.cell
def _(conversion_log, mo):
    _dropped = (conversion_log["source_cells"] - conversion_log["cells"]).clip(lower=0)
    _total_dropped = int(_dropped.sum())
    mo.callout(
        mo.md(
            f"**{int(conversion_log['cells'].sum()):,} cells** in {len(conversion_log)} Parquet file(s). "
            + (
                f"The join dropped **{_total_dropped:,}** cytoplasm object(s) that have no parent cell, "
                "parent nucleus or image. A few are normal; many mean the Cells, Nuclei and Cytoplasm "
                "tables come from different CellProfiler runs."
                if _total_dropped
                else "No objects were dropped."
            )
            + (" ⚠️ Some files are older than their raw input (see above)." if conversion_log["stale"].any() else "")
        ),
        kind="warn" if (_total_dropped or conversion_log["stale"].any()) else "success",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 — Check the Parquet files: what NB01 will find

    **Question:** can the data-QC notebook use these files as they are?

    Each check reads only the Parquet footers or a handful of columns.

    | Result | Meaning |
    |---|---|
    | **PASS** | NB01 finds what it needs |
    | **WARN** | NB01 runs with less: it falls back or skips a figure |
    | **FAIL** | NB01 will stop; fix the input or the CellProfiler pipeline first |

    The most important check is that **every cell is unique** by plate, well, site, image
    and object number. If two plates share a `Metadata_Plate` value (for example two
    folders that are both called `…_Plate_1`), their cells collide, and QC flags would be
    attached to the wrong cells.
    """)
    return


@app.cell
def _(PARQUET_FILES, inventory):
    parquet_inventory = inventory(PARQUET_FILES)
    parquet_inventory
    return (parquet_inventory,)


@app.cell
def _(
    PARQUET_FILES,
    SITE_KEYS,
    cellprofiler_quality_channels,
    ensure_core_metadata,
    mo,
    pd,
    pq,
    profile_qc_conditions,
    read_qc_columns,
    resolve_dna_channel,
):
    _columns = sorted({name for _path in PARQUET_FILES for name in pq.read_schema(_path).names})
    _checks = []

    def _add(name, status, detail):
        _checks.append({"check": name, "result": status, "detail": detail})

    # Keys: Plate/Well/Site may still carry CellProfiler's Image_Metadata_ prefix; NB01 renames them.
    _key_columns = ["ImageNumber", "Cells_Number_Object_Number"]
    _probe = read_qc_columns(
        PARQUET_FILES,
        prefixes=(),
        exact=_key_columns + ["Image_Metadata_Plate", "Image_Metadata_Well", "Image_Metadata_Site"] + SITE_KEYS,
    )
    _probe = ensure_core_metadata(_probe)
    _missing_meta = [c for c in SITE_KEYS if c not in _probe.columns]
    _missing_keys = [c for c in _key_columns if c not in _probe.columns]
    _add(
        "Plate, well and site columns",
        "PASS" if not _missing_meta else "FAIL",
        "found" if not _missing_meta else f"missing {_missing_meta}: image QC works per site",
    )
    if _missing_meta or _missing_keys:
        _add("Cells are unique", "FAIL", f"cannot check; missing {_missing_meta + _missing_keys}")
    else:
        _duplicates = int(_probe.duplicated(SITE_KEYS + _key_columns).sum())
        _plates = _probe["Metadata_Plate"].astype(str).nunique()
        _add(
            "Cells are unique (plate, well, site, image, object)",
            "PASS" if not _duplicates else "FAIL",
            f"{len(_probe):,} cells, {_plates} plate(s)"
            if not _duplicates
            else f"{_duplicates:,} repeated cells: check that each plate has its own Metadata_Plate value",
        )

    try:
        _dna = resolve_dna_channel(_columns)
        _add("Nuclear stain for DNA features", "PASS", f"{_dna} (Hoechst, DNA, DAPI or AOGFP, in that order)")
        _needed = sorted({f for t in profile_qc_conditions(_dna).values() for f in t})
        _absent = [f for f in _needed if f not in _columns]
        _add(
            "coSMicQC features",
            "PASS" if not _absent else "FAIL",
            f"{len(_needed)} features found" if not _absent else f"missing {_absent}",
        )
        _add(
            "Image file names for cell crops",
            "PASS" if f"Image_FileName_{_dna}" in _columns else "WARN",
            f"Image_FileName_{_dna}" if f"Image_FileName_{_dna}" in _columns else "no crops without file names",
        )
    except ValueError as _error:
        _add("Nuclear stain for DNA features", "FAIL", str(_error))

    _channels = cellprofiler_quality_channels(_columns)
    _add(
        "CellProfiler image quality",
        "PASS" if _channels else "WARN",
        f"channels: {', '.join(_channels)}"
        if _channels
        else "no Image_ImageQuality_* columns: NB01 measures blur and saturation from the raw images "
        "(slower) or skips image QC. Add MeasureImageQuality to the CellProfiler pipeline.",
    )
    _position = [c for c in ("Nuclei_Location_Center_X", "Nuclei_AreaShape_Center_X") if c in _columns]
    _add(
        "Nucleus position for cell crops",
        "PASS" if _position else "WARN",
        _position[0] if _position else "no crops without a nucleus position",
    )

    input_checks = pd.DataFrame(_checks)
    _worst = "FAIL" if (input_checks["result"] == "FAIL").any() else ("WARN" if (input_checks["result"] == "WARN").any() else "PASS")
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    {"PASS": "**All checks pass.** NB01 can read these files.",
                     "WARN": "**NB01 will run with reduced features** (see WARN rows).",
                     "FAIL": "**NB01 will stop.** Fix the FAIL rows first."}[_worst]
                ),
                kind={"PASS": "success", "WARN": "warn", "FAIL": "danger"}[_worst],
            ),
            input_checks,
        ]
    )
    return (input_checks,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 — Write the summary tables

    | File | Rows | Used by |
    |---|---|---|
    | `conversion_summary.csv` | one per input: raw files, raw cells, Parquet file, cells, columns | review |
    | `input_checks.csv` | the checks of section 4 | review |

    The summary holds only values that do not change between runs (no timings), so a rerun
    without overwrite does not fail on a harmless difference. Existing files are protected: a file
    whose content differs is not replaced unless **Overwrite existing outputs** is on.
    """)
    return


@app.cell
def _(
    OVERWRITE,
    RESULTS_DIR,
    conversion_log,
    input_checks,
    parquet_inventory,
    print,
    write_summary_table_protected,
):
    conversion_summary = conversion_log.drop(columns=["status", "stale"]).merge(
        parquet_inventory.rename(columns={"file": "parquet_file", "cells": "cells_check"}).drop(columns=["size_mb"]),
        on="parquet_file",
    ).drop(columns=["cells_check"])
    SUMMARY_PATHS = {
        "conversion_summary": RESULTS_DIR / "conversion_summary.csv",
        "input_checks": RESULTS_DIR / "input_checks.csv",
    }
    _status = {
        "conversion_summary": write_summary_table_protected(
            conversion_summary, SUMMARY_PATHS["conversion_summary"], overwrite=OVERWRITE
        ),
        "input_checks": write_summary_table_protected(input_checks, SUMMARY_PATHS["input_checks"], overwrite=OVERWRITE),
    }
    for _name, _state in _status.items():
        print(f"  {_state:>9}  {SUMMARY_PATHS[_name]}")
    return SUMMARY_PATHS, conversion_summary


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 — Provenance

    Schema-v2 provenance record: raw inputs and outputs with SHA-256, settings, software
    versions and git state. Read-only git commands only.
    """)
    return


@app.cell
def _(
    CHUNK_SIZE,
    EXPERIMENT_ID,
    MAX_THREADS,
    OVERWRITE,
    PARQUET_FILES,
    RAW_INPUTS,
    REPO_ROOT,
    RESULTS_DIR,
    SUMMARY_PATHS,
    conversion_summary,
    datetime,
    pd,
    platform,
    print,
    timezone,
):
    import cytotable
    import duckdb
    import pyarrow
    from hca_pipeline.provenance import canonicalize_provenance, provenance_json

    _record = canonicalize_provenance(
        {
            "pipeline": {"executed_at_utc": datetime.now(timezone.utc).isoformat()},
            "configuration": {
                "converter": "CytoTable",
                "input_kinds": sorted({r.kind for r in RAW_INPUTS}),
                "rows_per_chunk": CHUNK_SIZE,
                "threads": MAX_THREADS,
                "join": "inner: cytoplasm + cells + nuclei + image (all image columns kept)",
                "overwrite_existing_outputs": OVERWRITE,
            },
            "dataset": {
                "inputs": int(len(conversion_summary)),
                "cells": int(conversion_summary["cells"].sum()),
                "source_cells": int(conversion_summary["source_cells"].sum()),
            },
            "environment": {
                "python_version": platform.python_version(),
                "pandas_version": pd.__version__,
                "pyarrow_version": pyarrow.__version__,
                "cytotable_version": getattr(cytotable, "__version__", "unknown"),
                "duckdb_version": duckdb.__version__,
            },
        },
        notebook="00_convert_to_parquet.py",
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
        dependencies=[p for r in RAW_INPUTS for p in r.paths],
        outputs=[*PARQUET_FILES, *SUMMARY_PATHS.values()],
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
    ## 7 — Final integrity checks
    """)
    return


@app.cell
def _(PARQUET_DIR, PARQUET_FILES, conversion_summary, parquet_inventory, print):
    _checks = [
        ("every Parquet file exists", all(p.is_file() for p in PARQUET_FILES)),
        ("every Parquet file has cells", bool((parquet_inventory["cells"] > 0).all())),
        ("Parquet cells ≤ raw cells (the join only drops objects)",
         bool((conversion_summary["cells"] <= conversion_summary["source_cells"]).all())),
        ("no scratch folders left behind", not list(PARQUET_DIR.glob(".cytotable_work_*"))),
        ("all outputs are in the experiment's analysis folder", all(PARQUET_DIR in p.parents for p in PARQUET_FILES)),
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

    This notebook converted the raw CellProfiler output to Parquet, without changing any
    raw file or any column name:

    1. **Found** the input (SQLite, `Per_*` CSV tables or one merged CSV).
    2. **Converted** it with CytoTable to one row per cell, keeping all image columns.
    3. **Checked** that NB01 will find its columns and that every cell is unique.

    It did not do QC, normalise values or remove cells. Next: open NB01 (data QC).

    ### Key outputs
    - `analysis/<EXPERIMENT_ID>/outputs/parquet/*.parquet` — one row per cell; read by NB01
    - `results/parquet_conversion/conversion_summary.csv` — raw vs. converted cells per input
    - `results/parquet_conversion/input_checks.csv` — what NB01 will find (PASS / WARN / FAIL)
    - `results/parquet_conversion/provenance_nb00_latest.json` — run provenance
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
        basename=f"{EXPERIMENT_ID}_00_convert_to_parquet",
        suggested_directory=f"workspace/analysis/{EXPERIMENT_ID}/reports",
    )
    mo.ui.anywidget(_report_saver)
    return


if __name__ == "__main__":
    app.run()
