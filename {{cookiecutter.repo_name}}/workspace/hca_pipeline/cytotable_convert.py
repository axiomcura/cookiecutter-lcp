"""Convert CellProfiler output (SQLite, Per_* CSV tables, or one merged CSV) to Parquet with CytoTable.

The Parquet files hold one row per cell with the **same columns and values** as the pandas readers
in :mod:`hca_pipeline.io` (``read_cellprofiler_sqlite``, ``read_cellprofiler_csvs``). Column names
are not changed (``ImageNumber``, ``Image_Metadata_Plate``, ``Cells_Number_Object_Number``, ...),
so later notebooks can read only the columns they need.

Why a custom join: CytoTable's built-in CellProfiler join keeps only ``Image_FileName_*`` from the
image table. Data QC also needs ``Image_ImageQuality_*``, ``Image_PathName_*`` and
``Image_Count_*``, so the join here keeps every image column.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq

from .io import discover_single_cell_inputs

KINDS = ("sqlite", "csv_tables", "csv_merged")

# Same table aliases as hca_pipeline.io (logical name -> accepted table / file names, lower case).
TABLE_ALIASES = {
    "image": ("per_image", "image"),
    "cells": ("per_cells", "cells"),
    "cytoplasm": ("per_cytoplasm", "cytoplasm"),
    "nuclei": ("per_nuclei", "nuclei"),
}
COMPARTMENTS = ("cells", "nuclei", "cytoplasm")
PAGE_KEYS = {
    "image": "ImageNumber",
    "cells": "Cells_Number_Object_Number",
    "nuclei": "Nuclei_Number_Object_Number",
    "cytoplasm": "Cytoplasm_Number_Object_Number",
    "join": "Cytoplasm_Number_Object_Number",
}
# A merged table needs a numeric column to page through; the first one present is used.
MERGED_PAGE_KEYS = ("ImageNumber", "Cells_Number_Object_Number", "Nuclei_Number_Object_Number")


@dataclass(frozen=True)
class RawInput:
    """One thing to convert: a SQLite file, a set of Per_* CSV tables, or one merged CSV."""

    kind: str
    name: str  # output file stem
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class ConversionResult:
    path: Path
    status: str  # "converted", "replaced" or "preserved"
    rows: int
    seconds: float


def parquet_dir(analysis_dir: Path) -> Path:
    """Where the converted Parquet files live: ``analysis/<EXPERIMENT_ID>/outputs/parquet``."""
    return Path(analysis_dir) / "outputs" / "parquet"


# --------------------------------------------------------------------------- finding inputs
def _logical_table(stem: str) -> str | None:
    stem = stem.lower()
    return next((name for name, aliases in TABLE_ALIASES.items() if stem in aliases), None)


def detect_raw_inputs(backend_dir: Path) -> list[RawInput]:
    """Find what to convert in ``workspace/backend/<EXPERIMENT_ID>/``.

    SQLite files win over CSV files (as in the other notebooks). CSV files must be either the four
    CellProfiler tables (Per_Image, Per_Cells, Per_Cytoplasm, Per_Nuclei) or exactly one merged
    single-cell CSV; anything else is rejected rather than guessed.
    """
    found = discover_single_cell_inputs(Path(backend_dir))
    if found["sqlite"]:
        return [RawInput("sqlite", p.stem, (p,)) for p in found["sqlite"]]
    csvs = found["csv"]
    if not csvs:
        return []
    tables = {_logical_table(p.stem): p for p in csvs if _logical_table(p.stem)}
    if len(tables) == len(TABLE_ALIASES):
        return [RawInput("csv_tables", Path(backend_dir).name, tuple(tables[k] for k in TABLE_ALIASES))]
    if len(csvs) == 1:
        return [RawInput("csv_merged", csvs[0].stem, (csvs[0],))]
    raise ValueError(
        f"Cannot tell how to convert {[p.name for p in csvs]} in {backend_dir}. Expected either the four "
        "CellProfiler tables (Per_Image.csv, Per_Cells.csv, Per_Cytoplasm.csv, Per_Nuclei.csv) or exactly "
        "one merged CSV. Move the other CSV files out of this folder."
    )


# --------------------------------------------------------------------------- conversion
def _cytotable_config(max_threads: int, run_dir: Path):
    """Run CytoTable in threads: no subprocesses (works on Windows, no PATH lookup of parsl helpers).

    Parsl writes a ``runinfo`` folder into the current directory by default; ``run_dir`` sends it to
    the scratch folder that is deleted after the conversion.
    """
    from parsl.config import Config
    from parsl.executors.threads import ThreadPoolExecutor

    return Config(
        executors=[ThreadPoolExecutor(max_threads=max_threads, label="cytotable")], run_dir=str(run_dir)
    )


def _reset_parsl() -> None:
    """Drop a half-finished parsl session from an earlier failed call, so the next call can start."""
    import parsl

    try:
        parsl.dfk().cleanup()
    except Exception:  # no session loaded: nothing to clean
        pass
    try:
        parsl.clear()
    except Exception:
        pass


def _join_sql(stems: dict[str, str]) -> str:
    """Inner join of the four tables: one row per cell (same rule as ``io._read_cellprofiler_tables``).

    ``stems`` maps logical table -> CytoTable's Parquet file stem. The image key is renamed back
    from CytoTable's ``Metadata_ImageNumber`` to ``ImageNumber``.
    """
    for stem in stems.values():
        if not re.fullmatch(r"\w+", stem):
            raise ValueError(f"Unsupported table name {stem!r}: use letters, digits and underscores.")
    cyto, cells, nuclei, image = (stems[k] for k in ("cytoplasm", "cells", "nuclei", "image"))
    key = "Metadata_ImageNumber"
    return f"""
SELECT {cyto}.{key} AS ImageNumber,
       {cyto}.* EXCLUDE ({key}),
       {cells}.* EXCLUDE ({key}),
       {nuclei}.* EXCLUDE ({key}),
       {image}.* EXCLUDE ({key})
FROM read_parquet('{cyto}.parquet') AS {cyto}
JOIN read_parquet('{cells}.parquet') AS {cells}
  ON {cyto}.{key} = {cells}.{key} AND {cyto}.Cytoplasm_Parent_Cells = {cells}.Cells_Number_Object_Number
JOIN read_parquet('{nuclei}.parquet') AS {nuclei}
  ON {cyto}.{key} = {nuclei}.{key} AND {cyto}.Cytoplasm_Parent_Nuclei = {nuclei}.Nuclei_Number_Object_Number
JOIN read_parquet('{image}.parquet') AS {image}
  ON {cyto}.{key} = {image}.{key}
"""


def _stage(source: Path, target: Path) -> None:
    """Make ``target`` show the same bytes as ``source`` without copying when possible."""
    for make in (os.link, os.symlink):
        try:
            make(source, target)
            return
        except OSError:
            continue
    shutil.copy2(source, target)


def _sqlite_table_names(path: Path) -> dict[str, str]:
    """Logical table -> real table name in the SQLite file (accepts Per_Cells or Cells, any case)."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        available = {r[0].lower(): r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    names = {}
    for logical, aliases in TABLE_ALIASES.items():
        match = next((a for a in aliases if a in available), None)
        if match is None:
            raise ValueError(f"{path.name}: no table for '{logical}' (looked for {', '.join(aliases)}).")
        names[logical] = available[match]
    return names


def _sqlite_stems(path: Path) -> dict[str, str]:
    """CytoTable names each table's Parquet file after the lower-case table name."""
    return {logical: name.lower() for logical, name in _sqlite_table_names(path).items()}


def _convert_tables(source_path: Path, datatype: str, stems: dict[str, str], work: Path, chunk_size: int,
                    max_threads: int) -> Path:
    import cytotable

    out = cytotable.convert(
        source_path=str(source_path),
        dest_path=str(work / "joined.parquet"),
        dest_datatype="parquet",
        source_datatype=datatype,
        preset=None,
        compartments=COMPARTMENTS,
        metadata=("image",),
        identifying_columns=("ImageNumber",),
        page_keys=PAGE_KEYS,
        chunk_size=chunk_size,
        join=True,
        joins=_join_sql(stems),
        parsl_config=_cytotable_config(max_threads, work / "parsl_run"),
    )
    return Path(out)


def _convert_merged(csv_path: Path, work: Path, chunk_size: int, max_threads: int) -> Path:
    import cytotable

    header = list(pd.read_csv(csv_path, nrows=0).columns)
    page_key = next((c for c in MERGED_PAGE_KEYS if c in header), None)
    if page_key is None:
        raise ValueError(
            f"{csv_path.name} has none of {', '.join(MERGED_PAGE_KEYS)}. A merged CSV needs a numeric "
            "ImageNumber column (as written by CellProfiler) so it can be converted in chunks."
        )
    stage = work / "stage"
    stage.mkdir(parents=True)
    _stage(csv_path, stage / "merged.csv")
    cytotable.convert(
        source_path=str(stage),
        dest_path=str(work / "merged_out"),
        dest_datatype="parquet",
        source_datatype="csv",
        preset=None,
        compartments=[],  # no compartments: CytoTable then leaves every column name unchanged
        metadata=[],
        page_keys={"merged": page_key},
        chunk_size=chunk_size,
        join=False,
        concat=True,
        parsl_config=_cytotable_config(max_threads, work / "parsl_run"),
    )
    files = [p for p in (work / "merged_out").rglob("*.parquet") if p.is_file()]
    if len(files) != 1:
        raise RuntimeError(f"Expected one Parquet file from {csv_path.name}, found {len(files)}.")
    return files[0]


def convert_raw_input(raw: RawInput, dest_dir: Path, *, overwrite: bool = False, chunk_size: int = 10_000,
                      max_threads: int = 4) -> ConversionResult:
    """Convert one input to ``dest_dir/<name>.parquet`` with CytoTable.

    An existing output is kept unless ``overwrite`` is true (use :func:`is_stale` to warn when the
    raw file is newer). Scratch files go to a temporary folder inside ``dest_dir`` that is removed.
    """
    if raw.kind not in KINDS:
        raise ValueError(f"Unknown input kind {raw.kind!r}; expected one of {KINDS}.")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f"{raw.name}.parquet"
    started = time.perf_counter()
    existed = final.is_file()
    if existed and not overwrite:
        return ConversionResult(final, "preserved", pq.ParquetFile(final).metadata.num_rows, 0.0)

    work = dest_dir / f".cytotable_work_{raw.name}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    _reset_parsl()
    try:
        if raw.kind == "sqlite":
            produced = _convert_tables(raw.paths[0], "sqlite", _sqlite_stems(raw.paths[0]), work, chunk_size, max_threads)
        elif raw.kind == "csv_tables":
            stage = work / "stage"
            stage.mkdir()
            for logical in TABLE_ALIASES:
                source = next(p for p in raw.paths if p.stem.lower() in TABLE_ALIASES[logical])
                _stage(source, stage / f"{logical}.csv")
            produced = _convert_tables(stage, "csv", {k: k for k in TABLE_ALIASES}, work, chunk_size, max_threads)
        else:
            produced = _convert_merged(raw.paths[0], work, chunk_size, max_threads)
        os.replace(produced, final)
    finally:
        _reset_parsl()  # stop parsl first, so it cannot write into the scratch folder we delete next
        shutil.rmtree(work, ignore_errors=True)
    return ConversionResult(
        final, "replaced" if existed else "converted", pq.ParquetFile(final).metadata.num_rows,
        time.perf_counter() - started,
    )


def count_source_cells(raw: RawInput) -> int:
    """Cells in the raw input before conversion: cytoplasm objects (SQLite, Per_* CSV) or rows (merged CSV).

    The Parquet has one row per cell that has a cell, a nucleus and an image, so it can be smaller
    than this number; the difference is the objects the join dropped.
    """
    if raw.kind == "sqlite":
        table = _sqlite_table_names(raw.paths[0])["cytoplasm"]
        with sqlite3.connect(f"file:{raw.paths[0]}?mode=ro", uri=True) as connection:
            return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    if raw.kind == "csv_tables":
        source = next(p for p in raw.paths if p.stem.lower() in TABLE_ALIASES["cytoplasm"])
    else:
        source = raw.paths[0]
    return int(sum(len(chunk) for chunk in pd.read_csv(source, usecols=[0], chunksize=500_000)))


def is_stale(parquet: Path, sources: Sequence[Path]) -> bool:
    """True when a raw file was modified after the Parquet file was written."""
    written = Path(parquet).stat().st_mtime
    return any(Path(s).stat().st_mtime > written for s in sources)


# --------------------------------------------------------------------------- reading the result
def inventory(paths: Sequence[Path]) -> pd.DataFrame:
    """One row per Parquet file: cells (rows), columns and size in MB (from the file footer, no data read)."""
    rows = []
    for path in map(Path, paths):
        meta = pq.ParquetFile(path).metadata
        rows.append(
            {
                "file": path.name,
                "cells": meta.num_rows,
                "columns": meta.num_columns,
                "size_mb": round(path.stat().st_size / 1e6, 2),
            }
        )
    return pd.DataFrame(rows, columns=["file", "cells", "columns", "size_mb"])


def read_qc_columns(paths: Sequence[Path], prefixes: Sequence[str], exact: Sequence[str] = ()) -> pd.DataFrame:
    """Read only the columns that start with one of ``prefixes`` or equal one of ``exact``.

    Files are stacked; a column missing from one file (another channel on another plate) is NaN there.
    Parquet reads just the requested columns, so a table with thousands of features loads quickly.
    """
    prefixes, exact = tuple(prefixes), set(exact)
    frames = []
    for path in map(Path, paths):
        names = pq.read_schema(path).names
        wanted = [c for c in names if c.startswith(prefixes) or c in exact]
        frames.append(pq.read_table(path, columns=wanted).to_pandas())
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
