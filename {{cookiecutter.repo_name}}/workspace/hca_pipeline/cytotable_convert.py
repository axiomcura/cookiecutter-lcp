"""Convert raw CellProfiler output to Parquet files with CytoTable.

This module turns the tables that CellProfiler writes into one Parquet file per CellProfiler run,
with one row per cell. The notebook ``00_convert_to_parquet.py`` calls :func:`convert_to_parquet`.
The data quality control (QC) notebook, ``01_data_qc.py``, reads the result with
:func:`read_qc_columns`.

Notes
-----
Terms used in this module:

run
    One execution of a CellProfiler pipeline, usually one plate. CellProfiler restarts
    ``ImageNumber`` at 1 in every run, so each run is converted on its own.
preset
    A saved set of CytoTable settings for one data layout: which tables hold cells, nuclei and
    cytoplasm, and which columns identify an object.
join
    The step that combines the Image, Cells, Cytoplasm and Nuclei tables into one row per cell.

CytoTable has one preset for each CellProfiler export module:

=====================  ==========================================================  =======================
CellProfiler module    Files                                                       CytoTable preset
=====================  ==========================================================  =======================
ExportToDatabase       ``<name>.sqlite`` with the tables Per_Image, Per_Cells,     ``cellprofiler_sqlite``
                       Per_Cytoplasm and Per_Nuclei
ExportToSpreadsheet    a folder with Image.csv, Cells.csv, Cytoplasm.csv and       ``cellprofiler_csv``
                       Nuclei.csv
=====================  ==========================================================  =======================

This module uses both presets as they are, except for the join. The preset joins keep only the
image file names from the Image table (the SQLite preset also keeps well and plate). Data QC needs
more image columns: plate, well and site, ``Image_ImageQuality_*`` (blur and saturation) and
``Image_PathName_*``. The joins in this module therefore keep every image column. They keep a cell
under the same rule as the presets: its cytoplasm object must have a parent cell, a parent nucleus
and an image.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import cytotable
import pandas as pd
import parsl
import pyarrow.parquet as pq
from parsl.config import Config
from parsl.errors import NoDataFlowKernelError
from parsl.executors.threads import ThreadPoolExecutor

# CytoTable finds a CSV table by its file name without the extension, in any case.
CSV_TABLES = ("image", "cells", "cytoplasm", "nuclei")

# Setting modified join queres for CytoTable. The preset joins keep only the image
# file names from the Image table (the SQLite preset also keeps well and plate).
# Data QC needs more image columns: plate, well and site, "Image_ImageQuality_*"
# (blur and saturation) and "Image_PathName_*". (requirded for qc notebook)

# Join for ExportToDatabase (SQLite). Column names already start with the table name
# (Cells_Number_Object_Number). The preset renames ImageNumber to Metadata_ImageNumber;
# this join names it ImageNumber again, because the data-QC notebook matches cells on it.
SQLITE_JOIN = """
SELECT image.Metadata_ImageNumber AS ImageNumber,
       image.* EXCLUDE (Metadata_ImageNumber),
       cytoplasm.* EXCLUDE (Metadata_ImageNumber),
       cells.* EXCLUDE (Metadata_ImageNumber),
       nuclei.* EXCLUDE (Metadata_ImageNumber)
FROM read_parquet('per_cytoplasm.parquet') AS cytoplasm
JOIN read_parquet('per_cells.parquet') AS cells
  ON cells.Metadata_ImageNumber = cytoplasm.Metadata_ImageNumber
 AND cells.Cells_Number_Object_Number = cytoplasm.Cytoplasm_Parent_Cells
JOIN read_parquet('per_nuclei.parquet') AS nuclei
  ON nuclei.Metadata_ImageNumber = cytoplasm.Metadata_ImageNumber
 AND nuclei.Nuclei_Number_Object_Number = cytoplasm.Cytoplasm_Parent_Nuclei
JOIN read_parquet('per_image.parquet') AS image
  ON image.Metadata_ImageNumber = cytoplasm.Metadata_ImageNumber
"""

# Join for ExportToSpreadsheet (CSV). Column names do not start with the table name; the preset
# adds it (AreaShape_Area in Cells.csv becomes Cells_AreaShape_Area) and renames ObjectNumber and
# Parent_* to Metadata_ObjectNumber and Metadata_<Table>_Parent_*.
CSV_JOIN = """
SELECT image.*,
       cytoplasm.* EXCLUDE (Metadata_ImageNumber),
       cells.* EXCLUDE (Metadata_ImageNumber, Metadata_ObjectNumber),
       nuclei.* EXCLUDE (Metadata_ImageNumber, Metadata_ObjectNumber)
FROM read_parquet('cytoplasm.parquet') AS cytoplasm
JOIN read_parquet('cells.parquet') AS cells
  ON cells.Metadata_ImageNumber = cytoplasm.Metadata_ImageNumber
 AND cells.Metadata_ObjectNumber = cytoplasm.Metadata_Cytoplasm_Parent_Cells
JOIN read_parquet('nuclei.parquet') AS nuclei
  ON nuclei.Metadata_ImageNumber = cytoplasm.Metadata_ImageNumber
 AND nuclei.Metadata_ObjectNumber = cytoplasm.Metadata_Cytoplasm_Parent_Nuclei
JOIN ({image_columns} FROM read_parquet('image.parquet')) AS image
  ON image.ImageNumber = cytoplasm.Metadata_ImageNumber
"""
# The CSV preset adds "Image_" only to image columns that do not already start with "Image".
# ImageQuality_PowerLogLogSlope_DNA therefore keeps its raw name. These two parts of the image
# query rename it to Image_ImageQuality_PowerLogLogSlope_DNA, the name the SQLite output uses.
CSV_IMAGE_COLUMNS = (
    "SELECT Metadata_ImageNumber AS ImageNumber, "
    "COLUMNS(c -> c <> 'Metadata_ImageNumber' AND NOT starts_with(c, 'ImageQuality_'))"
)
CSV_IMAGE_QUALITY = r", COLUMNS('^ImageQuality_(.*)') AS 'Image_ImageQuality_\1'"


def parquet_dir(analysis_dir: Path) -> Path:
    """Return the folder for the converted Parquet files of one experiment.

    Parameters
    ----------
    analysis_dir : Path
        The experiment's analysis folder, ``workspace/analysis/<EXPERIMENT_ID>``.

    Returns
    -------
    Path
        ``<analysis_dir>/outputs/parquet``. The data-QC notebook reads its input from here.
    """
    return Path(analysis_dir) / "outputs" / "parquet"


def convert_to_parquet(
    source_dir: Path,
    dest_dir: Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 10_000,
    max_threads: int = 4,
    progress: Callable[[str], None] = print,
) -> list[Path]:
    """Convert every CellProfiler run in a folder to one Parquet file per run.

    The function searches ``source_dir`` and its subfolders. Each ``.sqlite`` file is one run.
    Each folder that holds Image.csv is one run. CytoTable converts each run with its CellProfiler
    preset and the join of this module (see the module notes).

    Parameters
    ----------
    source_dir : Path
        Folder with the raw CellProfiler output, usually ``workspace/backend/<EXPERIMENT_ID>``.
    dest_dir : Path
        Folder for the Parquet files. It is created if it does not exist.
    overwrite : bool, default False
        If False, an existing Parquet file is kept and not converted again. Set it to True after
        you replace the raw files.
    chunk_size : int, default 10_000
        Number of rows CytoTable reads at a time. A larger value is faster and uses more memory.
    max_threads : int, default 4
        Number of threads CytoTable uses.
    progress : callable, default print
        Receives one line of text for each step, for example
        ``"wrote      PlateA.parquet: 48,210 cells"``.

    Returns
    -------
    list of Path
        One Parquet file per run, including files that were kept. The file name is the run's path
        below ``source_dir``, joined with "_": ``PlateA/P1.sqlite`` gives ``PlateA_P1.parquet``.
        The list is empty if the folder holds no CellProfiler output.

    Raises
    ------
    ValueError
        If a CSV folder misses one of the four tables, if one CSV run folder lies inside another,
        or if the folder holds ``.db`` files but no ``.sqlite`` file.

    Notes
    -----
    If the folder holds both SQLite files and CSV files, only the SQLite files are converted.
    The function writes temporary files to ``dest_dir/.cytotable_work_<name>`` and deletes them
    afterwards, also when the conversion fails.

    Examples
    --------
    >>> convert_to_parquet(Path("workspace/backend/EXP"), Path("workspace/analysis/EXP/outputs/parquet"))
    [PosixPath('workspace/analysis/EXP/outputs/parquet/PlateA.parquet')]
    """

    # setting paths
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # iterating over the runs in the source folder
    outputs = []
    for name, source, kind in _find_runs(Path(source_dir)):
        final = dest_dir / f"{name}.parquet"
        outputs.append(final)
        if final.exists() and not overwrite:
            progress(
                f"kept       {final.name} (exists; turn on overwrite to convert again)"
            )
            continue

        # set up the temporary folder and the progress message
        progress(f"converting {source.name} ({kind}) -> {final.name}")
        work = (
            dest_dir / f".cytotable_work_{name}"
        )  # temporary: CytoTable chunks and parsl logs
        shutil.rmtree(work, ignore_errors=True)

        # stop any parsl session left from an earlier conversion, then delete the folder it wrote to
        _reset_parsl()

        # run conversion
        try:
            result = cytotable.convert(
                source_path=str(source),
                dest_path=str(work / final.name),
                dest_datatype="parquet",
                source_datatype=kind,
                preset=f"cellprofiler_{kind}",
                joins=SQLITE_JOIN if kind == "sqlite" else _csv_join(source),
                chunk_size=chunk_size,
                # Threads instead of separate processes, so it also works on Windows. Parsl (the task
                # runner CytoTable uses) writes its logs to the temporary folder, not to ./runinfo.
                parsl_config=Config(
                    executors=[
                        ThreadPoolExecutor(max_threads=max_threads, label="cytotable")
                    ],
                    run_dir=str(work / "parsl_run"),
                ),
            )
            # With a join, CytoTable returns the path of the one Parquet file it wrote (a Path, although
            # its type hint says str; other settings return a list or a dict). The file is in the
            # temporary folder; move it to its final place.
            if not isinstance(result, (str, os.PathLike)):
                raise TypeError(
                    f"CytoTable returned {type(result).__name__}, expected the path of one Parquet file."
                )
            os.replace(result, final)
        finally:
            # stop parsl before deleting the folder it writes to,
            # then delete the temporary folder. It is not needed after the conversion.
            _reset_parsl()
            shutil.rmtree(work, ignore_errors=True)
        progress(
            f"wrote      {final.name}: {pq.ParquetFile(final).metadata.num_rows:,} cells"
        )
    return outputs


def read_qc_columns(
    paths: Sequence[Path], prefixes: Sequence[str], exact: Sequence[str] = ()
) -> pd.DataFrame:
    """Read selected columns from one or more Parquet files into one table.

    Parquet stores each column separately, so reading 200 of 3,000 columns reads only those 200.

    Parameters
    ----------
    paths : sequence of Path
        Parquet files, for example the list returned by :func:`convert_to_parquet`.
    prefixes : sequence of str
        Read every column whose name starts with one of these, for example ``"Image_ImageQuality_"``.
    exact : sequence of str, default ()
        Read these columns as well. A name that no file has is skipped.

    Returns
    -------
    pandas.DataFrame
        The rows of all files, one below the other. If a column is missing from one file (for
        example a Hoechst channel that only plate B has), its values are NaN for that file's rows.
        The table is empty if ``paths`` is empty.
    """
    # setting the columns to read: every column that starts with one of the prefixes, plus the exact names
    starts, names = tuple(prefixes), set(exact)
    frames = []

    # iterate over the paths, read the schema of each file, then read only the wanted columns.
    # The schema is small and fast to read.
    for path in map(Path, paths):
        wanted = [
            c for c in pq.read_schema(path).names if c.startswith(starts) or c in names
        ]
        frames.append(pq.read_table(path, columns=wanted).to_pandas())

    # return the concatenation of all frames, or an empty frame if there are no frames.
    # The concatenation keeps all columns, even if some frames do not have them. Missing values are NaN.
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


# --------------------------------------------------------------------------- helpers
def _find_runs(source_dir: Path) -> list[tuple[str, Path, str]]:
    """Find the CellProfiler runs in a folder and its subfolders.

    Parameters
    ----------
    source_dir : Path
        Folder with the raw CellProfiler output.

    Returns
    -------
    list of tuple (str, Path, str)
        For each run: the output name, the source (a ``.sqlite`` file or a CSV folder) and the
        kind, ``"sqlite"`` or ``"csv"``. The list is empty if the folder does not exist or holds
        no CellProfiler output.

    Raises
    ------
    ValueError
        If a CSV folder misses one of the four tables, if one CSV run folder lies inside another
        (CytoTable reads subfolders too, so the two runs would be mixed), or if there are ``.db``
        files but no ``.sqlite`` file (CytoTable reads SQLite only with the ``.sqlite`` extension).
    """
    if not source_dir.is_dir():
        return []

    # set funcction to build output name from the run's path
    # The name is the run's path below source_dir, joined with "_": PlateA/P1.sqlite gives PlateA_P1.parquet.
    def name(path: Path) -> str:
        parts = path.relative_to(source_dir).parts
        if path.is_file():
            parts = (*parts[:-1], path.stem)
        return "_".join(parts) or source_dir.name

    # find the runs. If there are SQLite files, return them. If there are CSV folders, check that they
    # have all four tables and that no folder lies inside another, then return them.
    # If there are .db files but no .sqlite file, raise an error.
    databases = sorted(source_dir.rglob("*.sqlite"))
    if databases:
        return [(name(p), p, "sqlite") for p in databases]
    if any(source_dir.rglob("*.db")):
        raise ValueError(
            f"CytoTable reads SQLite files only with the .sqlite extension. Rename the .db files in {source_dir}."
        )

    # sort folders by their path, so a folder inside another comes after it.
    # Check that each folder has all four tables and that no folder lies inside another.
    folders = sorted(
        {p.parent for p in source_dir.rglob("*.csv") if p.stem.lower() == "image"}
    )

    # iterate each folder, check that it has all four tables and that no other folder is inside it.
    for folder in folders:
        # set found and missing tables. The CSV preset finds tables by their file name without
        # the extension, in any case.
        found = {p.stem.lower() for p in folder.glob("*.csv")}
        missing = [f"{t.capitalize()}.csv" for t in CSV_TABLES if t not in found]

        # raise an error if a table is missing or if another run is in a subfolder.
        if missing:
            raise ValueError(
                f"{folder} has Image.csv but no {', '.join(missing)} (CellProfiler ExportToSpreadsheet)."
            )

        # raise an error if another run is in a subfolder. CytoTable reads subfolders too, 
        # so the two runs would be mixed.
        if any(other != folder and folder in other.parents for other in folders):
            raise ValueError(
                f"{folder} holds a CellProfiler run and another run in a subfolder. Give each run its own folder."
            )

    # reeturn the list of runs
    return [(name(f), f, "csv") for f in folders]


def _csv_join(folder: Path) -> str:
    """Build the join query for one CSV run.

    Parameters
    ----------
    folder : Path
        The run's folder with Image.csv, Cells.csv, Cytoplasm.csv and Nuclei.csv.

    Returns
    -------
    str
        ``CSV_JOIN`` with the image query filled in. The query renames ``ImageQuality_*`` columns
        only if Image.csv has them, because DuckDB stops with an error when the rename finds no
        column (for example, in a pipeline without the MeasureImageQuality module).
    """
    # set the image query in the join. If Image.csv has ImageQuality_* columns, rename them to Image_ImageQuality_*.
    image_csv = next(p for p in folder.glob("*.csv") if p.stem.lower() == "image")

    # check if Image.csv has any ImageQuality_* columns. DuckDB stops with an error when the rename finds no column.
    has_quality = any(
        c.startswith("ImageQuality_") for c in pd.read_csv(image_csv, nrows=0).columns
    )

    # return the join query with the image query filled in. The query renames ImageQuality_* columns only if Image.csv has them.
    return CSV_JOIN.format(
        image_columns=CSV_IMAGE_COLUMNS + (CSV_IMAGE_QUALITY if has_quality else "")
    )


def _reset_parsl() -> None:
    """Stop any parsl session left from an earlier conversion.

    Parsl is the task runner that CytoTable uses. After a failed conversion it keeps its session
    open, and the next conversion in the same notebook session stops with an error. This function
    closes that session. It does nothing if no session is open.
    """
    try:
        session = parsl.dfk()
    except NoDataFlowKernelError:
        return  # no session is open: nothing to close
    if not session.cleanup_called:  # cleanup() raises if it already ran
        session.cleanup()
    parsl.clear()
