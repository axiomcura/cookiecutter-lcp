"""Tests for hca_pipeline.cytotable_convert: raw CellProfiler output to Parquet with CytoTable presets.

Fixtures follow what CellProfiler writes:

- ExportToDatabase: one SQLite file with Per_Image, Per_Cells, Per_Cytoplasm, Per_Nuclei (columns carry the
  table prefix). Reference: the project's pandas reader ``read_cellprofiler_sqlite``.
- ExportToSpreadsheet: Image.csv, Cells.csv, Cytoplasm.csv, Nuclei.csv (no table prefix, plus ObjectNumber).
  The CSV fixture is the same run as the SQLite fixture, so both presets must give the same cells and values.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hca_pipeline.cytotable_convert import convert_to_parquet, parquet_dir, read_qc_columns
from hca_pipeline.io import read_cellprofiler_sqlite

KEYS = ["ImageNumber", "Cells_Number_Object_Number"]


def make_tables(seed: int = 0, n_images: int = 3, cells_per_image: int = 9) -> dict[str, pd.DataFrame]:
    """Make the four tables of one CellProfiler run, as ExportToDatabase writes them.

    Parameters
    ----------
    seed : int, default 0
        Seed for the random feature values, so each test gets the same numbers.
    n_images : int, default 3
        Number of images (sites). Wells are B02, B02 and B03.
    cells_per_image : int, default 9
        Number of cells in each image. Object numbers restart at 1 in every image.

    Returns
    -------
    dict of str to pandas.DataFrame
        The tables Per_Image, Per_Cells, Per_Nuclei and Per_Cytoplasm. Column names start with the
        table name (``Cells_Number_Object_Number``). The image table has ImageQuality, FileName and
        PathName columns. The cytoplasm table has one extra object whose cell and nucleus do not
        exist, so a correct join gives ``n_images * cells_per_image`` cells (27 by default).
    """
    rng = np.random.default_rng(seed)
    image = pd.DataFrame(
        {
            "ImageNumber": np.arange(1, n_images + 1),
            "Image_Metadata_Plate": "PlateA",
            "Image_Metadata_Well": ["B02", "B02", "B03"][:n_images],
            "Image_Metadata_Site": np.arange(1, n_images + 1),
            "Image_FileName_AOGFP": [f"img{i}_GFP.tif" for i in range(n_images)],
            "Image_PathName_AOGFP": "/scope/images",
            "Image_ImageQuality_PowerLogLogSlope_AOGFP": rng.normal(-2, 0.1, n_images),
            "Image_ImageQuality_PercentMaximal_AOGFP": rng.uniform(0, 0.5, n_images),
            "Image_ImageQuality_MaxIntensity_AOGFP": rng.uniform(0.5, 1, n_images),
        }
    )
    n = n_images * cells_per_image
    numbers = np.tile(np.arange(1, cells_per_image + 1), n_images)  # object numbers repeat per image
    images = np.repeat(np.arange(1, n_images + 1), cells_per_image)
    cells = pd.DataFrame(
        {
            "ImageNumber": images,
            "Cells_Number_Object_Number": numbers,
            "Cells_Intensity_IntegratedIntensity_AOGFP": rng.uniform(100, 200, n),
        }
    )
    nuclei = pd.DataFrame(
        {
            "ImageNumber": images,
            "Nuclei_Number_Object_Number": numbers,
            "Nuclei_AreaShape_Solidity": rng.uniform(0.8, 1, n),
            "Nuclei_Intensity_IntegratedIntensity_AOGFP": rng.uniform(50, 100, n),
        }
    )
    cytoplasm = pd.DataFrame(
        {
            "ImageNumber": images,
            "Cytoplasm_Number_Object_Number": numbers,
            "Cytoplasm_Parent_Cells": numbers,
            "Cytoplasm_Parent_Nuclei": numbers,
        }
    )
    # An orphan cytoplasm object (its cell and nucleus are not in the tables): both the pandas merge and
    # the SQL join drop it.
    orphan = cytoplasm.iloc[[0]].assign(
        Cytoplasm_Number_Object_Number=999, Cytoplasm_Parent_Cells=999, Cytoplasm_Parent_Nuclei=999
    )
    return {"Per_Image": image, "Per_Cells": cells, "Per_Nuclei": nuclei,
            "Per_Cytoplasm": pd.concat([cytoplasm, orphan], ignore_index=True)}


def write_sqlite(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Write tables to a SQLite file, one table per entry.

    Parameters
    ----------
    path : Path
        The ``.sqlite`` file to create.
    tables : dict of str to pandas.DataFrame
        Table name to table, for example the output of :func:`make_tables`.
    """
    with sqlite3.connect(path) as con:
        for name, frame in tables.items():
            frame.to_sql(name, con, index=False)


def write_spreadsheets(folder: Path, tables: dict[str, pd.DataFrame], quality: bool = True) -> None:
    """Write the same run as CSV files, as ExportToSpreadsheet writes them.

    Parameters
    ----------
    folder : Path
        Folder for Image.csv, Cells.csv, Cytoplasm.csv, Nuclei.csv and Experiment.csv.
    tables : dict of str to pandas.DataFrame
        The output of :func:`make_tables`.
    quality : bool, default True
        If False, leave out the ImageQuality columns, as in a pipeline without MeasureImageQuality.

    Notes
    -----
    ExportToSpreadsheet writes column names without the table name (``AreaShape_Area`` in
    Cells.csv) and adds an ``ObjectNumber`` column to each object table.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        table = name.removeprefix("Per_")
        out = frame.rename(columns=lambda c: c.removeprefix(f"{table}_"))
        if table != "Image":
            out.insert(1, "ObjectNumber", out["Number_Object_Number"])
        elif not quality:
            out = out.drop(columns=[c for c in out.columns if c.startswith("ImageQuality_")])
        out.to_csv(folder / f"{table}.csv", index=False)
    pd.DataFrame({"Key": ["CellProfiler_Version"], "Value": ["4.2.8"]}).to_csv(folder / "Experiment.csv", index=False)


def quiet(*_):
    """Ignore progress messages, so the test output stays short."""


def sqlite_folder(tmp: Path, *names: str) -> Path:
    """Make a folder with one SQLite run per name.

    Parameters
    ----------
    tmp : Path
        Temporary folder; the runs go to ``tmp/raw``.
    *names : str
        One file stem per run, for example ``"PlateA"`` gives ``raw/PlateA.sqlite``.

    Returns
    -------
    Path
        The ``raw`` folder.
    """
    raw = tmp / "raw"
    raw.mkdir(exist_ok=True)
    for i, name in enumerate(names):
        write_sqlite(raw / f"{name}.sqlite", make_tables(i))
    return raw


def assert_same_cells(case: unittest.TestCase, got: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Check that two cell tables have the same columns, cells and values.

    Parameters
    ----------
    case : unittest.TestCase
        The running test, used for its assertions.
    got, expected : pandas.DataFrame
        The tables to compare. Rows are matched on ImageNumber and Cells_Number_Object_Number.
        Numbers must agree to a relative tolerance of 1e-12; other values must be equal as text.
    """
    case.assertEqual(sorted(got.columns), sorted(expected.columns))
    case.assertEqual(len(got), len(expected))
    a = got.sort_values(KEYS).reset_index(drop=True)
    b = expected.sort_values(KEYS).reset_index(drop=True)[list(a.columns)]
    for column in a.columns:
        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
            np.testing.assert_allclose(a[column].astype(float), b[column].astype(float), rtol=1e-12, err_msg=column)
        else:
            case.assertEqual(a[column].astype(str).tolist(), b[column].astype(str).tolist(), column)


class SqlitePresetTests(unittest.TestCase):
    """Conversion of ExportToDatabase output (preset ``cellprofiler_sqlite``)."""

    def test_parquet_has_the_same_cells_columns_and_values_as_the_pandas_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = sqlite_folder(tmp, "PlateA")
            (path,) = convert_to_parquet(raw, tmp / "out", progress=quiet)
            expected = read_cellprofiler_sqlite(raw / "PlateA.sqlite", progress=quiet)
            got = pq.read_table(path).to_pandas()
        assert_same_cells(self, got, expected)
        self.assertEqual(len(got), 27)  # 27 real cells; the orphan cytoplasm object is dropped

    def test_image_columns_dropped_by_the_preset_join_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (path,) = convert_to_parquet(sqlite_folder(tmp, "PlateA"), tmp / "out", progress=quiet)
            names = pq.read_schema(path).names
        for column in ("ImageNumber", "Image_Metadata_Site", "Image_ImageQuality_PowerLogLogSlope_AOGFP",
                       "Image_PathName_AOGFP"):
            self.assertIn(column, names)

    def test_small_chunks_neither_drop_nor_duplicate_cells(self):
        # Object numbers repeat in every image, so the paging key is not unique.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (path,) = convert_to_parquet(sqlite_folder(tmp, "PlateA"), tmp / "out", chunk_size=4, progress=quiet)
            got = pq.read_table(path, columns=KEYS).to_pandas()
        self.assertEqual(len(got), 27)
        self.assertFalse(got.duplicated(KEYS).any())

    def test_one_file_per_database_and_no_scratch_folder_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = sqlite_folder(tmp, "P1", "P2")
            (raw / "PlateC").mkdir()
            write_sqlite(raw / "PlateC" / "P1.sqlite", make_tables(3))  # same file name in a plate folder
            paths = convert_to_parquet(raw, tmp / "out", progress=quiet)
            self.assertEqual([p.name for p in paths], ["P1.parquet", "P2.parquet", "PlateC_P1.parquet"])
            self.assertEqual(sorted(p.name for p in (tmp / "out").iterdir()), [p.name for p in paths])

    def test_a_db_file_is_rejected_with_a_rename_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_sqlite(tmp / "DefaultDB.db", make_tables())
            with self.assertRaisesRegex(ValueError, r"\.sqlite"):
                convert_to_parquet(tmp, tmp / "out", progress=quiet)


class CsvPresetTests(unittest.TestCase):
    """Conversion of ExportToSpreadsheet output (preset ``cellprofiler_csv``) and how runs are found."""

    def test_spreadsheets_give_the_same_cells_and_values_as_the_sqlite_of_the_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tables = make_tables()
            write_sqlite(tmp / "A.sqlite", tables)
            write_spreadsheets(tmp / "exp", tables)
            (path,) = convert_to_parquet(tmp / "exp", tmp / "out", chunk_size=4, progress=quiet)
            got = pq.read_table(path).to_pandas()
            reference = read_cellprofiler_sqlite(tmp / "A.sqlite", progress=quiet)
        self.assertEqual(path.name, "exp.parquet")  # named after the run folder
        self.assertEqual(len(got), 27)
        self.assertFalse(got.duplicated(KEYS).any())
        # The CSV preset names Plate and Well Metadata_*; Site keeps the Image_ prefix (NB01 renames it).
        same_name = ["Image_ImageQuality_PowerLogLogSlope_AOGFP", "Image_ImageQuality_PercentMaximal_AOGFP",
                     "Image_PathName_AOGFP", "Image_FileName_AOGFP", "Image_Metadata_Site",
                     "Cells_Intensity_IntegratedIntensity_AOGFP", "Nuclei_AreaShape_Solidity",
                     "Nuclei_Intensity_IntegratedIntensity_AOGFP", "Nuclei_Number_Object_Number"]
        a = got.rename(columns={"Metadata_Plate": "Image_Metadata_Plate", "Metadata_Well": "Image_Metadata_Well"})
        a = a.sort_values(KEYS).reset_index(drop=True)
        b = reference.sort_values(KEYS).reset_index(drop=True)
        for column in same_name + ["Image_Metadata_Plate", "Image_Metadata_Well"]:
            self.assertIn(column, a.columns)
            self.assertEqual(a[column].astype(str).tolist(), b[column].astype(str).tolist(), column)

    def test_a_run_without_image_quality_columns_converts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_spreadsheets(tmp / "exp", make_tables(), quality=False)
            (path,) = convert_to_parquet(tmp / "exp", tmp / "out", progress=quiet)
            names = pq.read_schema(path).names
        self.assertFalse([c for c in names if "ImageQuality" in c])
        self.assertIn("Image_PathName_AOGFP", names)

    def test_one_file_per_plate_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_spreadsheets(tmp / "exp" / "PlateA", make_tables(0))
            write_spreadsheets(tmp / "exp" / "PlateB", make_tables(1))
            paths = convert_to_parquet(tmp / "exp", tmp / "out", progress=quiet)
            rows = [pq.ParquetFile(p).metadata.num_rows for p in paths]
        self.assertEqual([p.name for p in paths], ["PlateA.parquet", "PlateB.parquet"])
        self.assertEqual(rows, [27, 27])  # plates are not mixed although ImageNumber restarts in each

    def test_a_missing_table_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_spreadsheets(tmp / "exp", make_tables())
            (tmp / "exp" / "Nuclei.csv").unlink()
            with self.assertRaisesRegex(ValueError, "Nuclei.csv"):
                convert_to_parquet(tmp / "exp", tmp / "out", progress=quiet)

    def test_a_run_folder_inside_another_run_folder_is_rejected(self):
        # CytoTable reads subfolders too, so the two runs would be mixed.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_spreadsheets(tmp / "exp", make_tables(0))
            write_spreadsheets(tmp / "exp" / "rerun", make_tables(1))
            with self.assertRaisesRegex(ValueError, "subfolder"):
                convert_to_parquet(tmp / "exp", tmp / "out", progress=quiet)

    def test_sqlite_wins_over_csv_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = sqlite_folder(tmp, "PlateA")
            write_spreadsheets(raw / "csv", make_tables())
            paths = convert_to_parquet(raw, tmp / "out", progress=quiet)
        self.assertEqual([p.name for p in paths], ["PlateA.parquet"])

    def test_an_empty_or_missing_folder_gives_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(convert_to_parquet(Path(tmp), Path(tmp) / "out", progress=quiet), [])
            self.assertEqual(convert_to_parquet(Path(tmp) / "nope", Path(tmp) / "out", progress=quiet), [])


def make_labelled_run(plate: int, n_images: int = 4, cells_per_image: int = 5) -> dict[str, pd.DataFrame]:
    """Make one CellProfiler run in which every value says which object it belongs to.

    Parameters
    ----------
    plate : int
        Plate code, from 1 to 9. It is part of every value, so values of two plates never match.
    n_images : int, default 4
        Number of images. The last image has no objects (an empty field of view).
    cells_per_image : int, default 5
        Number of complete cells (cell + nucleus + cytoplasm) in each of the other images.

    Returns
    -------
    dict of str to pandas.DataFrame
        The tables Per_Image, Per_Cells, Per_Nuclei and Per_Cytoplasm, as ExportToDatabase writes them.

    Notes
    -----
    The values follow ``label = plate * 1_000_000 + image * 1_000 + object``. For example, the cell
    intensity 1_002_003.0 belongs to plate 1, image 2, cell 3. A join error therefore shows up as a
    wrong label in the output row.

    Each image (except the last) holds, with ``n = cells_per_image``:

    - cells 1 to n; cell ``c`` owns nucleus ``n + 1 - c``, so cell and nucleus numbers differ;
    - cytoplasm 1 to n, cytoplasm ``c`` with parent cell ``c`` and parent nucleus ``n + 1 - c``;
    - cytoplasm n + 1 with parent cell 0 and parent nucleus 0 (0 is CellProfiler's "no parent");
    - cell n + 1 and nucleus n + 1, which have no cytoplasm.

    A correct join keeps the n complete cells per image and drops the other objects. In image 2 of
    plate 1, cell 2 has a missing value (NaN) for ``Cells_AreaShape_Area``; the cell must stay.
    """
    n = cells_per_image
    full = list(range(1, n_images))  # images with objects; the last image is empty

    def label(image, obj):
        return float(plate * 1_000_000 + image * 1_000 + obj)

    image = pd.DataFrame(
        {
            "ImageNumber": range(1, n_images + 1),
            "Image_Metadata_Plate": f"Plate{plate}",
            "Image_Metadata_Well": [f"B{i:02d}" for i in range(1, n_images + 1)],
            "Image_Metadata_Site": 1,
            "Image_FileName_DNA": [f"p{plate}_i{i}.tif" for i in range(1, n_images + 1)],
            "Image_PathName_DNA": f"/scope/plate{plate}",
            "Image_ImageQuality_PowerLogLogSlope_DNA": [label(i, 0) for i in range(1, n_images + 1)],
            "Image_Count_Cells": [n + 1 if i in full else 0 for i in range(1, n_images + 1)],
        }
    )
    cells = pd.DataFrame(
        [
            {"ImageNumber": i, "Cells_Number_Object_Number": c, "Cells_Parent_Nuclei": n + 1 - c if c <= n else n + 1,
             "Cells_Intensity_MeanIntensity_DNA": label(i, c),
             "Cells_AreaShape_Area": np.nan if (plate, i, c) == (1, 2, 2) else label(i, c)}
            for i in full for c in range(1, n + 2)
        ]
    )
    nuclei = pd.DataFrame(
        [
            {"ImageNumber": i, "Nuclei_Number_Object_Number": k, "Nuclei_Intensity_MeanIntensity_DNA": label(i, k)}
            for i in full for k in range(1, n + 2)
        ]
    )
    cytoplasm = pd.DataFrame(
        [
            {"ImageNumber": i, "Cytoplasm_Number_Object_Number": c,
             "Cytoplasm_Parent_Cells": c if c <= n else 0, "Cytoplasm_Parent_Nuclei": n + 1 - c if c <= n else 0,
             "Cytoplasm_AreaShape_Area": label(i, c)}
            for i in full for c in range(1, n + 2)
        ]
    )
    return {"Per_Image": image, "Per_Cells": cells, "Per_Nuclei": nuclei, "Per_Cytoplasm": cytoplasm}


def expected_cells(plate: int, n_images: int = 4, cells_per_image: int = 5) -> pd.DataFrame:
    """List the cells a correct join of :func:`make_labelled_run` keeps.

    Parameters
    ----------
    plate, n_images, cells_per_image
        The same values as given to :func:`make_labelled_run`.

    Returns
    -------
    pandas.DataFrame
        One row per kept cell, with ImageNumber, the cell number and the number of its nucleus.
        It is written from the fixture rules, not from any join code.
    """
    n = cells_per_image
    return pd.DataFrame(
        [{"ImageNumber": i, "cell": c, "nucleus": n + 1 - c} for i in range(1, n_images) for c in range(1, n + 1)]
    )


def write_run(kind: str, folder: Path, name: str, tables: dict[str, pd.DataFrame]) -> None:
    """Write one run as CellProfiler would, as SQLite or as CSV files.

    Parameters
    ----------
    kind : {"sqlite", "csv"}
        ``"sqlite"`` writes ``folder/<name>.sqlite``; ``"csv"`` writes the CSV files to ``folder/<name>/``.
    folder : Path
        The experiment folder.
    name : str
        Name of the run, for example ``"Plate1"``.
    tables : dict of str to pandas.DataFrame
        The output of :func:`make_labelled_run`.
    """
    folder.mkdir(parents=True, exist_ok=True)
    if kind == "sqlite":
        write_sqlite(folder / f"{name}.sqlite", tables)
    else:
        write_spreadsheets(folder / name, tables)


# The same measurement has a different column name in the SQLite and the CSV output.
COLUMN_NAMES = {
    "sqlite": {"parent_nuclei": "Cytoplasm_Parent_Nuclei", "plate": "Image_Metadata_Plate"},
    "csv": {"parent_nuclei": "Metadata_Cytoplasm_Parent_Nuclei", "plate": "Metadata_Plate"},
}


class JoinCorrectnessTests(unittest.TestCase):
    """The join puts the right nucleus, cytoplasm and image values in each cell's row.

    Every test runs for both CellProfiler layouts (SQLite and CSV) and with chunks smaller than an
    image, so that CytoTable's chunked reading is part of the test.
    """

    def convert(self, tmp: Path, kind: str, plates=(1,), chunk_size: int = 3) -> dict[int, pd.DataFrame]:
        raw = tmp / f"raw_{kind}"
        for plate in plates:
            write_run(kind, raw, f"Plate{plate}", make_labelled_run(plate))
        paths = convert_to_parquet(raw, tmp / f"out_{kind}", chunk_size=chunk_size, progress=quiet)
        self.assertEqual([p.name for p in paths], [f"Plate{plate}.parquet" for plate in plates])
        return {plate: pq.read_table(path).to_pandas() for plate, path in zip(plates, paths)}

    def test_each_row_holds_the_values_of_one_cell_its_own_nucleus_and_its_own_image(self):
        for kind in ("sqlite", "csv"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                table = self.convert(Path(tmp), kind)[1]
                image, cell = table["ImageNumber"], table["Cells_Number_Object_Number"]
                nucleus = table["Nuclei_Number_Object_Number"]
                base = 1_000_000 + image * 1_000
                np.testing.assert_array_equal(table["Cells_Intensity_MeanIntensity_DNA"], base + cell)
                np.testing.assert_array_equal(table["Nuclei_Intensity_MeanIntensity_DNA"], base + nucleus)
                np.testing.assert_array_equal(table["Cytoplasm_AreaShape_Area"], base + cell)
                np.testing.assert_array_equal(table["Image_ImageQuality_PowerLogLogSlope_DNA"], base)
                np.testing.assert_array_equal(table[COLUMN_NAMES[kind]["parent_nuclei"]], nucleus)
                self.assertTrue((table["Image_FileName_DNA"] == "p1_i" + image.astype(str) + ".tif").all())

    def test_the_kept_cells_are_exactly_the_complete_cells(self):
        for kind in ("sqlite", "csv"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                table = self.convert(Path(tmp), kind)[1]
                got = table[["ImageNumber", "Cells_Number_Object_Number", "Nuclei_Number_Object_Number"]]
                got = got.set_axis(["ImageNumber", "cell", "nucleus"], axis=1)
                got = got.sort_values(["ImageNumber", "cell"]).reset_index(drop=True).astype("int64")
                pd.testing.assert_frame_equal(got, expected_cells(1))
                # Dropped: cytoplasm 6 (no parent), cell 6 and nucleus 6 (no cytoplasm), and image 4 (empty).
                self.assertNotIn(6, set(table["Cells_Number_Object_Number"]))
                self.assertNotIn(6, set(table["Nuclei_Number_Object_Number"]))
                self.assertNotIn(4, set(table["ImageNumber"]))

    def test_a_missing_value_keeps_its_cell(self):
        for kind in ("sqlite", "csv"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                table = self.convert(Path(tmp), kind)[1]
                row = table[(table["ImageNumber"] == 2) & (table["Cells_Number_Object_Number"] == 2)]
                self.assertEqual(len(row), 1)
                self.assertTrue(row["Cells_AreaShape_Area"].isna().all())
                self.assertEqual(int(table["Cells_AreaShape_Area"].isna().sum()), 1)

    def test_small_chunks_give_the_same_table_as_one_chunk(self):
        for kind in ("sqlite", "csv"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                small = self.convert(Path(tmp) / "small", kind, chunk_size=2)[1]
                whole = self.convert(Path(tmp) / "whole", kind, chunk_size=100_000)[1]
                order = ["ImageNumber", "Cells_Number_Object_Number"]
                pd.testing.assert_frame_equal(
                    small.sort_values(order).reset_index(drop=True)[sorted(small.columns)],
                    whole.sort_values(order).reset_index(drop=True)[sorted(whole.columns)],
                )

    def test_two_plates_with_the_same_image_numbers_are_not_mixed(self):
        for kind in ("sqlite", "csv"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                tables = self.convert(Path(tmp), kind, plates=(1, 2))
                for plate, table in tables.items():
                    self.assertEqual(len(table), 15)
                    self.assertEqual(set(table[COLUMN_NAMES[kind]["plate"]]), {f"Plate{plate}"})
                    for column in ("Cells_Intensity_MeanIntensity_DNA", "Nuclei_Intensity_MeanIntensity_DNA",
                                   "Image_ImageQuality_PowerLogLogSlope_DNA"):
                        self.assertTrue((table[column] // 1_000_000 == plate).all(), f"plate {plate}: {column}")

    def test_every_measurement_appears_once(self):
        for kind in ("sqlite", "csv"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                table = self.convert(Path(tmp), kind)[1]
                self.assertEqual(len(table.columns), len(set(table.columns)))
                for column in ("Cells_Intensity_MeanIntensity_DNA", "Cells_AreaShape_Area",
                               "Nuclei_Intensity_MeanIntensity_DNA", "Cytoplasm_AreaShape_Area",
                               "Image_ImageQuality_PowerLogLogSlope_DNA", "Image_PathName_DNA",
                               "Image_FileName_DNA", "Image_Count_Cells", "ImageNumber"):
                    self.assertIn(column, table.columns)


class NoLitterTests(unittest.TestCase):
    """The conversion leaves no files outside the output folder."""

    def test_conversion_does_not_write_a_runinfo_folder_into_the_working_directory(self):
        # Parsl (used by CytoTable) writes ./runinfo by default; in a notebook that would land in the repo.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = sqlite_folder(tmp, "PlateA")
            (tmp / "cwd").mkdir()
            previous = Path.cwd()
            os.chdir(tmp / "cwd")
            try:
                convert_to_parquet(raw, tmp / "out", progress=quiet)
                left_behind = sorted(p.name for p in (tmp / "cwd").iterdir())
            finally:
                os.chdir(previous)
        self.assertEqual(left_behind, [])


class OverwritePolicyTests(unittest.TestCase):
    """An existing Parquet file is kept unless overwrite is on."""

    def test_existing_output_is_kept_unless_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = sqlite_folder(tmp, "PlateA")
            (path,) = convert_to_parquet(raw, tmp / "out", progress=quiet)
            stamp = path.stat().st_mtime_ns
            self.assertEqual(convert_to_parquet(raw, tmp / "out", progress=quiet), [path])
            self.assertEqual(path.stat().st_mtime_ns, stamp)
            convert_to_parquet(raw, tmp / "out", overwrite=True, progress=quiet)
            self.assertNotEqual(path.stat().st_mtime_ns, stamp)
            self.assertEqual(pq.ParquetFile(path).metadata.num_rows, 27)


class ReadQcColumnsTests(unittest.TestCase):
    """Reading selected columns from several Parquet files."""

    def test_reads_only_the_requested_columns_and_stacks_files_with_different_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = make_tables(0)
            b = make_tables(1)
            b["Per_Nuclei"] = b["Per_Nuclei"].rename(
                columns={"Nuclei_Intensity_IntegratedIntensity_AOGFP": "Nuclei_Intensity_IntegratedIntensity_Hoechst"}
            )
            (tmp / "raw").mkdir()
            write_sqlite(tmp / "raw" / "A.sqlite", a)
            write_sqlite(tmp / "raw" / "B.sqlite", b)
            paths = convert_to_parquet(tmp / "raw", tmp / "out", progress=quiet)
            table = read_qc_columns(paths, prefixes=("Nuclei_Intensity_IntegratedIntensity_",), exact=("ImageNumber", "Not_There"))
        self.assertEqual(
            sorted(table.columns),
            ["ImageNumber", "Nuclei_Intensity_IntegratedIntensity_AOGFP", "Nuclei_Intensity_IntegratedIntensity_Hoechst"],
        )
        self.assertEqual(len(table), 54)
        self.assertTrue(table["Nuclei_Intensity_IntegratedIntensity_Hoechst"].iloc[:27].isna().all())


class LocationTests(unittest.TestCase):
    """Where the Parquet files are stored."""

    def test_parquet_dir_is_inside_the_experiment_analysis_outputs(self):
        self.assertEqual(
            parquet_dir(Path("workspace/analysis/EXP")), Path("workspace/analysis/EXP/outputs/parquet")
        )


if __name__ == "__main__":
    unittest.main()
