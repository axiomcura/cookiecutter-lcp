"""Tests for hca_pipeline.cytotable_convert (CellProfiler SQLite/CSV -> Parquet with CytoTable).

The reference is the project's own pandas readers (`read_cellprofiler_sqlite`,
`read_cellprofiler_csvs`): the Parquet that CytoTable writes must hold the same cells, columns and
values, because the data-QC notebook reads the Parquet instead of the raw files.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hca_pipeline.cytotable_convert import (
    RawInput,
    convert_raw_input,
    count_source_cells,
    detect_raw_inputs,
    inventory,
    is_stale,
    parquet_dir,
    read_qc_columns,
)
from hca_pipeline.io import read_cellprofiler_csvs, read_cellprofiler_sqlite

KEYS = ["ImageNumber", "Cells_Number_Object_Number"]


def make_tables(seed: int = 0, n_images: int = 3, cells_per_image: int = 9) -> dict[str, pd.DataFrame]:
    """CellProfiler ExportToDatabase-style tables with ImageQuality and path columns."""
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
    with sqlite3.connect(path) as con:
        for name, frame in tables.items():
            frame.to_sql(name, con, index=False)


def write_csv_tables(folder: Path, tables: dict[str, pd.DataFrame]) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in tables.items():
        paths.append(folder / f"{name}.csv")
        frame.to_csv(paths[-1], index=False)
    return paths


def assert_same_cells(case: unittest.TestCase, got: pd.DataFrame, expected: pd.DataFrame) -> None:
    case.assertEqual(sorted(got.columns), sorted(expected.columns))
    case.assertEqual(len(got), len(expected))
    a = got.sort_values(KEYS).reset_index(drop=True)
    b = expected.sort_values(KEYS).reset_index(drop=True)[list(a.columns)]
    for column in a.columns:
        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
            np.testing.assert_allclose(a[column].astype(float), b[column].astype(float), rtol=1e-12, err_msg=column)
        else:
            case.assertEqual(a[column].astype(str).tolist(), b[column].astype(str).tolist(), column)


class SqliteConversionTests(unittest.TestCase):
    def test_parquet_has_the_same_cells_columns_and_values_as_the_pandas_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_sqlite(tmp / "PlateA.sqlite", make_tables())
            result = convert_raw_input(RawInput("sqlite", "PlateA", (tmp / "PlateA.sqlite",)), tmp / "out")
            expected = read_cellprofiler_sqlite(tmp / "PlateA.sqlite", progress=lambda *_: None)
            got = pq.read_table(result.path).to_pandas()
        assert_same_cells(self, got, expected)
        self.assertEqual(len(got), 27)  # 27 real cells; the orphan cytoplasm object is dropped

    def test_image_quality_and_path_columns_survive_the_join(self):
        # CytoTable's built-in CellProfiler join keeps only Image_FileName_*; the data-QC
        # notebook needs ImageQuality (blur, saturation), PathName and Count columns too.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_sqlite(tmp / "PlateA.sqlite", make_tables())
            result = convert_raw_input(RawInput("sqlite", "PlateA", (tmp / "PlateA.sqlite",)), tmp / "out")
            names = pq.read_schema(result.path).names
        for column in ("Image_ImageQuality_PowerLogLogSlope_AOGFP", "Image_PathName_AOGFP", "Image_Metadata_Plate"):
            self.assertIn(column, names)
        self.assertIn("ImageNumber", names)  # not CytoTable's "Metadata_ImageNumber"

    def test_small_chunks_neither_drop_nor_duplicate_cells(self):
        # Object numbers repeat in every image, so the paging key is not unique.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_sqlite(tmp / "PlateA.sqlite", make_tables(n_images=3, cells_per_image=9))
            result = convert_raw_input(RawInput("sqlite", "PlateA", (tmp / "PlateA.sqlite",)), tmp / "out", chunk_size=4)
            got = pq.read_table(result.path, columns=KEYS).to_pandas()
        self.assertEqual(len(got), 27)
        self.assertFalse(got.duplicated(KEYS).any())

    def test_writes_one_file_named_after_the_input_and_leaves_no_scratch_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_sqlite(tmp / "PlateA.sqlite", make_tables())
            result = convert_raw_input(RawInput("sqlite", "PlateA", (tmp / "PlateA.sqlite",)), tmp / "out")
            self.assertEqual(result.path, tmp / "out" / "PlateA.parquet")
            self.assertEqual(sorted(p.name for p in (tmp / "out").iterdir()), ["PlateA.parquet"])
            self.assertEqual(result.rows, 27)
            self.assertEqual(result.status, "converted")


class CsvConversionTests(unittest.TestCase):
    def test_per_table_csvs_match_the_pandas_reader_and_sources_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = write_csv_tables(tmp / "raw", make_tables())
            before = {p.name: p.stat().st_mtime_ns for p in paths}
            result = convert_raw_input(RawInput("csv_tables", "exp", tuple(paths)), tmp / "out")
            expected = read_cellprofiler_csvs(paths, progress=lambda *_: None)
            got = pq.read_table(result.path).to_pandas()
            self.assertEqual(sorted(p.name for p in (tmp / "raw").iterdir()), sorted(before))
            self.assertEqual({p.name: p.stat().st_mtime_ns for p in paths}, before)
            self.assertEqual(sorted(p.name for p in (tmp / "out").iterdir()), ["exp.parquet"])
        assert_same_cells(self, got, expected)

    def test_merged_csv_keeps_every_column_name_and_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tables = make_tables()
            paths = write_csv_tables(tmp / "t", tables)
            merged = read_cellprofiler_csvs(paths, progress=lambda *_: None)
            merged_path = tmp / "raw" / "exp_single_cell.csv"
            merged_path.parent.mkdir()
            merged.to_csv(merged_path, index=False)
            result = convert_raw_input(RawInput("csv_merged", "exp_single_cell", (merged_path,)), tmp / "out", chunk_size=5)
            got = pq.read_table(result.path).to_pandas()
        assert_same_cells(self, got, merged)  # no "Merged_single_cell_" prefixes, no lost rows

    def test_merged_csv_without_a_numeric_key_column_is_rejected_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "m.csv"
            pd.DataFrame({"Metadata_Plate": ["A"], "Some_Feature": [1.0]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "ImageNumber"):
                convert_raw_input(RawInput("csv_merged", "m", (path,)), tmp / "out")


class NoLitterTests(unittest.TestCase):
    def test_conversion_does_not_write_a_runinfo_folder_into_the_working_directory(self):
        # Parsl (used by CytoTable) writes ./runinfo by default; in a notebook that would land in the repo.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_sqlite(tmp / "PlateA.sqlite", make_tables())
            (tmp / "cwd").mkdir()
            previous = Path.cwd()
            os.chdir(tmp / "cwd")
            try:
                convert_raw_input(RawInput("sqlite", "PlateA", (tmp / "PlateA.sqlite",)), tmp / "out")
                left_behind = sorted(p.name for p in (tmp / "cwd").iterdir())
            finally:
                os.chdir(previous)
        self.assertEqual(left_behind, [])


class DetectRawInputsTests(unittest.TestCase):
    def test_classifies_sqlite_csv_tables_and_a_merged_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "a").mkdir(), (tmp / "b").mkdir(), (tmp / "c").mkdir()
            write_sqlite(tmp / "a" / "P1.sqlite", make_tables())
            write_sqlite(tmp / "a" / "P2.sqlite", make_tables(1))
            write_csv_tables(tmp / "b", make_tables())
            (tmp / "c" / "dormant_single_cell.csv").write_text("ImageNumber\n1\n")
            sqlite = detect_raw_inputs(tmp / "a")
            tables = detect_raw_inputs(tmp / "b")
            merged = detect_raw_inputs(tmp / "c")
        self.assertEqual([(r.kind, r.name) for r in sqlite], [("sqlite", "P1"), ("sqlite", "P2")])
        self.assertEqual([(r.kind, len(r.paths)) for r in tables], [("csv_tables", 4)])
        self.assertEqual([(r.kind, r.name) for r in merged], [("csv_merged", "dormant_single_cell")])

    def test_an_empty_or_missing_folder_gives_no_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(detect_raw_inputs(Path(tmp)), [])
            self.assertEqual(detect_raw_inputs(Path(tmp) / "nope"), [])

    def test_ambiguous_csv_files_are_rejected_not_guessed(self):
        # Two CSVs that are not a Per_* table set: converting both as "merged" tables would mix schemas.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "one.csv").write_text("ImageNumber\n1\n")
            (tmp / "two.csv").write_text("ImageNumber\n1\n")
            with self.assertRaisesRegex(ValueError, "one merged CSV"):
                detect_raw_inputs(tmp)


class CountSourceCellsTests(unittest.TestCase):
    def test_counts_the_cytoplasm_objects_of_sqlite_and_csv_tables_and_the_rows_of_a_merged_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tables = make_tables()
            write_sqlite(tmp / "A.sqlite", tables)
            paths = write_csv_tables(tmp / "t", tables)
            merged = tmp / "m.csv"
            read_cellprofiler_csvs(paths, progress=lambda *_: None).to_csv(merged, index=False)
            counts = (
                count_source_cells(RawInput("sqlite", "A", (tmp / "A.sqlite",))),
                count_source_cells(RawInput("csv_tables", "t", tuple(paths))),
                count_source_cells(RawInput("csv_merged", "m", (merged,))),
            )
        self.assertEqual(counts, (28, 28, 27))  # 27 cells + 1 orphan cytoplasm object; the merged CSV has 27 rows


class OverwritePolicyTests(unittest.TestCase):
    def test_existing_output_is_kept_unless_overwrite_and_stale_sources_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "PlateA.sqlite"
            write_sqlite(source, make_tables())
            raw = RawInput("sqlite", "PlateA", (source,))
            first = convert_raw_input(raw, tmp / "out")
            stamp = first.path.stat().st_mtime_ns
            again = convert_raw_input(raw, tmp / "out")
            self.assertEqual(again.status, "preserved")
            self.assertEqual(again.rows, first.rows)  # row count is read back from the kept file
            self.assertEqual(first.path.stat().st_mtime_ns, stamp)
            self.assertFalse(is_stale(first.path, raw.paths))
            time.sleep(0.05)
            os.utime(source, None)  # the raw file changed after the conversion
            self.assertTrue(is_stale(first.path, raw.paths))
            replaced = convert_raw_input(raw, tmp / "out", overwrite=True)
            self.assertEqual(replaced.status, "replaced")
            self.assertFalse(is_stale(replaced.path, raw.paths))


class ReadQcColumnsTests(unittest.TestCase):
    def test_reads_only_the_requested_columns_and_stacks_files_with_different_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = make_tables(0)
            b = make_tables(1)
            b["Per_Nuclei"] = b["Per_Nuclei"].rename(
                columns={"Nuclei_Intensity_IntegratedIntensity_AOGFP": "Nuclei_Intensity_IntegratedIntensity_Hoechst"}
            )
            paths = []
            for name, tables in (("A", a), ("B", b)):
                write_sqlite(tmp / f"{name}.sqlite", tables)
                paths.append(convert_raw_input(RawInput("sqlite", name, (tmp / f"{name}.sqlite",)), tmp / "out").path)
            table = read_qc_columns(paths, prefixes=("Nuclei_Intensity_IntegratedIntensity_",), exact=("ImageNumber", "Not_There"))
        self.assertEqual(
            sorted(table.columns),
            ["ImageNumber", "Nuclei_Intensity_IntegratedIntensity_AOGFP", "Nuclei_Intensity_IntegratedIntensity_Hoechst"],
        )
        self.assertEqual(len(table), 54)
        self.assertTrue(table["Nuclei_Intensity_IntegratedIntensity_Hoechst"].iloc[:27].isna().all())

    def test_inventory_reports_rows_columns_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_sqlite(tmp / "A.sqlite", make_tables())
            path = convert_raw_input(RawInput("sqlite", "A", (tmp / "A.sqlite",)), tmp / "out").path
            table = inventory([path])
            expected_columns = len(pq.read_schema(path).names)
        self.assertEqual(table.loc[0, "file"], "A.parquet")
        self.assertEqual(table.loc[0, "cells"], 27)
        self.assertEqual(table.loc[0, "columns"], expected_columns)
        self.assertGreater(expected_columns, 10)


class LocationTests(unittest.TestCase):
    def test_parquet_dir_is_inside_the_experiment_analysis_outputs(self):
        self.assertEqual(
            parquet_dir(Path("workspace/analysis/EXP")), Path("workspace/analysis/EXP/outputs/parquet")
        )


if __name__ == "__main__":
    unittest.main()
