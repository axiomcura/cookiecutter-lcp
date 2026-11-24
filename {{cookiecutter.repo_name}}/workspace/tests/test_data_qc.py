"""Tests for NB00 data-QC helpers (image-level, cell-level and DNA-content QC).

Each test states the scientific expectation it encodes. Synthetic data with a
known answer stands in for real plates: a positive control (a defect the
check must find) and a negative control (a look-alike it must not flag).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hca_pipeline.data_qc import (
    dna_content_groups,
    lookup_qc_status,
    flag_sites,
    label_cell_outliers,
    profile_qc_conditions,
    resolve_dna_channel,
    robust_z,
    site_quality_from_cellprofiler,
    site_quality_from_pixels,
)


class RobustZTests(unittest.TestCase):
    def test_median_is_zero_and_outlier_does_not_shrink_the_scale(self):
        # Expectation: robust z uses median/MAD, so one extreme value neither
        # moves the centre nor inflates the spread for the other values.
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
        z = robust_z(values)
        self.assertAlmostEqual(z.iloc[2], 0.0)
        # MAD of [1,2,3,4,1000] around median 3 is 1 -> scale 1.4826.
        self.assertAlmostEqual(z.iloc[3], 1 / 1.4826, places=6)
        self.assertGreater(z.iloc[4], 100)

    def test_groups_are_scored_independently(self):
        # Expectation: a plate-wide intensity shift does not change within-plate z.
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        groups = pd.Series(["A", "A", "A", "B", "B", "B"])
        z = robust_z(values, groups)
        np.testing.assert_allclose(z.iloc[:3].to_numpy(), z.iloc[3:].to_numpy())

    def test_zero_mad_gives_nan_not_infinity(self):
        z = robust_z(pd.Series([5.0, 5.0, 5.0]))
        self.assertTrue(z.isna().all())


class DnaChannelTests(unittest.TestCase):
    def test_prefers_hoechst_over_acridine_orange(self):
        columns = [
            "Nuclei_Intensity_IntegratedIntensity_AOGFP",
            "Nuclei_Intensity_IntegratedIntensity_Hoechst",
        ]
        self.assertEqual(resolve_dna_channel(columns), "Hoechst")

    def test_falls_back_to_aogfp(self):
        self.assertEqual(resolve_dna_channel(["Nuclei_Intensity_IntegratedIntensity_AOGFP"]), "AOGFP")

    def test_raises_when_no_nuclear_intensity_exists(self):
        with self.assertRaises(ValueError):
            resolve_dna_channel(["Nuclei_AreaShape_Area"])


def _cellprofiler_sites(n_sites: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sites = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"B{i % 10 + 2:02d}" for i in range(n_sites)],
            "Metadata_Site": np.arange(n_sites) // 10 + 1,
            "Image_ImageQuality_PowerLogLogSlope_AOGFP": rng.normal(-1.8, 0.05, n_sites),
            "Image_ImageQuality_PercentMaximal_AOGFP": rng.uniform(0, 0.01, n_sites),
            "Image_ImageQuality_MaxIntensity_AOGFP": rng.uniform(0.3, 0.6, n_sites),
        }
    )
    return sites


class ImageQcTests(unittest.TestCase):
    def test_blurred_and_saturated_sites_are_flagged_with_reasons(self):
        sites = _cellprofiler_sites()
        sites.loc[0, "Image_ImageQuality_PowerLogLogSlope_AOGFP"] = -4.0  # positive control: blur
        sites.loc[1, "Image_ImageQuality_MaxIntensity_AOGFP"] = 1.0  # positive control: saturation
        sites.loc[1, "Image_ImageQuality_PercentMaximal_AOGFP"] = 5.0
        # Negative control: many pixels at the image maximum, but that maximum is
        # far below the detector ceiling, so nothing is clipped.
        sites.loc[2, "Image_ImageQuality_MaxIntensity_AOGFP"] = 0.5
        sites.loc[2, "Image_ImageQuality_PercentMaximal_AOGFP"] = 5.0

        quality = site_quality_from_cellprofiler(sites, ["AOGFP"])
        flagged = flag_sites(quality, ["AOGFP"], group_cols=["Metadata_Plate"])
        # Output is sorted by site key, so look rows up by key, not position.
        keys = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
        flagged = flagged.set_index(keys)
        blurred, saturated, unclipped = (tuple(sites.loc[i, keys]) for i in (0, 1, 2))

        self.assertEqual(flagged.loc[blurred, "Metadata_image_qc"], "fail")
        self.assertIn("blurred (AOGFP)", flagged.loc[blurred, "Metadata_image_qc_reason"])
        self.assertEqual(flagged.loc[saturated, "Metadata_image_qc"], "fail")
        self.assertIn("saturated (AOGFP)", flagged.loc[saturated, "Metadata_image_qc_reason"])
        self.assertEqual(flagged.loc[unclipped, "Metadata_image_qc"], "pass")
        self.assertEqual(int(flagged["Metadata_image_qc"].eq("fail").sum()), 2)

    def test_one_row_per_site_even_when_input_has_one_row_per_cell(self):
        sites = _cellprofiler_sites(n_sites=5)
        per_cell = pd.concat([sites] * 3, ignore_index=True)
        quality = site_quality_from_cellprofiler(per_cell, ["AOGFP"])
        self.assertEqual(len(quality), 5)

    def test_pixel_fallback_matches_the_same_flag_contract(self):
        # Expectation: the pixel path produces the same columns, so flag_sites
        # works unchanged; focus score is log10-transformed (it is right-skewed).
        long = pd.DataFrame(
            {
                "Metadata_Plate": ["P1"] * 4,
                "Metadata_Well": ["B02", "B02", "B03", "B03"],
                "Metadata_Site": [1, 1, 1, 1],
                "Metadata_Channel": ["AOGFP", "AOPI", "AOGFP", "AOPI"],
                "focus_score": [100.0, 10.0, 1000.0, 10.0],
                "saturated_fraction": [0.0, 0.02, 0.0, 0.0],
            }
        )
        quality = site_quality_from_pixels(long, ["AOGFP", "AOPI"])
        self.assertEqual(len(quality), 2)
        self.assertAlmostEqual(quality.loc[0, "blur_metric_AOGFP"], 2.0)
        self.assertAlmostEqual(quality.loc[0, "pct_saturated_AOPI"], 2.0)
        flagged = flag_sites(quality, ["AOGFP", "AOPI"], group_cols=["Metadata_Plate"])
        self.assertIn("saturated (AOPI)", flagged.loc[0, "Metadata_image_qc_reason"])


def _single_cells(n: int = 400, plate: str = "P1", scale: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = pd.DataFrame(
        {
            "Metadata_Plate": plate,
            "Nuclei_Intensity_MassDisplacement_AOGFP": rng.normal(1.0, 0.2, n) * scale,
            "Nuclei_Intensity_IntegratedIntensity_AOGFP": rng.normal(100.0, 10.0, n) * scale,
            "Nuclei_AreaShape_Solidity": rng.normal(0.95, 0.01, n),
            "Cells_Intensity_IntegratedIntensity_AOGFP": rng.normal(100.0, 10.0, n) * scale,
        }
    )
    return cells


class ProfileQcTests(unittest.TestCase):
    def setUp(self):
        self.conditions = profile_qc_conditions("AOGFP")

    def test_conditions_use_the_resolved_channel(self):
        features = {f for thresholds in self.conditions.values() for f in thresholds}
        self.assertIn("Nuclei_Intensity_IntegratedIntensity_AOGFP", features)
        self.assertIn("Nuclei_AreaShape_Solidity", features)
        self.assertFalse(any(f.endswith("_DNA") for f in features))

    def test_merged_nuclei_need_both_high_mass_displacement_and_high_dna(self):
        cells = _single_cells()
        # Positive control: merged nuclei = off-centre mass AND too much DNA.
        cells.loc[0, "Nuclei_Intensity_MassDisplacement_AOGFP"] = 3.0
        cells.loc[0, "Nuclei_Intensity_IntegratedIntensity_AOGFP"] = 200.0
        # Negative control: off-centre mass only (AND rule must not fire on
        # mass displacement; DNA set to the mean, z ~ 0 < 1.5).
        cells.loc[1, "Nuclei_Intensity_MassDisplacement_AOGFP"] = 3.0
        cells.loc[1, "Nuclei_Intensity_IntegratedIntensity_AOGFP"] = 100.0
        flags = label_cell_outliers(cells, self.conditions, group_col="Metadata_Plate")
        column = "Metadata_cqc_under_segmented_nuclei_is_outlier"
        self.assertTrue(flags.loc[0, column])
        self.assertFalse(flags.loc[1, column])

    def test_low_solidity_nucleus_is_flagged(self):
        cells = _single_cells()
        cells.loc[5, "Nuclei_AreaShape_Solidity"] = 0.6
        flags = label_cell_outliers(cells, self.conditions, group_col="Metadata_Plate")
        self.assertTrue(flags.loc[5, "Metadata_cqc_poorly_segmented_nuclei_is_outlier"])

    def test_z_scores_are_per_plate(self):
        # Metamorphic: the same plate imaged twice as bright gets the same flags.
        plate_a = _single_cells(plate="A", seed=1)
        plate_b = _single_cells(plate="B", seed=1, scale=2.0)
        both = pd.concat([plate_a, plate_b], ignore_index=True)
        flags = label_cell_outliers(both, self.conditions, group_col="Metadata_Plate")
        n = len(plate_a)
        pd.testing.assert_frame_equal(
            flags.iloc[:n].reset_index(drop=True), flags.iloc[n:].reset_index(drop=True)
        )

    def test_missing_value_does_not_disable_the_whole_plate(self):
        # Regression guard: scipy.stats.zscore returns all-NaN if any input is
        # NaN, which would silently flag nothing on that plate.
        cells = _single_cells()
        cells.loc[5, "Nuclei_AreaShape_Solidity"] = 0.6
        cells.loc[10, "Nuclei_AreaShape_Solidity"] = np.nan
        flags = label_cell_outliers(cells, self.conditions, group_col="Metadata_Plate")
        self.assertTrue(flags.loc[5, "Metadata_cqc_poorly_segmented_nuclei_is_outlier"])
        self.assertFalse(flags.loc[10, "Metadata_cqc_poorly_segmented_nuclei_is_outlier"])
        self.assertTrue(flags.loc[10, "Metadata_cqc_features_missing"])

    def test_output_is_aligned_to_input_index(self):
        cells = _single_cells(n=50)
        cells.index = cells.index + 1000
        flags = label_cell_outliers(cells, self.conditions, group_col="Metadata_Plate")
        self.assertTrue(flags.index.equals(cells.index))


def _dna_mixture(n: int = 6000, g1: float = 1000.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_g1, n_g2 = int(n * 0.65), int(n * 0.25)
    n_s = n - n_g1 - n_g2 - 60
    return np.concatenate(
        [
            g1 * rng.lognormal(0, 0.08, n_g1),
            2 * g1 * rng.lognormal(0, 0.08, n_g2),
            rng.uniform(1.3 * g1, 1.7 * g1, n_s),
            0.3 * g1 * rng.lognormal(0, 0.1, 60),  # sub-G1 debris / fragments
        ]
    )


class DnaContentTests(unittest.TestCase):
    def test_positive_control_g2_peak_is_twice_g1(self):
        # Expectation: for a stoichiometric DNA stain, G2/M nuclei carry 2x the
        # DNA of G1 nuclei, so the second peak sits at ~2x the first.
        dna = pd.Series(_dna_mixture())
        labels, peaks = dna_content_groups(dna, pd.Series("P1", index=dna.index))
        row = peaks.iloc[0]
        self.assertAlmostEqual(row["g1_peak"], 1000.0, delta=60.0)
        self.assertTrue(row["has_g2_peak"])
        self.assertAlmostEqual(row["g2_g1_ratio"], 2.0, delta=0.15)
        self.assertGreater((labels == "G1").mean(), 0.5)
        self.assertGreater((labels == "G2/M").mean(), 0.15)
        self.assertGreaterEqual((labels == "sub-G1").sum(), 50)

    def test_negative_control_unimodal_has_no_g2_peak(self):
        rng = np.random.default_rng(3)
        dna = pd.Series(1000.0 * rng.lognormal(0, 0.08, 5000))
        _, peaks = dna_content_groups(dna, pd.Series("P1", index=dna.index))
        self.assertFalse(bool(peaks.iloc[0]["has_g2_peak"]))
        self.assertTrue(np.isnan(peaks.iloc[0]["g2_g1_ratio"]))

    def test_labels_do_not_depend_on_intensity_units(self):
        # Metamorphic: multiplying every intensity by 10 must not change groups.
        dna = pd.Series(_dna_mixture())
        groups = pd.Series("P1", index=dna.index)
        labels, _ = dna_content_groups(dna, groups)
        labels_scaled, _ = dna_content_groups(dna * 10, groups)
        pd.testing.assert_series_equal(labels, labels_scaled)

    def test_reference_cells_define_the_g1_peak(self):
        # A treatment that arrests everything in G2 must not move the G1 anchor
        # when negative controls are given as the reference.
        rng = np.random.default_rng(5)
        control = 1000.0 * rng.lognormal(0, 0.08, 3000)
        arrested = 2000.0 * rng.lognormal(0, 0.08, 6000)
        dna = pd.Series(np.concatenate([control, arrested]))
        reference = pd.Series([True] * 3000 + [False] * 6000)
        labels, peaks = dna_content_groups(
            dna, pd.Series("P1", index=dna.index), reference_mask=reference
        )
        self.assertAlmostEqual(peaks.iloc[0]["g1_peak"], 1000.0, delta=60.0)
        self.assertGreater((labels.iloc[3000:] == "G2/M").mean(), 0.9)


class LookupQcStatusTests(unittest.TestCase):
    KEYS = ["Metadata_Plate", "ImageNumber", "Cells_Number_Object_Number"]

    def test_labels_each_row_and_marks_unmatched_rows_not_assessed(self):
        # Non-default index: the result must align to the input rows.
        cells = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1", "P1", "P2"],
                "ImageNumber": [1, 1, 2, 1],
                "Cells_Number_Object_Number": [1, 2, 1, 1],
            },
            index=[10, 11, 12, 13],
        )
        # Same object number on another plate must not leak (key collision guard);
        # P2 is absent from the table, so it was never assessed.
        table = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1", "P1", "P9"],
                "ImageNumber": [1, 1, 2, 1],
                "Cells_Number_Object_Number": [1, 2, 1, 1],
                "Metadata_profile_qc": ["pass", "fail", "pass", "fail"],
            }
        )
        status = lookup_qc_status(cells, table, self.KEYS, "Metadata_profile_qc")
        self.assertEqual(status.index.tolist(), [10, 11, 12, 13])
        self.assertEqual(status.tolist(), ["pass", "fail", "pass", "not assessed"])

    def test_key_dtypes_are_harmonised(self):
        # CSV round-trips turn integer keys into strings or floats; still match.
        cells = pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Site": ["1", "2"]})
        table = pd.DataFrame(
            {"Metadata_Plate": ["P1", "P1"], "Metadata_Site": [1.0, 2.0], "Metadata_image_qc": ["pass", "fail"]}
        )
        status = lookup_qc_status(cells, table, ["Metadata_Plate", "Metadata_Site"], "Metadata_image_qc")
        self.assertEqual(status.tolist(), ["pass", "fail"])

    def test_conflicting_status_for_one_key_raises(self):
        cells = pd.DataFrame({"Metadata_Plate": ["P1"], "Metadata_Site": [1]})
        table = pd.DataFrame(
            {"Metadata_Plate": ["P1", "P1"], "Metadata_Site": [1, 1], "Metadata_image_qc": ["pass", "fail"]}
        )
        with self.assertRaises(ValueError):
            lookup_qc_status(cells, table, ["Metadata_Plate", "Metadata_Site"], "Metadata_image_qc")


if __name__ == "__main__":
    unittest.main()
