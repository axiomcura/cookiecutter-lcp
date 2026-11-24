"""Tests for the NB00 figure functions (hca_pipeline.data_qc_figures).

Each figure is built from a tiny fixture, must pass the publication-figures audit, and must
show the numbers it should (read back from the drawn artists, not from the inputs).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hca_pipeline import pubfig as pf  # noqa: E402
from hca_pipeline.data_qc_figures import (  # noqa: E402
    fig_cell_crops,
    fig_dna_content,
    fig_feature_z,
    fig_image_montage,
    fig_image_qc_distributions,
    fig_image_qc_plate_maps,
    fig_profile_qc_rates,
)


def headline(fig) -> str:
    """Text of the explain-mode headline + guide (the only artists with gid 'explain')."""
    return " ".join(" ".join(t.get_text().split()) for t in fig.texts if t.get_gid() == "explain")


def built(make_fig, *args, explain: bool, **kwargs):
    pf.use_style("generic", explain=explain)
    with pf.explain_mode(explain):
        return make_fig(*args, **kwargs)


def site_fixture() -> pd.DataFrame:
    """Six sites on one 96-well plate: one blurred in AOGFP, one saturated in AOPI."""
    rng = np.random.default_rng(0)
    sites = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": ["B03", "B03", "B05", "C04", "C04", "D06"],
            "Metadata_Site": ["1", "2", "1", "1", "2", "1"],
        }
    )
    for channel in ("AOGFP", "AOPI"):
        sites[f"blur_z_{channel}"] = rng.normal(0, 1, len(sites))
        sites[f"pct_saturated_{channel}"] = rng.uniform(0, 0.2, len(sites))
        sites[f"blurred_{channel}"] = False
        sites[f"saturated_{channel}"] = False
    sites.loc[1, ["blur_z_AOGFP", "blurred_AOGFP"]] = [-9.0, True]
    sites.loc[2, ["pct_saturated_AOPI", "saturated_AOPI"]] = [6.0, True]
    failed = sites[["blurred_AOGFP", "saturated_AOPI"]].any(axis=1)
    sites["Metadata_image_qc"] = np.where(failed, "fail", "pass")
    return sites


class ImageQcFigureTests(unittest.TestCase):
    def test_distributions_pass_audit_and_draw_the_limits_given(self):
        sites = site_fixture()
        fig = built(fig_image_qc_distributions, sites, ["AOGFP", "AOPI"], -5.0, 1.0, explain=False)
        pf.assert_figure_ok(fig)
        limits = sorted(
            float(line.get_xdata()[0]) for ax in fig.axes for line in ax.lines if len(set(line.get_xdata())) == 1
        )
        self.assertEqual(limits, [-5.0, -5.0, 1.0, 1.0])  # one blur + one saturation limit per channel
        plt.close(fig)

    def test_headline_counts_failed_sites_from_the_data(self):
        fig = built(fig_image_qc_distributions, site_fixture(), ["AOGFP", "AOPI"], -5.0, 1.0, explain=True)
        text = headline(fig)
        self.assertIn("2 of 6 sites fail image QC", text)
        self.assertIn("1 blurred", text)
        self.assertIn("1 saturated", text)
        plt.close(fig)

    def test_negative_control_no_failures_says_so(self):
        sites = site_fixture()
        sites[[c for c in sites if c.startswith(("blurred_", "saturated_"))]] = False
        sites["Metadata_image_qc"] = "pass"
        fig = built(fig_image_qc_distributions, sites, ["AOGFP", "AOPI"], -5.0, 1.0, explain=True)
        self.assertIn("0 of 6 sites fail image QC", headline(fig))
        plt.close(fig)

    def test_plate_map_shows_failed_sites_per_well(self):
        sites = site_fixture()
        sites.loc[0, "Metadata_image_qc"] = "fail"  # B03 now has 2 failed sites
        fig = built(fig_image_qc_plate_maps, sites, 96, explain=False)
        pf.assert_figure_ok(fig)
        grid = fig.axes[0].images[0].get_array()
        self.assertEqual(grid.shape, (8, 12))
        self.assertEqual(grid[1, 2], 2)  # row B, column 3
        self.assertEqual(grid[1, 4], 1)  # B05
        self.assertEqual(grid[2, 3], 0)  # C04: imaged, none failed
        self.assertTrue(np.ma.is_masked(grid[0, 0]) or np.isnan(grid[0, 0]))  # A01 not imaged
        plt.close(fig)


class MontageTests(unittest.TestCase):
    def tiles(self):
        rng = np.random.default_rng(1)
        failed = [("B03 site 2\nblurred", rng.uniform(0, 1, (40, 40))), ("B05 site 1\nsaturated", np.ones((40, 40)))]
        passed = [("C04 site 1", rng.uniform(0, 1, (40, 40))), ("D06 site 1", None)]
        return failed, passed

    def test_passes_audit_and_all_panels_share_one_brightness_scale(self):
        failed, passed = self.tiles()
        fig = built(fig_image_montage, "AOGFP", failed, passed, (0.0, 0.8), explain=False)
        pf.assert_figure_ok(fig)
        scales = {tuple(ax.images[0].get_clim()) for ax in fig.axes if ax.images}
        self.assertEqual(scales, {(0.0, 0.8)})
        plt.close(fig)

    def test_pixels_at_the_ceiling_are_marked_and_missing_images_are_labelled(self):
        failed, passed = self.tiles()
        fig = built(fig_image_montage, "AOGFP", failed, passed, (0.0, 1.0), explain=False)
        marked = lambda ax: int((~np.ma.getmaskarray(ax.images[-1].get_array())).sum()) if len(ax.images) > 1 else 0  # noqa: E731
        self.assertEqual(marked(fig.axes[1]), 40 * 40)  # all-ones tile: every pixel at the ceiling
        self.assertLess(marked(fig.axes[0]), 40)  # random tile: only a few pixels near 1.0
        missing = fig.axes[4]  # bottom row, second tile (3 columns minimum)
        self.assertTrue(any("not found" in t.get_text() for t in missing.texts))
        plt.close(fig)

    def test_scale_bar_only_when_pixel_size_is_known(self):
        failed, passed = self.tiles()
        without = built(fig_image_montage, "AOGFP", failed, passed, (0, 1), explain=False)
        with_bar = built(fig_image_montage, "AOGFP", failed, passed, (0, 1), pixel_size_um=0.65, explain=False)
        self.assertFalse(any("µm" in t.get_text() for t in without.findobj(plt.Text)))
        self.assertTrue(any("µm" in t.get_text() for t in with_bar.findobj(plt.Text)))
        plt.close(without)
        plt.close(with_bar)


class ProfileQcFigureTests(unittest.TestCase):
    RATES = pd.DataFrame(
        {"under_segmented_nuclei": [4.0, 6.0], "poorly_segmented_nuclei": [1.0, 2.0], "under_segmented_cells": [20.0, 30.0]},
        index=pd.Index(["P1", "P2"], name="Metadata_Plate"),
    )

    def test_rates_bars_start_at_zero_and_show_the_plate_values(self):
        fig = built(fig_profile_qc_rates, self.RATES, 22.5, ["under_segmented_nuclei", "poorly_segmented_nuclei"], explain=False)
        pf.assert_figure_ok(fig)
        heights = sorted(p.get_height() for p in fig.axes[0].patches if p.get_height() > 0)
        self.assertEqual(heights, [1.0, 2.0, 4.0, 6.0, 20.0, 30.0])
        plt.close(fig)

    def test_headline_states_overall_fail_rate_and_marks_inactive_conditions(self):
        fig = built(fig_profile_qc_rates, self.RATES, 22.5, ["under_segmented_nuclei", "poorly_segmented_nuclei"], explain=True)
        self.assertIn("22.5 % of cells fail profile QC", headline(fig))
        legend = [t.get_text() for t in fig.legends[0].get_texts()]
        self.assertTrue(any("under segmented cells" in t and "not applied" in t for t in legend))
        plt.close(fig)

    def test_feature_z_marks_each_threshold_and_reports_tail_share(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, 10_000)
        z_by_feature = {
            "Nuclei_Intensity_IntegratedIntensity_AOGFP": (z, 1.5),
            "Nuclei_AreaShape_Solidity": (-z, -1.6),
        }
        fig = built(fig_feature_z, z_by_feature, explain=False)
        pf.assert_figure_ok(fig)
        positions = sorted(float(ax.lines[0].get_xdata()[0]) for ax in fig.axes if ax.lines)
        self.assertEqual(positions, [-1.6, 1.5])
        plt.close(fig)
        fig = built(fig_feature_z, z_by_feature, explain=True)
        share_above = 100 * np.mean(z > 1.5)
        self.assertIn(f"{share_above:.1f}%", headline(fig))
        plt.close(fig)


class CropAndDnaFigureTests(unittest.TestCase):
    def test_crops_pass_audit_and_leave_missing_images_blank(self):
        rng = np.random.default_rng(2)
        blocks = [
            ("FAIL: under segmented nuclei", [rng.uniform(0, 1, (80, 80)), None, rng.uniform(0, 1, (80, 80))]),
            ("pass", [rng.uniform(0, 1, (80, 80))] * 3),
        ]
        fig = built(fig_cell_crops, blocks, 3, explain=False)
        pf.assert_figure_ok(fig)
        shown = sum(1 for ax in fig.axes if ax.images)
        self.assertEqual(shown, 5)
        plt.close(fig)

    def test_dna_content_headline_counts_plates_with_a_g2_peak(self):
        rng = np.random.default_rng(3)
        g1 = rng.normal(1000, 40, 600)
        two_peaks = np.r_[g1, rng.normal(2000, 80, 300)]
        peaks = pd.DataFrame(
            {
                "Metadata_Plate": ["P1", "P2"],
                "g1_peak": [1000.0, 1000.0],
                "g2_peak": [2000.0, np.nan],
                "g2_g1_ratio": [2.0, np.nan],
                "has_g2_peak": [True, False],
            }
        )
        values = {"P1": two_peaks, "P2": g1}
        fig = built(fig_dna_content, peaks, values, "AOGFP", explain=False)
        pf.assert_figure_ok(fig)
        plt.close(fig)
        fig = built(fig_dna_content, peaks, values, "AOGFP", explain=True)
        self.assertIn("1 of 2 plates show a G2/M peak", headline(fig))
        plt.close(fig)


class ExportTests(unittest.TestCase):
    def test_same_input_gives_identical_files_and_overwrite_false_keeps_existing(self):
        sites = site_fixture()
        pf.use_style("generic", explain=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pf.export(fig_image_qc_distributions, "dist", sites, ["AOGFP", "AOPI"], -5.0, 1.0, outdir=out, formats=("png",))
            first = (out / "dist.png").read_bytes()
            plt.close("all")
            pf.export(fig_image_qc_distributions, "dist", sites, ["AOGFP", "AOPI"], -5.0, 1.0, outdir=out, formats=("png",))
            self.assertEqual(first, (out / "dist.png").read_bytes())
            plt.close("all")
            (out / "dist.png").write_bytes(b"keep me")
            pf.export(
                fig_image_qc_distributions, "dist", sites, ["AOGFP", "AOPI"], -5.0, 1.0,
                outdir=out, formats=("png",), overwrite=False,
            )
            self.assertEqual((out / "dist.png").read_bytes(), b"keep me")
            self.assertTrue((out / "dist_explained.png").exists())
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
