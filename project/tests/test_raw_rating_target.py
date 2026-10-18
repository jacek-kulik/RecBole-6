from pathlib import Path
import tempfile
import unittest

import numpy as np

from project.recsys.hybrids import fit_oof_regression
from project.recsys.raw_rating_target import raw_targets


class TestRawRatingTarget(unittest.TestCase):
    def test_external_pairs_recover_raw_targets_and_reject_misalignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.inter"
            path.write_text("user_id:token\titem_id:token\trating:float\nu\ta\t1\nu\tb\t3\nv\tc\t5\n")
            pairs = [{"user": "v", "item": "c"}, {"user": "u", "item": "a"}, {"user": "u", "item": "b"}]
            np.testing.assert_array_equal(raw_targets(pairs, [1, 0, .5], path), [5, 1, 3])
            with self.assertRaises(ValueError):
                raw_targets(pairs, [0, 1, .5], path)

    def test_affine_rating_scale_preserves_ridge_score_order(self):
        x = np.array([[1, 3], [2, 1], [4, -1], [-2, 2]], dtype=float)
        y = np.array([0, .25, 1, .5])
        normalized = fit_oof_regression(x, y, ["a", "b"], 100)
        raw = fit_oof_regression(x, 4 * y + 1, ["a", "b"], 100)
        np.testing.assert_allclose(raw["coefficients"], 4 * np.asarray(normalized["coefficients"]))
        self.assertAlmostEqual(raw["intercept"], 4 * normalized["intercept"] + 1)
