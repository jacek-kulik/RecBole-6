"""Saved-result plots tolerate methods that become unavailable on test."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.scripts.plot_reranking import METHODS, plot


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Plotting requires matplotlib")
class PlotRerankingTests(unittest.TestCase):
    def test_missing_test_method_is_marked_unavailable_and_other_results_are_saved(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.addCleanup(plt.close, "all")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "manifest.json").write_text(json.dumps({"models": ["fixture"]}))
            valid = run / "fixture/valid"
            test = run / "fixture/test"
            valid.mkdir(parents=True)
            test.mkdir()
            baseline = {"method": "original", "strength": 0, "ndcg": .4,
                        "GenreMiscalibration@10": 1., "UPD@10": .5, "ExposureGini@10": .8}
            trials = [{**baseline, "method": method, "strength": .5, "ndcg": .39,
                       metric: baseline[metric] / 2} for method, (_, metric, _) in METHODS.items()]
            (valid / "summary.json").write_text(json.dumps([baseline, *trials]))
            (valid / "selection.json").write_text(json.dumps({"methods": {
                method: {"status": "complete", "selected": {"strength": .5}} for method in METHODS}}))
            # A new test candidate lacks genres, so only genre calibration is absent.
            (test / "summary.json").write_text(json.dumps([
                baseline, *(row for row in trials if row["method"] != "genre_calibration")]))
            output = run / "figures"
            with patch.object(plt, "close", wraps=plt.close) as close:
                plot(run, output)

            for name in ("validation-tradeoffs", "test-effects"):
                for suffix in ("png", "pdf"):
                    self.assertGreater((output / f"{name}.{suffix}").stat().st_size, 0)
            validation, tested = [call.args[0] for call in close.call_args_list[-2:]]
            self.assertEqual(len(validation.axes[0].collections), 3)
            test_ax = tested.axes[0]
            self.assertEqual(len(test_ax.patches), 4)
            self.assertEqual(sum(text.get_text() == "strength 0.5" for text in test_ax.texts), 2)
            unavailable = next(text for text in test_ax.texts if text.get_text() == "Unavailable")
            self.assertEqual(unavailable.get_position()[0], 0)
            bounds = unavailable.get_window_extent()
            self.assertGreaterEqual(bounds.x0, test_ax.get_window_extent().x0)
            self.assertLessEqual(bounds.x1, test_ax.get_window_extent().x1)


if __name__ == "__main__":
    unittest.main()
