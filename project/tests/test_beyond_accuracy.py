"""Small known-answer cases for metadata, metrics, and independent groupings."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from project.recsys.analysis.groups import compare_groups, group_definitions
from project.recsys.data import Interaction, activity_terciles, item_popularity_groups, prepare_metadata
from project.recsys.metrics.accuracy import evaluate_accuracy
from project.recsys.metrics.beyond_accuracy import evaluate_beyond_accuracy, exposure_gini, jensen_shannon


class BeyondAccuracyTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        item_file, user_file = root / "movies.item", root / "users.user"
        item_file.write_text("item_id:token\tclass:token_seq\na\tAction Comedy\nb\tDrama\nc\tComedy\nd\tunknown\n")
        user_file.write_text("user_id:token\tgender:token\nf\tF\nm\tM\nmissing\t?\ncold\tF\n")
        self.statistics = prepare_metadata(
            [Interaction("f", "a"), Interaction("m", "b"), Interaction("missing", "a")], item_file, user_file,
        )
        self.rankings = {"f": ["a", "b"], "m": ["b", "c"], "missing": ["c", "d"]}
        self.relevant = {"f": {"a", "b"}, "m": {"c"}, "missing": {"a"}}

    def test_hand_calculated_metrics_and_unknown_gender(self) -> None:
        accuracy = evaluate_accuracy(self.rankings, self.relevant, 2)
        result = evaluate_beyond_accuracy(self.rankings, self.statistics, accuracy["per_user"], 2)
        female = result["per_user"]["f"]
        self.assertEqual(female["ILD@2"], 1)
        self.assertAlmostEqual(female["MeanSelfInformation@2"], (math.log2(7 / 3) + math.log2(7 / 2)) / 2)
        self.assertAlmostEqual(female["GenreMiscalibration@2"], -math.log(.505))
        self.assertEqual(jensen_shannon({"head": 1}, {"tail": 1}), 1)
        self.assertAlmostEqual(female["UPD@2"], .5 * math.log2(4 / 3) + .25 * math.log2(2 / 3) + .25)
        self.assertEqual(result["aggregate"]["CatalogueCoverage@2"], 1)
        self.assertAlmostEqual(exposure_gini({"a": 1, "b": 0, "c": 0, "d": 0}), .75)
        discount = 1 / math.log2(3)
        exposures = {"a": 1, "b": 1 + discount, "c": 1 + discount, "d": discount}
        pairwise_gini = sum(abs(a - b) for a in exposures.values() for b in exposures.values()) / (2 * 4 * sum(exposures.values()))
        self.assertAlmostEqual(result["aggregate"]["ExposureGini@2"], pairwise_gini)
        self.assertAlmostEqual(sum(row["share"] for row in result["item_group_exposure"].values()), 1)
        fairness = result["gender_fairness"]
        self.assertEqual(fairness["counts"], {"F": 1, "M": 1, "unknown": 1})
        self.assertEqual(fairness["gaps"]["MRR@2"], {"absolute": .5, "signed_f_minus_m": .5})
        self.assertEqual(self.statistics.genre_profiles["f"], {"Action": .5, "Comedy": .5})
        json.dumps(result, allow_nan=False)

    def test_count_ties_and_popularity_share_buckets(self) -> None:
        boundaries, groups = activity_terciles({"a": 1, "b": 1, "c": 3, "d": 4})
        self.assertEqual(boundaries, (1.0, 3.0))
        self.assertEqual(groups, {"a": "low", "b": "low", "c": "medium", "d": "high"})
        bands = item_popularity_groups({"a": 4, "b": 4, "c": 1, "cold": 0})
        self.assertEqual(bands, {"a": "head", "b": "head", "c": "tail", "cold": "tail"})
        self.assertEqual(self.statistics.item_counts["d"], 0)
        self.assertEqual(self.statistics.item_features["d"], ("unknown",))

    def test_activity_cohorts_use_gender_fairness_and_recompute_coverage(self) -> None:
        statistics = replace(self.statistics, activity_groups={"f": "low", "m": "low", "missing": "high"})
        reports = compare_groups(self.rankings, self.relevant, statistics, 2)
        self.assertEqual(reports["low"]["n_users"], 2)
        self.assertEqual(reports["low"]["beyond_accuracy"]["gender_fairness"]["gaps"]["MRR@2"]["absolute"], .5)
        self.assertEqual(reports["low"]["beyond_accuracy"]["aggregate"]["CatalogueCoverage@2"], .75)
        self.assertEqual(reports["overall"]["beyond_accuracy"]["aggregate"]["CatalogueCoverage@2"], 1)
        self.assertIsNone(reports["high"]["beyond_accuracy"]["gender_fairness"]["gaps"]["MRR@2"]["absolute"])
        self.assertIsNone(reports["medium"]["beyond_accuracy"]["aggregate"]["CatalogueCoverage@2"])
        self.assertEqual(reports["medium"]["n_users"], 0)

    def test_unavailable_profiles_and_single_item_diversity(self) -> None:
        rankings = {"cold": ["d"]}
        accuracy = evaluate_accuracy(rankings, {"cold": {"d"}}, 1)
        result = evaluate_beyond_accuracy(rankings, self.statistics, accuracy["per_user"], 1)
        for metric in ("ILD@1", "GenreMiscalibration@1", "UPD@1"):
            self.assertIsNone(result["aggregate"][metric])
            self.assertEqual(result["availability"][metric]["n_users"], 0)
            self.assertIsNotNone(result["availability"][metric]["reason"])
        self.assertAlmostEqual(result["aggregate"]["MeanSelfInformation@1"], math.log2(7))
        definitions = group_definitions(self.statistics, rankings)
        self.assertEqual(definitions["activity"]["members"]["cold"], "low")
        self.assertEqual(definitions["activity"]["training_counts"]["cold"], 0)


if __name__ == "__main__":
    unittest.main()
