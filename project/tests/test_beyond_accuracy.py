"""Small known-answer cases for metadata, metrics, and independent groupings."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from project.recsys.analysis.groups import compare_groups, group_definitions
from project.recsys.data import (
    Interaction,
    activity_terciles,
    genre_distribution,
    item_popularity_groups,
    prepare_metadata,
)
from project.recsys.metrics.accuracy import evaluate_accuracy
from project.recsys.metrics.beyond_accuracy import (
    evaluate_beyond_accuracy,
    exposure_gini,
    jensen_shannon,
)


class BeyondAccuracyTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        item_file, user_file = root / "movies.item", root / "users.user"
        item_file.write_text("item_id:token\tclass:token_seq\na\tAction Comedy\nb\tDrama\nc\tComedy\nd\tunknown\n")
        user_file.write_text("user_id:token\tgender:token\nf\tF\nm\tM\nmissing\t?\ncold\tF\n")
        rows = [Interaction("f", "a"), Interaction("m", "b"), Interaction("missing", "a")]
        self.statistics = prepare_metadata(rows, item_file, user_file, catalogue=("a", "b", "c", "d"))
        self.partial_statistics = prepare_metadata(rows, root / "absent.item", root / "absent.user", catalogue=("a", "b", "c", "d"))
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

    def test_missing_metadata_files_run_without_fabricating_metadata_or_reports(self) -> None:
        accuracy = evaluate_accuracy(self.rankings, self.relevant, 2)
        result = evaluate_beyond_accuracy(self.rankings, self.partial_statistics, accuracy["per_user"], 2)
        self.assertEqual(self.partial_statistics.item_features, {})
        self.assertEqual(self.partial_statistics.gender_groups, {})
        self.assertIsNone(result["aggregate"]["ILD@2"])
        self.assertIsNone(result["aggregate"]["GenreMiscalibration@2"])
        self.assertIsNone(result["gender_fairness"]["gaps"]["MRR@2"]["absolute"])
        self.assertEqual(result["aggregate"]["CatalogueCoverage@2"], 1)
        self.assertIsNotNone(result["aggregate"]["UPD@2"])
        report = compare_groups(self.rankings, self.relevant, self.partial_statistics, 2,
                                minimum_group_size=1)
        self.assertEqual((report["status"], report["owner"]), ("complete", "Victor"))
        self.assertEqual(report["pending"], {"gender": self.partial_statistics.pending["metadata"]})
        gender = report["user_groups"]["gender"]
        self.assertEqual(gender["F"]["n_users"], 0)
        self.assertEqual(gender["F"]["reason"], "No evaluation users in this group")
        self.assertEqual(gender["unknown"]["n_users"], 3)
        self.assertIsNone(gender["unknown"]["beyond_accuracy"]["ILD@2"])
        self.assertEqual(gender["unknown"]["availability"]["ILD@2"]["n_users"], 0)
        self.assertIsNotNone(report["overall"]["beyond_accuracy"]["MeanSelfInformation@2"])
        json.dumps(report, allow_nan=False)

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
        self.assertEqual(definitions["gender"]["members"]["cold"], "F")
        self.assertEqual(definitions["status"], "complete")

    def test_metadata_loaded_from_atomic_files(self) -> None:
        features = {"a": ("Action", "Comedy"), "b": ("Drama",), "c": ("Comedy",), "d": ("unknown",)}
        self.assertEqual(self.statistics.item_features, features)
        self.assertEqual(self.statistics.genre_profiles,
                         {user: genre_distribution(history, features)
                          for user, history in self.statistics.histories.items()})
        self.assertEqual(self.statistics.gender_groups, {"f": "F", "m": "M", "cold": "F"})
        self.assertEqual(self.statistics.pending, {})

    def test_missing_items_get_no_genres(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        item_file = Path(directory.name) / "few.item"
        item_file.write_text("item_id:token\tmovie_title:token_seq\tclass:token_seq\na\tA Film\tDrama Drama\n")
        statistics = prepare_metadata([Interaction("u", "a"), Interaction("v", "b")], item_file, None, catalogue=("a", "b"))
        self.assertEqual(statistics.item_features, {"a": ("Drama",)})
        self.assertEqual(set(statistics.genre_profiles), {"u"})
        self.assertIn("item_genres", statistics.pending)
        self.assertIn("genre_profiles", statistics.pending)
        self.assertIn("metadata", statistics.pending) 

if __name__ == "__main__":
    unittest.main()