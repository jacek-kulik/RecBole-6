import unittest

from project.recsys.contracts import ScoreTable
from project.recsys.hybrid_report import test_candidates


class TestHybridReport(unittest.TestCase):
    def test_test_mask_changes_eligibility_without_changing_scores(self):
        table = ScoreTable("m", "split", "valid", {"u": {"v": 9., "t": 8., "n": 2.}})
        result = test_candidates(table, {"u": {"h"}}, {"u": {"v"}},
                                 {"u": {"t"}}, {"h", "v", "t", "n"})
        self.assertEqual(result.partition, "test")
        self.assertEqual(result.scores, {"u": {"t": 8., "n": 2.}})
        self.assertEqual(table.scores["u"]["v"], 9.)

    def test_rejects_overlapping_partitions_and_incomplete_candidates(self):
        table = ScoreTable("m", "split", "valid", {"u": {"v": 9., "t": 8.}})
        with self.assertRaises(ValueError):
            test_candidates(table, {"u": {"h"}}, {"u": {"v"}},
                            {"u": {"v"}}, {"h", "v", "t"})
        with self.assertRaises(ValueError):
            test_candidates(table, {"u": {"h"}}, {"u": {"v"}},
                            {"u": {"t"}}, {"h", "v", "t", "missing"})
