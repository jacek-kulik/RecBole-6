"""Input-contract checks shared by independent ranking metrics."""

import unittest

from project.recsys.metrics._validation import external_id, validate_rankings


class RankingValidationTests(unittest.TestCase):
    def test_external_ids_are_nonempty_strings(self):
        for value in ("u", "42", "0"):
            with self.subTest(value=value):
                self.assertTrue(external_id(value))
        for value in ("", 42, 0, True, None, ["u"]):
            with self.subTest(value=value):
                self.assertFalse(external_id(value))

    def test_truncates_sequences_without_mutating_input(self):
        rankings = {"u1": ["b", "a", "b"], "u2": ("a", "c", "")}
        result = validate_rankings(rankings, 2, {"a", "b", "c"})
        self.assertEqual(result, {"u1": ["b", "a"], "u2": ["a", "c"]})
        result["u1"].append("c")
        self.assertEqual(rankings["u1"], ["b", "a", "b"])

    def test_rejects_invalid_cutoffs(self):
        for k in (0, -1, True, False, 1.0, "1", None):
            with self.subTest(k=k), self.assertRaises(ValueError):
                validate_rankings({"u": ["a"]}, k)

    def test_rejects_invalid_rankings_and_ids(self):
        cases = (
            None, [], {}, [("u", ["a"])], {"": ["a"]}, {1: ["a"]},
            {"u": "a"}, {"u": b"a"}, {"u": {"a"}}, {"u": {"a": 1}},
            {"u": None}, {"u": []}, {"u": [""]}, {"u": [1]},
        )
        for rankings in cases:
            with self.subTest(rankings=rankings), self.assertRaises(ValueError):
                validate_rankings(rankings, 1)

    def test_every_user_needs_k_distinct_items(self):
        for items in (["a"], ["a", "a"]):
            with self.subTest(items=items), self.assertRaises(ValueError):
                validate_rankings({"good": ["a", "b"], "bad": items}, 2)

    def test_catalogue_membership_applies_to_retained_items(self):
        self.assertEqual(validate_rankings({"u": ["a", "outside"]}, 1, {"a"}), {"u": ["a"]})
        for catalogue in (set(), {"b"}):
            with self.subTest(catalogue=catalogue), self.assertRaisesRegex(ValueError, "catalogue"):
                validate_rankings({"u": ["a"]}, 1, catalogue)


if __name__ == "__main__":
    unittest.main()
