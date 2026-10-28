"""Regressions for the baselines used in the Task 2.2 comparison."""

import unittest

import torch

from recbole.model.general_recommender.pop import Pop
from recbole.model.general_recommender.random import Random


class TinyDataset:
    def __init__(self):
        self.inter_feat = {"item_id": torch.tensor([1, 1, 1, 2, 2, 3])}

    def num(self, field):
        return {"user_id": 4, "item_id": 5}[field]


CONFIG = {
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "NEG_PREFIX": "neg_",
    "device": "cpu",
    "seed": 42,
}


class BaselineModelTests(unittest.TestCase):
    def test_random_gives_each_user_independent_scores(self):
        model = Random(CONFIG, TinyDataset())
        scores = model.full_sort_predict({"user_id": torch.tensor([1, 2, 3])}).reshape(
            3, 5
        )
        self.assertFalse(torch.equal(scores[0], scores[1]))
        self.assertFalse(torch.equal(scores[1], scores[2]))
        self.assertTrue(((scores >= 0) & (scores < 1)).all())

    def test_random_reproduces_scores_across_cpu_batch_sizes(self):
        users = torch.tensor([1, 2, 3])
        model = Random(CONFIG, TinyDataset())
        together = model.full_sort_predict({"user_id": users}).reshape(3, 5)
        model = Random(CONFIG, TinyDataset())
        separately = torch.stack(
            [model.full_sort_predict({"user_id": user.reshape(1)}) for user in users]
        )
        torch.testing.assert_close(together, separately)

    def test_popularity_counts_repeated_training_items(self):
        model = Pop(CONFIG, TinyDataset())
        torch.testing.assert_close(
            model.item_cnt.flatten(), torch.tensor([0, 3, 2, 1, 0])
        )

    def test_popularity_ignores_sampled_training_batches(self):
        model = Pop(CONFIG, TinyDataset())
        batch = {
            "item_id": torch.tensor([1, 1, 2, 4, 4, 4]),
            "label": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        }
        for _ in range(2):
            model.calculate_loss(batch).backward()
        torch.testing.assert_close(
            model.item_cnt.flatten(), torch.tensor([0, 3, 2, 1, 0])
        )

    def test_popularity_pointwise_and_full_sort_use_training_frequencies(self):
        model = Pop(CONFIG, TinyDataset())
        expected = torch.tensor([0.0, 1.0, 2 / 3, 1 / 3, 0.0], dtype=torch.float64)
        full = model.full_sort_predict({"user_id": torch.tensor([1, 2])}).reshape(2, 5)
        torch.testing.assert_close(full, expected.expand(2, -1))
        pointwise = model.predict({"item_id": torch.tensor([1, 2, 3, 4])})
        torch.testing.assert_close(pointwise, expected[1:].float())


if __name__ == "__main__":
    unittest.main()
