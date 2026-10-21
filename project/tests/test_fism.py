"""Regression checks for FISM's training logits and inference scores."""

import unittest

import torch

from recbole.model.general_recommender.fism import FISM


class TinyDataset:
    def num(self, field):
        return {"user_id": 3, "item_id": 5}[field]

    def history_item_matrix(self):
        return (
            torch.tensor([[0, 0], [1, 2], [2, 0]]),
            None,
            torch.tensor([0, 2, 1]),
        )


class TestFISM(unittest.TestCase):
    def setUp(self):
        self.model = FISM(
            {
                "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
                "NEG_PREFIX": "neg_", "LABEL_FIELD": "label",
                "device": "cpu", "embedding_size": 2,
                "reg_weights": [0.0, 0.0], "alpha": 0.5, "split_to": 2,
            },
            TinyDataset(),
        )
        with torch.no_grad():
            self.model.item_src_embedding.weight.fill_(0.5)
            self.model.item_dst_embedding.weight.fill_(0.25)
            self.model.user_bias.copy_(torch.tensor([0., -3., 2.]))
            self.model.item_bias.copy_(torch.tensor([0., 0., 0., -1., 1.]))
        self.users = torch.tensor([1, 2])
        self.items = torch.tensor([3, 4])
        self.logits = torch.tensor([0.5 ** 0.5 * 0.5 - 4., 3.25])

    def test_training_uses_logits_and_correct_gradients(self):
        labels = torch.tensor([1., 0.])
        torch.testing.assert_close(self.model(self.users, self.items), self.logits)
        loss = self.model.calculate_loss(
            {"user_id": self.users, "item_id": self.items, "label": labels}
        )
        torch.testing.assert_close(
            loss, torch.nn.functional.binary_cross_entropy_with_logits(self.logits, labels)
        )
        loss.backward()
        torch.testing.assert_close(
            self.model.user_bias.grad[1:], (self.logits.sigmoid() - labels) / 2
        )

    def test_candidate_exclusion_matches_explicit_history_and_gradients(self):
        torch.manual_seed(42)
        with torch.no_grad():
            self.model.item_src_embedding.weight.normal_()
            self.model.item_dst_embedding.weight.normal_()
        # Include repeated candidates, a singleton, and an empty history.
        for history in ([1, 2], [1, 1], [1], []):
            self.model.history_item_matrix[1] = torch.tensor((history + [0, 0])[:2])
            self.model.history_lens[1] = len(history)
            for item in (1, 3):
                for alpha in (0.0, 0.5, 1.0):
                    with self.subTest(history=history, item=item, alpha=alpha):
                        self.model.alpha = alpha
                        retained = torch.tensor([i for i in history if i != item], dtype=torch.long)
                        source = self.model.item_src_embedding(retained).sum(0)
                        expected = (
                            (source * self.model.item_dst_embedding.weight[item]).sum()
                            * max(len(retained), 1) ** (-alpha)
                            + self.model.user_bias[1] + self.model.item_bias[item]
                        )
                        actual = self.model(torch.tensor([1]), torch.tensor([item]))[0]
                        torch.testing.assert_close(actual, expected)
                        parameters = tuple(self.model.parameters())
                        for actual_grad, expected_grad in zip(
                            torch.autograd.grad(actual, parameters),
                            torch.autograd.grad(expected, parameters),
                        ):
                            torch.testing.assert_close(actual_grad, expected_grad)
                        for split_to in (0, 2):
                            self.model.split_to = split_to
                            full = self.model.full_sort_predict({"user_id": torch.tensor([1])})
                            torch.testing.assert_close(full[item], expected)

    def test_pointwise_probability_agrees_with_full_sort_logits(self):
        interaction = {"user_id": self.users, "item_id": self.items}
        torch.testing.assert_close(self.model.predict(interaction), self.logits.sigmoid())
        for split_to in (0, 2):
            self.model.split_to = split_to
            full = self.model.full_sort_predict(interaction).reshape(2, 5)
            torch.testing.assert_close(full[torch.arange(2), self.items], self.logits)

    def test_history_sum_matches_expanded_scores_and_gradients(self):
        torch.manual_seed(42)
        with torch.no_grad():
            self.model.item_src_embedding.weight.normal_()
            self.model.item_dst_embedding.weight.normal_()
        for alpha in (0.0, 0.5, 1.0):
            self.model.alpha = alpha
            histories = self.model.history_item_matrix[self.users]
            expanded = self.model.item_src_embedding(histories)
            target = self.model.item_dst_embedding(self.items)
            similarities = torch.bmm(expanded, target.unsqueeze(2)).squeeze(2)
            reference = (
                (similarities * self.model.mask_mat[self.users]).sum(1)
                * self.model.history_lens[self.users].float().pow(-alpha)
                + self.model.user_bias[self.users] + self.model.item_bias[self.items]
            )
            actual = self.model(self.users, self.items)
            torch.testing.assert_close(actual, reference)
            parameters = tuple(self.model.parameters())
            for actual_grad, reference_grad in zip(
                torch.autograd.grad(actual.sum(), parameters),
                torch.autograd.grad(reference.sum(), parameters),
            ):
                torch.testing.assert_close(actual_grad, reference_grad)


if __name__ == "__main__":
    unittest.main()
