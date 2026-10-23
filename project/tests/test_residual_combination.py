import unittest

import numpy as np
import torch

from project.recsys.residual_combination import ResidualMLP


class ResidualTests(unittest.TestCase):
    def test_initialization_is_exact_mf_and_features_remain_frozen(self):
        torch.manual_seed(42)
        users, items = torch.randn(3, 4), torch.randn(5, 4)
        baseline = users @ items.T
        model = ResidualMLP(users, items, 8, baseline)
        u, i = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
        torch.testing.assert_close(model(u, i), baseline[u, i], rtol=0, atol=0)
        np.testing.assert_array_equal(model.full_scores(), baseline.numpy())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model(u, i).sum().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(model(u, i), baseline[u, i]))
        torch.testing.assert_close(model.users, users, rtol=0, atol=0)
        torch.testing.assert_close(model.items, items, rtol=0, atol=0)
        torch.testing.assert_close(model.baseline, baseline, rtol=0, atol=0)

    def test_nonzero_correction_full_scores_and_checkpoint_reload(self):
        torch.manual_seed(43)
        model = ResidualMLP(torch.randn(3, 4), torch.randn(5, 7), 8, torch.randn(3, 5))
        torch.nn.init.normal_(model.output.weight)
        torch.nn.init.normal_(model.output.bias)
        users, items = torch.arange(3).repeat_interleave(5), torch.arange(5).repeat(3)
        expected = model(users, items).detach().numpy().reshape(3, 5)
        np.testing.assert_allclose(model.full_scores(2), expected, atol=1e-6)
        restored = ResidualMLP(model.users, model.items, 8, model.baseline)
        restored.load_state_dict(model.state_dict())
        np.testing.assert_array_equal(restored.full_scores(), model.full_scores())


if __name__ == '__main__':
    unittest.main()
