"""Verify update eligibility, activation derivatives, and optimizer resets."""
import copy
import sys
from pathlib import Path
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_scripts"))
from build_tta_replay import AdaptiveActivation, BufferResidual, make_adapter, adapt_step, frozen_parameter_check
from tta_resnet import cifar10_resnet20


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(27)
        cls.source = cifar10_resnet20().eval().requires_grad_(False)
        cls.x = torch.randn(4, 3, 32, 32)

    def test_identity_and_derivatives(self):
        activation = AdaptiveActivation(16).double()
        x = torch.randn(2, 16, 2, 2, dtype=torch.double) + 0.4
        self.assertTrue(torch.equal(activation(x), torch.relu(x)))
        with torch.no_grad():
            activation.center.fill_(0.05)
            activation.positive.fill_(0.12)
            activation.negative.fill_(-0.07)
        self.assertTrue(torch.autograd.gradcheck(activation, (x.requires_grad_(),), eps=1e-6, atol=1e-4))

    def test_frozen_backbone_and_eligible_parameters(self):
        for method, count in {"tent": 1568, "buffer": 10252, "actta": 288}.items():
            with self.subTest(method=method):
                model, optimizer = make_adapter(self.source, method)
                learned = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), count)
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        self.assertEqual(module.weight.requires_grad, method == "tent")
                        self.assertIsNone(module.running_mean)
                before = model(self.x).detach().clone()
                adapt_step(model, optimizer, self.x)
                frozen_parameter_check(model, frozen)
                self.assertTrue(any(not torch.equal(p.detach(), learned[n]) for n, p in model.named_parameters() if p.requires_grad))
                self.assertFalse(torch.equal(before, model(self.x).detach()))

    def test_reset_restores_parameters_and_optimizer(self):
        for method in ["tent", "buffer", "actta"]:
            first, optimizer = make_adapter(self.source, method)
            initial = copy.deepcopy(first.state_dict())
            adapt_step(first, optimizer, self.x)
            self.assertTrue(optimizer.state)
            fresh, fresh_optimizer = make_adapter(self.source, method)
            self.assertFalse(fresh_optimizer.state)
            self.assertTrue(all(torch.equal(v, fresh.state_dict()[n]) for n, v in initial.items()))

    def test_buffer_residual_branch(self):
        layer = BufferResidual(16)
        x = torch.randn(2, 16, 8, 8)
        base = torch.relu(x)
        expected = base + layer.alpha * (layer.scale1 * layer.conv1(base) + layer.scale2 * layer.conv3(base))
        self.assertTrue(torch.equal(layer(x), expected))


if __name__ == "__main__":
    unittest.main()
