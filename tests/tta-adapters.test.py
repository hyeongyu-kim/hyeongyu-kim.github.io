"""Verify public model placement, update parity, frozen weights, and resets."""
import copy
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_scripts"))
from build_tta_replay import make_adapter, adapt_step, frozen_parameter_check, inspect, input_tensor, learning_rates, enable_checkpointing
from tta_vendor.activation import AcTTAActivation
from tta_vendor.adaptation import EntropyAdapter
from tta_vendor.buffer import BufferLayer
from tta_vendor.wide_resnet import WideResNet


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)
        torch.manual_seed(1)
        cls.source = WideResNet().eval().requires_grad_(False).to(memory_format=torch.channels_last)
        for module in cls.source.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        cls.x = torch.rand(2, 3, 32, 32).contiguous(memory_format=torch.channels_last)

    def test_public_activation_identity_and_derivatives(self):
        activation = AcTTAActivation(16).double()
        x = torch.randn(2, 16, 2, 2, dtype=torch.double) + 0.4
        self.assertTrue(torch.equal(activation(x), torch.relu(x)))
        self.assertEqual(set(dict(activation.named_parameters())), {"shift_gsL", "pos_gsL", "neg_gsL"})
        self.assertTrue(torch.equal(activation.beta, torch.ones(16, dtype=torch.double)))
        with torch.no_grad():
            activation.shift_gsL.fill_(0.05)
            activation.pos_gsL.fill_(0.12)
            activation.neg_gsL.fill_(-0.07)
        self.assertTrue(torch.autograd.gradcheck(activation, (x.requires_grad_(),), eps=1e-6, atol=1e-4))

    def test_frozen_backbone_and_eligible_parameters(self):
        for method, count in {"tent": 17952, "buffer": 9221171, "actta": 3408}.items():
            with self.subTest(method=method):
                model, optimizer = make_adapter(self.source, method)
                learned = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), count)
                for name, parameter in model.named_parameters():
                    if method == "buffer":
                        self.assertEqual(parameter.requires_grad, "buffer" in name)
                    elif method == "actta":
                        self.assertEqual(parameter.requires_grad, name.startswith("block1.") and "_gsL" in name)
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        self.assertEqual(module.weight.requires_grad, method == "tent")
                        self.assertIsNone(module.running_mean)
                with torch.no_grad():
                    before = model(self.x).clone()
                adapt_step(model, optimizer, self.x)
                frozen_parameter_check(model, frozen)
                self.assertTrue(any(not torch.equal(p.detach(), learned[n]) for n, p in model.named_parameters() if p.requires_grad))
                with torch.no_grad():
                    self.assertFalse(torch.equal(before, model(self.x)))

    def test_public_buffer_backbone_and_independent_sites(self):
        model, _ = make_adapter(self.source, "buffer")
        branches = [module for module in model.modules() if isinstance(module, BufferLayer)]
        self.assertEqual(len(branches), 17)  # initial conv + 8 stage-one + 8 stage-two
        self.assertEqual(len({module.conv1.weight.data_ptr() for module in branches}), 17)
        self.assertIsNot(model.block1.layer[0].buffer_1, model.block1.layer[0].buffer_2)
        self.assertTrue(all(not block.use_buffer for block in model.block3.layer))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "alpha_buffer" in name:
                    parameter.zero_()
            reference, _ = make_adapter(self.source, "tent")
            torch.testing.assert_close(model(self.x), reference(self.x), rtol=0, atol=0)
        for name, parameter in self.source.named_parameters():
            self.assertTrue(torch.equal(parameter, dict(model.named_parameters())[name]))

    def test_recorder_step_matches_public_entropy_adapter(self):
        for method in ["tent", "actta"]:
            with self.subTest(method=method):
                model, optimizer = make_adapter(self.source, method)
                public = EntropyAdapter(copy.deepcopy(model), learning_rates(16)[method], "Adam",
                                        adapt_norm=method == "tent", adapt_activation=method == "actta")
                enable_checkpointing(model)
                with torch.no_grad():
                    expected_before = model(self.x).softmax(1).numpy()
                recorded = inspect(model, self.x, np.array([0, 1]), method, optimizer)
                logits = public(self.x)
                np.testing.assert_allclose(recorded["probabilities"], expected_before, rtol=0, atol=6e-8)
                torch.testing.assert_close(logits.softmax(1), torch.tensor(expected_before), rtol=0, atol=0)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, dict(public.model.named_parameters())[name], rtol=0, atol=0)

    def test_checkpointed_buffer_update_matches_native_public_model(self):
        model, optimizer = make_adapter(self.source, "buffer")
        reference = copy.deepcopy(model)
        ref_optimizer = torch.optim.Adam([p for p in reference.parameters() if p.requires_grad], lr=.001)
        enable_checkpointing(model)
        inspect(model, self.x, np.array([0, 1]), "buffer", optimizer)
        adapt_step(reference, ref_optimizer, self.x)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, dict(reference.named_parameters())[name], rtol=0, atol=0)

    def test_reset_restores_parameters_and_optimizer(self):
        for method in ["tent", "buffer", "actta"]:
            first, optimizer = make_adapter(self.source, method)
            initial = copy.deepcopy(first.state_dict())
            adapt_step(first, optimizer, self.x)
            self.assertTrue(optimizer.state)
            fresh, fresh_optimizer = make_adapter(self.source, method)
            self.assertFalse(fresh_optimizer.state)
            self.assertTrue(all(torch.equal(v, fresh.state_dict()[n]) for n, v in initial.items()))

    def test_public_profiles_and_input_preprocessing(self):
        self.assertEqual(learning_rates(4), {"tent": .0001, "buffer": .001, "actta": .001})
        self.assertEqual(learning_rates(128), {"tent": .001, "buffer": .001, "actta": .01})
        self.assertEqual(learning_rates(16), learning_rates(128))
        pixels = np.array([[[[0, 127, 255]]]], dtype=np.uint8)
        torch.testing.assert_close(input_tensor(pixels), torch.tensor([[[[0.]], [[127 / 255]], [[1.]]]]))


if __name__ == "__main__":
    unittest.main()
