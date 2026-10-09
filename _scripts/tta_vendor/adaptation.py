"""Entropy objective from methods/tent.py and methods/tent_gs.py.

Predictions are from the forward used for adaptation, before its optimizer step.
"""

from copy import deepcopy

import torch
from torch import nn

from .activation import AcTTAActivation


class EntropyAdapter(nn.Module):
    """Online TENT or AcTTA_TENT with explicit trainable parameter selection."""

    def __init__(self, model, lr, optimizer="Adam", momentum=0.9,
                 adapt_norm=False, adapt_activation=True):
        super().__init__()
        self.model = model.train()
        self.model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Frozen BN affine still uses target batch statistics.
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            if adapt_norm and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                                                  nn.LayerNorm, nn.GroupNorm)):
                for p in module.parameters(recurse=False):
                    p.requires_grad_(True)
            if adapt_activation and isinstance(module, AcTTAActivation):
                for p in module.parameters(recurse=False):
                    p.requires_grad_(True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("No adaptation parameters selected")
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=(momentum, .999))
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
        else:
            raise ValueError("optimizer must be Adam or SGD")
        self._initial_model = deepcopy(model.state_dict())
        self._initial_optimizer = deepcopy(self.optimizer.state_dict())

    @torch.enable_grad()
    def forward(self, images):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        if not torch.isfinite(entropy):
            raise FloatingPointError("Non-finite adaptation loss")
        entropy.backward()
        self.optimizer.step()
        return logits.detach()

    def reset(self):
        self.model.load_state_dict(self._initial_model, strict=True)
        self.optimizer.load_state_dict(deepcopy(self._initial_optimizer))
        self.optimizer.zero_grad(set_to_none=True)

    def trainable_parameters(self):
        return {name: p.numel() for name, p in self.model.named_parameters() if p.requires_grad}
