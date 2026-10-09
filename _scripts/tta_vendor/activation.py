"""Eq. (1), cleaned from activation_bs/models/gs_model.py.

Parameter names remain compatible with the research GS2 implementation.
The default GELU uses its original tanh formula; gate beta is fixed by default.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class AcTTAActivation(nn.Module):
    """g(x) = phi(x-c) + [neg + (pos-neg)*sigmoid(beta*(x-c))]*(x-c)."""

    def __init__(self, channels, share="channel", preset="relu", learn_neg=True,
                 learn_pos=True, learn_shift=True, learn_beta=False,
                 gelu_approximation="tanh"):
        super().__init__()
        if share not in {"channel", "layer"}:
            raise ValueError("share must be 'channel' or 'layer'")
        if preset not in {"relu", "gelu", None}:
            raise ValueError("preset must be 'relu', 'gelu', or None")
        if channels < 1 or gelu_approximation not in {"tanh", "none"}:
            raise ValueError("invalid channels or GELU approximation")
        self.share = share
        self.preset = preset
        self.gelu_approximation = gelu_approximation
        shape = (channels,) if share == "channel" else ()
        for name, learn, initial in (("neg", learn_neg, 0.), ("pos", learn_pos, 0.),
                                     ("shift", learn_shift, 0.), ("beta", learn_beta, 1.)):
            value = torch.full(shape, initial)
            if learn:
                self.register_parameter(name + "_gsL", nn.Parameter(value))
            else:
                self.register_buffer(name, value)

    def _value(self, name, x):
        value = getattr(self, name + "_gsL", None)
        if value is None:
            value = getattr(self, name)
        if self.share == "layer":
            return value.reshape([1] * x.ndim)
        axis = 1 if x.ndim == 4 else x.ndim - 1
        if x.ndim < 2 or x.shape[axis] != value.numel():
            raise ValueError(f"AcTTA channel mismatch: input {tuple(x.shape)}, "
                             f"parameters {value.numel()}; adaptation cannot be skipped")
        shape = [1] * x.ndim
        shape[axis] = value.numel()
        return value.reshape(shape)

    def forward(self, x):
        neg, pos, shift, beta = [self._value(name, x)
                                 for name in ("neg", "pos", "shift", "beta")]
        xc = x - shift
        if self.preset == "relu":
            base = F.relu(xc)
        elif self.preset == "gelu" and self.gelu_approximation == "tanh":
            # Keep the same operation order as the original GS2, for parity.
            base = 0.5 * xc * (1. + torch.tanh(
                math.sqrt(2. / math.pi) * (xc + 0.044715 * xc ** 3)))
        elif self.preset == "gelu":
            base = F.gelu(xc, approximate="none")
        else:
            base = xc
        gate = torch.sigmoid(beta * xc)
        return base + (neg + (pos - neg) * gate) * xc


GS2 = AcTTAActivation


def replace_activations(model, example_input, prefixes, share="channel",
                        learn_neg=True, learn_pos=True, learn_shift=True,
                        learn_beta=False, gelu_approximation="tanh"):
    """Replace selected ReLU/GELU modules using observed input dimensions.

Run one unlabeled calibration forward in eval mode. Exact module prefixes are
required. Shared activations with different channel counts are rejected before
any replacements. This avoids the research code's silent shape fallback.
Returns a manifest of replaced module paths and dimensions.
"""
    if not prefixes or isinstance(prefixes, str):
        raise ValueError("provide a nonempty list of activation module prefixes")
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, (nn.ReLU, nn.GELU))
                  and any(name == p.rstrip(".") or name.startswith(p.rstrip(".") + ".")
                          for p in prefixes)]
    if not candidates:
        raise ValueError(f"No ReLU/GELU modules match {prefixes}")
    observed = {name: set() for name, _ in candidates}
    hooks = []

    def record(name):
        def hook(module, inputs):
            x = inputs[0]
            if x.ndim < 2:
                raise ValueError(f"Unsupported activation input at {name}: {tuple(x.shape)}")
            observed[name].add(int(x.shape[1 if x.ndim == 4 else -1]))
        return hook

    modes = [(m, m.training) for m in model.modules()]
    try:
        hooks = [m.register_forward_pre_hook(record(name)) for name, m in candidates]
        model.eval()
        with torch.no_grad():
            model(example_input)
    finally:
        for hook in hooks:
            hook.remove()
        for module, mode in modes:
            module.training = mode
    for name, _ in candidates:
        widths = observed[name]
        if len(widths) != 1:
            raise ValueError(f"Activation {name} has observed channel counts {sorted(widths)}. "
                             "Use one activation module per call site before replacement.")

    manifest = []
    for name, original in candidates:
        channels = next(iter(observed[name]))
        replacement = AcTTAActivation(
            channels, share, "relu" if isinstance(original, nn.ReLU) else "gelu",
            learn_neg, learn_pos, learn_shift, learn_beta, gelu_approximation,
        ).to(device=example_input.device, dtype=example_input.dtype)
        replacement.training = original.training
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)
        manifest.append({"path": name, "channels": channels,
                         "preset": replacement.preset, "share": share})
    return manifest
