"""Record real entropy updates; the browser only inspects measured outputs."""
import base64
import copy
import gzip
import hashlib
import io
import json
import pathlib
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from tta_resnet import cifar10_resnet20

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "tta-data"
OUT = ROOT / "assets/lab/tta"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["source", "tent", "buffer", "actta"]
DOMAINS = ["gaussian_noise", "brightness", "defocus_blur", "jpeg_compression"]
torch.set_num_threads(2)
torch.manual_seed(27)


class AdaptiveActivation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(channels))
        self.positive = nn.Parameter(torch.zeros(channels))
        self.negative = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        z = x - self.center[None, :, None, None]
        gate = torch.sigmoid(z)
        slope = self.negative[None, :, None, None] + (self.positive - self.negative)[None, :, None, None] * gate
        return F.relu(z) + slope * z


class BufferResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.scale1 = nn.Parameter(torch.tensor(0.5))
        self.scale2 = nn.Parameter(torch.tensor(0.5))
        self.alpha = nn.Parameter(torch.tensor(1e-5))
        self.last_residual = None

    def forward(self, x):
        base = F.relu(x)
        self.last_residual = self.alpha * (self.scale1 * self.conv1(base) + self.scale2 * self.conv3(base))
        return base + self.last_residual


def make_adapter(source, method):
    torch.manual_seed(27)
    model = copy.deepcopy(source).train().requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            if method == "tent":
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
    if method in ["buffer", "actta"]:
        factory = BufferResidual if method == "buffer" else AdaptiveActivation
        model.relu = factory(16)
        for block in model.layer1:
            block.relu = factory(16)
        if method == "actta":
            model.layer2[0].relu = factory(32)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    return model, optimizer


def frozen_parameter_check(model, frozen):
    for name, value in model.named_parameters():
        if name in frozen and not torch.equal(value.detach(), frozen[name]):
            raise AssertionError("Frozen backbone changed: " + name)


def adapt_step(model, optimizer, x):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    log_prob = logits.log_softmax(-1)
    entropy = -(log_prob.exp() * log_prob).sum(-1).mean()
    entropy.backward()
    optimizer.step()


def stats(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    confidence = p.max(-1)
    correct = p.argmax(-1) == labels
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.
    for b in range(10):
        inside = bins == b
        if inside.any():
            ece += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return {"accuracy": float(correct.mean()), "ece": float(ece), "confidence": float(confidence.mean()),
            "entropy": float(-(p * np.log(np.maximum(p, 1e-30))).sum(-1).mean())}


def numbers(value):
    return np.round(value.detach().cpu().numpy(), 7).tolist()


def inspect(model, x, labels, method):
    captured = {}
    layers = {"stem": (model.relu, model.bn1), "early": (model.layer1[0].relu, model.layer1[0].bn1)}
    handles = []
    for name, (activation, bn) in layers.items():
        def hook(module, inputs, output, name=name, bn=bn):
            if name in captured:
                return
            item = {"map": F.adaptive_avg_pool2d(output[:4], 8).detach().clone().numpy(),
                    "gamma": numbers(bn.weight), "beta": numbers(bn.bias)}
            if isinstance(module, BufferResidual):
                item.update(alpha=float(module.alpha.detach()), scale1=float(module.scale1.detach()), scale2=float(module.scale2.detach()),
                            residual=F.adaptive_avg_pool2d(module.last_residual[:4], 8).detach().clone().numpy())
            if isinstance(module, AdaptiveActivation):
                item.update(center=numbers(module.center), positive=numbers(module.positive), negative=numbers(module.negative))
            captured[name] = item
        handles.append(activation.register_forward_hook(hook))
    with torch.no_grad():
        raw = model(x).softmax(-1)
        probabilities = numbers(raw)
    for handle in handles:
        handle.remove()
    if not np.array_equal(raw.argmax(-1).numpy(), np.argmax(probabilities, axis=-1)):
        raise AssertionError("Decimal storage changed a prediction")
    return {"probabilities": probabilities, "metrics": stats(probabilities, labels), "layers": captured}


def input_tensor(images):
    x = torch.from_numpy(images.copy()).float().permute(0, 3, 1, 2) / 255.
    mean = torch.tensor([0.4914, 0.4822, 0.4465])[None, :, None, None]
    std = torch.tensor([0.2023, 0.1994, 0.2010])[None, :, None, None]
    return (x - mean) / std


def png(image):
    file = io.BytesIO()
    Image.fromarray(image).save(file, format="PNG")
    return "data:image/png;base64," + base64.b64encode(file.getvalue()).decode()


def encode_trace(trace):
    scales, residual_scales = {}, {}
    for layer in ["stem", "early"]:
        scales[layer] = max(1e-8, max(float(np.abs(f["methods"][m]["layers"][layer]["map"]).max()) for f in trace["frames"] for m in METHODS))
        residual_scales[layer] = max(1e-12, max(float(np.abs(f["methods"]["buffer"]["layers"][layer]["residual"]).max()) for f in trace["frames"]))
    trace["mapScales"] = scales
    trace["residualScales"] = residual_scales
    trace["mapCodec"] = "xor-reference"
    for frame in trace["frames"]:
        frame["featureReference"] = {}
        for layer in ["stem", "early"]:
            reference = None
            for method in METHODS:
                item = frame["methods"][method]["layers"][layer]
                q = np.rint(np.clip(item["map"] / scales[layer], -1, 1) * 127 + 128).astype(np.uint8)
                if reference is None:
                    reference = q
                    frame["featureReference"][layer] = base64.b64encode(reference.tobytes()).decode()
                delta = np.bitwise_xor(q, reference)
                if not np.array_equal(np.bitwise_xor(delta, reference), q):
                    raise AssertionError("Map compression lost data")
                item["map"] = base64.b64encode(delta.tobytes()).decode()
                if "residual" in item:
                    residual = np.rint(np.clip(item["residual"] / residual_scales[layer], -1, 1) * 127 + 128).astype(np.uint8)
                    item["residual"] = base64.b64encode(residual.tobytes()).decode()


def prepare_models():
    models = {"source": source}
    optimizers, frozen = {}, {}
    for method in METHODS[1:]:
        models[method], optimizers[method] = make_adapter(source, method)
        frozen[method] = {n: p.detach().clone() for n, p in models[method].named_parameters() if not p.requires_grad}
    return models, optimizers, frozen


def frame_for(models, images, indices, domain, severity, position, online=None):
    x = input_tensor(images)
    target = labels[indices]
    return {"position": position, "domain": domain, "severity": severity, "indices": indices.tolist(), "labels": target.tolist(),
            "original": [png(i) for i in clean[indices][:4]], "images": [png(i) for i in images[:4]],
            "methods": {m: inspect(models[m], x, target, m) for m in METHODS}, "online": online}


def record_batch(batch, domain, severity):
    images = arrays[domain if domain == "clean" else f"{domain}-{severity}"][:batch]
    indices = np.arange(batch)
    models, optimizers, frozen = prepare_models()
    frames = []
    for update in range(9):
        frames.append(frame_for(models, images, indices, domain, severity, update))
        if update < 8:
            for method in METHODS[1:]:
                adapt_step(models[method], optimizers[method], input_tensor(images))
                frozen_parameter_check(models[method], frozen[method])
    return {"kind": "batch", "batch": batch, "domain": domain, "severity": severity, "frames": frames}


def record_stream(batch, policy):
    models, optimizers, frozen = prepare_models()
    phases = [("clean", 0), ("gaussian_noise", 3), ("defocus_blur", 5), ("clean", 0)]
    frames = []
    for phase, (domain, severity) in enumerate(phases):
        for step in range(4):
            if policy == "reset":
                models, optimizers, frozen = prepare_models()
            indices = np.arange(step * batch, (step + 1) * batch)
            images = arrays[domain if domain == "clean" else f"{domain}-{severity}"][indices]
            x = input_tensor(images)
            online = {}
            with torch.no_grad():
                for method in METHODS:
                    prob = numbers(models[method](x).softmax(-1))
                    online[method] = {"probabilities": prob, "metrics": stats(prob, labels[indices])}
            for method in METHODS[1:]:
                adapt_step(models[method], optimizers[method], x)
                frozen_parameter_check(models[method], frozen[method])
            frame = frame_for(models, images, indices, domain, severity, len(frames), online)
            frame["phase"] = phase
            frames.append(frame)
    return {"kind": "stream", "batch": batch, "policy": policy, "frames": frames}


if __name__ == "__main__":
    clean = np.load(DATA / "clean.npy")
    labels = np.load(DATA / "labels.npy")
    arrays = {"clean": clean}
    for domain in DOMAINS:
        for severity in [1, 3, 5]:
            arrays[f"{domain}-{severity}"] = np.load(DATA / f"{domain}-{severity}.npy")
    source = cifar10_resnet20().eval().requires_grad_(False)
    weights = (DATA / "resnet20.pt").read_bytes()
    if not hashlib.sha256(weights).hexdigest().startswith("4118986f"):
        raise AssertionError("Checkpoint checksum mismatch")
    source.load_state_dict(torch.load(io.BytesIO(weights), weights_only=True, map_location="cpu"))
    for module in source.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    counts = {m: sum(p.numel() for p in make_adapter(source, m)[0].parameters() if p.requires_grad) for m in METHODS[1:]}
    if counts != {"tent": 1568, "buffer": 10252, "actta": 288}:
        raise AssertionError("Unexpected eligible parameter counts " + str(counts))
    with torch.no_grad():
        clean_prob = numbers(source(input_tensor(clean)).softmax(-1))
    manifest = {"schema": 1, "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                "channels": 16, "inspectableImages": 4, "trainableParameters": {"source": 0, **counts}, "sourceParameters": sum(p.numel() for p in source.parameters()),
                "cleanSubset": stats(clean_prob, labels), "protocol": {"seed": 27, "optimizer": "Adam", "learningRate": 0.001, "betas": [0.9, 0.999], "epsilon": 1e-8, "weightDecay": 0,
                "objective": "mean prediction entropy; no labels", "acTtaBeta": 1, "normalizationMean": [0.4914, 0.4822, 0.4465], "normalizationStd": [0.2023, 0.1994, 0.2010],
                "data": json.loads((DATA / "provenance.json").read_text())}, "traces": []}
    conditions = [(f"batch-{b}-clean-0", lambda b=b: record_batch(b, "clean", 0)) for b in [4, 16, 64]]
    for batch in [4, 16, 64]:
        for domain in DOMAINS:
            for severity in [1, 3, 5]:
                conditions.append((f"batch-{batch}-{domain}-{severity}", lambda b=batch, d=domain, s=severity: record_batch(b, d, s)))
        for policy in ["continual", "reset"]:
            conditions.append((f"stream-{batch}-{policy}", lambda b=batch, p=policy: record_stream(b, p)))
    started = time.time()
    for index, (key, record) in enumerate(conditions):
        trace = record()
        encode_trace(trace)
        raw = json.dumps(trace, separators=(",", ":"), allow_nan=False).encode()
        packed = gzip.compress(raw, compresslevel=9, mtime=0)
        filename = key + ".json.gz"
        (OUT / filename).write_bytes(packed)
        item = {"id": key, "file": filename, "kind": trace["kind"], "batch": trace["batch"], "frames": len(trace["frames"]), "bytes": len(packed), "sha256": hashlib.sha256(packed).hexdigest()}
        for field in ["domain", "severity", "policy"]:
            if field in trace:
                item[field] = trace[field]
        manifest["traces"].append(item)
        print(f"{index+1}/45 {key} {len(packed):,} bytes ({time.time()-started:.1f}s)", flush=True)
    manifest["conditions"] = len(manifest["traces"])
    manifest["frames"] = sum(t["frames"] for t in manifest["traces"])
    (ROOT / "assets/lab/tta-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Recorded", manifest["conditions"], "conditions and", manifest["frames"], "frames", flush=True)
