"""Record real entropy updates; the browser only inspects measured outputs."""
import base64
import copy
import gc
import gzip
import hashlib
import io
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from tta_vendor.activation import AcTTAActivation, replace_activations
from tta_vendor.adaptation import EntropyAdapter
from tta_vendor.buffer import BufferLayer, WideResNetTTA
from tta_vendor.wide_resnet import WideResNet

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "tta-data"
OUT = ROOT / "assets/lab/tta"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["source", "tent", "buffer", "actta"]
DOMAINS = ["gaussian_noise", "brightness", "defocus_blur", "jpeg_compression"]
SEED = 1
BATCHES = [4, 16, 128]
INSPECT_CHANNELS = 16
CHECKPOINT_SHA256 = "6ed1c75dad63e8ebdbef365020dd1a7eee5e79cf8a1438bd3bdc3b9c7aab3d44"
ACTTA_COMMIT = "6bd3b9ab090240ec67038257c5fd9d8f11a283fc"
BUFFER_COMMIT = "b266f3c7904aa760b21618ebe5fba0fcd864bc16"
torch.set_num_threads(8)
torch.manual_seed(SEED)


def learning_rates(batch):
    # Public CIFAR profiles: BS4 and BS128. BS16 reuses the main BS128 rates.
    return {"tent": 0.0001 if batch == 4 else 0.001,
            "buffer": 0.001, "actta": 0.001 if batch == 4 else 0.01}


def make_adapter(source, method, batch=16):
    torch.manual_seed(SEED)
    if method == "buffer":
        model = WideResNetTTA(use_buffers=[True, True, False], alpha_init=1e-5)
        missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
        if unexpected or any("buffer" not in key for key in missing):
            raise AssertionError("Buffer backbone checkpoint mismatch")
        model.to(memory_format=torch.channels_last)
    else:
        model = copy.deepcopy(source)
    if method == "actta":
        model.activation_manifest = replace_activations(
            model, torch.zeros(1, 3, 32, 32), ["block1"], share="channel", learn_beta=False)
    if method in ["tent", "actta"]:
        adapter = EntropyAdapter(model, learning_rates(batch)[method], "Adam",
                                 adapt_norm=method == "tent", adapt_activation=method == "actta")
        return adapter.model, adapter.optimizer
    model.train().requires_grad_(False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("buffer" in name)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rates(batch)[method], betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    return model, optimizer


def frozen_parameter_check(model, frozen):
    for name, value in model.named_parameters():
        if name in frozen and not torch.equal(value.detach(), frozen[name]):
            raise AssertionError("Frozen backbone changed: " + name)


def enable_checkpointing(model):
    """Recompute pure residual blocks during backward; BN running state is disabled."""
    for stage in [model.block1, model.block2, model.block3]:
        for block in stage.layer:
            original = block.forward
            def forward(x, original=original):
                if torch.is_grad_enabled():
                    return gradient_checkpoint(original, x, use_reentrant=False, preserve_rng_state=False)
                return original(x)
            block.forward = forward


def adapt_step(model, optimizer, x):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
    if not torch.isfinite(entropy):
        raise FloatingPointError("Non-finite adaptation entropy")
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


def inspect(model, x, labels, method, optimizer=None):
    captured = {}
    # Both sites are inside the official AcTTA block1 prefix. Show channels 0–15.
    blocks = {"stem": model.block1.layer[0], "early": model.block1.layer[1]}
    handles = []
    for name, block in blocks.items():
        @torch.no_grad()
        def hook(module, inputs, output, name=name, block=block):
            if name in captured:
                return  # checkpoint backward may invoke this site again
            item = {"map": F.adaptive_avg_pool2d(output[:4, :INSPECT_CHANNELS], 8).detach().clone().numpy(),
                    "gamma": numbers(block.bn1.weight[:INSPECT_CHANNELS]),
                    "beta": numbers(block.bn1.bias[:INSPECT_CHANNELS])}
            if method == "buffer":
                # Capture the branch's real output; add it exactly as public forward does.
                residual = output * block.alpha_buffer_1
                corrected = inputs[0] + residual
                item.update(map=F.adaptive_avg_pool2d(corrected[:4, :INSPECT_CHANNELS], 8).detach().clone().numpy(),
                            alpha=float(block.alpha_buffer_1.detach()), scale1=float(module.scale1.detach()), scale2=float(module.scale2.detach()),
                            residual=F.adaptive_avg_pool2d(residual[:4, :INSPECT_CHANNELS], 8).detach().clone().numpy())
            if isinstance(module, AcTTAActivation):
                item.update(center=numbers(module.shift_gsL[:INSPECT_CHANNELS]),
                            positive=numbers(module.pos_gsL[:INSPECT_CHANNELS]), negative=numbers(module.neg_gsL[:INSPECT_CHANNELS]))
            captured[name] = item
        activation = block.buffer_1 if method == "buffer" else block.relu1
        handles.append(activation.register_forward_hook(hook))
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(optimizer is not None):
        logits = model(x)
        raw = logits.softmax(-1)
        probabilities = numbers(raw)
    for handle in handles:
        handle.remove()
    if not np.array_equal(raw.argmax(-1).numpy(), np.argmax(probabilities, axis=-1)):
        raise AssertionError("Decimal storage changed a prediction")
    if optimizer is not None:
        entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        if not torch.isfinite(entropy):
            raise FloatingPointError("Non-finite adaptation entropy")
        entropy.backward()
        optimizer.step()
    return {"probabilities": probabilities, "metrics": stats(probabilities, labels), "layers": captured}


def input_tensor(images):
    x = torch.from_numpy(images.copy()).float().permute(0, 3, 1, 2) / 255.
    # The public Standard WRN checkpoint consumes unnormalized [0, 1] RGB.
    return x.contiguous(memory_format=torch.channels_last)


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


def prepare_method(method, batch):
    model, optimizer = make_adapter(source, method, batch)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    enable_checkpointing(model)
    return model, optimizer, frozen


def frame_data(images, indices, domain, severity, position, source_state):
    return {"position": position, "domain": domain, "severity": severity, "indices": indices.tolist(), "labels": labels[indices].tolist(),
            "original": [png(i) for i in clean[indices][:4]], "images": [png(i) for i in images[:4]],
            "methods": {"source": copy.deepcopy(source_state)}, "online": None}


def record_batch(batch, domain, severity):
    images = arrays[domain if domain == "clean" else f"{domain}-{severity}"][:batch]
    indices = np.arange(batch)
    x = input_tensor(images)
    source_state = inspect(source, x, labels[indices], "source")
    frames = [frame_data(images, indices, domain, severity, update, source_state) for update in range(9)]
    for method in METHODS[1:]:
        model, optimizer, frozen = prepare_method(method, batch)
        for update, frame in enumerate(frames):
            frame["methods"][method] = inspect(model, x, labels[indices], method, optimizer if update < 8 else None)
            if update < 8:
                frozen_parameter_check(model, frozen)
        del model, optimizer, frozen
        gc.collect()
        if batch == 128:
            print(f"  B128 {domain}-{severity}: {method} complete", flush=True)
    return {"kind": "batch", "batch": batch, "domain": domain, "severity": severity, "frames": frames}


def record_stream(batch, policy):
    phases = [("clean", 0), ("gaussian_noise", 3), ("defocus_blur", 5), ("clean", 0)]
    frames = []
    for phase, (domain, severity) in enumerate(phases):
        for step in range(4):
            indices = np.arange(step * batch, (step + 1) * batch)
            images = arrays[domain if domain == "clean" else f"{domain}-{severity}"][indices]
            x = input_tensor(images)
            source_state = frames[step]["methods"]["source"] if phase == 3 else inspect(source, x, labels[indices], "source")
            frame = frame_data(images, indices, domain, severity, len(frames), source_state)
            frame["online"] = {"source": {"probabilities": source_state["probabilities"], "metrics": source_state["metrics"]}}
            frame["phase"] = phase
            frames.append(frame)
    for method in METHODS[1:]:
        model, optimizer, frozen = prepare_method(method, batch)
        for frame in frames:
            if policy == "reset" and frame["position"] > 0:
                del model, optimizer, frozen
                gc.collect()
                model, optimizer, frozen = prepare_method(method, batch)
            indices = np.asarray(frame["indices"])
            domain, severity = frame["domain"], frame["severity"]
            images = arrays[domain if domain == "clean" else f"{domain}-{severity}"][indices]
            x = input_tensor(images)
            pre = inspect(model, x, labels[indices], method, optimizer)
            frame["online"][method] = {"probabilities": pre["probabilities"], "metrics": pre["metrics"]}
            frozen_parameter_check(model, frozen)
            frame["methods"][method] = inspect(model, x, labels[indices], method)
        del model, optimizer, frozen
        gc.collect()
        if batch == 128:
            print(f"  B128 stream-{policy}: {method} complete", flush=True)
    return {"kind": "stream", "batch": batch, "policy": policy, "frames": frames}


if __name__ == "__main__":
    clean = np.load(DATA / "clean.npy")
    labels = np.load(DATA / "labels.npy")
    arrays = {"clean": clean}
    for domain in DOMAINS:
        for severity in [1, 3, 5]:
            arrays[f"{domain}-{severity}"] = np.load(DATA / f"{domain}-{severity}.npy")
    source = WideResNet().eval().requires_grad_(False).to(memory_format=torch.channels_last)
    weights = (DATA / "Standard.pt").read_bytes()
    if hashlib.sha256(weights).hexdigest() != CHECKPOINT_SHA256:
        raise AssertionError("Checkpoint checksum mismatch")
    checkpoint = torch.load(io.BytesIO(weights), weights_only=False, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    source.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    del checkpoint, state, weights
    for module in source.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    counts = {m: sum(p.numel() for p in make_adapter(source, m)[0].parameters() if p.requires_grad) for m in METHODS[1:]}
    if counts != {"tent": 17952, "buffer": 9221171, "actta": 3408}:
        raise AssertionError("Unexpected eligible parameter counts " + str(counts))
    with torch.no_grad():
        clean_prob = sum([numbers(source(input_tensor(chunk)).softmax(-1)) for chunk in np.array_split(clean, 4)], [])
    manifest = {"schema": 2, "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                "channels": 16, "inspectableImages": 4, "trainableParameters": {"source": 0, **counts}, "sourceParameters": sum(p.numel() for p in source.parameters()),
                "model": "RobustBench Standard WRN-28-10", "batches": BATCHES,
                "inspectedLayers": {"stem": {"path": "block1.layer.0.relu1", "channels": 16},
                                    "early": {"path": "block1.layer.1.relu1", "channels": 160}},
                "implementation": {"actta": {"repository": "hyeongyu-kim/actta", "commit": ACTTA_COMMIT, "prefixes": ["block1"]},
                                   "buffer": {"repository": "hyeongyu-kim/Buffer_TTA", "commit": BUFFER_COMMIT, "stages": [True, True, False], "alphaInit": 1e-5}},
                "cleanSubset": stats(clean_prob, labels), "protocol": {"seed": SEED, "optimizer": "Adam",
                "learningRatesByBatch": {str(b): learning_rates(b) for b in BATCHES}, "betas": [0.9, 0.999], "epsilon": 1e-8, "weightDecay": 0,
                "objective": "mean prediction entropy; no labels", "acTtaBeta": 1, "inputRange": [0, 1],
                "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "dtype": "float32", "memoryFormat": "channels_last", "threads": torch.get_num_threads(),
                            "gradientCheckpointing": "nonreentrant residual blocks", "armsRecordedSequentially": True},
                "scope": "mechanism demo; fixed-order subsets and repeated batches, not a full paper benchmark",
                "data": json.loads((DATA / "provenance.json").read_text())}, "traces": []}
    conditions = [(f"batch-{b}-clean-0", lambda b=b: record_batch(b, "clean", 0)) for b in BATCHES]
    for batch in BATCHES:
        for domain in DOMAINS:
            for severity in [1, 3, 5]:
                conditions.append((f"batch-{batch}-{domain}-{severity}", lambda b=batch, d=domain, s=severity: record_batch(b, d, s)))
        for policy in ["continual", "reset"]:
            conditions.append((f"stream-{batch}-{policy}", lambda b=batch, p=policy: record_stream(b, p)))
    protocol_hash = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "traces"}, sort_keys=True).encode()
                                   + pathlib.Path(__file__).read_bytes()).hexdigest()
    started = time.time()
    for index, (key, record) in enumerate(conditions):
        print(f"Recording {index+1}/45 {key}", flush=True)
        filename = key + ".json.gz"
        destination = OUT / filename
        trace = None
        if "--resume" in sys.argv and destination.exists():
            packed = destination.read_bytes()
            previous = json.loads(gzip.decompress(packed))
            if previous.get("recordingProtocolSha256") == protocol_hash:
                trace = previous
        if trace is None:
            trace = record()
            trace["recordingProtocolSha256"] = protocol_hash
            encode_trace(trace)
            raw = json.dumps(trace, separators=(",", ":"), allow_nan=False).encode()
            packed = gzip.compress(raw, compresslevel=9, mtime=0)
            destination.write_bytes(packed)
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
