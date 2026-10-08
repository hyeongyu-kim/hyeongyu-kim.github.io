# Reproducing the TTA comparison

The Lab replays measured PyTorch updates on CIFAR-10/C. It does not fit a substitute browser model or synthesize scores. A small manifest loads first; only the selected experiment is downloaded. Save an offline copy includes all 45 recorded experiments in one self-contained HTML file.

## Setup and record

Use Python 3.12, `torch==2.5.1+cpu`, NumPy, Pillow, and PyArrow. From the repository root:

```bash
python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy Pillow pyarrow
python _scripts/fetch_tta_data.py --mirror
python _scripts/build_tta_replay.py
python tests/tta-adapters.test.py
npm ci
npm run test:lab
node _scripts/build_lab_offline.mjs
```

The fetcher writes to `../tta-data/`. Without `--mirror`, it uses the original CIFAR-10 Python archive and verifies MD5 `c58f30108f718f92721af3b95e74349a`. The mirror option verifies the complete uoft-cs test Parquet SHA-256 `841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de`. Both preserve original test order. Corrupted labels are checked against the selected clean labels before recording.

Corruptions are read by verified byte ranges from the original [CIFAR-10-C archive](https://zenodo.org/records/2535967): Gaussian noise, brightness, defocus blur, and JPEG compression, at severities 1, 3, and 5. Array shape, dtype, and layout are validated. Slice SHA-256 checksums and the complete checkpoint hash are recorded in the manifest. The source checkpoint must start with SHA-256 `4118986f` and is loaded with `weights_only=True`.

The model code is vendored from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models), commit `786c16252c0fc58ee9adac063f8337cc4a7a497a`, with its BSD-3-Clause notice retained. Model: CIFAR-10 ResNet-20 (272,474 source parameters).

## Controlled interventions

| Arm | Updated parameters | Placement | Count |
| --- | --- | --- | ---: |
| Source | None | Stored source BN statistics | 0 |
| Tent | BN gamma and beta | All BN layers | 1,568 |
| Buffer | New convolutions, mixing coefficients, residual alpha | Stem and stage-one shared activation modules | 10,252 |
| AcTTA | Per-channel center and positive/negative response | Stem, stage one, first stage-two block | 288 |

All adaptive arms use current-batch BN statistics, including before their first gradient update. Their step-zero difference from Source is therefore not a learned change. Every update minimizes mean prediction entropy with Adam: learning rate 0.001, betas `(0.9, 0.999)`, epsilon `1e-8`, no weight decay, seed 27. Labels are used only to evaluate recorded probabilities. All ineligible backbone parameters are checked for bitwise equality after each update.

Buffer follows its [public residual branch](https://github.com/hyeongyu-kim/Buffer_TTA/blob/main/models/custom_standard.py): ReLU followed by alpha times a learned mixture of 1×1 and 3×3 convolutions. Alpha starts at `1e-5`; both mixing coefficients start at 0.5 and remain trainable. AcTTA follows [Eq. 1](https://arxiv.org/html/2603.26096v1), with a ReLU base and fixed beta=1. Its zero initialization exactly recovers ReLU. Shared activations within a residual block share parameters. Inspected maps capture the first invocation only.

The 39 one-batch conditions cover clean inputs plus four corruptions at three severities, for B=4/16/64, and updates 0–8. Evaluation uses the same batch after its recorded update; these numbers are not unseen-data generalization estimates.

Six stream conditions cover B=4/16/64 with continual and reset policies. Each phase contains four batches: clean → noise severity 3 → defocus blur severity 5 → clean. Returning clean inputs reuse the original indices. Each incoming batch gets one update. History uses predictions before its update; the images, internal state, and table use post-update diagnostics. Reset restores both parameters and optimizer state. CSV export explicitly distinguishes the measurements.

## Measurements and files

Accuracy, confidence, entropy in nats, and ECE with ten equal-width bins are independently recoverable from all B stored probability vectors. Four images, two layers, and 16 channels are inspectable. Feature maps are average-pooled to 8×8 and display-quantized on one signed scale shared across methods and steps within a trace. Buffer residuals use their own fixed scale. Parameters and probabilities retain decimal precision. Quantized map bytes use lossless XOR against a measured Source reference; reconstruction is checked during recording.

Each experiment lives in `assets/lab/tta/*.json.gz`; `assets/lab/tta-manifest.json` lists hashes, sizes, counts, source provenance, and the optimizer protocol. The generated `assets/lab/research-lab.html` is a small standalone shell. Clicking Save an offline copy on the live page embeds the exact same measured gzip files; the saved HTML works without a server or connection.

This common ResNet-20 configuration demonstrates the update mechanisms. It does not reproduce the papers' full architectures, tuning, or benchmarks, and it does not establish a general method ranking.
