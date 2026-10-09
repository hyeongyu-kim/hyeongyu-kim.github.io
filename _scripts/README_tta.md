# Reproducing the public-code TTA comparison

The Lab replays measured PyTorch updates on CIFAR-10/C. It uses the public AcTTA activation and entropy adapter, the public Buffer WRN model, and one shared RobustBench Standard WRN-28-10 checkpoint. The browser reads stored predictions and internal states. Save an offline copy includes all 45 recorded experiments.

## Setup and record

Use Python 3.12 and the recorded dependency versions. From the repository root:

```bash
python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 Pillow==12.3.0 pyarrow==25.0.1 gdown==6.4.2
python _scripts/fetch_tta_data.py --mirror
python tests/tta-adapters.test.py
python _scripts/build_tta_replay.py --resume
npm ci
npm run test:lab
node _scripts/build_lab_offline.mjs
```

`--resume` reuses a trace only if its recording-protocol hash matches the current recorder, model, input provenance, runtime, and settings. It does not reuse the former ResNet-20 recordings. Keep the manifest and its trace files together when publishing.

The published set retains all 45 conditions recorded on CPU GitHub Actions workers with `_scripts/record_tta_condition.py` and the unchanged recorder. Each worker runs all four arms on the same machine, verifies checkpoint and input hashes, and pins the math-library versions. `assets/lab/tta-worker-receipts.json` records each condition's recorder hash, runtime, and output checksum. The common protocol identifier matches an independently rebuilt local profile; minor CPU floating-point differences remain possible. The commands above reproduce the full protocol locally without Actions. No condition is selected based on its outcome.

## Pinned public implementations

- [AcTTA](https://github.com/hyeongyu-kim/actta/tree/6bd3b9ab090240ec67038257c5fd9d8f11a283fc): `actta/activation.py`, `actta/adaptation.py`, and `vendor/wide_resnet.py` are copied unchanged to `_scripts/tta_vendor/`. MIT and RobustBench notices are retained. Configuration: `configs/cifar10_wrn28_bs128.yaml` and the batch-size rules in `docs/REPRODUCIBILITY.md`.
- [Buffer](https://github.com/hyeongyu-kim/Buffer_TTA/tree/b266f3c7904aa760b21618ebe5fba0fcd864bc16): the model classes from `models/custom_standard.py` are unchanged; unused checkpoint helper functions are omitted. Its default `conf.py` uses stages `[True, True, False]`, alpha `1e-5`, frozen BN affine, and Adam LR `0.001`. Source weights are loaded by matching names; only added Buffer weights may be missing.
- `_scripts/tta_vendor/provenance.json` records upstream file hashes and local copies. The run manifest links both commits.

The model is WRN-28-10: **36,479,194 source parameters**. The [RobustBench CIFAR-10 corruption Standard checkpoint](https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/cifar10.py) is `Standard.pt`, Google Drive ID `1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC`. Its complete SHA-256 is `6ed1c75dad63e8ebdbef365020dd1a7eee5e79cf8a1438bd3bdc3b9c7aab3d44`, matching the AcTTA release's measured CIFAR-10 runs. This checksum is verified before loading the original checkpoint, which includes training metadata. All arms consume RGB tensors in `[0,1]`, with no extra normalization.

## Data

The fetcher writes to `../tta-data/`. It preserves the first **512** original test indices. Without `--mirror`, it verifies the original CIFAR-10 archive MD5 `c58f30108f718f92721af3b95e74349a`. The Parquet mirror verifies complete SHA-256 `841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de`. Corrupted labels must match the selected clean labels.

Verified byte ranges from the original [CIFAR-10-C archive](https://zenodo.org/records/2535967) provide Gaussian noise, brightness, defocus blur, and JPEG compression, at severities 1, 3, and 5. Array shape, dtype, layout, HTTP range, and lengths are checked; subset hashes are stored in the manifest.

## Parameters and placements

| Arm | Updated parameters | Placement | Count |
| --- | --- | --- | ---: |
| Source | None | Stored source BN statistics | 0 |
| Tent | BN gamma and beta | Every BN layer | 17,952 |
| Buffer | Added convolutions, mixing scales, and alphas | After initial conv; after both ReLUs in stages 1 and 2 | 9,221,171 |
| AcTTA | Channel-wise `shift_gsL`, `pos_gsL`, `neg_gsL` | All eight independent ReLUs in stage 1 | 3,408 |

Buffer has **17 independent branches**, including its initial branch. There is no shared Buffer parameter between activation sites. AcTTA uses the released `replace_activations(..., prefixes=["block1"])`; beta is a fixed buffer at 1. Zero activation vectors exactly recover ReLU. The classifier and original convolution weights remain frozen in every adaptive arm.

All adaptive arms use target-batch BN statistics, including update zero. Source uses the checkpoint's stored statistics. This initial difference is not an entropy update. BN affine values remain frozen in Buffer and AcTTA. Frozen model parameters are checked bitwise after every update.

The shared objective is unlabeled mean prediction entropy. Adam uses betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, and seed 1. **Learning rates follow each method's public configuration**, rather than forcing one common value:

| Batch | Tent LR | Buffer LR | AcTTA LR | Scope |
| --- | ---: | ---: | ---: | --- |
| 4 | 0.0001 | 0.001 | 0.001 | Public small-batch rates |
| 16 | 0.001 | 0.001 | 0.01 | Explicit demo override using public batch-128 rates |
| 128 | 0.001 | 0.001 | 0.01 | Public main CIFAR-10 rates |

Runtime is CPU float32, channels-last tensors, and eight PyTorch threads. Arms are recorded sequentially, and nonreentrant gradient checkpointing recomputes residual blocks during backward to stay within memory limits. BN running state is disabled; there is no dropout. Checkpointed updates are tested against the public adapter for exact parameter parity. The release's measured environment uses PyTorch 2.2.1 CUDA on an RTX A6000. Runtime, order, and subset differences are declared; this is not a bitwise reproduction of those full runs. Labels only compute evaluation metrics.

## Replay protocol

The **39 one-batch conditions** cover clean inputs plus four corruptions at three severities, for B=4/16/128 and updates 0–8. Predictions and internals show the same batch after that many updates. These are adaptation diagnostics, not held-out generalization estimates.

The **six streams** cover the same batch sizes with continual and reset policies. Four batches per phase follow clean → noise severity 3 → blur severity 5 → clean. The final clean phase reuses original indices. Each incoming batch gets one update. History uses predictions from the adaptation forward **before** its optimizer step, matching the public online adapter. Images, internals, and the table show the post-update diagnostic. Reset restores both model and optimizer. CSV explicitly labels pre/post measurements.

The recorder's combined inspection/update forward is tested for exact parameter-update and prediction parity with the released `EntropyAdapter`, for both Tent and AcTTA. Buffer with alpha zero is tested for exact output parity with the identical checkpoint backbone using target-batch BN.

## Display and integrity

Accuracy, confidence, entropy in nats, and ECE with ten equal-width bins are independently recoverable from all stored probability vectors. Accuracy also shows the number of correct images and the percentage-point change from one image. The learning-change caption compares one-batch states with update zero, when target-batch BN is already active; stream states compare with the same batch's pre-update prediction. The UI inspects four images and **channels 0–15** at `block1.layer.0.relu1` (16 actual channels) and `block1.layer.1.relu1` (160 actual channels). Buffer maps show the activation plus its correction at the matching site. The displayed BN and activation vectors cover those same channels.

Maps are pooled to 8×8 and quantized on a fixed signed scale shared across methods and steps within the trace. Buffer residuals have a separate fixed scale. Parameters and probabilities retain decimal precision. Map bytes use lossless XOR against the Source reference; reconstruction is checked while recording.

`assets/lab/tta/*.json.gz` holds the experiments. `assets/lab/tta-manifest.json` lists hashes, sizes, counts, source commits, model/input provenance, and optimizer settings. The small standalone shell is `assets/lab/research-lab.html`. The offline download embeds the same exact gzip files.

The public modules, model, and placements are now used directly. Fixed-order subsets, 0–8 repeated-batch updates, four selected corruptions, and one seed **remain a mechanism demonstration**. Full AcTTA results use shuffled 15-corruption, 10,000-image-per-corruption benchmarks and are available separately in the [release results](https://github.com/hyeongyu-kim/actta/blob/main/docs/RESULTS.md). No outcomes are selected or modified to favor a method.
