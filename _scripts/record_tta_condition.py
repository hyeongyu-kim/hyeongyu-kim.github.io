"""Run one independent condition from the unchanged Lab recorder on a CI worker.

The canonical protocol identifier comes from the native recording. It identifies
shared settings, not a worker's recomputed clean-subset score. Receipts preserve
the actual recorder hash, runtime, and output checksum. No scores are modified.
"""
import argparse
import gzip
import hashlib
import io
import json
import pathlib
import re

import numpy as np
import torch
from torch import nn

import build_tta_replay as replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--protocol-sha', required=True)
    parser.add_argument('--recorder-sha', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-dir', default=str(replay.DATA))
    args = parser.parse_args()
    if not re.fullmatch(r'(batch-(4|16|128)-((gaussian_noise|brightness|defocus_blur|jpeg_compression)-[135]|clean-0)|stream-(4|16|128)-(continual|reset))', args.condition):
        parser.error('Only the declared 45-condition protocol is supported')
    if not re.fullmatch('[a-f0-9]{64}', args.protocol_sha):
        parser.error('Expected a canonical protocol SHA-256')
    recorder_sha = hashlib.sha256(pathlib.Path(replay.__file__).read_bytes()).hexdigest()
    if recorder_sha != args.recorder_sha:
        raise AssertionError('CI recorder differs from the native source')
    if (torch.__version__, np.__version__) != ('2.5.1+cpu', '2.3.5'):
        raise AssertionError('Unexpected worker math-library versions')
    replay.DATA = pathlib.Path(args.data_dir)
    provenance = json.loads((replay.DATA / 'provenance.json').read_text())
    replay.clean = np.load(replay.DATA / 'clean.npy')
    replay.labels = np.load(replay.DATA / 'labels.npy')
    replay.arrays = {'clean': replay.clean}
    for domain in replay.DOMAINS:
        for severity in [1, 3, 5]:
            key = f'{domain}-{severity}'
            images = np.load(replay.DATA / f'{key}.npy')
            if hashlib.sha256(images.tobytes()).hexdigest() != provenance['subsetSha256'][key]:
                raise AssertionError('Changed input bytes: ' + key)
            replay.arrays[key] = images
    weights = (replay.DATA / 'Standard.pt').read_bytes()
    if hashlib.sha256(weights).hexdigest() != replay.CHECKPOINT_SHA256:
        raise AssertionError('Changed WRN checkpoint')
    checkpoint = torch.load(io.BytesIO(weights), weights_only=False, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    replay.source = replay.WideResNet().eval().requires_grad_(False).to(memory_format=torch.channels_last)
    replay.source.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
    del weights, checkpoint, state
    for module in replay.source.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    if args.condition.startswith('batch-'):
        _, batch, domain, severity = args.condition.split('-')
        trace = replay.record_batch(int(batch), domain, int(severity))
    else:
        _, batch, policy = args.condition.split('-')
        trace = replay.record_stream(int(batch), policy)
    trace['recordingProtocolSha256'] = args.protocol_sha
    trace['recorderSourceSha256'] = recorder_sha
    replay.encode_trace(trace)
    packed = gzip.compress(json.dumps(trace, separators=(',', ':'), allow_nan=False).encode(), compresslevel=9, mtime=0)
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    filename = args.condition + '.json.gz'
    (output / filename).write_bytes(packed)
    receipt = {'condition': args.condition, 'file': filename, 'sha256': hashlib.sha256(packed).hexdigest(),
               'bytes': len(packed), 'frames': len(trace['frames']), 'recordingProtocolSha256': args.protocol_sha,
               'recorderSourceSha256': recorder_sha, 'torch': torch.__version__, 'numpy': np.__version__,
               'threads': torch.get_num_threads(), 'device': 'cpu', 'dtype': 'float32',
               'allArmsOnSameWorker': True}
    (output / (args.condition + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
