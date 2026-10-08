"""Fetch index-preserving CIFAR-10/C subsets from the original releases."""
import hashlib
import io
import json
import pathlib
import pickle
import sys
import tarfile
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT.parent / "tta-data"
DEST.mkdir(exist_ok=True)
COUNT = 256
CLEAN = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CORRUPT = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
DOMAINS = ["gaussian_noise", "brightness", "defocus_blur", "jpeg_compression"]


def download(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as source, path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
    return path.read_bytes()


def ranged(start, count):
    request = urllib.request.Request(CORRUPT, headers={"Range": f"bytes={start}-{start + count - 1}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206:
            raise RuntimeError("The original corruption archive did not honor its byte range")
        expected = f"bytes {start}-{start + count - 1}/"
        if not response.headers.get("Content-Range", "").startswith(expected):
            raise RuntimeError("Unexpected corruption archive range")
        data = response.read(count + 1)
    if len(data) != count:
        raise RuntimeError("Truncated corruption archive")
    return data


if "--mirror" in sys.argv:
    import pyarrow.parquet as parquet
    CLEAN = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"
    blob = download(CLEAN, DEST / "cifar10-test.parquet")
    if hashlib.sha256(blob).hexdigest() != "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de":
        raise RuntimeError("CIFAR-10 Parquet mirror checksum mismatch")
    from PIL import Image
    table = parquet.read_table(io.BytesIO(blob)).slice(0, COUNT).to_pylist()
    clean = np.stack([np.asarray(Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")) for row in table])
    labels = np.asarray([row["label"] for row in table], dtype=np.int64)
else:
    archive = download(CLEAN, DEST / "cifar-10-python.tar.gz")
    if hashlib.md5(archive).hexdigest() != "c58f30108f718f92721af3b95e74349a":
        raise RuntimeError("Original CIFAR-10 archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        batch = pickle.load(tar.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")
    clean = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)[:COUNT]
    labels = np.asarray(batch[b"labels"], dtype=np.int64)[:COUNT]
np.save(DEST / "clean.npy", clean)
np.save(DEST / "labels.npy", labels)
print("Original clean images verified", flush=True)

needed = set(DOMAINS + ["labels"])
offset = 0
long_name = None
provenance = {}
while needed:
    header = ranged(offset, 512)
    if not any(header):
        raise RuntimeError("Missing corruption arrays: " + str(needed))
    size = int(header[124:136].rstrip(b"\0 ") or b"0", 8)
    name = header[:100].split(b"\0")[0].decode()
    prefix = header[345:500].split(b"\0")[0].decode()
    if prefix:
        name = prefix + "/" + name
    if header[156:157] == b"L":
        long_name = ranged(offset + 512, size).rstrip(b"\0").decode()
    else:
        if long_name:
            name, long_name = long_name, None
        stem = pathlib.PurePosixPath(name).stem
        if stem in needed:
            beginning = ranged(offset + 512, 256)
            stream = io.BytesIO(beginning)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise RuntimeError("Unsupported original NumPy header version")
            if fortran:
                raise RuntimeError("Unexpected Fortran corruption layout")
            header_size = stream.tell()
            if stem == "labels":
                full = np.load(io.BytesIO(ranged(offset + 512, size)), allow_pickle=False)
                if not np.array_equal(full[:COUNT], labels):
                    raise RuntimeError("Clean and corrupted image indices do not match")
            else:
                if shape != (50000, 32, 32, 3) or dtype != np.uint8:
                    raise RuntimeError("Unexpected corruption array " + str((shape, dtype)))
                for severity in [1, 3, 5]:
                    start = offset + 512 + header_size + (severity - 1) * 10000 * 3072
                    raw = ranged(start, COUNT * 3072)
                    np.save(DEST / f"{stem}-{severity}.npy", np.frombuffer(raw, dtype=np.uint8).reshape(COUNT, 32, 32, 3))
                    provenance[f"{stem}-{severity}"] = hashlib.sha256(raw).hexdigest()
            needed.remove(stem)
            print("Verified", stem, flush=True)
    offset += 512 + ((size + 511) // 512) * 512

weights = download("https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt", DEST / "resnet20.pt")
weight_hash = hashlib.sha256(weights).hexdigest()
if not weight_hash.startswith("4118986f"):
    raise RuntimeError("Source checkpoint checksum mismatch")
(DEST / "provenance.json").write_text(json.dumps({"clean": CLEAN, "corruptions": CORRUPT, "checkpointSha256": weight_hash, "subsetSha256": provenance}, indent=2))
print("Data and checkpoint ready", flush=True)
