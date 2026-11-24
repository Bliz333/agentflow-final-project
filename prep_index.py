"""Convert the Search-R1 retrieval files into memory-mappable arrays (run once).

Inputs are the files Search-R1 publishes on Hugging Face:
  PeterJinGo/wiki-18-e5-index   part_aa + part_ab = one faiss IndexFlatIP (E5-base-v2, 21M x 768, fp32)
  PeterJinGo/wiki-18-corpus     wiki-18.jsonl.gz, a gzipped tar holding one JSON-lines file, one passage
                                per line, in index order

Outputs in --out:
  e5_fp16.npy      (N, 768) float16 passage embeddings, so the whole index fits on one GPU (32 GB)
  corpus.bin       UTF-8 "contents" strings back to back
  corpus_offsets.npy  (N + 1,) int64 byte offsets into corpus.bin

    python prep_index.py --index-dir data/wiki-18-e5-index --corpus data/wiki-18-corpus/wiki-18.jsonl.gz --out data/retrieval
"""
import argparse
import json
import struct
import tarfile
from pathlib import Path

import numpy as np

# faiss write_index for IndexFlatIP: fourcc, d (int32), ntotal (int64), two unused int64,
# is_trained (uint8), metric_type (int32), then the vector buffer as a size (uint64) + floats.
HEADER = struct.Struct("<4siqqqBi")
CHUNK_ROWS = 1 << 18


class Concat:
    """Read several files as one stream (the index is split into part_aa, part_ab)."""

    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]

    def read(self, n):
        out = b""
        while len(out) < n and self.files:
            data = self.files[0].read(n - len(out))
            if not data:
                self.files.pop(0).close()
                continue
            out += data
        return out


def convert_index(index_dir, out):
    parts = sorted(Path(index_dir).glob("part_*"))
    stream = Concat(parts)
    fourcc, d, ntotal, _, _, _, metric = HEADER.unpack(stream.read(HEADER.size))
    (size,) = struct.unpack("<Q", stream.read(8))
    assert fourcc == b"IxFI" and metric == 0, (fourcc, metric)  # METRIC_INNER_PRODUCT
    assert size == ntotal * d, (size, ntotal, d)
    total = sum(p.stat().st_size for p in parts)
    assert total == HEADER.size + 8 + 4 * size, (total, size)
    print(f"index: {ntotal} x {d}", flush=True)

    emb = np.lib.format.open_memmap(out / "e5_fp16.npy", mode="w+", dtype=np.float16, shape=(ntotal, d))
    for start in range(0, ntotal, CHUNK_ROWS):
        rows = min(CHUNK_ROWS, ntotal - start)
        chunk = np.frombuffer(stream.read(4 * rows * d), dtype=np.float32).reshape(rows, d)
        emb[start:start + rows] = chunk.astype(np.float16)
    emb.flush()
    return ntotal


def convert_corpus(corpus_path, out):
    offsets = [0]
    with tarfile.open(corpus_path, "r:gz") as tar, open(out / "corpus.bin", "wb") as dst:
        for i, line in enumerate(tar.extractfile(tar.next())):
            doc = json.loads(line)
            assert int(doc["id"]) == i, (doc["id"], i)
            data = doc["contents"].encode("utf-8")
            dst.write(data)
            offsets.append(offsets[-1] + len(data))
    np.save(out / "corpus_offsets.npy", np.asarray(offsets, dtype=np.int64))
    return len(offsets) - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Each output is finished before the next starts, so a rerun skips what is already done.
    if not (out / "corpus_offsets.npy").exists():
        convert_corpus(args.corpus, out)
    n_docs = len(np.load(out / "corpus_offsets.npy", mmap_mode="r")) - 1
    print(f"corpus: {n_docs} passages", flush=True)
    if not (out / "e5_fp16.done").exists():
        convert_index(args.index_dir, out)
        (out / "e5_fp16.done").touch()
    n_vectors = np.load(out / "e5_fp16.npy", mmap_mode="r").shape[0]
    print(f"index: {n_vectors} vectors", flush=True)
    assert n_docs == n_vectors, (n_docs, n_vectors)


if __name__ == "__main__":
    main()
