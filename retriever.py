"""Exact dense retrieval over the 2018 Wikipedia dump, the Search-R1 search engine.

Same retriever as Search-R1 (E5-base-v2, inner product over the 21M passage embeddings they
publish, top 3), but searched with a float16 matrix product on the GPU instead of a faiss
server, so training and evaluation need no separate process. prep_index.py builds the files.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ENCODER = "intfloat/e5-base-v2"
SHARD_ROWS = 1 << 22


class Retriever:
    def __init__(self, data_dir, device="cuda", max_length=256):
        data_dir = Path(data_dir)
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        self.encoder = AutoModel.from_pretrained(ENCODER, dtype=torch.float16).to(device).eval()
        emb = np.load(data_dir / "e5_fp16.npy", mmap_mode="r")
        # Copy the 32 GB matrix to the GPU shard by shard so host memory stays small.
        self.shards = [torch.from_numpy(np.array(emb[i:i + SHARD_ROWS])).to(device)
                       for i in range(0, emb.shape[0], SHARD_ROWS)]
        self.offsets = np.load(data_dir / "corpus_offsets.npy")
        self.corpus = np.memmap(data_dir / "corpus.bin", dtype=np.uint8, mode="r")
        assert len(self.offsets) == emb.shape[0] + 1

    @torch.no_grad()
    def encode(self, queries):
        batch = self.tokenizer([f"query: {q}" for q in queries], max_length=self.max_length, padding=True,
                               truncation=True, return_tensors="pt").to(self.device)
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1)
        return F.normalize(mean.float(), dim=-1).half()

    def search(self, queries, topk=3, chunk=128):
        """Return, for each query, the top-k passages as raw "contents" strings ("Title"\\ntext).

        At most `chunk` queries at a time: the fp32 scores of 512 queries against one 4M-row shard
        alone take 8 GB, more than is left on the GPU next to the model and vLLM."""
        return [row for i in range(0, len(queries), chunk) for row in self._search(queries[i:i + chunk], topk)]

    @torch.no_grad()
    def _search(self, queries, topk):
        q = self.encode(queries)
        best_scores, best_ids, base = None, None, 0
        for shard in self.shards:
            scores, ids = (q @ shard.T).float().topk(topk, dim=1)
            ids = ids + base
            base += shard.shape[0]
            if best_scores is None:
                best_scores, best_ids = scores, ids
            else:
                scores = torch.cat([best_scores, scores], 1)
                ids = torch.cat([best_ids, ids], 1)
                best_scores, order = scores.topk(topk, dim=1)
                best_ids = ids.gather(1, order)
        return [[self.passage(i) for i in row] for row in best_ids.tolist()]

    def passage(self, i):
        return bytes(self.corpus[self.offsets[i]:self.offsets[i + 1]]).decode("utf-8")


def format_passages(passages):
    """Search-R1's formatting of the retrieved passages inside <information>."""
    lines = []
    for i, contents in enumerate(passages):
        title, _, text = contents.partition("\n")
        lines.append(f"Doc {i + 1}(Title: {title}) {text}\n")
    return "".join(lines)
