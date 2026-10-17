"""
Real weights: GPT-2 small (124M). Every Linear/Conv1D weight inside the 12 transformer blocks
(attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj; 84.9M weights) is magnitude-pruned per matrix
to sparsity s, then passed through each method (encode -> stored bytes -> decode).

Reports total stored bytes of those weights and WikiText-2 (raw, test) perplexity with
non-overlapping 1024-token windows. Embeddings, LayerNorms and biases stay float32 and are
identical across methods. Perplexity runs in float32 on the GPU when one is present. No training, no fine-tuning after pruning.

Writes experiments/runs/gpt2.json.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from methods import METHODS

OUT = Path(__file__).resolve().parent / "runs"
OUT.mkdir(exist_ok=True)

SPARSITY = [0.0, 0.5, 0.7, 0.9]
STORAGE = ["fp32", "fp16", "int8_rowwise", "csr_fp32", "csr_int8", "bitmap_int8",
           "bie_binary", "bie_bitplane8", "bie2_8", "bie2_4"]
# methods whose reconstruction equals another's, so perplexity is evaluated once
SAME_AS = {"csr_fp32": "fp32", "csr_int8": "bie2_8", "bitmap_int8": "bie2_8"}
WINDOW = 1024

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, torch.get_num_threads()))


def block_weights(model):
    out = []
    for i, blk in enumerate(model.transformer.h):
        for name in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
            mod = blk.get_submodule(name)
            out.append((f"h.{i}.{name}", mod))
    return out


def prune(W, s):
    if s == 0:
        return W.copy()
    k = int(round(s * W.size))
    thr = np.partition(np.abs(W).ravel(), k - 1)[k - 1]
    P = W.copy()
    P[np.abs(P) <= thr] = 0
    return P


@torch.no_grad()
def perplexity(model, ids):
    nll, n = 0.0, 0
    for start in range(0, ids.numel() - 1, WINDOW):
        x = ids[start:start + WINDOW + 1]
        if x.numel() < 2:
            break
        x = x.to(DEVICE)
        logits = model(x[:-1].unsqueeze(0)).logits[0]
        loss = torch.nn.functional.cross_entropy(logits.float(), x[1:], reduction="sum")
        nll += loss.item()
        n += x.numel() - 1
    return math.exp(nll / n) if nll / n < 700 else float("inf")


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = torch.tensor(tok(text)["input_ids"])
    layers = block_weights(model)
    # Conv1D stores (in, out); encode as (out, in) so "per-row" means per output channel
    originals = {n: m.weight.detach().cpu().numpy().T.copy() for n, m in layers}
    n_weights = sum(W.size for W in originals.values())
    print(f"{len(layers)} matrices, {n_weights:,} weights, {ids.numel():,} eval tokens", flush=True)

    rows = []
    for s in SPARSITY:
        pruned = {n: prune(W, s) for n, W in originals.items()}
        ppl_cache = {}
        for meth in STORAGE:
            t = time.perf_counter()
            total, recon = 0, {}
            for n, P in pruned.items():
                b, R = METHODS[meth](P)
                total += b
                recon[n] = R
            rel = math.sqrt(sum(float(np.sum((pruned[n] - recon[n]) ** 2)) for n in pruned)
                            / sum(float(np.sum(P ** 2)) for P in pruned.values()))
            key = SAME_AS.get(meth, meth)
            if key not in ppl_cache:
                for n, m in layers:
                    m.weight.data = torch.from_numpy(np.ascontiguousarray(recon[n].T, dtype=np.float32)).to(DEVICE)
                ppl_cache[key] = perplexity(model, ids)
            row = dict(sparsity=s, method=meth, bytes=int(total), ratio=4 * n_weights / total,
                       rel_err=rel, ppl=ppl_cache[key], seconds=time.perf_counter() - t)
            rows.append(row)
            print(json.dumps(row), flush=True)
            (OUT / "gpt2.json").write_text(json.dumps(dict(n_weights=n_weights, eval_tokens=int(ids.numel()),
                                                            rows=rows), indent=1))


if __name__ == "__main__":
    main()
