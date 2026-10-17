"""
Synthetic benchmark, paper protocol: Gaussian matrices 256x256 to 2048x1024, sparsity 0-95%,
random (unstructured) zero mask. Three seeds per setting.

Writes experiments/runs/synthetic.json (storage + error), timing.json (multiply speed and
kernel correctness) and progressive.json (BIE-v2 error vs number of bitplanes read).
"""

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from methods import METHODS, bie2, BIEEncoder, BIEDecoder
from src.bie.sparse_kernels import BIEMatMul

OUT = Path(__file__).resolve().parent / "runs"
OUT.mkdir(exist_ok=True)

SIZES = [(256, 256), (512, 512), (1024, 1024), (2048, 1024)]
SPARSITY = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]
SEEDS = [0, 1, 2]


def make(shape, s, seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(shape).astype(np.float32)
    if s > 0:
        W *= rng.random(shape) >= s
    return W


def rel_err(W, R):
    return float(np.linalg.norm(W - R) / np.linalg.norm(W))


def storage_and_error():
    rows = []
    for shape in SIZES:
        for s in SPARSITY:
            for seed in SEEDS:
                W = make(shape, s, seed)
                for name, fn in METHODS.items():
                    t = time.perf_counter()
                    b, R = fn(W)
                    rows.append(dict(rows=shape[0], cols=shape[1], sparsity=s, seed=seed, method=name,
                                     bytes=int(b), ratio=W.nbytes / b, rel_err=rel_err(W, R),
                                     seconds=time.perf_counter() - t))
            print(f"storage {shape} s={s} done", flush=True)
    (OUT / "synthetic.json").write_text(json.dumps(rows, indent=1))


def median_time(f, reps=7):
    f()  # warm-up (JIT compile, caches)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def timing():
    shape, ncols = (2048, 1024), 64
    rows = []
    for s in [0.5, 0.7, 0.9, 0.95, 0.99]:
        W = make(shape, s, 0)
        X = np.random.default_rng(1).standard_normal((shape[1], ncols)).astype(np.float32)
        ref = W @ X
        A = sp.csr_matrix(W)
        E = bie2.encode(W, 8)
        op = bie2.BIE2Operator(E)
        eb = BIEEncoder("binary").encode(W)
        mm = BIEMatMul()
        Yv2 = op.matmul(X)
        Dv2 = bie2.decode(E)
        Yb = mm.matmul(eb, X)
        Db = BIEDecoder.decode(eb).astype(np.float32)
        t_idx = median_time(lambda: bie2.decode_indices(E), reps=3)
        rows.append(dict(
            sparsity=s,
            dense_fp32=median_time(lambda: W @ X),
            scipy_csr_fp32=median_time(lambda: A @ X),
            bie_binary_kernel=median_time(lambda: mm.matmul(eb, X)),
            bie2_kernel=median_time(lambda: op.matmul(X)),
            bie2_index_decode_once=t_idx,
            bie2_kernel_vs_its_decode=float(np.abs(Yv2 - Dv2 @ X).max() / np.abs(ref).max()),
            bie_binary_kernel_vs_its_decode=float(np.abs(Yb - Db @ X).max() / np.abs(ref).max()),
            bie2_output_rel_err=rel_err(ref, Yv2),
            bie_binary_output_rel_err=rel_err(ref, Yb),
        ))
        print(f"timing s={s} done", flush=True)
    (OUT / "timing.json").write_text(json.dumps(dict(shape=shape, x_cols=ncols, rows=rows), indent=1))


def progressive():
    rows = []
    for s in [0.5, 0.9]:
        for seed in SEEDS:
            W = make((1024, 1024), s, seed)
            E = bie2.encode(W, 8)
            full = E.nbytes()
            plane = E.mag_planes[0].nbytes
            for p in range(1, 8):
                rows.append(dict(sparsity=s, seed=seed, planes=p, value_bits=p + 1,
                                 bytes_read=full - (7 - p) * plane,
                                 rel_err=rel_err(W, bie2.decode(E, planes=p))))
    (OUT / "progressive.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    storage_and_error()
    timing()
    progressive()
