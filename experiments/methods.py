"""
Every compression method in the re-run, behind one interface:
    fn(W) -> (stored_bytes, reconstructed_W)
Stored bytes are the nbytes of the arrays a method keeps, so ratios are measured, not modelled.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bie.encoder import BIEEncoder, BIEDecoder  # original BIE (paper version)
from src.bie import bie2                             # improved BIE-v2


def fp32(W):
    return W.nbytes, W


def fp16(W):
    H = W.astype(np.float16)
    return H.nbytes, H.astype(np.float32)


def int8_rowwise(W):
    s = (np.abs(W).max(axis=1) / 127).astype(np.float16)
    sf = s.astype(np.float32)[:, None]
    q = np.clip(np.rint(np.divide(W, sf, out=np.zeros_like(W), where=sf > 0)), -127, 127).astype(np.int8)
    return q.nbytes + s.nbytes, q.astype(np.float32) * sf


def csr_fp32(W):
    A = sp.csr_matrix(W)
    return A.data.nbytes + A.indices.astype(np.int32).nbytes + A.indptr.astype(np.int32).nbytes, W


def _int8_nonzeros(W):
    """Same per-row int8 quantisation of the non-zeros that BIE-v2 uses, so errors match exactly."""
    return bie2.decode(bie2.encode(W, 8))


def csr_int8(W):
    rows, cols = W.shape
    nnz = int(np.count_nonzero(W))
    col_bytes = 2 if cols < 65536 else 4
    return nnz + nnz * col_bytes + 4 * (rows + 1) + 2 * rows, _int8_nonzeros(W)


def bitmap_int8(W):
    rows, cols = W.shape
    nnz = int(np.count_nonzero(W))
    return (rows * cols + 7) // 8 + nnz + 2 * rows, _int8_nonzeros(W)


def bie_binary(W):
    e = BIEEncoder("binary").encode(W)
    b = e["indices"].nbytes + (e["sign_indices"].nbytes if e["sign_indices"] is not None else 0)
    return b, BIEDecoder.decode(e).astype(np.float32)


def bie_bitplane8(W):
    e = BIEEncoder("bitplane").encode(W, num_bits=8)
    b = sum(p.nbytes for p in e["bitplanes"]) + 8  # + min/max
    return b, BIEDecoder.decode(e).astype(np.float32)


def _blocked(W, kind):
    e = BIEEncoder(kind, block_size=64).encode(W)
    b = 0
    for blk in e["blocks"]:
        x = blk["encoded"]
        if x["type"] == "binary":
            b += x["indices"].nbytes + (x["sign_indices"].nbytes if x["sign_indices"] is not None else 0)
        else:
            b += sum(p.nbytes for p in x["bitplanes"]) + 8
        b += 4  # block offsets
    return b, BIEDecoder.decode(e).astype(np.float32)


def bie_blocked_binary(W):
    return _blocked(W, "binary")


def bie_blocked_bitplane(W):
    return _blocked(W, "bitplane")


def bie2_8(W):
    E = bie2.encode(W, 8)
    return E.nbytes(), bie2.decode(E)


def bie2_4(W):
    E = bie2.encode(W, 4)
    return E.nbytes(), bie2.decode(E)


METHODS = {
    "fp32": fp32,
    "fp16": fp16,
    "int8_rowwise": int8_rowwise,
    "csr_fp32": csr_fp32,
    "csr_int8": csr_int8,
    "bitmap_int8": bitmap_int8,
    "bie_binary": bie_binary,
    "bie_bitplane8": bie_bitplane8,
    "bie_blocked_binary": bie_blocked_binary,
    "bie_blocked_bitplane": bie_blocked_bitplane,
    "bie2_8": bie2_8,
    "bie2_4": bie2_4,
}

LABELS = {
    "fp32": "Dense float32",
    "fp16": "Dense float16",
    "int8_rowwise": "Dense int8 (per-row)",
    "csr_fp32": "CSR float32",
    "csr_int8": "CSR int8 + uint16 cols",
    "bitmap_int8": "Bitmap + int8 values",
    "bie_binary": "BIE binary (paper)",
    "bie_bitplane8": "BIE bitplane 8-bit (paper)",
    "bie_blocked_binary": "BIE blocked binary (paper)",
    "bie_blocked_bitplane": "BIE blocked bitplane (paper)",
    "bie2_8": "BIE-v2 8-bit (this re-run)",
    "bie2_4": "BIE-v2 4-bit (this re-run)",
}
