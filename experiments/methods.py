"""
Every storage format in the experiments, behind one interface:
    fn(W) -> (stored_bytes, reconstructed_W)
Stored bytes are the nbytes of the arrays a method keeps, so ratios are measured, not modelled.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bie.early_encoder import BIEEncoder, BIEDecoder  # early variants
from src.bie import codec                                 # final BIE format


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
    """Same per-row int8 quantisation of the non-zeros that BIE uses, so errors match exactly."""
    return codec.decode(codec.encode(W, 8))


def csr_int8(W):
    rows, cols = W.shape
    nnz = int(np.count_nonzero(W))
    col_bytes = 2 if cols < 65536 else 4
    return nnz + nnz * col_bytes + 4 * (rows + 1) + 2 * rows, _int8_nonzeros(W)


def bitmap_int8(W):
    rows, cols = W.shape
    nnz = int(np.count_nonzero(W))
    return (rows * cols + 7) // 8 + nnz + 2 * rows, _int8_nonzeros(W)


def early_positions(W):
    e = BIEEncoder("binary").encode(W)
    b = e["indices"].nbytes + (e["sign_indices"].nbytes if e["sign_indices"] is not None else 0)
    return b, BIEDecoder.decode(e).astype(np.float32)


def early_bitplane8(W):
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


def early_blocked_positions(W):
    return _blocked(W, "binary")


def early_blocked_bitplane(W):
    return _blocked(W, "bitplane")


def bie_8bit(W):
    E = codec.encode(W, 8)
    return E.nbytes(), codec.decode(E)


def bie_4bit(W):
    E = codec.encode(W, 4)
    return E.nbytes(), codec.decode(E)


METHODS = {
    "fp32": fp32,
    "fp16": fp16,
    "int8_rowwise": int8_rowwise,
    "csr_fp32": csr_fp32,
    "csr_int8": csr_int8,
    "bitmap_int8": bitmap_int8,
    "early_positions": early_positions,
    "early_bitplane8": early_bitplane8,
    "early_blocked_positions": early_blocked_positions,
    "early_blocked_bitplane": early_blocked_bitplane,
    "bie_8bit": bie_8bit,
    "bie_4bit": bie_4bit,
}

LABELS = {
    "fp32": "Dense float32",
    "fp16": "Dense float16",
    "int8_rowwise": "Dense int8 (per-row)",
    "csr_fp32": "CSR float32",
    "csr_int8": "CSR int8 + uint16 cols",
    "bitmap_int8": "Bitmap + int8 values",
    "early_positions": "Early variant: positions + signs only",
    "early_bitplane8": "Early variant: bitplanes over all weights",
    "early_blocked_positions": "Early variant: positions only, 64x64 tiles",
    "early_blocked_bitplane": "Early variant: bitplanes, 64x64 tiles",
    "bie_8bit": "BIE 8-bit",
    "bie_4bit": "BIE 4-bit",
}
