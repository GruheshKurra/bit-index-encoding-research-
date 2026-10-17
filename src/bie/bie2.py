"""
BIE-v2: Bit-Index Encoding that keeps the weight values.

What changed from the original BIE (src/bie/encoder.py):
  * Positions of non-zeros are stored as per-row gaps, Golomb-Rice coded at the
    bit level (or as a plain bitmap when that is smaller). Indices are lossless.
  * Values are kept: each non-zero is quantised to a signed b-bit integer with a
    per-row scale (fp16). The original binary variant dropped magnitudes.
  * Zero stays zero. The original bitplane variant quantised over [min, max], so
    zeros became non-zero codes and the encoding was larger than float32.
  * Magnitudes are stored as bitplanes over the non-zero list, MSB first, so a
    reader can use only the top p planes (progressive precision).
  * The multiply kernel is parallel over output rows, so threads never write to
    the same row (the original binary kernel had a race and gave wrong outputs).

Storage is counted from the arrays actually stored (nbytes), not from a formula.
"""

from dataclasses import dataclass
import numpy as np
from numba import njit, prange


# ---------- bit-level index coding ----------

@njit(cache=True)
def _rice_encode(gaps, k, total_bits):
    out = np.zeros((total_bits + 7) // 8, np.uint8)
    pos = 0
    for i in range(gaps.shape[0]):
        g = gaps[i]
        q = g >> k
        for _ in range(q):
            out[pos >> 3] |= np.uint8(1 << (7 - (pos & 7)))
            pos += 1
        pos += 1  # terminating zero of the unary part
        for b in range(k - 1, -1, -1):
            if (g >> b) & 1:
                out[pos >> 3] |= np.uint8(1 << (7 - (pos & 7)))
            pos += 1
    return out


@njit(cache=True)
def _rice_decode(stream, k, n):
    gaps = np.empty(n, np.int64)
    pos = 0
    for i in range(n):
        q = 0
        while (stream[pos >> 3] >> (7 - (pos & 7))) & 1:
            q += 1
            pos += 1
        pos += 1
        r = 0
        for _ in range(k):
            r = (r << 1) | ((stream[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
        gaps[i] = (q << k) | r
    return gaps


def _row_gaps(cols, row_nnz):
    """Gap of each non-zero from the previous one in its row (first: from column -1)."""
    gaps = np.diff(cols, prepend=-1)
    starts = np.concatenate([[0], np.cumsum(row_nnz)[:-1]]).astype(np.int64)
    starts = starts[row_nnz > 0]
    gaps[starts] = cols[starts] + 1
    return (gaps - 1).astype(np.int64)


def _gaps_to_cols(gaps, row_nnz):
    g = gaps + 1
    starts = np.concatenate([[0], np.cumsum(row_nnz)[:-1]]).astype(np.int64)
    starts = starts[row_nnz > 0]
    # cumulative sum restarted at every row start
    c = np.cumsum(g)
    offs = np.zeros_like(c)
    offs[starts] = c[starts] - g[starts]
    offs = np.maximum.accumulate(offs)
    return (c - offs - 1).astype(np.int64)


# ---------- container ----------

@dataclass
class BIE2Matrix:
    shape: tuple
    bits: int                 # value bits incl. sign
    index_mode: str           # "rice" or "bitmap"
    rice_k: int
    index_stream: np.ndarray  # uint8, packed bits
    row_nnz: np.ndarray       # uint16 / uint32
    scales: np.ndarray        # float16, one per row
    sign_plane: np.ndarray    # uint8, packed, one bit per non-zero
    mag_planes: list          # (bits-1) packed uint8 arrays, MSB first
    nnz: int

    HEADER_BYTES = 16  # shape, bits, mode, k

    def nbytes(self):
        return int(self.HEADER_BYTES + self.index_stream.nbytes + self.row_nnz.nbytes
                   + self.scales.nbytes + self.sign_plane.nbytes
                   + sum(p.nbytes for p in self.mag_planes))


def encode(W, bits=8):
    W = np.asarray(W, dtype=np.float32)
    rows, cols = W.shape
    m = bits - 1
    qmax = (1 << m) - 1

    r_idx, c_idx = np.nonzero(W)
    vals = W[r_idx, c_idx]
    row_nnz = np.bincount(r_idx, minlength=rows)
    nnz = int(vals.size)

    # values: per-row symmetric scale, sign + magnitude
    row_max = np.zeros(rows, np.float32)
    np.maximum.at(row_max, r_idx, np.abs(vals))
    scales = (row_max / qmax).astype(np.float16)
    s = scales.astype(np.float32)[r_idx]
    q = np.zeros(nnz, np.int64)
    ok = s > 0
    q[ok] = np.clip(np.rint(np.abs(vals[ok]) / s[ok]), 0, qmax)
    sign_plane = np.packbits(vals < 0)
    mag_planes = [np.packbits((q >> (m - 1 - j)) & 1) for j in range(m)]

    # indices: Rice-coded gaps or bitmap, whichever is smaller
    gaps = _row_gaps(c_idx.astype(np.int64), row_nnz)
    best_k, best_bits = 0, None
    for k in range(0, 17):
        tb = int(np.sum((gaps >> k) + 1 + k)) if nnz else 0
        if best_bits is None or tb < best_bits:
            best_k, best_bits = k, tb
    bitmap_bits = rows * cols
    if nnz and best_bits < bitmap_bits:
        mode, stream = "rice", _rice_encode(gaps, best_k, best_bits)
    else:
        mode, stream = "bitmap", np.packbits(W != 0)

    rn_dtype = np.uint16 if cols < 65536 else np.uint32
    return BIE2Matrix((rows, cols), bits, mode, best_k, stream,
                      row_nnz.astype(rn_dtype), scales, sign_plane, mag_planes, nnz)


def decode_indices(E):
    rows, cols = E.shape
    row_nnz = E.row_nnz.astype(np.int64)
    if E.index_mode == "rice":
        gaps = _rice_decode(E.index_stream, E.rice_k, E.nnz)
        c = _gaps_to_cols(gaps, row_nnz)
    else:
        mask = np.unpackbits(E.index_stream, count=rows * cols).reshape(rows, cols)
        _, c = np.nonzero(mask)
    indptr = np.concatenate([[0], np.cumsum(row_nnz)]).astype(np.int64)
    return indptr, c.astype(np.int32)


def decode_values(E, planes=None):
    """Signed integer codes; planes < bits-1 reads only the top planes (progressive)."""
    m = E.bits - 1
    p = m if planes is None else min(planes, m)
    q = np.zeros(E.nnz, np.int64)
    for j in range(p):
        q |= np.unpackbits(E.mag_planes[j], count=E.nnz).astype(np.int64) << (m - 1 - j)
    if p < m:  # mid-point of the truncated interval
        q += 1 << (m - p - 1)
    sign = np.unpackbits(E.sign_plane, count=E.nnz).astype(bool)
    q[sign] *= -1
    return q


def decode(E, planes=None):
    rows, cols = E.shape
    indptr, c = decode_indices(E)
    q = decode_values(E, planes)
    r = np.repeat(np.arange(rows), np.diff(indptr))
    W = np.zeros((rows, cols), np.float32)
    W[r, c] = q * E.scales.astype(np.float32)[r]
    return W


# ---------- multiply on the compressed (decoded-index) form ----------

@njit(parallel=True, cache=True)
def _csr_int_matmul(indptr, cols, q, scales, X, out):
    for r in prange(out.shape[0]):
        s = scales[r]
        for t in range(indptr[r], indptr[r + 1]):
            v = q[t] * s
            c = cols[t]
            for j in range(X.shape[1]):
                out[r, j] += v * X[c, j]


class BIE2Operator:
    """Indices decoded once at load; values stay as int8 codes plus row scales."""

    def __init__(self, E, planes=None):
        self.shape = E.shape
        self.indptr, self.cols = decode_indices(E)
        self.q = decode_values(E, planes).astype(np.int8 if E.bits <= 8 else np.int16)
        self.scales = E.scales.astype(np.float32)

    def matmul(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = np.zeros((self.shape[0], X.shape[1]), np.float32)
        _csr_int_matmul(self.indptr, self.cols, self.q, self.scales, X, out)
        return out
