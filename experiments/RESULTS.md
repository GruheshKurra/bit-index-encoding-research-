# Results

Measured on 17 Oct 2026 by `python experiments/run_synthetic.py`, `python experiments/run_gpt2.py`, then `python experiments/make_report.py` on a RunPod pod (16 vCPU, RTX 4090; matrices and timings on the CPU, GPT-2 perplexity on the GPU).

## 1. Synthetic matrices (paper protocol)

Gaussian 1024×1024 matrices with a random zero mask; mean ± sd over seeds 0, 1, 2. Ratio = float32 bytes / bytes actually stored (higher is smaller). Other sizes (256×256 to 2048×1024) give the same picture; all rows are in `runs/synthetic.json`.

**Compression ratio**

| Method | 0% | 30% | 50% | 70% | 90% | 95% |
|---|---|---|---|---|---|---|
| Dense float32 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| Dense float16 | 2.00 ± 0.00 | 2.00 ± 0.00 | 2.00 ± 0.00 | 2.00 ± 0.00 | 2.00 ± 0.00 | 2.00 ± 0.00 |
| Dense int8 (per-row) | 3.99 ± 0.00 | 3.99 ± 0.00 | 3.99 ± 0.00 | 3.99 ± 0.00 | 3.99 ± 0.00 | 3.99 ± 0.00 |
| CSR float32 | 0.50 ± 0.00 | 0.71 ± 0.00 | 1.00 ± 0.00 | 1.66 ± 0.00 | 4.98 ± 0.01 | 9.91 ± 0.03 |
| CSR int8 + uint16 cols | 1.33 ± 0.00 | 1.90 ± 0.00 | 2.65 ± 0.00 | 4.42 ± 0.01 | 13.09 ± 0.02 | 25.68 ± 0.07 |
| Bitmap + int8 values | 3.55 ± 0.00 | 4.84 ± 0.00 | 6.38 ± 0.00 | 9.37 ± 0.01 | 17.63 ± 0.01 | 22.61 ± 0.02 |
| BIE binary (paper) | 0.67 ± 0.00 | 0.95 ± 0.00 | 1.33 ± 0.00 | 2.22 ± 0.00 | 6.67 ± 0.00 | 13.34 ± 0.04 |
| BIE bitplane 8-bit (paper) | 0.25 ± 0.00 | 0.26 ± 0.05 | 0.27 ± 0.11 | 0.29 ± 0.20 | 0.18 ± 0.02 | 0.21 ± 0.04 |
| BIE blocked binary (paper) | 0.67 ± 0.00 | 0.95 ± 0.00 | 1.33 ± 0.00 | 2.22 ± 0.00 | 6.66 ± 0.00 | 13.30 ± 0.04 |
| BIE blocked bitplane (paper) | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.26 ± 0.01 | 0.25 ± 0.01 |
| BIE-v2 8-bit (this re-run) | 3.54 ± 0.00 | 4.83 ± 0.00 | 6.36 ± 0.00 | 9.64 ± 0.01 | 24.52 ± 0.03 | 44.50 ± 0.11 |
| BIE-v2 4-bit (this re-run) | 6.36 ± 0.00 | 8.35 ± 0.00 | 10.55 ± 0.01 | 15.11 ± 0.02 | 35.34 ± 0.04 | 61.63 ± 0.14 |

**Relative reconstruction error** ‖W − Ŵ‖ / ‖W‖ (0 = exact)

| Method | 0% | 30% | 50% | 70% | 90% | 95% |
|---|---|---|---|---|---|---|
| Dense float32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense float16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense int8 (per-row) | 0.008 | 0.008 | 0.007 | 0.007 | 0.006 | 0.006 |
| CSR float32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CSR int8 + uint16 cols | 0.008 | 0.008 | 0.007 | 0.007 | 0.006 | 0.006 |
| Bitmap + int8 values | 0.008 | 0.008 | 0.007 | 0.007 | 0.006 | 0.006 |
| BIE binary (paper) | 0.636 | 0.636 | 0.636 | 0.636 | 0.637 | 0.636 |
| BIE bitplane 8-bit (paper) | 0.011 | 0.013 | 0.017 | 0.023 | 0.043 | 0.055 |
| BIE blocked binary (paper) | 0.636 | 0.636 | 0.636 | 0.636 | 0.637 | 0.636 |
| BIE blocked bitplane (paper) | 0.008 | 0.010 | 0.011 | 0.014 | 0.022 | 0.028 |
| BIE-v2 8-bit (this re-run) | 0.008 | 0.008 | 0.007 | 0.007 | 0.006 | 0.006 |
| BIE-v2 4-bit (this re-run) | 0.143 | 0.139 | 0.135 | 0.129 | 0.114 | 0.105 |

## 2. Multiply speed and kernel correctness

W is 2048×1024, X has 64 columns; median of 7 runs after a warm-up, milliseconds, pod CPU (16 vCPU). Dense float32 uses NumPy's BLAS.

| Sparsity | Dense fp32 | SciPy CSR fp32 | BIE binary kernel (paper, race fixed) | BIE-v2 kernel | BIE-v2 index decode (once, at load) | BIE-v2 output error | BIE binary output error |
|---|---|---|---|---|---|---|---|
| 50% | 0.43 | 8.13 | 14.92 | 0.68 | 12.67 | 0.0074 | 0.636 |
| 70% | 0.36 | 5.03 | 9.15 | 0.49 | 7.23 | 0.0071 | 0.638 |
| 90% | 0.27 | 1.72 | 2.99 | 0.16 | 2.47 | 0.0063 | 0.635 |
| 95% | 0.24 | 0.85 | 1.47 | 0.11 | 1.38 | 0.0058 | 0.637 |
| 99% | 0.28 | 0.25 | 0.39 | 0.10 | 0.38 | 0.0043 | 0.631 |

BIE-v2 kernel vs multiplying its own decoded matrix: max difference 8.6e-07 of the output scale (correct).

## 3. Progressive precision (BIE-v2 bitplanes)

One 8-bit BIE-v2 encoding of a 1024×1024 matrix, read with only the top p magnitude planes. Mean over 3 seeds.

| Planes read | Value bits | Error at 50% sparsity | Error at 90% sparsity |
|---|---|---|---|
| 1 | 2 | 0.478 | 0.410 |
| 2 | 3 | 0.239 | 0.205 |
| 3 | 4 | 0.120 | 0.103 |
| 4 | 5 | 0.061 | 0.052 |
| 5 | 6 | 0.032 | 0.028 |
| 6 | 7 | 0.020 | 0.017 |
| 7 | 8 | 0.007 | 0.006 |

## 4. Real weights: GPT-2 small

All 84,934,656 weights of the 48 attention/MLP matrices, magnitude-pruned per matrix (no retraining), then compressed. Perplexity on WikiText-2 raw test (287,644 tokens, 1024-token windows). Lower perplexity is better; the float32 row at each sparsity is the reference for that sparsity.

| Method | 0%: ratio / ppl | 50%: ratio / ppl | 70%: ratio / ppl | 90%: ratio / ppl |
|---|---|---|---|---|
| Dense float32 | 1.00× / 29.9 | 1.00× / 534.8 | 1.00× / 5700.3 | 1.00× / 14615.1 |
| Dense float16 | 2.00× / 29.9 | 2.00× / 534.7 | 2.00× / 5698.0 | 2.00× / 14616.2 |
| Dense int8 (per-row) | 3.99× / 30.0 | 3.99× / 526.5 | 3.99× / 5765.3 | 3.99× / 14670.6 |
| CSR float32 | 0.50× / 29.9 | 1.00× / 534.8 | 1.66× / 5700.3 | 4.98× / 14615.1 |
| CSR int8 + uint16 cols | 1.33× / 30.0 | 2.66× / 526.5 | 4.42× / 5765.3 | 13.08× / 14670.6 |
| Bitmap + int8 values | 3.55× / 30.0 | 6.38× / 526.5 | 9.37× / 5765.3 | 17.62× / 14670.6 |
| BIE binary (paper) | 0.67× / >10⁵ | 1.33× / >10⁵ | 2.21× / >10⁵ | 6.63× / >10⁵ |
| BIE bitplane 8-bit (paper) | 0.27× / 58.9 | 0.27× / 1490.1 | 0.28× / 3986.5 | 0.28× / 12280.6 |
| BIE-v2 8-bit (this re-run) | 3.54× / 30.0 | 6.36× / 526.5 | 9.64× / 5765.3 | 24.54× / 14670.6 |
| BIE-v2 4-bit (this re-run) | 6.36× / 46.5 | 10.56× / 701.5 | 15.11× / 6699.6 | 35.41× / 11622.7 |
