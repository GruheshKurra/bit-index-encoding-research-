# BIE: Bit-Index Encoding for Neural Network Weight Compression

Code for the preprint **"BIE: Bit-Index Encoding for Efficient Neural Network Weight Compression"** (G. S. S. K. Kurra, Zenodo, September 2025, [doi:10.5281/zenodo.17217218](https://doi.org/10.5281/zenodo.17217218)). Preprint, not peer reviewed.

The idea: a sparse weight matrix is stored as the **positions of its non-zeros** instead of as a dense array, and matrix products run directly on that index form.

This repository now has two parts:

1. **The original BIE code** (`src/bie/encoder.py`, `src/bie/sparse_kernels.py`) with its binary, bitplane and blocked variants, as described in the preprint.
2. **A measured re-run (17 Oct 2026)** that re-tests those variants against standard formats and adds **BIE-v2**, a corrected version of the method (`src/bie/bie2.py`, `experiments/`). Every number below comes from `experiments/runs/*.json`.

## Re-run findings in brief

- **The preprint's "up to 40×" does not hold up.** The binary variant stores only positions and signs, so every weight comes back as ±1 (63.6% reconstruction error on Gaussian weights, and GPT-2 perplexity above 10⁵). With signs stored too, it reaches **13.3×** at 95% sparsity, not 40×.
- **The bitplane variant is larger than float32** (0.21–0.29×), because its min–max quantisation maps zero to a non-zero code, so the zeros are not exploited.
- **The binary multiply kernel gave wrong results:** a parallel loop let several threads add into the same output row. Fixed here (serial loop); it now matches its own decoded matrix.
- **BIE-v2 keeps the idea and fixes these problems.** At 95% sparsity it stores a matrix **44.5× smaller than float32 with 0.6% error**, against 25.7× for CSR with int8 values and 22.6× for bitmap + int8. On GPT-2 its 8-bit form gives the same perplexity as dense int8 (30.0 vs 29.9 for float32).
- **Limit:** below about 50% sparsity BIE-v2 is not smaller than plain dense int8 (3.5× vs 4.0× at 0%). GPT-2 small pruned by magnitude without retraining is already unusable at 50% (perplexity 535), so the high-sparsity gains need a model trained to be sparse; that test is not done here.

## BIE-v2: what changed

| | Original BIE (preprint) | BIE-v2 (this re-run) |
|---|---|---|
| Positions | flat uint32 index per non-zero (4 bytes each) | gap from the previous non-zero in the row, Golomb-Rice coded at the bit level; a plain bitmap when that is smaller. Lossless |
| Values | binary: dropped (±1); bitplane: 8-bit codes over [min, max] | signed 8-bit (or 4-bit) code per non-zero with one fp16 scale per row |
| Zeros | bitplane turns zeros into non-zero codes | zero stays zero |
| Bitplanes | over every weight | over the non-zeros only, most significant first, so a reader can stop after p planes (progressive precision) |
| Multiply kernel | parallel over non-zeros (race) | parallel over output rows (no shared writes) |

BIE-v2 is built from known parts. Coding the gaps between non-zero indices follows Deep Compression (Han et al., 2015), which stores index differences and entropy-codes them. Golomb-Rice coding is a standard code for gaps that are roughly geometric. Per-channel int8 is standard post-training quantisation. The contribution here is putting these together as a measured, working form of the preprint's idea, not a new compression principle.

## Results

All measured on a RunPod pod (AMD EPYC 75F3, 16 vCPU, RTX 4090). Matrix experiments and timings ran on the CPU and GPT-2 perplexity on the GPU. The full tables are in [experiments/RESULTS.md](experiments/RESULTS.md).

### 1. Synthetic matrices (the preprint's protocol)

Gaussian matrices with a random zero mask, 256×256 to 2048×1024, sparsity 0–95%, seeds 0, 1 and 2. Ratio = float32 bytes ÷ bytes actually stored. Error = ‖W − Ŵ‖ / ‖W‖. Shown for 1024×1024, mean of 3 seeds (sd ≤ 0.2 everywhere).

![Size and error vs sparsity](docs/figures/synthetic_ratio_error.png)

| Method | Ratio at 50% | Ratio at 90% | Ratio at 95% | Error |
|---|---|---|---|---|
| Dense int8 (per-row) | 3.99× | 3.99× | 3.99× | 0.006–0.007 |
| CSR float32 | 1.00× | 4.98× | 9.91× | 0 |
| CSR int8 + uint16 columns | 2.65× | 13.09× | 25.68× | 0.006–0.007 |
| Bitmap + int8 values | 6.38× | 17.63× | 22.61× | 0.006–0.007 |
| BIE binary (preprint) | 1.33× | 6.67× | 13.34× | **0.636** |
| BIE bitplane 8-bit (preprint) | 0.27× | 0.18× | 0.21× | 0.017–0.055 |
| **BIE-v2 8-bit** | 6.36× | **24.52×** | **44.50×** | 0.006–0.007 |
| BIE-v2 4-bit | 10.55× | 35.34× | 61.63× | 0.105–0.135 |

### 2. Multiply speed

W is 2048×1024 and X has 64 columns; median of 7 runs, in milliseconds.

| Sparsity | Dense float32 (BLAS) | SciPy CSR float32 | BIE binary kernel (fixed) | BIE-v2 kernel |
|---|---|---|---|---|
| 50% | **0.43** | 8.13 | 14.92 | 0.68 |
| 90% | 0.27 | 1.72 | 2.99 | **0.16** |
| 95% | 0.24 | 0.85 | 1.47 | **0.11** |

The BIE-v2 kernel is faster than SciPy CSR at every sparsity tested. It beats dense BLAS from 90% sparsity upward and loses to it at 50–70%. Positions are decoded once when a matrix is loaded (1.4 ms at 95%, 12.7 ms at 50%). The kernel output matches multiplying its own decoded matrix (maximum difference 8.6 × 10⁻⁷ of the output scale).

### 3. Progressive precision

One 8-bit BIE-v2 encoding read with only its top p magnitude planes. The error roughly halves with each extra plane (1024×1024, 90% sparsity: 0.41, 0.21, 0.10, 0.05, 0.03, 0.02, 0.006 for 1–7 planes). One stored file can therefore serve several precisions.

### 4. Real weights: GPT-2 small

All 84.9M weights of GPT-2's 48 attention and MLP matrices were magnitude-pruned per matrix (no retraining), then compressed. Perplexity is measured on WikiText-2 raw test (287,644 tokens).

![GPT-2 size vs perplexity](docs/figures/gpt2_ratio_ppl.png)

| Method | 0%: size / ppl | 50% | 70% | 90% |
|---|---|---|---|---|
| Dense float32 (reference) | 1.00× / 29.9 | 1.00× / 534.8 | 1.00× / 5,700 | 1.00× / 14,615 |
| Dense int8 (per-row) | 3.99× / 30.0 | 3.99× / 526.5 | 3.99× / 5,765 | 3.99× / 14,671 |
| Bitmap + int8 values | 3.55× / 30.0 | 6.38× / 526.5 | 9.37× / 5,765 | 17.62× / 14,671 |
| BIE binary (preprint) | 0.67× / >10⁵ | 1.33× / >10⁵ | 2.21× / >10⁵ | 6.63× / >10⁵ |
| BIE bitplane 8-bit (preprint) | 0.27× / 58.9 | 0.27× / 1,490 | 0.28× / 3,987 | 0.28× / 12,281 |
| **BIE-v2 8-bit** | 3.54× / 30.0 | 6.36× / 526.5 | 9.64× / 5,765 | **24.54×** / 14,671 |
| BIE-v2 4-bit | 6.36× / 46.5 | 10.56× / 701.5 | 15.11× / 6,700 | 35.41× / 11,623 |

- Compare each row with the float32 reference at the same sparsity. Compression should not move perplexity away from it.
- BIE-v2 8-bit, bitmap + int8 and CSR int8 quantise the non-zeros identically, so their perplexities are equal by construction; they differ only in size.
- The very high perplexities at 50–90% come from the pruning, not the storage format. GPT-2 small does not survive unstructured magnitude pruning without retraining.

## Reproduce

```bash
pip install -r requirements-experiments.txt
bash experiments/run_all.sh
```

`run_all.sh` runs `run_synthetic.py` (storage, error, timing, progressive), then `run_gpt2.py` (downloads GPT-2 and WikiText-2 from Hugging Face), then `make_report.py`, which writes `experiments/RESULTS.md` and `docs/figures/`. The whole run took about 4 minutes on the pod described above.

## Repository layout

```
src/bie/encoder.py          original BIE encoders/decoders (binary, bitplane, blocked)
src/bie/sparse_kernels.py   original kernels (race in the binary kernel fixed 17 Oct 2026)
src/bie/bie2.py             BIE-v2: Rice-coded positions, int8/4-bit values, bitplanes, row-parallel kernel
experiments/methods.py      every method behind one interface, storage counted from stored arrays
experiments/run_synthetic.py, run_gpt2.py, make_report.py, run_all.sh
experiments/runs/           raw results (JSON)
experiments/RESULTS.md      full tables
src/baseline, src/benchmarks, src/utils, src/visualization, run_experiments.py
                            original benchmark suite (Sept 2025)
results/, results_demo/     outputs of the original Sept 2025 run, kept for the record; superseded by experiments/
```

## Citation

```bibtex
@misc{kurra2025bie,
  title  = {BIE: Bit-Index Encoding for Efficient Neural Network Weight Compression},
  author = {Kurra, Gruhesh Sri Sai Karthik},
  year   = {2025},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.17217218},
  note   = {Preprint}
}
```

## License

MIT.
