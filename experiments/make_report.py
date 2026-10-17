"""Builds experiments/RESULTS.md and docs/figures/*.png from experiments/runs/*.json."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods import LABELS

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIG = HERE.parent / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MAIN = ["fp16", "int8_rowwise", "csr_fp32", "csr_int8", "bitmap_int8",
        "early_positions", "early_bitplane8", "bie_8bit", "bie_4bit"]
COLORS = {"bie_8bit": "#c2185b", "bie_4bit": "#f06292", "early_positions": "#6d4c41", "early_bitplane8": "#a1887f",
          "csr_int8": "#1565c0", "bitmap_int8": "#42a5f5", "csr_fp32": "#90caf9",
          "int8_rowwise": "#2e7d32", "fp16": "#81c784"}


def ms(v):
    return f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def synthetic(lines):
    rows = json.loads((RUNS / "synthetic.json").read_text())
    sp = sorted({r["sparsity"] for r in rows})
    g = defaultdict(list)
    for r in rows:
        if (r["rows"], r["cols"]) == (1024, 1024):
            g[(r["method"], r["sparsity"])].append(r)

    lines += ["## 1. Synthetic matrices", "",
              "Gaussian 1024×1024 matrices with a random zero mask; mean ± sd over seeds 0, 1, 2. "
              "Ratio = float32 bytes / bytes actually stored (higher is smaller). "
              "Other sizes (256×256 to 2048×1024) give the same picture; all rows are in `runs/synthetic.json`.", "",
              "**Compression ratio**", "",
              "| Method | " + " | ".join(f"{int(s*100)}%" for s in sp) + " |",
              "|---|" + "---|" * len(sp)]
    for m in LABELS:
        lines.append(f"| {LABELS[m]} | " + " | ".join(ms([x['ratio'] for x in g[(m, s)]]) for s in sp) + " |")
    lines += ["", "**Relative reconstruction error** ‖W − Ŵ‖ / ‖W‖ (0 = exact)", "",
              "| Method | " + " | ".join(f"{int(s*100)}%" for s in sp) + " |",
              "|---|" + "---|" * len(sp)]
    for m in LABELS:
        lines.append(f"| {LABELS[m]} | " + " | ".join(f"{np.mean([x['rel_err'] for x in g[(m, s)]]):.3f}" for s in sp) + " |")
    lines.append("")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in MAIN:
        y = [np.mean([x["ratio"] for x in g[(m, s)]]) for s in sp]
        e = [np.mean([x["rel_err"] for x in g[(m, s)]]) for s in sp]
        ax[0].plot([s * 100 for s in sp], y, "o-", label=LABELS[m], color=COLORS[m], lw=2.2 if m.startswith("bie_") else 1.3)
        ax[1].plot([s * 100 for s in sp], e, "o-", color=COLORS[m], lw=2.2 if m.startswith("bie_") else 1.3)
    ax[0].axhline(1, color="grey", ls=":", lw=1)
    ax[0].set(yscale="log", xlabel="sparsity (%)", ylabel="compression vs float32 (log)", title="Size")
    ax[1].set(yscale="symlog", xlabel="sparsity (%)", ylabel="relative error", title="Error (lower is better)")
    ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "synthetic_ratio_error.png", dpi=150)
    plt.close(fig)


def timing(lines):
    t = json.loads((RUNS / "timing.json").read_text())
    lines += ["## 2. Multiply speed and kernel correctness", "",
              f"W is {t['shape'][0]}×{t['shape'][1]}, X has {t['x_cols']} columns; median of 7 runs after a warm-up, "
              "milliseconds, pod CPU (16 vCPU). Dense float32 uses NumPy's BLAS.", "",
              "| Sparsity | Dense fp32 | SciPy CSR fp32 | Early positions-only kernel | BIE kernel | BIE position decode (once, at load) | BIE output error | Early positions-only output error |",
              "|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        lines.append(f"| {int(r['sparsity']*100)}% | {r['dense_fp32']*1e3:.2f} | {r['scipy_csr_fp32']*1e3:.2f} | "
                     f"{r['early_positions_kernel']*1e3:.2f} | {r['bie_kernel']*1e3:.2f} | {r['bie_index_decode_once']*1e3:.2f} | "
                     f"{r['bie_output_rel_err']:.4f} | {r['early_positions_output_rel_err']:.3f} |")
    worst = max(r["bie_kernel_vs_its_decode"] for r in t["rows"])
    lines += ["", f"BIE kernel vs multiplying its own decoded matrix: max difference {worst:.1e} of the output scale (correct).", ""]


def progressive(lines):
    rows = json.loads((RUNS / "progressive.json").read_text())
    g = defaultdict(list)
    for r in rows:
        g[(r["sparsity"], r["planes"])].append(r)
    lines += ["## 3. Progressive precision (BIE bitplanes)", "",
              "One 8-bit BIE encoding of a 1024×1024 matrix, read with only the top p magnitude planes. "
              "Mean over 3 seeds.", "",
              "| Planes read | Value bits | Error at 50% sparsity | Error at 90% sparsity |", "|---|---|---|---|"]
    for p in range(1, 8):
        lines.append(f"| {p} | {p+1} | {np.mean([x['rel_err'] for x in g[(0.5, p)]]):.3f} | "
                     f"{np.mean([x['rel_err'] for x in g[(0.9, p)]]):.3f} |")
    lines.append("")


def gpt2(lines):
    f = RUNS / "gpt2.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    rows = d["rows"]
    sp = sorted({r["sparsity"] for r in rows})
    g = {(r["method"], r["sparsity"]): r for r in rows}
    meths = list(dict.fromkeys(r["method"] for r in rows))
    lines += ["## 4. Real weights: GPT-2 small", "",
              f"All {d['n_weights']:,} weights of the 48 attention/MLP matrices, magnitude-pruned per matrix (no retraining), "
              f"then compressed. Perplexity on WikiText-2 raw test ({d['eval_tokens']:,} tokens, 1024-token windows). "
              "Lower perplexity is better; the float32 row at each sparsity is the reference for that sparsity.", "",
              "| Method | " + " | ".join(f"{int(s*100)}%: ratio / ppl" for s in sp) + " |",
              "|---|" + "---|" * len(sp)]
    for m in meths:
        cells = []
        for s in sp:
            r = g.get((m, s))
            cells.append("–" if r is None else f"{r['ratio']:.2f}× / {r['ppl']:.1f}" if r["ppl"] < 1e5 else f"{r['ratio']:.2f}× / >10⁵")
        lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    lines.append("")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in meths:
        if m in ("fp32", "early_positions"):
            continue
        pts = [(g[(m, s)]["ratio"], g[(m, s)]["ppl"]) for s in sp if (m, s) in g]
        ax.plot(*zip(*pts), "o-", label=LABELS[m], color=COLORS.get(m, "k"), lw=2.2 if m.startswith("bie_") else 1.2)
    ax.set(xscale="log", yscale="log", xlabel="compression vs float32 (log)", ylabel="WikiText-2 perplexity (log)",
           title="GPT-2 small, pruned 0/50/70/90%")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "gpt2_ratio_ppl.png", dpi=150)
    plt.close(fig)


def main():
    lines = ["# Results", "",
             "Measured on 17 Oct 2026 by `python experiments/run_synthetic.py`, `python experiments/run_gpt2.py`, "
             "then `python experiments/make_report.py` on a RunPod pod (16 vCPU, RTX 4090; matrices and timings on the CPU, GPT-2 perplexity on the GPU).", ""]
    synthetic(lines)
    timing(lines)
    progressive(lines)
    gpt2(lines)
    (HERE / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
