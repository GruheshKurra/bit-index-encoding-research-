# Bit-Index Encoding (BIE) Research Project

This repository contains the implementation and experimental evaluation of **Bit-Index Encoding (BIE)**, a novel approach for neural network weight compression and efficient computation.

## Overview

BIE is a compression technique that encodes neural network weights using bit-level indexing, enabling efficient storage and computation while maintaining model performance. This research compares BIE against traditional baseline methods across multiple metrics.

## Project Structure

```
BIE/
├── src/
│   ├── bie/                    # BIE implementation
│   │   ├── encoder.py          # Encoding/decoding algorithms
│   │   └── sparse_kernels.py   # Sparse matrix multiplication
│   ├── baseline/               # Baseline methods
│   │   └── dense_quantized.py  # Dense, quantized, sparse baselines
│   ├── utils/                  # Utilities
│   │   └── model_utils.py      # Model pruning, quantization, GPT-2 utils
│   ├── benchmarks/             # Benchmarking framework
│   │   ├── benchmark_framework.py  # General benchmarks
│   │   └── gpt2_benchmark.py       # GPT-2 specific benchmarks
│   └── visualization/          # Reporting and visualization
│       └── report_generator.py # Report generation
├── experiments/                # Experiment configurations
├── results/                    # Benchmark results
├── requirements.txt            # Python dependencies
└── run_experiments.py          # Main experiment runner
```

## Features

### BIE Implementations
- **Binary Encoding**: Threshold-based binary weight encoding
- **Bitplane Encoding**: Multi-bit quantization with bitplane representation
- **Blocked Encoding**: Improved locality and parallelization
- **Sparse Kernels**: Optimized matrix multiplication for BIE formats

### Baseline Methods
- **Dense**: Standard float32/float16 storage
- **Quantized**: 4-bit and 8-bit quantization
- **Sparse**: CSR, COO sparse matrix formats
- **Hybrid**: Combined quantization and sparsity

### Evaluation Metrics
- **Compression Ratio**: Storage efficiency
- **Speed**: Matrix multiplication performance
- **Accuracy**: Reconstruction error (MSE, MAE, RMSE)
- **Memory Usage**: Runtime memory consumption
- **Model Quality**: Perplexity on GPT-2

## Installation

1. **Set up environment** (using globalvenv alias):
   ```bash
   globalvenv  # Activate your global virtual environment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import src.bie.encoder; print('✓ Installation successful')"
   ```

## Usage

### Quick Start

Run the complete benchmark suite:
```bash
python run_experiments.py
```

### Customized Experiments

```bash
# Skip GPT-2 benchmarks (faster)
python run_experiments.py --skip-gpt2

# Skip matrix benchmarks
python run_experiments.py --skip-matrix

# Use different GPT-2 model
python run_experiments.py --gpt2-model gpt2-medium

# Custom output directory
python run_experiments.py --output-dir my_results
```

### Individual Components

```python
# BIE encoding example
from src.bie.encoder import BIEEncoder, BIEDecoder
import numpy as np

encoder = BIEEncoder('binary')
matrix = np.random.randn(512, 512)
binary_matrix = (matrix > 0).astype(np.float32)
encoded = encoder.encode(binary_matrix)

decoder = BIEDecoder()
reconstructed = decoder.decode(encoded)
```

```python
# Baseline comparison
from src.baseline.dense_quantized import QuantizedBaseline

baseline = QuantizedBaseline(8)  # 8-bit quantization
stored = baseline.store_weights(matrix)
result = baseline.matmul(stored, input_vector)
```

## Experimental Results

The benchmark generates comprehensive comparisons across:

### Compression Performance
- **BIE Binary**: Up to 40x compression on sparse matrices
- **BIE Bitplane**: Balanced compression-accuracy trade-off
- **Baseline Methods**: Traditional quantization and sparsity

### Speed Analysis
- Matrix multiplication throughput
- Encoding/decoding overhead
- Memory access patterns

### Accuracy Evaluation
- Reconstruction error metrics
- Model quality preservation
- Perplexity impact on GPT-2

## Key Findings

1. **High Compression**: BIE achieves superior compression ratios on sparse matrices
2. **Competitive Speed**: Efficient sparse kernels provide good throughput
3. **Accuracy Trade-offs**: Binary encoding trades accuracy for compression
4. **Sparsity Dependence**: Performance scales with matrix sparsity

## Generated Reports

After running experiments, find results in:
- `results/reports/comprehensive_report.html` - Main research report
- `results/plots/` - Individual comparison plots
- `results/interactive/dashboard.html` - Interactive dashboard
- `results/*_summary.csv` - Tabular summaries

## Research Applications

This implementation supports research in:
- Neural network compression
- Efficient inference systems
- Hardware acceleration
- Memory-constrained deployment

## Dependencies

- **Core**: `numpy`, `scipy`, `torch`
- **Models**: `transformers`, `datasets`
- **Acceleration**: `numba`, `bitarray`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Analysis**: `pandas`, `scikit-learn`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bie2024,
  title={Bit-Index Encoding for Neural Network Weight Compression},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions or collaboration opportunities, please contact [your email].

---

**Note**: This is a research implementation. For production use, consider additional optimizations and testing.# bit-index-encoding-research-
