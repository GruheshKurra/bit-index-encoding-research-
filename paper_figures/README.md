# Publication-Quality Figures for BIE Research Paper

This directory contains publication-ready visualizations generated from the BIE (Bit Index Encoding) benchmark results.

## Generated Figures

### Figure 1: Compression Comparison (`figure1_compression_comparison.*`)
- **Panel (a)**: Compression ratio vs matrix sparsity for different encoding methods
- **Panel (b)**: Space savings percentage for dense matrices
- Shows how BIE methods perform compared to baseline compression techniques

### Figure 2: Speed Performance (`figure2_speed_performance.*`)
- **Panel (a)**: Average execution time comparison with error bars
- **Panel (b)**: Relative speedup compared to Dense FP32 baseline
- Demonstrates the computational efficiency of different encoding methods

### Figure 3: Accuracy Analysis (`figure3_accuracy_analysis.*`)
- **Panel (a)**: Mean Squared Error (MSE) comparison
- **Panel (b)**: Mean Absolute Error (MAE) comparison  
- **Panel (c)**: Root Mean Squared Error (RMSE) comparison
- **Panel (d)**: Maximum error comparison
- All metrics shown on logarithmic scale for better visualization

### Figure 4: Pareto Frontier Analysis (`figure4_pareto_frontier.*`)
- **Panel (a)**: Compression ratio vs accuracy trade-off (RMSE)
- **Panel (b)**: Space savings vs accuracy trade-off
- Helps identify optimal methods balancing compression and accuracy

### Figure 5: Scalability Analysis (`figure5_scalability_analysis.*`)
- **Panel (a)**: Compression ratio across different matrix sizes
- **Panel (b)**: Encoding time scalability (logarithmic scale)
- Shows how BIE methods scale with increasing matrix dimensions

### Table 1: Performance Summary (`table1_performance_summary.*`)
- Comprehensive comparison table for dense 512×512 matrices
- Available in both CSV and LaTeX formats
- Includes compression ratio, space savings, encoding time, and RMSE

## File Formats

Each figure is provided in two formats:
- **PDF**: Vector format suitable for high-quality publication printing
- **PNG**: Raster format for web display and presentations

## Usage in Papers

These figures are designed to meet publication standards for academic journals and conferences:
- High resolution (300 DPI)
- Professional typography (Times New Roman font)
- Clear axis labels and legends
- Consistent color scheme
- Publication-ready sizing

## Regenerating Figures

To regenerate the figures with updated data:

```bash
python paper_figures/create_paper_figures.py [path_to_results_file]
```

If no path is specified, it defaults to `results/benchmark_results_1758633103.json`.

## Dependencies

The figure generation script requires:
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- pandas >= 1.4.0
- numpy >= 1.21.0

All dependencies are listed in the main `requirements.txt` file.