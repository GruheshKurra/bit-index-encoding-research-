#!/usr/bin/env python3
"""
Publication-Quality Figure Generator for BIE Research Paper
Creates professional visualizations suitable for academic publication.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path    
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class PaperFigureGenerator:
    """Generate publication-quality figures for BIE research paper."""
    
    def __init__(self, results_file: str, output_dir: str = "paper_figures"):
        """Initialize with benchmark results."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load benchmark results
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Define method categories and colors
        self.bie_methods = {
            'bie_binary': 'BIE Binary',
            'bie_bitplane_8bit': 'BIE Bitplane 8-bit',
            'bie_bitplane_4bit': 'BIE Bitplane 4-bit',
            'bie_blocked_binary': 'BIE Blocked Binary',
            'bie_blocked_bitplane': 'BIE Blocked Bitplane'
        }
        
        self.baseline_methods = {
            'baseline_dense_fp32': 'Dense FP32',
            'baseline_dense_fp16': 'Dense FP16',
            'baseline_quantized_8bit': 'Quantized 8-bit',
            'baseline_quantized_4bit': 'Quantized 4-bit'
        }
        
        # Color scheme for methods
        self.colors = {
            'BIE Binary': '#2E86AB',
            'BIE Bitplane 8-bit': '#A23B72',
            'BIE Bitplane 4-bit': '#F18F01',
            'BIE Blocked Binary': '#C73E1D',
            'BIE Blocked Bitplane': '#592E83',
            'Dense FP32': '#7D8491',
            'Dense FP16': '#A8DADC',
            'Quantized 8-bit': '#457B9D',
            'Quantized 4-bit': '#1D3557'
        }
    
    def extract_compression_data(self) -> pd.DataFrame:
        """Extract compression data for analysis."""
        data = []
        
        for matrix_type, methods in self.results['compression'].items():
            # Parse matrix info
            if 'sparse' in matrix_type:
                parts = matrix_type.split('_')
                sparsity = float(parts[1])
                size = parts[2]
            else:
                sparsity = 0.0
                size = matrix_type.replace('dense_', '')
            
            for method_key, metrics in methods.items():
                if 'error' in metrics:
                    continue
                    
                method_name = self.bie_methods.get(method_key) or self.baseline_methods.get(method_key)
                if not method_name:
                    continue
                
                data.append({
                    'method': method_name,
                    'method_type': 'BIE' if method_key.startswith('bie_') else 'Baseline',
                    'matrix_size': size,
                    'sparsity': sparsity,
                    'compression_ratio': metrics['compression_ratio'],
                    'space_savings': metrics['space_savings_percent'],
                    'encode_time': metrics['encode_time']
                })
        
        return pd.DataFrame(data)
    
    def extract_speed_data(self) -> pd.DataFrame:
        """Extract speed benchmark data."""
        data = []
        
        for matrix_type, size_data in self.results['speed'].items():
            if not size_data:  # Skip empty entries
                continue
                
            # Parse matrix info
            if 'sparse' in matrix_type:
                parts = matrix_type.split('_')
                sparsity = float(parts[1])
                size = parts[2]
            else:
                sparsity = 0.0
                size = matrix_type.replace('dense_', '')
            
            for input_size, methods in size_data.items():
                for method_key, metrics in methods.items():
                    if 'error' in metrics:
                        continue
                    
                    method_name = self.bie_methods.get(method_key) or self.baseline_methods.get(method_key)
                    if not method_name:
                        continue
                    
                    data.append({
                        'method': method_name,
                        'method_type': 'BIE' if method_key.startswith('bie_') else 'Baseline',
                        'matrix_size': size,
                        'sparsity': sparsity,
                        'avg_time': metrics['avg_time'],
                        'std_time': metrics['std_time'],
                        'min_time': metrics['min_time'],
                        'max_time': metrics['max_time']
                    })
        
        return pd.DataFrame(data)
    
    def extract_accuracy_data(self) -> pd.DataFrame:
        """Extract accuracy benchmark data."""
        data = []
        
        for matrix_type, methods in self.results['accuracy'].items():
            # Parse matrix info
            if 'sparse' in matrix_type:
                parts = matrix_type.split('_')
                sparsity = float(parts[1])
                size = parts[2]
            else:
                sparsity = 0.0
                size = matrix_type.replace('dense_', '')
            
            for method_key, metrics in methods.items():
                if 'error' in metrics:
                    continue
                
                method_name = self.bie_methods.get(method_key) or self.baseline_methods.get(method_key)
                if not method_name:
                    continue
                
                # Convert string numbers to float
                mse = float(metrics['mse']) if metrics['mse'] != 'Infinity' else np.inf
                mae = float(metrics['mae']) if metrics['mae'] != 'Infinity' else np.inf
                rmse = float(metrics['rmse']) if metrics['rmse'] != 'Infinity' else np.inf
                max_error = float(metrics['max_error']) if metrics['max_error'] != 'Infinity' else np.inf
                
                data.append({
                    'method': method_name,
                    'method_type': 'BIE' if method_key.startswith('bie_') else 'Baseline',
                    'matrix_size': size,
                    'sparsity': sparsity,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'max_error': max_error
                })
        
        return pd.DataFrame(data)
    
    def create_compression_comparison(self):
        """Create Figure 1: Compression Ratio Comparison."""
        df = self.extract_compression_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Filter for 512x512 matrices for cleaner visualization
        df_filtered = df[df['matrix_size'] == '512x512']
        
        # Plot 1: Compression ratio vs sparsity
        for method in df_filtered['method'].unique():
            method_data = df_filtered[df_filtered['method'] == method]
            ax1.plot(method_data['sparsity'], method_data['compression_ratio'], 
                    'o-', label=method, color=self.colors[method], linewidth=2, markersize=6)
        
        ax1.set_xlabel('Matrix Sparsity')
        ax1.set_ylabel('Compression Ratio')
        ax1.set_title('(a) Compression Ratio vs Sparsity')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.0)
        
        # Plot 2: Space savings comparison
        dense_data = df_filtered[df_filtered['sparsity'] == 0.0]
        methods = dense_data['method'].tolist()
        savings = dense_data['space_savings'].tolist()
        
        bars = ax2.bar(range(len(methods)), savings, 
                      color=[self.colors[m] for m in methods])
        ax2.set_xlabel('Compression Method')
        ax2.set_ylabel('Space Savings (%)')
        ax2.set_title('(b) Space Savings for Dense Matrices')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, savings):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure1_compression_comparison.pdf')
        plt.savefig(self.output_dir / 'figure1_compression_comparison.png')
        plt.close()
    
    def create_speed_performance(self):
        """Create Figure 2: Speed Performance Analysis."""
        df = self.extract_speed_data()
        
        if df.empty:
            print("No speed data available for visualization")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Average execution time comparison
        df_512 = df[df['matrix_size'] == '512x512']
        if not df_512.empty:
            methods = df_512['method'].unique()
            avg_times = []
            std_times = []
            
            for method in methods:
                method_data = df_512[df_512['method'] == method]
                avg_times.append(method_data['avg_time'].mean())
                std_times.append(method_data['std_time'].mean())
            
            bars = ax1.bar(range(len(methods)), avg_times, 
                          yerr=std_times, capsize=5,
                          color=[self.colors[m] for m in methods])
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Average Execution Time (s)')
            ax1.set_title('(a) Execution Time Comparison')
            ax1.set_xticks(range(len(methods)))
            ax1.set_xticklabels(methods, rotation=45, ha='right')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup relative to Dense FP32
        if 'Dense FP32' in df['method'].values:
            baseline_time = df[df['method'] == 'Dense FP32']['avg_time'].mean()
            
            speedups = []
            method_names = []
            for method in df['method'].unique():
                if method != 'Dense FP32':
                    method_time = df[df['method'] == method]['avg_time'].mean()
                    speedup = baseline_time / method_time
                    speedups.append(speedup)
                    method_names.append(method)
            
            bars = ax2.bar(range(len(method_names)), speedups,
                          color=[self.colors[m] for m in method_names])
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Speedup vs Dense FP32')
            ax2.set_title('(b) Relative Speedup')
            ax2.set_xticks(range(len(method_names)))
            ax2.set_xticklabels(method_names, rotation=45, ha='right')
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure2_speed_performance.pdf')
        plt.savefig(self.output_dir / 'figure2_speed_performance.png')
        plt.close()
    
    def create_accuracy_analysis(self):
        """Create Figure 3: Accuracy Preservation Analysis."""
        df = self.extract_accuracy_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Filter for 512x512 dense matrices
        df_dense = df[(df['matrix_size'] == '512x512') & (df['sparsity'] == 0.0)]
        
        if not df_dense.empty:
            methods = df_dense['method'].tolist()
            
            # Plot 1: MSE comparison (log scale)
            mse_values = df_dense['mse'].tolist()
            bars1 = ax1.bar(range(len(methods)), mse_values,
                           color=[self.colors[m] for m in methods])
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Mean Squared Error')
            ax1.set_title('(a) Mean Squared Error')
            ax1.set_xticks(range(len(methods)))
            ax1.set_xticklabels(methods, rotation=45, ha='right')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: MAE comparison
            mae_values = df_dense['mae'].tolist()
            bars2 = ax2.bar(range(len(methods)), mae_values,
                           color=[self.colors[m] for m in methods])
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.set_title('(b) Mean Absolute Error')
            ax2.set_xticks(range(len(methods)))
            ax2.set_xticklabels(methods, rotation=45, ha='right')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: RMSE comparison
            rmse_values = df_dense['rmse'].tolist()
            bars3 = ax3.bar(range(len(methods)), rmse_values,
                           color=[self.colors[m] for m in methods])
            ax3.set_xlabel('Method')
            ax3.set_ylabel('Root Mean Squared Error')
            ax3.set_title('(c) Root Mean Squared Error')
            ax3.set_xticks(range(len(methods)))
            ax3.set_xticklabels(methods, rotation=45, ha='right')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Max error comparison
            max_error_values = df_dense['max_error'].tolist()
            bars4 = ax4.bar(range(len(methods)), max_error_values,
                           color=[self.colors[m] for m in methods])
            ax4.set_xlabel('Method')
            ax4.set_ylabel('Maximum Error')
            ax4.set_title('(d) Maximum Error')
            ax4.set_xticks(range(len(methods)))
            ax4.set_xticklabels(methods, rotation=45, ha='right')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure3_accuracy_analysis.pdf')
        plt.savefig(self.output_dir / 'figure3_accuracy_analysis.png')
        plt.close()
    
    def create_pareto_frontier(self):
        """Create Figure 4: Pareto Frontier Analysis."""
        compression_df = self.extract_compression_data()
        accuracy_df = self.extract_accuracy_data()
        
        # Merge compression and accuracy data
        merged_df = pd.merge(compression_df, accuracy_df, 
                           on=['method', 'method_type', 'matrix_size', 'sparsity'])
        
        # Filter for dense 512x512 matrices
        df_plot = merged_df[(merged_df['matrix_size'] == '512x512') & 
                           (merged_df['sparsity'] == 0.0)]
        
        if df_plot.empty:
            print("No data available for Pareto frontier analysis")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Compression vs Accuracy (RMSE)
        for method_type in ['BIE', 'Baseline']:
            subset = df_plot[df_plot['method_type'] == method_type]
            ax1.scatter(subset['compression_ratio'], subset['rmse'], 
                       s=100, alpha=0.7, label=method_type)
            
            # Add method labels
            for _, row in subset.iterrows():
                ax1.annotate(row['method'], 
                           (row['compression_ratio'], row['rmse']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
        
        ax1.set_xlabel('Compression Ratio')
        ax1.set_ylabel('Root Mean Squared Error')
        ax1.set_title('(a) Compression vs Accuracy Trade-off')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Space Savings vs Accuracy
        ax2.scatter(df_plot['space_savings'], df_plot['rmse'], 
                   c=[self.colors[m] for m in df_plot['method']], 
                   s=100, alpha=0.7)
        
        # Add method labels
        for _, row in df_plot.iterrows():
            ax2.annotate(row['method'], 
                       (row['space_savings'], row['rmse']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)
        
        ax2.set_xlabel('Space Savings (%)')
        ax2.set_ylabel('Root Mean Squared Error')
        ax2.set_title('(b) Space Savings vs Accuracy')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure4_pareto_frontier.pdf')
        plt.savefig(self.output_dir / 'figure4_pareto_frontier.png')
        plt.close()
    
    def create_scalability_analysis(self):
        """Create Figure 5: Scalability Analysis."""
        compression_df = self.extract_compression_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Compression ratio across matrix sizes
        matrix_sizes = ['256x256', '512x512', '1024x1024', '2048x1024']
        available_sizes = compression_df['matrix_size'].unique()
        
        for method in compression_df['method'].unique():
            if method in self.bie_methods.values():  # Only plot BIE methods
                ratios = []
                sizes_to_plot = []
                
                for size in matrix_sizes:
                    if size in available_sizes:
                        method_data = compression_df[
                            (compression_df['method'] == method) & 
                            (compression_df['matrix_size'] == size) &
                            (compression_df['sparsity'] == 0.0)
                        ]
                        if not method_data.empty:
                            ratios.append(method_data['compression_ratio'].iloc[0])
                            sizes_to_plot.append(size)
                
                if ratios:
                    ax1.plot(range(len(sizes_to_plot)), ratios, 'o-', 
                            label=method, color=self.colors[method], 
                            linewidth=2, markersize=6)
        
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Compression Ratio')
        ax1.set_title('(a) Compression Scalability')
        ax1.set_xticks(range(len(sizes_to_plot)))
        ax1.set_xticklabels(sizes_to_plot)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Encoding time across matrix sizes
        for method in compression_df['method'].unique():
            if method in self.bie_methods.values():  # Only plot BIE methods
                times = []
                sizes_to_plot = []
                
                for size in matrix_sizes:
                    if size in available_sizes:
                        method_data = compression_df[
                            (compression_df['method'] == method) & 
                            (compression_df['matrix_size'] == size) &
                            (compression_df['sparsity'] == 0.0)
                        ]
                        if not method_data.empty:
                            times.append(method_data['encode_time'].iloc[0])
                            sizes_to_plot.append(size)
                
                if times:
                    ax2.plot(range(len(sizes_to_plot)), times, 'o-', 
                            label=method, color=self.colors[method], 
                            linewidth=2, markersize=6)
        
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Encoding Time (s)')
        ax2.set_title('(b) Encoding Time Scalability')
        ax2.set_xticks(range(len(sizes_to_plot)))
        ax2.set_xticklabels(sizes_to_plot)
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure5_scalability_analysis.pdf')
        plt.savefig(self.output_dir / 'figure5_scalability_analysis.png')
        plt.close()
    
    def create_summary_table(self):
        """Create a summary table for the paper."""
        compression_df = self.extract_compression_data()
        accuracy_df = self.extract_accuracy_data()
        
        # Merge data for 512x512 dense matrices
        merged_df = pd.merge(compression_df, accuracy_df, 
                           on=['method', 'method_type', 'matrix_size', 'sparsity'])
        
        summary_df = merged_df[
            (merged_df['matrix_size'] == '512x512') & 
            (merged_df['sparsity'] == 0.0)
        ][['method', 'compression_ratio', 'space_savings', 'encode_time', 'rmse']]
        
        # Round values for presentation
        summary_df['compression_ratio'] = summary_df['compression_ratio'].round(3)
        summary_df['space_savings'] = summary_df['space_savings'].round(1)
        summary_df['encode_time'] = summary_df['encode_time'].apply(lambda x: f"{x:.2e}")
        summary_df['rmse'] = summary_df['rmse'].apply(lambda x: f"{x:.2e}")
        
        # Save as CSV
        summary_df.to_csv(self.output_dir / 'table1_performance_summary.csv', index=False)
        
        # Create LaTeX table
        latex_table = summary_df.to_latex(index=False, 
                                         column_format='lcccc',
                                         caption='Performance Summary for Dense 512×512 Matrices',
                                         label='tab:performance_summary')
        
        with open(self.output_dir / 'table1_performance_summary.tex', 'w') as f:
            f.write(latex_table)
    
    def generate_all_figures(self):
        """Generate all publication figures."""
        print("Generating publication-quality figures...")
        
        print("Creating Figure 1: Compression Comparison...")
        self.create_compression_comparison()
        
        print("Creating Figure 2: Speed Performance...")
        self.create_speed_performance()
        
        print("Creating Figure 3: Accuracy Analysis...")
        self.create_accuracy_analysis()
        
        print("Creating Figure 4: Pareto Frontier...")
        self.create_pareto_frontier()
        
        print("Creating Figure 5: Scalability Analysis...")
        self.create_scalability_analysis()
        
        print("Creating Summary Table...")
        self.create_summary_table()
        
        print(f"\nAll figures saved to: {self.output_dir.absolute()}")
        print("Files generated:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")


def main():
    """Main function to generate all paper figures."""
    import sys
    
    # Default to the most recent results file
    results_file = "results/benchmark_results_1758633103.json"
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: Results file {results_file} not found!")
        return
    
    generator = PaperFigureGenerator(results_file)
    generator.generate_all_figures()


if __name__ == "__main__":
    main()