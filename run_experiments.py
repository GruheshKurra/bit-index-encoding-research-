#!/usr/bin/env python3
"""
My main experiment runner - this is where I orchestrate all the BIE research.
Runs comprehensive benchmarks comparing BIE with baseline methods.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from benchmarks.benchmark_framework import BenchmarkRunner
from benchmarks.gpt2_benchmark import GPT2BenchmarkRunner
from visualization.report_generator import BIEReportGenerator


def run_matrix_benchmarks(output_dir: str = "results"):
    """
    Run my comprehensive matrix benchmarks - testing BIE on various matrix types.
    """
    print("=" * 60)
    print("RUNNING MATRIX BENCHMARKS")
    print("=" * 60)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(output_dir)
    
    # Define test configurations - these are the matrix sizes I want to test
    matrix_sizes = [
        (256, 256),   # Small
        (512, 512),   # Medium
        (1024, 1024), # Large
        (2048, 1024)  # Rectangular
    ]
    
    sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]
    
    print(f"Testing matrix sizes: {matrix_sizes}")
    print(f"Testing sparsity levels: {sparsity_levels}")
    print()
    
    # Run comprehensive benchmark
    start_time = time.time()
    
    try:
        results = runner.run_comprehensive_benchmark(
            matrix_sizes=matrix_sizes,
            sparsity_levels=sparsity_levels,
            num_trials=3
        )
        
        elapsed = time.time() - start_time
        print(f"\nMatrix benchmarks completed in {elapsed:.2f} seconds")
        
        # Save results
        results_file = Path(output_dir) / f"benchmark_results_{int(time.time())}.json"
        runner.save_results(results, results_file)
        print(f"Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"Error during matrix benchmarks: {e}")
        return None


def run_gpt2_benchmarks(output_dir: str = "results", model_name: str = "gpt2"):
    """
    Run GPT-2 model benchmarks - testing BIE on real transformer weights.
    """
    print("=" * 60)
    print("RUNNING GPT-2 BENCHMARKS")
    print("=" * 60)
    
    try:
        # Initialize GPT-2 benchmark runner
        gpt2_runner = GPT2BenchmarkRunner(output_dir, model_name)
        
        print(f"Testing model: {model_name}")
        print("This will test BIE compression on actual transformer weights...")
        print()
        
        start_time = time.time()
        
        # Run comprehensive GPT-2 benchmark
        results = gpt2_runner.run_comprehensive_gpt2_benchmark()
        
        elapsed = time.time() - start_time
        print(f"\nGPT-2 benchmarks completed in {elapsed:.2f} seconds")
        
        # Save results
        results_file = Path(output_dir) / f"gpt2_benchmark_results_{int(time.time())}.json"
        gpt2_runner.save_results(results, results_file)
        print(f"GPT-2 results saved to: {results_file}")
        
        return results
        
    except ImportError as e:
        print(f"Warning: Could not run GPT-2 benchmarks - missing dependencies: {e}")
        print("Install transformers and torch to enable GPT-2 testing")
        return None
    except Exception as e:
        print(f"Error during GPT-2 benchmarks: {e}")
        return None


def generate_reports(matrix_results, gpt2_results=None, output_dir: str = "results"):
    """
    Generate all the visualization reports from my benchmark results.
    """
    print("=" * 60)
    print("GENERATING REPORTS AND VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Initialize report generator
        generator = BIEReportGenerator(output_dir)
        
        if matrix_results:
            print("Creating matrix benchmark plots...")
            
            # Create all the standard plots
            plots_dir = Path(output_dir) / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            generator.create_compression_comparison_plot(
                matrix_results, plots_dir / "compression_comparison.png"
            )
            
            generator.create_speed_comparison_plot(
                matrix_results, plots_dir / "speed_comparison.png"
            )
            
            generator.create_accuracy_comparison_plot(
                matrix_results, plots_dir / "accuracy_comparison.png"
            )
            
            generator.create_pareto_frontier_plot(
                matrix_results, plots_dir / "pareto_frontier.png"
            )
            
            print("Creating interactive dashboard...")
            generator.create_interactive_dashboard(matrix_results)
        
        if gpt2_results:
            print("Creating GPT-2 analysis plots...")
            generator.create_gpt2_analysis_plots(gpt2_results)
        
        # Generate comprehensive report
        print("Generating comprehensive HTML report...")
        report_path = generator.generate_comprehensive_report(matrix_results, gpt2_results)
        
        print(f"\nAll reports generated successfully!")
        print(f"Main report: {report_path}")
        print(f"Interactive dashboard: {Path(output_dir) / 'interactive' / 'dashboard.html'}")
        
    except Exception as e:
        print(f"Error generating reports: {e}")


def main():
    """
    Main function - orchestrates the entire BIE research experiment.
    """
    parser = argparse.ArgumentParser(description="Run BIE research experiments")
    parser.add_argument("--output-dir", default="results", 
                       help="Directory to save results (default: results)")
    parser.add_argument("--skip-matrix", action="store_true",
                       help="Skip matrix benchmarks")
    parser.add_argument("--skip-gpt2", action="store_true",
                       help="Skip GPT-2 benchmarks")
    parser.add_argument("--gpt2-model", default="gpt2",
                       help="GPT-2 model to use (default: gpt2)")
    parser.add_argument("--skip-reports", action="store_true",
                       help="Skip report generation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("BIE Research Experiment Runner")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Skip matrix benchmarks: {args.skip_matrix}")
    print(f"Skip GPT-2 benchmarks: {args.skip_gpt2}")
    print(f"GPT-2 model: {args.gpt2_model}")
    print(f"Skip reports: {args.skip_reports}")
    print()
    
    # Run experiments
    matrix_results = None
    gpt2_results = None
    
    if not args.skip_matrix:
        matrix_results = run_matrix_benchmarks(args.output_dir)
    
    if not args.skip_gpt2:
        gpt2_results = run_gpt2_benchmarks(args.output_dir, args.gpt2_model)
    
    # Generate reports
    if not args.skip_reports and (matrix_results or gpt2_results):
        generate_reports(matrix_results, gpt2_results, args.output_dir)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    if matrix_results:
        print("✓ Matrix benchmarks completed")
    if gpt2_results:
        print("✓ GPT-2 benchmarks completed")
    if not args.skip_reports:
        print("✓ Reports and visualizations generated")
    
    print(f"\nAll results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()