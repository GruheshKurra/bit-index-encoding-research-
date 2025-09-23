"""
My comprehensive benchmarking setup to prove BIE works better than existing methods.
This is where I test everything - compression, speed, accuracy, memory usage.
"""

import numpy as np
import torch
import time
import psutil
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our implementations
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bie.encoder import BIEEncoder, BIEDecoder, get_compression_stats
from bie.sparse_kernels import BIEMatMul, benchmark_sparse_kernels
from baseline.dense_quantized import (DenseBaseline, QuantizedBaseline, 
                                     SparseBaseline, HybridBaseline, benchmark_baselines)
from utils.model_utils import ModelPruner, ModelQuantizer, GPT2Utils, create_test_matrices


class BenchmarkRunner:
    """
    This is my main testing framework - compares BIE against all the standard approaches.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Set up the benchmark runner with all the methods I want to compare.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # My BIE implementations - different variants to test
        self.bie_encoders = {
            'binary': BIEEncoder('binary'),
            'bitplane_8bit': BIEEncoder('bitplane'),
            'bitplane_4bit': BIEEncoder('bitplane'),
            'blocked_binary': BIEEncoder('binary', block_size=64),
            'blocked_bitplane': BIEEncoder('bitplane', block_size=64)
        }
        
        # Standard baseline methods for comparison
        self.baselines = {
            'dense_fp32': DenseBaseline('float32'),
            'dense_fp16': DenseBaseline('float16'),
            'quantized_8bit': QuantizedBaseline(8),
            'quantized_4bit': QuantizedBaseline(4),
            'sparse_csr': SparseBaseline('csr'),
            'hybrid_8bit': HybridBaseline(8)
        }
    
    def benchmark_compression(self, matrices: Dict[str, np.ndarray]) -> Dict:
        """
        Test how well each method compresses the matrices.
        This is where BIE should really shine, especially on sparse data.
        """
        results = {
            'bie_methods': {},
            'baseline_methods': {}
        }
        
        # Test BIE methods
        for method_name, encoder in self.bie_encoders.items():
            results['bie_methods'][method_name] = {}
            
            for matrix_name, matrix in matrices.items():
                try:
                    start_time = time.time()
                    encoded = encoder.encode(matrix)
                    encode_time = time.time() - start_time
                    
                    # Get compression stats
                    stats = get_compression_stats(encoded, matrix.shape)
                    
                    results['bie_methods'][method_name][matrix_name] = {
                        'compression_ratio': stats['compression_ratio'],
                        'space_savings_percent': stats['space_savings_percent'],
                        'encode_time': encode_time,
                        'size_bytes': stats['compressed_size']
                    }
                except Exception as e:
                    results['bie_methods'][method_name][matrix_name] = {
                        'error': str(e),
                        'compression_ratio': 0.0,
                        'space_savings_percent': 0.0,
                        'encode_time': float('inf')
                    }
        
        # Test baseline methods
        for method_name, baseline in self.baselines.items():
            results['baseline_methods'][method_name] = {}
            
            for matrix_name, matrix in matrices.items():
                try:
                    start_time = time.time()
                    stored = baseline.store_weights(matrix)
                    store_time = time.time() - start_time
                    
                    stats = baseline.get_compression_stats(stored)
                    
                    results['baseline_methods'][method_name][matrix_name] = {
                        'compression_ratio': stats['compression_ratio'],
                        'space_savings_percent': stats['space_savings_percent'],
                        'encode_time': store_time,
                        'size_bytes': stats['size_bytes']
                    }
                except Exception as e:
                    results['baseline_methods'][method_name][matrix_name] = {
                        'error': str(e),
                        'compression_ratio': 0.0,
                        'space_savings_percent': 0.0,
                        'encode_time': float('inf')
                    }
        
        return results
    
    def benchmark_speed(self, matrices: Dict[str, np.ndarray], 
                       input_sizes: List[int] = [128, 512, 1024]) -> Dict:
        """
        Test how fast matrix multiplication is with each method.
        BIE should be competitive here, especially with the optimized kernels.
        """
        results = {
            'bie_methods': {},
            'baseline_methods': {}
        }
        
        # Test BIE methods
        for method_name, encoder in self.bie_encoders.items():
            results['bie_methods'][method_name] = {}
            
            for matrix_name, matrix in matrices.items():
                results['bie_methods'][method_name][matrix_name] = {}
                
                try:
                    # Encode the matrix first
                    encoded = encoder.encode(matrix)
                    decoder = BIEDecoder()
                    matmul = BIEMatMul()
                    
                    for input_size in input_sizes:
                        # Create random input
                        input_tensor = np.random.randn(matrix.shape[1], input_size).astype(np.float32)
                        
                        # Time the matrix multiplication
                        times = []
                        for _ in range(5):  # Average over multiple runs
                            start_time = time.time()
                            if encoded['type'] == 'binary':
                                result = matmul.binary_matmul(encoded, input_tensor)
                            elif encoded['type'] == 'bitplane':
                                result = matmul.bitplane_matmul(encoded, input_tensor)
                            else:
                                # Fallback to standard multiplication
                                decoded = decoder.decode(encoded)
                                result = np.dot(decoded, input_tensor)
                            end_time = time.time()
                            times.append(end_time - start_time)
                        
                        avg_time = np.mean(times)
                        std_time = np.std(times)
                        
                        results['bie_methods'][method_name][matrix_name][f'input_{input_size}'] = {
                            'avg_time': avg_time,
                            'std_time': std_time,
                            'throughput': input_size / avg_time  # operations per second
                        }
                        
                except Exception as e:
                    for input_size in input_sizes:
                        results['bie_methods'][method_name][matrix_name][f'input_{input_size}'] = {
                            'error': str(e),
                            'avg_time': float('inf'),
                            'std_time': 0.0,
                            'throughput': 0.0
                        }
        
        # Test baseline methods
        for method_name, baseline in self.baselines.items():
            results['baseline_methods'][method_name] = {}
            
            for matrix_name, matrix in matrices.items():
                results['baseline_methods'][method_name][matrix_name] = {}
                
                try:
                    stored = baseline.store_weights(matrix)
                    
                    for input_size in input_sizes:
                        input_tensor = np.random.randn(matrix.shape[1], input_size).astype(np.float32)
                        
                        times = []
                        for _ in range(5):
                            start_time = time.time()
                            result = baseline.matmul(stored, input_tensor)
                            end_time = time.time()
                            times.append(end_time - start_time)
                        
                        avg_time = np.mean(times)
                        std_time = np.std(times)
                        
                        results['baseline_methods'][method_name][matrix_name][f'input_{input_size}'] = {
                            'avg_time': avg_time,
                            'std_time': std_time,
                            'throughput': input_size / avg_time
                        }
                        
                except Exception as e:
                    for input_size in input_sizes:
                        results['baseline_methods'][method_name][matrix_name][f'input_{input_size}'] = {
                            'error': str(e),
                            'avg_time': float('inf'),
                            'std_time': 0.0,
                            'throughput': 0.0
                        }
        
        return results
    
    def benchmark_accuracy(self, matrices: Dict[str, np.ndarray],
                          input_size: int = 512) -> Dict:
        """
        Test how accurate each method is compared to the original.
        This is crucial - compression is useless if it destroys accuracy.
        """
        results = {
            'bie_methods': {},
            'baseline_methods': {}
        }
        
        # Create a consistent input for all tests
        input_tensors = {}
        for matrix_name, matrix in matrices.items():
            input_tensors[matrix_name] = np.random.randn(matrix.shape[1], input_size).astype(np.float32)
        
        # Get reference results (original dense fp32)
        reference_results = {}
        for matrix_name, matrix in matrices.items():
            reference_results[matrix_name] = np.dot(matrix.astype(np.float32), input_tensors[matrix_name])
        
        # Test BIE methods
        for method_name, encoder in self.bie_encoders.items():
            results['bie_methods'][method_name] = {}
            
            for matrix_name, matrix in matrices.items():
                try:
                    # Encode and decode
                    encoded = encoder.encode(matrix)
                    decoder = BIEDecoder()
                    matmul = BIEMatMul()
                    
                    # Get result
                    if encoded['type'] == 'binary':
                        result = matmul.binary_matmul(encoded, input_tensors[matrix_name])
                    elif encoded['type'] == 'bitplane':
                        result = matmul.bitplane_matmul(encoded, input_tensors[matrix_name])
                    else:
                        decoded = decoder.decode(encoded)
                        result = np.dot(decoded, input_tensors[matrix_name])
                    
                    # Calculate accuracy metrics
                    reference = reference_results[matrix_name]
                    mse = np.mean((reference - result) ** 2)
                    mae = np.mean(np.abs(reference - result))
                    rmse = np.sqrt(mse)
                    
                    # Relative error
                    ref_norm = np.linalg.norm(reference)
                    relative_error = np.linalg.norm(reference - result) / (ref_norm + 1e-8)
                    
                    # Signal-to-noise ratio
                    signal_power = np.mean(reference ** 2)
                    noise_power = np.mean((reference - result) ** 2)
                    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
                    
                    results['bie_methods'][method_name][matrix_name] = {
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse,
                        'relative_error': relative_error,
                        'snr_db': snr
                    }
                    
                except Exception as e:
                    results['bie_methods'][method_name][matrix_name] = {
                        'error': str(e),
                        'mse': float('inf'),
                        'mae': float('inf'),
                        'rmse': float('inf'),
                        'relative_error': float('inf'),
                        'snr_db': float('-inf')
                    }
        
        # Test baseline methods
        for method_name, baseline in self.baselines.items():
            results['baseline_methods'][method_name] = {}
            
            for matrix_name, matrix in matrices.items():
                try:
                    stored = baseline.store_weights(matrix)
                    result = baseline.matmul(stored, input_tensors[matrix_name])
                    
                    reference = reference_results[matrix_name]
                    mse = np.mean((reference - result) ** 2)
                    mae = np.mean(np.abs(reference - result))
                    rmse = np.sqrt(mse)
                    
                    ref_norm = np.linalg.norm(reference)
                    relative_error = np.linalg.norm(reference - result) / (ref_norm + 1e-8)
                    
                    signal_power = np.mean(reference ** 2)
                    noise_power = np.mean((reference - result) ** 2)
                    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
                    
                    results['baseline_methods'][method_name][matrix_name] = {
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse,
                        'relative_error': relative_error,
                        'snr_db': snr
                    }
                    
                except Exception as e:
                    results['baseline_methods'][method_name][matrix_name] = {
                        'error': str(e),
                        'mse': float('inf'),
                        'mae': float('inf'),
                        'rmse': float('inf'),
                        'relative_error': float('inf'),
                        'snr_db': float('-inf')
                    }
        
        return results
    
    def benchmark_memory_usage(self, matrices: Dict[str, np.ndarray]) -> Dict:
        """
        Measure actual memory usage during encoding/decoding.
        This is important for real-world deployment.
        """
        results = {
            'bie_methods': {},
            'baseline_methods': {}
        }
        
        def get_memory_usage():
            """Get current memory usage in MB"""
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        
        # Test BIE methods
        for method_name, encoder in self.bie_encoders.items():
            results['bie_methods'][method_name] = {}
            
            for matrix_name, matrix in matrices.items():
                try:
                    # Measure memory before
                    mem_before = get_memory_usage()
                    
                    # Encode
                    encoded = encoder.encode(matrix)
                    mem_after_encode = get_memory_usage()
                    
                    # Decode
                    decoder = BIEDecoder()
                    decoded = decoder.decode(encoded)
                    mem_after_decode = get_memory_usage()
                    
                    # Clean up
                    del encoded, decoded
                    
                    results['bie_methods'][method_name][matrix_name] = {
                        'memory_before_mb': mem_before,
                        'memory_after_encode_mb': mem_after_encode,
                        'memory_after_decode_mb': mem_after_decode,
                        'encode_memory_increase_mb': mem_after_encode - mem_before,
                        'decode_memory_increase_mb': mem_after_decode - mem_after_encode,
                        'peak_memory_mb': max(mem_after_encode, mem_after_decode)
                    }
                    
                except Exception as e:
                    results['bie_methods'][method_name][matrix_name] = {
                        'error': str(e),
                        'memory_before_mb': 0,
                        'memory_after_encode_mb': 0,
                        'memory_after_decode_mb': 0,
                        'encode_memory_increase_mb': 0,
                        'decode_memory_increase_mb': 0,
                        'peak_memory_mb': 0
                    }
        
        # Test baseline methods
        for method_name, baseline in self.baselines.items():
            results['baseline_methods'][method_name] = {}
            
            for matrix_name, matrix in matrices.items():
                try:
                    mem_before = get_memory_usage()
                    
                    stored = baseline.store_weights(matrix)
                    mem_after_store = get_memory_usage()
                    
                    # For baselines, "decode" is just accessing the weights
                    if hasattr(baseline, 'matmul'):
                        input_tensor = np.random.randn(matrix.shape[1], 128).astype(np.float32)
                        result = baseline.matmul(stored, input_tensor)
                        mem_after_use = get_memory_usage()
                    else:
                        mem_after_use = mem_after_store
                    
                    del stored
                    
                    results['baseline_methods'][method_name][matrix_name] = {
                        'memory_before_mb': mem_before,
                        'memory_after_encode_mb': mem_after_store,
                        'memory_after_decode_mb': mem_after_use,
                        'encode_memory_increase_mb': mem_after_store - mem_before,
                        'decode_memory_increase_mb': mem_after_use - mem_after_store,
                        'peak_memory_mb': max(mem_after_store, mem_after_use)
                    }
                    
                except Exception as e:
                    results['baseline_methods'][method_name][matrix_name] = {
                        'error': str(e),
                        'memory_before_mb': 0,
                        'memory_after_encode_mb': 0,
                        'memory_after_decode_mb': 0,
                        'encode_memory_increase_mb': 0,
                        'decode_memory_increase_mb': 0,
                        'peak_memory_mb': 0
                    }
        
        return results
    
    def run_comprehensive_benchmark(self, matrix_sizes: List[Tuple[int, int]] = None,
                                   sparsity_levels: List[float] = None) -> Dict:
        """
        Run the full benchmark suite - this is the main function I use.
        """
        if matrix_sizes is None:
            matrix_sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 1024)]
        
        if sparsity_levels is None:
            sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]
        
        print("Creating test matrices...")
        matrices = create_test_matrices(matrix_sizes, sparsity_levels)
        
        print("Running compression benchmarks...")
        compression_results = self.benchmark_compression(matrices)
        
        print("Running speed benchmarks...")
        speed_results = self.benchmark_speed(matrices)
        
        print("Running accuracy benchmarks...")
        accuracy_results = self.benchmark_accuracy(matrices)
        
        print("Running memory benchmarks...")
        memory_results = self.benchmark_memory_usage(matrices)
        
        # Combine all results
        full_results = {
            'compression': compression_results,
            'speed': speed_results,
            'accuracy': accuracy_results,
            'memory': memory_results,
            'metadata': {
                'matrix_sizes': matrix_sizes,
                'sparsity_levels': sparsity_levels,
                'timestamp': time.time()
            }
        }
        
        # Save results
        output_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")
        
        # Create summary
        summary_df = self.create_summary_report(full_results)
        summary_file = self.output_dir / "matrix_benchmark_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Summary saved to {summary_file}")
        
        return full_results
    
    def create_summary_report(self, results: Dict) -> pd.DataFrame:
        """
        Create a nice summary table of all the results.
        """
        summary_data = []
        
        # Process BIE results
        for method_name in results['compression']['bie_methods']:
            for matrix_name in results['compression']['bie_methods'][method_name]:
                comp_data = results['compression']['bie_methods'][method_name][matrix_name]
                acc_data = results['accuracy']['bie_methods'][method_name].get(matrix_name, {})
                
                # Get speed data (average across input sizes)
                speed_data = results['speed']['bie_methods'][method_name].get(matrix_name, {})
                avg_times = []
                for key, val in speed_data.items():
                    if key.startswith('input_') and isinstance(val, dict) and 'avg_time' in val:
                        avg_times.append(val['avg_time'])
                avg_time = np.mean(avg_times) if avg_times else float('inf')
                
                summary_data.append({
                    'matrix_type': matrix_name,
                    'method': f"bie_{method_name}",
                    'compression_ratio': comp_data.get('compression_ratio', 0),
                    'space_savings_percent': comp_data.get('space_savings_percent', 0),
                    'encode_time': comp_data.get('encode_time', float('inf')),
                    'avg_matmul_time': avg_time,
                    'mse': acc_data.get('mse', float('inf')),
                    'mae': acc_data.get('mae', float('inf')),
                    'rmse': acc_data.get('rmse', float('inf')),
                    'relative_error': acc_data.get('relative_error', float('inf'))
                })
        
        # Process baseline results
        for method_name in results['compression']['baseline_methods']:
            for matrix_name in results['compression']['baseline_methods'][method_name]:
                comp_data = results['compression']['baseline_methods'][method_name][matrix_name]
                acc_data = results['accuracy']['baseline_methods'][method_name].get(matrix_name, {})
                
                speed_data = results['speed']['baseline_methods'][method_name].get(matrix_name, {})
                avg_times = []
                for key, val in speed_data.items():
                    if key.startswith('input_') and isinstance(val, dict) and 'avg_time' in val:
                        avg_times.append(val['avg_time'])
                avg_time = np.mean(avg_times) if avg_times else float('inf')
                
                summary_data.append({
                    'matrix_type': matrix_name,
                    'method': f"baseline_{method_name}",
                    'compression_ratio': comp_data.get('compression_ratio', 0),
                    'space_savings_percent': comp_data.get('space_savings_percent', 0),
                    'encode_time': comp_data.get('encode_time', float('inf')),
                    'avg_matmul_time': avg_time,
                    'mse': acc_data.get('mse', float('inf')),
                    'mae': acc_data.get('mae', float('inf')),
                    'rmse': acc_data.get('rmse', float('inf')),
                    'relative_error': acc_data.get('relative_error', float('inf'))
                })
        
        return pd.DataFrame(summary_data)


def main():
    """
    Run the benchmarks - this is what I call to test everything.
    """
    runner = BenchmarkRunner()
    results = runner.run_comprehensive_benchmark()
    
    print("\nBenchmark completed!")
    print(f"Results saved in: {runner.output_dir}")
    
    # Print some key findings
    summary_df = runner.create_summary_report(results)
    print("\nTop compression ratios:")
    top_compression = summary_df.nlargest(5, 'compression_ratio')[['method', 'matrix_type', 'compression_ratio']]
    print(top_compression.to_string(index=False))


if __name__ == "__main__":
    main()