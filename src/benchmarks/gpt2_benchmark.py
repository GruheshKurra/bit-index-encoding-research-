"""
Testing BIE on real GPT-2 weights to see how it performs on actual transformer models.
This is where I validate that my approach works on production-scale neural networks.
"""

import numpy as np
import torch
import time
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import our implementations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bie.encoder import BIEEncoder, BIEDecoder, get_compression_stats
from bie.sparse_kernels import BIEMatMul
from baseline.dense_quantized import (DenseBaseline, QuantizedBaseline, 
                                     SparseBaseline, HybridBaseline)
from utils.model_utils import ModelPruner, ModelQuantizer, GPT2Utils


class GPT2BenchmarkRunner:
    """
    My specialized tester for GPT-2 models - this is where I prove BIE works on real transformers.
    """
    
    def __init__(self, model_name: str = "gpt2", output_dir: str = "results"):
        """
        Set up for testing on GPT-2 - loads the model and gets everything ready.
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # My GPT-2 helper utilities
        self.gpt2_utils = GPT2Utils()
        
        # Load the actual model
        print(f"Loading {model_name} model...")
        self.model, self.tokenizer = self.gpt2_utils.load_model(model_name)
        
        # Set up all my encoding methods
        self.bie_encoders = {
            'binary': BIEEncoder('binary'),
            'bitplane_8bit': BIEEncoder('bitplane'),
            'bitplane_4bit': BIEEncoder('bitplane'),
            'blocked_binary': BIEEncoder('binary', block_size=64),
            'blocked_bitplane': BIEEncoder('bitplane', block_size=64)
        }
        
        # Standard baseline methods
        self.baselines = {
            'dense_fp32': DenseBaseline('float32'),
            'dense_fp16': DenseBaseline('float16'),
            'quantized_8bit': QuantizedBaseline(8),
            'quantized_4bit': QuantizedBaseline(4),
            'sparse_csr': SparseBaseline('csr'),
            'hybrid_8bit': HybridBaseline(8)
        }
        
        # For matrix multiplication testing
        self.bie_matmul = BIEMatMul()
    
    def extract_layer_weights(self, layer_indices: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Pull out the actual weight matrices from GPT-2 layers.
        These are the real weights I want to compress.
        """
        if layer_indices is None:
            layer_indices = [0, 1, 2]  # First few layers by default
        
        weights = {}
        
        for layer_idx in layer_indices:
            if layer_idx >= len(self.model.transformer.h):
                continue
                
            layer = self.model.transformer.h[layer_idx]
            
            # Extract different weight matrices from this layer
            weights[f'layer_{layer_idx}_attn_c_attn'] = layer.attn.c_attn.weight.detach().cpu().numpy()
            weights[f'layer_{layer_idx}_attn_c_proj'] = layer.attn.c_proj.weight.detach().cpu().numpy()
            weights[f'layer_{layer_idx}_mlp_c_fc'] = layer.mlp.c_fc.weight.detach().cpu().numpy()
            weights[f'layer_{layer_idx}_mlp_c_proj'] = layer.mlp.c_proj.weight.detach().cpu().numpy()
        
        return weights
    
    def benchmark_model_compression(self, sparsity_levels: List[float] = [0.0, 0.5, 0.9]) -> Dict:
        """
        Test compression on actual GPT-2 weights, including pruned versions.
        This shows how BIE performs on real neural network weights.
        """
        results = {
            'bie_methods': {},
            'baseline_methods': {}
        }
        
        # Get the actual weights from GPT-2
        original_weights = self.extract_layer_weights()
        
        # Test on original weights and pruned versions
        test_weights = {'original': original_weights}
        
        # Create pruned versions
        for sparsity in sparsity_levels:
            if sparsity > 0:
                pruned_weights = {}
                for name, weight in original_weights.items():
                    # Apply magnitude pruning
                    pruned = ModelPruner.magnitude_pruning(torch.from_numpy(weight), sparsity)
                    pruned_weights[name] = pruned.numpy()
                test_weights[f'pruned_{sparsity}'] = pruned_weights
        
        # Test BIE methods
        for method_name, encoder in self.bie_encoders.items():
            results['bie_methods'][method_name] = {}
            
            for weight_set_name, weight_set in test_weights.items():
                results['bie_methods'][method_name][weight_set_name] = {}
                
                for weight_name, weight in weight_set.items():
                    try:
                        start_time = time.time()
                        
                        if 'bitplane_4bit' in method_name:
                            encoded = encoder.encode(weight, num_bits=4)
                        elif 'bitplane' in method_name:
                            encoded = encoder.encode(weight, num_bits=8)
                        else:
                            # For binary, threshold the weights
                            binary_weight = (weight > np.median(weight)).astype(np.float32)
                            encoded = encoder.encode(binary_weight)
                        
                        encode_time = time.time() - start_time
                        stats = get_compression_stats(encoded, weight.shape)
                        
                        results['bie_methods'][method_name][weight_set_name][weight_name] = {
                            'compression_ratio': stats['compression_ratio'],
                            'space_savings_percent': stats['space_savings_percent'],
                            'encode_time': encode_time,
                            'original_size_mb': weight.nbytes / (1024 * 1024),
                            'compressed_size_mb': stats['compressed_size'] / (1024 * 1024)
                        }
                        
                    except Exception as e:
                        results['bie_methods'][method_name][weight_set_name][weight_name] = {
                            'error': str(e),
                            'compression_ratio': 0.0,
                            'space_savings_percent': 0.0,
                            'encode_time': float('inf')
                        }
        
        # Test baseline methods
        for method_name, baseline in self.baselines.items():
            results['baseline_methods'][method_name] = {}
            
            for weight_set_name, weight_set in test_weights.items():
                results['baseline_methods'][method_name][weight_set_name] = {}
                
                for weight_name, weight in weight_set.items():
                    try:
                        start_time = time.time()
                        stored = baseline.store_weights(weight)
                        store_time = time.time() - start_time
                        
                        stats = baseline.get_compression_stats(stored)
                        
                        results['baseline_methods'][method_name][weight_set_name][weight_name] = {
                            'compression_ratio': stats['compression_ratio'],
                            'space_savings_percent': stats['space_savings_percent'],
                            'encode_time': store_time,
                            'original_size_mb': weight.nbytes / (1024 * 1024),
                            'compressed_size_mb': stats['size_bytes'] / (1024 * 1024)
                        }
                        
                    except Exception as e:
                        results['baseline_methods'][method_name][weight_set_name][weight_name] = {
                            'error': str(e),
                            'compression_ratio': 0.0,
                            'space_savings_percent': 0.0,
                            'encode_time': float('inf')
                        }
        
        return results
    
    def benchmark_model_quality(self, test_texts: List[str] = None,
                               sparsity_levels: List[float] = [0.0, 0.5, 0.9]) -> Dict:
        """
        Test how compression affects the actual model performance.
        This is the most important test - does BIE preserve model quality?
        """
        if test_texts is None:
            test_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "In a hole in the ground there lived a hobbit.",
                "It was the best of times, it was the worst of times.",
                "To be or not to be, that is the question.",
                "All happy families are alike; each unhappy family is unhappy in its own way."
            ]
        
        results = {
            'original_perplexity': {},
            'bie_methods': {},
            'baseline_methods': {}
        }
        
        # Get baseline perplexity with original model
        for text in test_texts:
            try:
                perplexity = self.gpt2_utils.calculate_perplexity(self.model, text)
                results['original_perplexity'][text] = perplexity
            except Exception as e:
                results['original_perplexity'][text] = {'error': str(e)}
        
        # Test with different compression methods
        # Note: This is a simplified version - in practice, you'd need to 
        # actually replace the model weights and test inference
        
        return results
    
    def benchmark_inference_speed(self, input_lengths: List[int] = [50, 100, 200],
                                 num_runs: int = 10) -> Dict:
        """
        Test how fast inference is with compressed weights.
        BIE should be competitive with standard approaches.
        """
        results = {
            'original_model': {},
            'bie_methods': {},
            'baseline_methods': {}
        }
        
        # Test original model speed
        for length in input_lengths:
            # Create random input of specified length
            input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, length))
            
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(input_ids)
                end_time = time.time()
                times.append(end_time - start_time)
            
            results['original_model'][f'length_{length}'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'tokens_per_second': length / np.mean(times)
            }
        
        # Note: Testing compressed models would require actually modifying
        # the model weights and implementing custom forward passes
        # This is a placeholder for that functionality
        
        return results
    
    def run_comprehensive_gpt2_benchmark(self) -> Dict:
        """
        Run the complete GPT-2 benchmark suite.
        This is my main function for testing BIE on real transformers.
        """
        print("Running GPT-2 compression benchmarks...")
        compression_results = self.benchmark_model_compression()
        
        print("Running GPT-2 quality benchmarks...")
        quality_results = self.benchmark_model_quality()
        
        print("Running GPT-2 speed benchmarks...")
        speed_results = self.benchmark_inference_speed()
        
        # Combine all results
        full_results = {
            'compression': compression_results,
            'quality': quality_results,
            'speed': speed_results,
            'metadata': {
                'model_name': self.model_name,
                'timestamp': time.time(),
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            }
        }
        
        # Save results
        output_file = self.output_dir / f"gpt2_benchmark_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"GPT-2 results saved to {output_file}")
        
        # Create summary
        summary_df = self.create_gpt2_summary_report(full_results)
        summary_file = self.output_dir / "gpt2_benchmark_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"GPT-2 summary saved to {summary_file}")
        
        return full_results
    
    def create_gpt2_summary_report(self, results: Dict) -> pd.DataFrame:
        """
        Create a summary of GPT-2 benchmark results.
        """
        summary_data = []
        
        # Process compression results
        for method_type in ['bie_methods', 'baseline_methods']:
            for method_name in results['compression'][method_type]:
                for weight_set_name in results['compression'][method_type][method_name]:
                    for weight_name in results['compression'][method_type][method_name][weight_set_name]:
                        data = results['compression'][method_type][method_name][weight_set_name][weight_name]
                        
                        if 'error' not in data:
                            summary_data.append({
                                'method_type': method_type.replace('_methods', ''),
                                'method_name': method_name,
                                'weight_set': weight_set_name,
                                'weight_layer': weight_name,
                                'compression_ratio': data['compression_ratio'],
                                'space_savings_percent': data['space_savings_percent'],
                                'encode_time': data['encode_time'],
                                'original_size_mb': data['original_size_mb'],
                                'compressed_size_mb': data['compressed_size_mb']
                            })
        
        return pd.DataFrame(summary_data)


def main():
    """
    Run the GPT-2 benchmarks - this tests BIE on real transformer weights.
    """
    runner = GPT2BenchmarkRunner()
    results = runner.run_comprehensive_gpt2_benchmark()
    
    print("\nGPT-2 benchmark completed!")
    print(f"Results saved in: {runner.output_dir}")
    
    # Show some key findings
    summary_df = runner.create_gpt2_summary_report(results)
    if not summary_df.empty:
        print("\nTop compression ratios on GPT-2 weights:")
        top_compression = summary_df.nlargest(5, 'compression_ratio')[
            ['method_name', 'weight_layer', 'compression_ratio', 'space_savings_percent']
        ]
        print(top_compression.to_string(index=False))


if __name__ == "__main__":
    main()