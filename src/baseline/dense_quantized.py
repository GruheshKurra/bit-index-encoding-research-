"""
My baseline implementations to compare against BIE.
Need these to show that BIE actually works better than standard approaches.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Union, Tuple, Optional
import time
import pickle
import gzip


class DenseBaseline:
    """
    Standard dense storage - the most basic approach.
    """
    
    def __init__(self, dtype: str = 'float32'):
        """Just store everything as-is in the specified precision"""
        self.dtype = dtype
        self.np_dtype = getattr(np, dtype)
    
    def store_weights(self, weights: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        Store weights without any compression - baseline for comparison.
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        
        weights = weights.astype(self.np_dtype)
        
        return {
            'type': 'dense',
            'weights': weights,
            'shape': weights.shape,
            'dtype': self.dtype,
            'size_bytes': weights.nbytes
        }
    
    def matmul(self, stored_weights: Dict, input_tensor: np.ndarray) -> np.ndarray:
        """Standard matrix multiplication - nothing fancy here"""
        weights = stored_weights['weights']
        return np.dot(weights, input_tensor)
    
    def get_compression_stats(self, stored_weights: Dict) -> Dict:
        """No compression, so ratio is always 1.0"""
        return {
            'compression_ratio': 1.0,
            'space_savings_percent': 0.0,
            'size_bytes': stored_weights['size_bytes']
        }


class QuantizedBaseline:
    """
    Standard quantization - reduce precision to save space.
    This is what most people use for compression.
    """
    
    def __init__(self, num_bits: int = 8, symmetric: bool = True):
        """
        Initialize quantization parameters.
        8-bit is pretty standard, symmetric is easier to implement.
        """
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.max_val = 2**(num_bits - 1) - 1
        self.min_val = -2**(num_bits - 1) if symmetric else 0
    
    def quantize(self, tensor: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, float, float]:
        """
        Quantize tensor to specified bit width.
        Had to be careful with the scaling to avoid overflow.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Calculate scale and zero point
        tensor_min, tensor_max = tensor.min(), tensor.max()
        
        if self.symmetric:
            # Symmetric quantization around zero
            abs_max = max(abs(tensor_min), abs(tensor_max))
            scale = abs_max / self.max_val if abs_max > 0 else 1.0
            zero_point = 0
        else:
            # Asymmetric quantization
            scale = (tensor_max - tensor_min) / (2**self.num_bits - 1) if tensor_max != tensor_min else 1.0
            zero_point = self.min_val - tensor_min / scale
        
        # Quantize
        quantized = np.round(tensor / scale + zero_point)
        quantized = np.clip(quantized, self.min_val, self.max_val)
        
        return quantized.astype(np.int8), scale, zero_point
    
    def dequantize(self, quantized: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
        """Convert back to float - loses some precision but that's the point"""
        return scale * (quantized.astype(np.float32) - zero_point)
    
    def store_weights(self, weights: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        Store quantized weights with scale/zero_point for reconstruction.
        """
        quantized, scale, zero_point = self.quantize(weights)
        
        return {
            'type': 'quantized',
            'quantized_weights': quantized,
            'scale': scale,
            'zero_point': zero_point,
            'shape': weights.shape if isinstance(weights, np.ndarray) else weights.shape,
            'num_bits': self.num_bits,
            'size_bytes': quantized.nbytes + 8  # +8 for scale and zero_point
        }
    
    def matmul(self, stored_weights: Dict, input_tensor: np.ndarray) -> np.ndarray:
        """
        Dequantize and multiply - could optimize this but keeping it simple.
        """
        quantized = stored_weights['quantized_weights']
        scale = stored_weights['scale']
        zero_point = stored_weights['zero_point']
        
        # Dequantize weights
        weights = self.dequantize(quantized, scale, zero_point)
        
        return np.dot(weights, input_tensor)
    
    def get_compression_stats(self, stored_weights: Dict) -> Dict:
        """Calculate compression compared to float32"""
        original_size = np.prod(stored_weights['shape']) * 4  # float32
        compressed_size = stored_weights['size_bytes']
        
        return {
            'compression_ratio': original_size / compressed_size,
            'space_savings_percent': (1 - compressed_size / original_size) * 100,
            'size_bytes': compressed_size
        }


class SparseBaseline:
    """
    Standard sparse matrix storage - only store non-zero elements.
    """
    
    def __init__(self, format: str = 'csr'):
        """CSR is usually the most efficient for matrix multiplication"""
        self.format = format
    
    def store_weights(self, weights: Union[np.ndarray, torch.Tensor], 
                     threshold: float = 1e-6) -> Dict:
        """
        Convert to sparse format - threshold determines what counts as 'zero'.
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        
        # Zero out small values
        weights_sparse = weights.copy()
        weights_sparse[np.abs(weights_sparse) < threshold] = 0
        
        # Find non-zero elements
        nonzero_mask = weights_sparse != 0
        nonzero_indices = np.where(nonzero_mask)
        nonzero_values = weights_sparse[nonzero_mask]
        
        # Calculate storage size
        num_nonzeros = len(nonzero_values)
        if self.format == 'coo':
            # COO: store row, col, value
            size_bytes = num_nonzeros * (4 + 4 + 4)  # int32 + int32 + float32
        elif self.format == 'csr':
            # CSR: more complex but more efficient
            size_bytes = num_nonzeros * 8 + weights.shape[0] * 4  # values + col_indices + row_ptr
        else:
            size_bytes = num_nonzeros * 12  # conservative estimate
        
        return {
            'type': 'sparse',
            'format': self.format,
            'nonzero_indices': nonzero_indices,
            'nonzero_values': nonzero_values,
            'shape': weights.shape,
            'nnz': num_nonzeros,
            'threshold': threshold,
            'size_bytes': size_bytes
        }
    
    def matmul(self, stored_weights: Dict, input_tensor: np.ndarray) -> np.ndarray:
        """
        Reconstruct sparse matrix and multiply.
        Could optimize this but it's just a baseline.
        """
        shape = stored_weights['shape']
        indices = stored_weights['nonzero_indices']
        values = stored_weights['nonzero_values']
        
        # Reconstruct sparse matrix
        weights = np.zeros(shape)
        weights[indices] = values
        
        return np.dot(weights, input_tensor)
    
    def get_compression_stats(self, stored_weights: Dict) -> Dict:
        """Calculate compression ratio"""
        original_size = np.prod(stored_weights['shape']) * 4
        compressed_size = stored_weights['size_bytes']
        
        return {
            'compression_ratio': original_size / compressed_size,
            'space_savings_percent': (1 - compressed_size / original_size) * 100,
            'size_bytes': compressed_size,
            'sparsity': 1 - stored_weights['nnz'] / np.prod(stored_weights['shape'])
        }


class HybridBaseline:
    """
    Combination of quantization and sparsity - best of both worlds?
    """
    
    def __init__(self, num_bits: int = 8, sparsity_threshold: float = 1e-6):
        """Combine quantization with sparsity"""
        self.quantizer = QuantizedBaseline(num_bits=num_bits)
        self.sparse = SparseBaseline()
        self.sparsity_threshold = sparsity_threshold
    
    def store_weights(self, weights: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        First make sparse, then quantize the remaining values.
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        
        # Apply sparsity threshold
        sparse_weights = weights.copy()
        sparse_weights[np.abs(sparse_weights) < self.sparsity_threshold] = 0
        
        # Get non-zero elements
        nonzero_mask = sparse_weights != 0
        nonzero_values = sparse_weights[nonzero_mask]
        nonzero_indices = np.where(nonzero_mask)
        
        # Quantize the non-zero values
        if len(nonzero_values) > 0:
            quantized_values, scale, zero_point = self.quantizer.quantize(nonzero_values)
        else:
            quantized_values, scale, zero_point = np.array([]), 1.0, 0.0
        
        size_bytes = len(quantized_values) * 1 + len(nonzero_indices[0]) * 8 + 8  # quantized + indices + metadata
        
        return {
            'type': 'hybrid',
            'quantized_values': quantized_values,
            'indices': nonzero_indices,
            'scale': scale,
            'zero_point': zero_point,
            'shape': weights.shape,
            'nnz': len(nonzero_values),
            'size_bytes': size_bytes
        }
    
    def matmul(self, stored_weights: Dict, input_tensor: np.ndarray) -> np.ndarray:
        """Reconstruct and multiply"""
        shape = stored_weights['shape']
        quantized_values = stored_weights['quantized_values']
        indices = stored_weights['indices']
        scale = stored_weights['scale']
        zero_point = stored_weights['zero_point']
        
        # Dequantize values
        if len(quantized_values) > 0:
            values = self.quantizer.dequantize(quantized_values, scale, zero_point)
        else:
            values = np.array([])
        
        # Reconstruct matrix
        weights = np.zeros(shape)
        if len(values) > 0:
            weights[indices] = values
        
        return np.dot(weights, input_tensor)
    
    def get_compression_stats(self, stored_weights: Dict) -> Dict:
        """Calculate compression stats"""
        original_size = np.prod(stored_weights['shape']) * 4
        compressed_size = stored_weights['size_bytes']
        
        return {
            'compression_ratio': original_size / compressed_size,
            'space_savings_percent': (1 - compressed_size / original_size) * 100,
            'size_bytes': compressed_size,
            'sparsity': 1 - stored_weights['nnz'] / np.prod(stored_weights['shape'])
        }


def benchmark_baselines(weight_matrix: np.ndarray, input_tensor: np.ndarray,
                       sparsity_levels: list = [0.0, 0.5, 0.7, 0.9, 0.95]) -> Dict:
    """
    Benchmark all baseline methods - this is how I validate BIE performance.
    """
    baselines = {
        'dense_fp32': DenseBaseline('float32'),
        'dense_fp16': DenseBaseline('float16'),
        'quantized_8bit': QuantizedBaseline(8),
        'quantized_4bit': QuantizedBaseline(4),
        'sparse_csr': SparseBaseline('csr'),
        'hybrid_8bit': HybridBaseline(8)
    }
    
    results = {
        'baselines': list(baselines.keys()),
        'sparsity_levels': sparsity_levels,
        'timings': {},
        'compression': {},
        'accuracy': {}
    }
    
    # Dense baseline result for accuracy comparison
    dense_result = np.dot(weight_matrix, input_tensor)
    
    for sparsity in sparsity_levels:
        # Create sparse version
        sparse_weight = weight_matrix.copy()
        if sparsity > 0:
            mask = np.random.random(weight_matrix.shape) < sparsity
            sparse_weight[mask] = 0
        
        for name, baseline in baselines.items():
            key = f"{name}_sparsity_{sparsity}"
            
            # Store weights
            start_time = time.time()
            stored = baseline.store_weights(sparse_weight)
            store_time = time.time() - start_time
            
            # Matrix multiplication
            start_time = time.time()
            result = baseline.matmul(stored, input_tensor)
            matmul_time = time.time() - start_time
            
            # Calculate metrics
            mse = np.mean((dense_result - result) ** 2)
            compression_stats = baseline.get_compression_stats(stored)
            
            results['timings'][key] = {
                'store_time': store_time,
                'matmul_time': matmul_time,
                'total_time': store_time + matmul_time
            }
            
            results['compression'][key] = compression_stats
            
            results['accuracy'][key] = {
                'mse': mse,
                'relative_error': np.sqrt(mse) / (np.linalg.norm(dense_result) + 1e-8)
            }
    
    return results


def save_baseline_results(results: Dict, filepath: str) -> None:
    """Save results for later analysis"""
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(results, f)


def load_baseline_results(filepath: str) -> Dict:
    """Load saved results"""
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)