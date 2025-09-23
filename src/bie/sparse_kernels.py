"""
My sparse matrix kernels for BIE - this was the hardest part to get right!
Had to optimize these operations to work directly on the compressed data.
"""

import numpy as np
import torch
from typing import Dict, Union, Tuple
import time
from numba import jit, prange
import scipy.sparse as sp


class BIESparseKernels:
    """
    My custom sparse kernels - much faster than converting back to dense first.
    """
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def sparse_dense_matmul_indices(indices: np.ndarray, values: np.ndarray, 
                                   dense_matrix: np.ndarray, output: np.ndarray,
                                   rows: int, cols: int) -> None:
        """
        The core sparse-dense multiplication - took forever to get the indexing right.
        Works directly with the BIE indices instead of reconstructing the full matrix.
        """
        for idx in prange(len(indices)):
            flat_idx = indices[idx]
            row = flat_idx // cols
            col = flat_idx % cols
            value = values[idx]
            
            # This is where the magic happens - accumulate without full reconstruction
            for j in range(dense_matrix.shape[1]):
                output[row, j] += value * dense_matrix[col, j]
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def binary_matmul_indices(indices: np.ndarray, sign_indices: np.ndarray,
                             dense_matrix: np.ndarray, output: np.ndarray,
                             rows: int, cols: int) -> None:
        """
        Binary version - handles {-1, 0, 1} weights efficiently.
        The sign handling was tricky but gives good speedup.
        """
        # First pass: add all positive contributions
        for idx in prange(len(indices)):
            flat_idx = indices[idx]
            row = flat_idx // cols
            col = flat_idx % cols
            
            for j in range(dense_matrix.shape[1]):
                output[row, j] += dense_matrix[col, j]
        
        # Second pass: subtract negative contributions
        if len(sign_indices) > 0:
            for idx in prange(len(sign_indices)):
                flat_idx = sign_indices[idx]
                row = flat_idx // cols
                col = flat_idx % cols
                
                # Subtract twice (since we added once above)
                for j in range(dense_matrix.shape[1]):
                    output[row, j] -= 2.0 * dense_matrix[col, j]
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def bitplane_matmul_progressive(bitplane_indices: Dict, bitplane_weights: np.ndarray,
                                   dense_matrix: np.ndarray, output: np.ndarray,
                                   rows: int, cols: int, max_bitplanes: int = -1) -> None:
        """
        Progressive bitplane multiplication - can stop early for approximate results.
        Each bitplane contributes with weight 2^bit_position.
        """
        num_bitplanes = len(bitplane_indices) if max_bitplanes == -1 else min(max_bitplanes, len(bitplane_indices))
        
        for bit_pos in range(num_bitplanes):
            if bit_pos in bitplane_indices:
                indices = bitplane_indices[bit_pos]
                weight = bitplane_weights[bit_pos]  # 2^bit_pos
                
                for idx in prange(len(indices)):
                    flat_idx = indices[idx]
                    row = flat_idx // cols
                    col = flat_idx % cols
                    
                    for j in range(dense_matrix.shape[1]):
                        output[row, j] += weight * dense_matrix[col, j]


class BIEMatMul:
    """
    Main interface for BIE matrix multiplication - handles all the encoding types.
    """
    
    def __init__(self, use_numba: bool = True):
        """Initialize with optional numba acceleration - much faster when available"""
        self.use_numba = use_numba
        
        # Fallback to Python if numba not available
        if not use_numba:
            self.kernels = None
        else:
            self.kernels = BIESparseKernels()
    
    def matmul_binary(self, encoded_weight: Dict, input_tensor: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication with binary BIE weights.
        This is usually the fastest since it's just additions/subtractions.
        """
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().cpu().numpy()
        
        weight_shape = encoded_weight['shape']
        output_shape = (weight_shape[0], input_tensor.shape[1])
        output = np.zeros(output_shape, dtype=np.float32)
        
        indices = encoded_weight['indices']
        sign_indices = encoded_weight.get('sign_indices', np.array([], dtype=np.uint32))
        
        if len(indices) == 0:
            return output  # All zeros
        
        if self.use_numba and self.kernels:
            # Use optimized numba kernel
            BIESparseKernels.binary_matmul_indices(
                indices, sign_indices, input_tensor, output,
                weight_shape[0], weight_shape[1]
            )
        else:
            # Fallback Python implementation
            self._binary_matmul_python(indices, sign_indices, input_tensor, output,
                                     weight_shape[0], weight_shape[1])
        
        return output
    
    def matmul_bitplane(self, encoded_weight: Dict, input_tensor: np.ndarray,
                       max_bitplanes: int = -1) -> np.ndarray:
        """
        Bitplane matrix multiplication - can trade accuracy for speed by using fewer bitplanes.
        Really useful for approximate inference!
        """
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().cpu().numpy()
        
        weight_shape = encoded_weight['shape']
        output_shape = (weight_shape[0], input_tensor.shape[1])
        output = np.zeros(output_shape, dtype=np.float32)
        
        bitplanes = encoded_weight['bitplanes']
        num_bits = encoded_weight['num_bits']
        min_val = encoded_weight['min_val']
        max_val = encoded_weight['max_val']
        
        # Handle edge case
        if max_val == min_val:
            return np.full(output_shape, min_val)
        
        # Calculate bitplane weights
        bitplane_weights = np.array([2**i for i in range(num_bits)], dtype=np.float32)
        
        # Convert bitplanes list to dict for numba compatibility
        bitplane_dict = {i: bitplanes[i] for i in range(len(bitplanes)) if len(bitplanes[i]) > 0}
        
        if self.use_numba and self.kernels:
            # This doesn't work with numba due to dict limitations, use Python fallback
            self._bitplane_matmul_python(bitplane_dict, bitplane_weights, input_tensor, 
                                       output, weight_shape[0], weight_shape[1], max_bitplanes)
        else:
            self._bitplane_matmul_python(bitplane_dict, bitplane_weights, input_tensor,
                                       output, weight_shape[0], weight_shape[1], max_bitplanes)
        
        # Dequantize the result
        scale = (max_val - min_val) / (2**num_bits - 1)
        output = output * scale + min_val
        
        return output
    
    def matmul_blocked(self, encoded_weight: Dict, input_tensor: np.ndarray) -> np.ndarray:
        """
        Blocked matrix multiplication - processes each block separately.
        Good for memory locality and can be parallelized easily.
        """
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().cpu().numpy()
        
        weight_shape = encoded_weight['shape']
        output_shape = (weight_shape[0], input_tensor.shape[1])
        output = np.zeros(output_shape, dtype=np.float32)
        
        # Process each block
        for block_info in encoded_weight['blocks']:
            encoded_block = block_info['encoded']
            row_offset = block_info['row_offset']
            col_offset = block_info['col_offset']
            block_shape = block_info['block_shape']
            
            # Extract relevant input slice
            input_slice = input_tensor[col_offset:col_offset+block_shape[1], :]
            
            # Compute block result
            if encoded_block['type'] == 'binary':
                block_output = self.matmul_binary(encoded_block, input_slice)
            elif encoded_block['type'] == 'bitplane':
                block_output = self.matmul_bitplane(encoded_block, input_slice)
            else:
                raise ValueError(f"Unknown block encoding: {encoded_block['type']}")
            
            # Add to output
            output[row_offset:row_offset+block_shape[0], :] += block_output
        
        return output
    
    def matmul(self, encoded_weight: Dict, input_tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Main matmul function - routes to the right implementation.
        """
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().cpu().numpy()
        
        if encoded_weight['type'] == 'binary':
            return self.matmul_binary(encoded_weight, input_tensor)
        elif encoded_weight['type'] == 'bitplane':
            return self.matmul_bitplane(encoded_weight, input_tensor)
        elif encoded_weight['type'] == 'blocked':
            return self.matmul_blocked(encoded_weight, input_tensor)
        else:
            raise ValueError(f"Unknown encoding type: {encoded_weight['type']}")
    
    def _sparse_matmul_python(self, indices: np.ndarray, values: np.ndarray,
                             dense_matrix: np.ndarray, output: np.ndarray,
                             rows: int, cols: int) -> None:
        """Python fallback when numba isn't available"""
        for idx in range(len(indices)):
            flat_idx = indices[idx]
            row = flat_idx // cols
            col = flat_idx % cols
            value = values[idx]
            output[row, :] += value * dense_matrix[col, :]
    
    def _binary_matmul_python(self, indices: np.ndarray, sign_indices: np.ndarray,
                             dense_matrix: np.ndarray, output: np.ndarray,
                             rows: int, cols: int) -> None:
        """Python fallback for binary matmul"""
        # Add positive contributions
        for idx in range(len(indices)):
            flat_idx = indices[idx]
            row = flat_idx // cols
            col = flat_idx % cols
            output[row, :] += dense_matrix[col, :]
        
        # Subtract negative contributions
        for idx in range(len(sign_indices)):
            flat_idx = sign_indices[idx]
            row = flat_idx // cols
            col = flat_idx % cols
            output[row, :] -= 2.0 * dense_matrix[col, :]
    
    def _bitplane_matmul_python(self, bitplane_dict: Dict, bitplane_weights: np.ndarray,
                               dense_matrix: np.ndarray, output: np.ndarray,
                               rows: int, cols: int, max_bitplanes: int = -1) -> None:
        """Python fallback for bitplane matmul"""
        num_bitplanes = len(bitplane_dict) if max_bitplanes == -1 else min(max_bitplanes, len(bitplane_dict))
        
        for bit_pos in range(num_bitplanes):
            if bit_pos in bitplane_dict:
                indices = bitplane_dict[bit_pos]
                weight = bitplane_weights[bit_pos]
                
                for idx in range(len(indices)):
                    flat_idx = indices[idx]
                    row = flat_idx // cols
                    col = flat_idx % cols
                    output[row, :] += weight * dense_matrix[col, :]


def benchmark_sparse_kernels(weight_matrix: np.ndarray, input_tensor: np.ndarray,
                            sparsity_levels: list = [0.5, 0.7, 0.9, 0.95],
                            encoding_types: list = ['binary', 'bitplane']) -> Dict:
    """
    Benchmark my sparse kernels against different baselines.
    This helped me validate that the optimizations actually work!
    """
    from .encoder import BIEEncoder
    
    results = {
        'sparsity_levels': sparsity_levels,
        'encoding_types': encoding_types,
        'timings': {},
        'accuracy': {},
        'compression': {}
    }
    
    matmul_engine = BIEMatMul(use_numba=True)
    
    # Baseline: dense multiplication
    start_time = time.time()
    dense_result = np.dot(weight_matrix, input_tensor)
    dense_time = time.time() - start_time
    results['timings']['dense'] = dense_time
    
    for sparsity in sparsity_levels:
        # Create sparse version
        sparse_weight = weight_matrix.copy()
        mask = np.random.random(weight_matrix.shape) < sparsity
        sparse_weight[mask] = 0
        
        for encoding_type in encoding_types:
            key = f"{encoding_type}_sparsity_{sparsity}"
            
            # Encode
            encoder = BIEEncoder(encoding_type=encoding_type)
            encoded = encoder.encode(sparse_weight)
            
            # Time the sparse multiplication
            start_time = time.time()
            sparse_result = matmul_engine.matmul(encoded, input_tensor)
            sparse_time = time.time() - start_time
            
            # Calculate metrics
            mse = np.mean((dense_result - sparse_result) ** 2)
            compression_ratio = encoded.get('compression_ratio', 1.0)
            
            results['timings'][key] = sparse_time
            results['accuracy'][key] = {
                'mse': mse,
                'relative_error': np.sqrt(mse) / (np.linalg.norm(dense_result) + 1e-8)
            }
            results['compression'][key] = {
                'ratio': compression_ratio,
                'speedup': dense_time / sparse_time if sparse_time > 0 else float('inf')
            }

    return results