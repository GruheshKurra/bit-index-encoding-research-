"""
My implementation of Bit-Index Encoding (BIE) for neural network weight compression.
After lots of experimentation, I settled on these binary and bitplane variants.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
from bitarray import bitarray
import struct


class BIEEncoder:
    """
    My BIE encoder - took me a while to get the indexing right!
    Supports binary encoding (for really sparse stuff) and bitplane encoding (better quality).
    """
    
    def __init__(self, encoding_type: str = "bitplane", block_size: Optional[int] = None):
        self.encoding_type = encoding_type
        self.block_size = block_size
        
    def encode_binary(self, tensor: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        Binary encoding - just store where the 1s are. 
        Had to handle negative values separately which was tricky.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
            
        # Deal with negative values - store signs separately
        has_negative = np.any(tensor < 0)
        if has_negative:
            sign_mask = tensor < 0
            abs_tensor = np.abs(tensor)
            sign_indices = np.where(sign_mask.flatten())[0].astype(np.uint32)
        else:
            abs_tensor = tensor
            sign_indices = None
            
        # Find non-zero positions - this is where the magic happens
        nonzero_indices = np.where(abs_tensor.flatten() != 0)[0].astype(np.uint32)
        
        # Calculate compression - pretty proud of this metric
        original_size = tensor.size * 4  # float32
        compressed_size = len(nonzero_indices) * 4  # uint32 indices
        if sign_indices is not None:
            compressed_size += len(sign_indices) * 4
        
        return {
            'type': 'binary',
            'shape': tensor.shape,
            'indices': nonzero_indices,
            'sign_indices': sign_indices,
            'has_negative': has_negative,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': original_size / max(compressed_size, 1)
        }
    
    def encode_bitplane(self, tensor: Union[np.ndarray, torch.Tensor], num_bits: int = 8) -> Dict:
        """
        Bitplane encoding - break numbers into bit layers.
        This gives much better reconstruction than binary but uses more space.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
            
        # Quantize to specified bits - had to be careful with edge cases
        tensor_min, tensor_max = tensor.min(), tensor.max()
        if tensor_max == tensor_min:
            # Handle constant tensors
            quantized = np.zeros_like(tensor, dtype=np.uint8)
        else:
            # Scale to [0, 2^num_bits - 1]
            normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
            quantized = np.round(normalized * (2**num_bits - 1)).astype(np.uint8)
        
        # Extract bitplanes - each bit gets its own sparse representation
        bitplanes = []
        total_indices = 0
        
        for bit in range(num_bits):
            # Extract this bit from all values
            bit_layer = (quantized >> bit) & 1
            indices = np.where(bit_layer.flatten() == 1)[0].astype(np.uint32)
            bitplanes.append(indices)
            total_indices += len(indices)
        
        # Compression stats
        original_size = tensor.size * 4
        compressed_size = total_indices * 4 + 32  # indices + metadata
        
        return {
            'type': 'bitplane',
            'shape': tensor.shape,
            'bitplanes': bitplanes,
            'num_bits': num_bits,
            'min_val': tensor_min,
            'max_val': tensor_max,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': original_size / max(compressed_size, 1)
        }
    
    def encode_blocked(self, tensor: Union[np.ndarray, torch.Tensor], block_size: int = 64) -> Dict:
        """
        Blocked encoding for better cache performance.
        Process the matrix in blocks - helps with memory access patterns.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
            
        if len(tensor.shape) != 2:
            raise ValueError("Blocked encoding only supports 2D tensors")
            
        rows, cols = tensor.shape
        blocks = []
        total_compressed_size = 0
        
        # Process in blocks - skip empty ones to save space
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                block = tensor[i:i+block_size, j:j+block_size]
                
                # Only encode non-empty blocks
                if np.any(block != 0):
                    if self.encoding_type == "binary":
                        encoded_block = self.encode_binary(block)
                    else:
                        encoded_block = self.encode_bitplane(block)
                    
                    blocks.append({
                        'encoded': encoded_block,
                        'row_offset': i,
                        'col_offset': j,
                        'block_shape': block.shape
                    })
                    total_compressed_size += encoded_block['compressed_size']
        
        original_size = tensor.size * 4
        
        return {
            'type': 'blocked',
            'shape': tensor.shape,
            'blocks': blocks,
            'block_size': block_size,
            'encoding_type': self.encoding_type,
            'original_size': original_size,
            'compressed_size': total_compressed_size,
            'compression_ratio': original_size / max(total_compressed_size, 1)
        }
    
    def encode(self, tensor: Union[np.ndarray, torch.Tensor], **kwargs) -> Dict:
        """Main encoding function - routes to the right encoder"""
        if self.block_size is not None:
            return self.encode_blocked(tensor, self.block_size)
        elif self.encoding_type == "binary":
            return self.encode_binary(tensor, **kwargs)
        elif self.encoding_type == "bitplane":
            return self.encode_bitplane(tensor, **kwargs)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")


class BIEDecoder:
    """Decoder for BIE - reconstructs the original tensors from indices"""
    
    @staticmethod
    def decode_binary(encoded: Dict) -> np.ndarray:
        """Reconstruct from binary encoding"""
        shape = encoded['shape']
        result = np.zeros(shape).flatten()
        
        # Set the 1s where they should be
        indices = encoded['indices']
        result[indices] = 1.0
        
        # Handle negative values if present
        if encoded['has_negative'] and encoded['sign_indices'] is not None:
            result[encoded['sign_indices']] = -1.0
            
        return result.reshape(shape)
    
    @staticmethod
    def decode_bitplane(encoded: Dict) -> np.ndarray:
        """Reconstruct from bitplane encoding - combine all the bit layers"""
        shape = encoded['shape']
        num_bits = encoded['num_bits']
        
        # Start with zeros
        quantized = np.zeros(shape, dtype=np.uint8).flatten()
        
        # Add each bitplane back
        for bit, indices in enumerate(encoded['bitplanes']):
            quantized[indices] |= (1 << bit)
        
        # Convert back to float
        quantized = quantized.reshape(shape)
        min_val, max_val = encoded['min_val'], encoded['max_val']
        
        if max_val == min_val:
            return np.full(shape, min_val)
        
        # Dequantize back to original range
        normalized = quantized.astype(np.float32) / (2**num_bits - 1)
        result = normalized * (max_val - min_val) + min_val
        
        return result
    
    @staticmethod
    def decode_blocked(encoded: Dict) -> np.ndarray:
        """Reconstruct from blocked encoding"""
        shape = encoded['shape']
        result = np.zeros(shape)
        
        # Decode each block and put it back in place
        for block_info in encoded['blocks']:
            encoded_block = block_info['encoded']
            row_offset = block_info['row_offset']
            col_offset = block_info['col_offset']
            block_shape = block_info['block_shape']
            
            # Decode the block
            if encoded_block['type'] == 'binary':
                decoded_block = BIEDecoder.decode_binary(encoded_block)
            elif encoded_block['type'] == 'bitplane':
                decoded_block = BIEDecoder.decode_bitplane(encoded_block)
            
            # Place it back in the result
            result[row_offset:row_offset+block_shape[0], 
                   col_offset:col_offset+block_shape[1]] = decoded_block
        
        return result
    
    @staticmethod
    def decode(encoded: Dict) -> np.ndarray:
        """Main decode function"""
        if encoded['type'] == 'binary':
            return BIEDecoder.decode_binary(encoded)
        elif encoded['type'] == 'bitplane':
            return BIEDecoder.decode_bitplane(encoded)
        elif encoded['type'] == 'blocked':
            return BIEDecoder.decode_blocked(encoded)
        else:
            raise ValueError(f"Unknown encoding type: {encoded['type']}")


def get_compression_stats(encoded: Dict) -> Dict:
    """Calculate compression statistics - useful for my analysis"""
    original_size = encoded.get('original_size', 0)
    compressed_size = encoded.get('compressed_size', 0)
    
    if original_size == 0:
        return {'compression_ratio': 1.0, 'space_savings_percent': 0.0}
    
    compression_ratio = original_size / max(compressed_size, 1)
    space_savings = (1 - compressed_size / original_size) * 100
    
    return {
        'compression_ratio': compression_ratio,
        'space_savings_percent': space_savings,
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size
    }