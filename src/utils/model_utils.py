"""
My helper functions for working with neural network models.
I need these for pruning, quantization, and testing on real models like GPT-2.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
import copy


class ModelPruner:
    """
    My pruning toolkit - different ways to make models sparse.
    """
    
    @staticmethod
    def magnitude_pruning(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
        """
        Remove the smallest weights - simple but effective pruning.
        """
        if sparsity <= 0:
            return tensor.clone()
        
        # Find the threshold for pruning
        flat_tensor = tensor.flatten()
        k = int(sparsity * len(flat_tensor))
        if k >= len(flat_tensor):
            return torch.zeros_like(tensor)
        
        threshold = torch.kthvalue(torch.abs(flat_tensor), k + 1)[0]
        
        # Zero out weights below threshold
        mask = torch.abs(tensor) >= threshold
        return tensor * mask.float()
    
    @staticmethod
    def structured_pruning(tensor: torch.Tensor, sparsity: float, 
                          block_size: int = 4) -> torch.Tensor:
        """
        Prune entire blocks instead of individual weights.
        Better for hardware acceleration.
        """
        if sparsity <= 0:
            return tensor.clone()
        
        # Reshape into blocks
        h, w = tensor.shape
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        if h_blocks == 0 or w_blocks == 0:
            return ModelPruner.magnitude_pruning(tensor, sparsity)
        
        # Calculate block norms
        blocks = tensor[:h_blocks*block_size, :w_blocks*block_size].view(
            h_blocks, block_size, w_blocks, block_size
        )
        block_norms = torch.norm(blocks, dim=(1, 3))
        
        # Determine blocks to prune
        num_blocks_to_prune = int(sparsity * h_blocks * w_blocks)
        if num_blocks_to_prune >= h_blocks * w_blocks:
            return torch.zeros_like(tensor)
        
        flat_norms = block_norms.flatten()
        threshold = torch.kthvalue(flat_norms, num_blocks_to_prune + 1)[0]
        
        # Create mask
        block_mask = (block_norms >= threshold).float()
        mask = block_mask.unsqueeze(1).unsqueeze(3).expand_as(blocks)
        
        # Apply mask
        result = tensor.clone()
        result[:h_blocks*block_size, :w_blocks*block_size] = (blocks * mask).view(
            h_blocks*block_size, w_blocks*block_size
        )
        
        return result
    
    @staticmethod
    def gradual_pruning(model: nn.Module, target_sparsity: float, 
                       num_steps: int = 10) -> List[nn.Module]:
        """
        Gradually increase sparsity over multiple steps.
        Sometimes works better than pruning all at once.
        """
        models = []
        current_model = copy.deepcopy(model)
        
        for step in range(num_steps + 1):
            current_sparsity = (step / num_steps) * target_sparsity
            
            # Apply pruning to all linear layers
            for name, module in current_model.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = ModelPruner.magnitude_pruning(
                        module.weight.data, current_sparsity
                    )
            
            models.append(copy.deepcopy(current_model))
        
        return models


class ModelQuantizer:
    """
    My quantization methods - reducing precision to save space.
    """
    
    @staticmethod
    def linear_quantization(tensor: torch.Tensor, num_bits: int = 8, 
                           symmetric: bool = True) -> Tuple[torch.Tensor, float, float]:
        """
        Basic linear quantization - map float values to integers.
        """
        if num_bits >= 32:
            return tensor, 1.0, 0.0
        
        # Calculate quantization parameters
        if symmetric:
            max_val = torch.max(torch.abs(tensor))
            min_val = -max_val
        else:
            max_val = torch.max(tensor)
            min_val = torch.min(tensor)
        
        # Avoid division by zero
        if max_val == min_val:
            return torch.zeros_like(tensor), 1.0, 0.0
        
        # Calculate scale and zero point
        qmax = 2**(num_bits - 1) - 1
        qmin = -2**(num_bits - 1)
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize for return
        dequantized = (quantized - zero_point) * scale
        
        return dequantized, scale.item(), zero_point.item()
    
    @staticmethod
    def quantize_model(model: nn.Module, num_bits: int = 8) -> nn.Module:
        """
        Quantize all weights in a model.
        """
        quantized_model = copy.deepcopy(model)
        
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                quantized_weight, _, _ = ModelQuantizer.linear_quantization(
                    module.weight.data, num_bits
                )
                module.weight.data = quantized_weight
                
                if module.bias is not None:
                    quantized_bias, _, _ = ModelQuantizer.linear_quantization(
                        module.bias.data, num_bits
                    )
                    module.bias.data = quantized_bias
        
        return quantized_model


class GPT2Utils:
    """
    My helper class for working with GPT-2 models.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Set up for working with a specific GPT-2 variant.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
    
    def load_model(self, model_name: str = None) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
        """
        Load the GPT-2 model and tokenizer.
        """
        if model_name is None:
            model_name = self.model_name
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def get_layer_weights(self, model: GPT2LMHeadModel) -> Dict[str, torch.Tensor]:
        """
        Extract all the weight matrices from GPT-2.
        """
        weights = {}
        
        # Extract weights from each transformer layer
        for i, layer in enumerate(model.transformer.h):
            weights[f'layer_{i}_attn_c_attn'] = layer.attn.c_attn.weight
            weights[f'layer_{i}_attn_c_proj'] = layer.attn.c_proj.weight
            weights[f'layer_{i}_mlp_c_fc'] = layer.mlp.c_fc.weight
            weights[f'layer_{i}_mlp_c_proj'] = layer.mlp.c_proj.weight
        
        # Add final layer norm and language model head
        weights['ln_f'] = model.transformer.ln_f.weight
        weights['lm_head'] = model.lm_head.weight
        
        return weights
    
    def calculate_perplexity(self, model: GPT2LMHeadModel, text: str) -> float:
        """
        Calculate perplexity on a text - measures how well the model predicts it.
        """
        if self.tokenizer is None:
            _, self.tokenizer = self.load_model()
        
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def generate_text(self, model: GPT2LMHeadModel, prompt: str, 
                     max_length: int = 100) -> str:
        """
        Generate text using the model.
        """
        if self.tokenizer is None:
            _, self.tokenizer = self.load_model()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def benchmark_model_quality(self, original_model: GPT2LMHeadModel, 
                               modified_model: GPT2LMHeadModel,
                               test_texts: List[str]) -> Dict:
        """
        Compare quality between original and modified models.
        """
        results = {
            'original_perplexities': [],
            'modified_perplexities': [],
            'perplexity_differences': []
        }
        
        for text in test_texts:
            orig_perp = self.calculate_perplexity(original_model, text)
            mod_perp = self.calculate_perplexity(modified_model, text)
            
            results['original_perplexities'].append(orig_perp)
            results['modified_perplexities'].append(mod_perp)
            results['perplexity_differences'].append(mod_perp - orig_perp)
        
        # Calculate summary statistics
        results['avg_original_perplexity'] = np.mean(results['original_perplexities'])
        results['avg_modified_perplexity'] = np.mean(results['modified_perplexities'])
        results['avg_perplexity_increase'] = np.mean(results['perplexity_differences'])
        results['relative_increase_percent'] = (results['avg_perplexity_increase'] / results['avg_original_perplexity']) * 100
        
        return results


def create_test_matrices(sizes: List[Tuple[int, int]], 
                        sparsity_levels: List[float]) -> Dict:
    """
    Generate test matrices with different sparsity patterns.
    Useful for benchmarking compression methods.
    """
    matrices = {}
    
    for height, width in sizes:
        for sparsity in sparsity_levels:
            # Create random matrix
            matrix = np.random.randn(height, width).astype(np.float32)
            
            if sparsity > 0:
                # Apply sparsity
                mask = np.random.random((height, width)) > sparsity
                matrix = matrix * mask
            
            key = f"matrix_{height}x{width}_sparsity_{sparsity:.1f}"
            matrices[key] = matrix
    
    return matrices


def analyze_model_sparsity(model: nn.Module) -> Dict:
    """
    Analyze the sparsity patterns in a model.
    """
    total_params = 0
    zero_params = 0
    layer_sparsities = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            layer_total = weight.numel()
            layer_zeros = (weight == 0).sum().item()
            
            total_params += layer_total
            zero_params += layer_zeros
            
            layer_sparsities[name] = {
                'total_params': layer_total,
                'zero_params': layer_zeros,
                'sparsity': layer_zeros / layer_total if layer_total > 0 else 0.0
            }
    
    overall_sparsity = zero_params / total_params if total_params > 0 else 0.0
    
    return {
        'overall_sparsity': overall_sparsity,
        'total_parameters': total_params,
        'zero_parameters': zero_params,
        'layer_sparsities': layer_sparsities
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.
    """
    total_params = sum(p.numel() for p in model.parameters())
    # Assuming float32 (4 bytes per parameter)
    size_mb = (total_params * 4) / (1024 * 1024)
    return size_mb