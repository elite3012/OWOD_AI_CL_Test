"""
LoRA (Low-Rank Adaptation) Implementation
Efficient parameter adaptation by adding low-rank matrices to model weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that can be inserted into existing linear/conv layers
    
    Implements: W' = W + BA where B∈R^(d×r), A∈R^(r×k), r << min(d,k)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
        
        Returns:
            LoRA adaptation
        """
        # x @ A^T @ B^T * scaling
        result = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return result


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        self.merge_weights = merge_weights
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)
        else:
            return self.linear(x) + self.lora(x)
    
    def merge(self):
        """Merge LoRA weights into base weights for inference"""
        if not self.merged:
            with torch.no_grad():
                self.linear.weight.data += (
                    self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
                )
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from base weights"""
        if self.merged:
            with torch.no_grad():
                self.linear.weight.data -= (
                    self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
                )
            self.merged = False


class Conv2dWithLoRA(nn.Module):
    """
    Conv2d layer with LoRA adaptation
    """
    
    def __init__(
        self,
        conv: nn.Conv2d,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = conv
        
        # Flatten conv for LoRA
        in_features = conv.in_channels
        out_features = conv.out_channels
        
        self.lora = LoRALayer(
            in_features,
            out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard convolution
        conv_out = self.conv(x)
        
        # LoRA adaptation
        # Apply LoRA on pooled features
        b, c, h, w = x.shape
        x_pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        lora_out = self.lora(x_pooled).view(b, -1, 1, 1)
        lora_out = lora_out.expand(-1, -1, h, w)
        
        return conv_out + lora_out


def inject_lora(
    model: nn.Module,
    config: Dict,
    target_modules: Optional[List[str]] = None,
    freeze_base: bool = True,
) -> nn.Module:
    """
    Inject LoRA layers into a model
    
    Args:
        model: The base model to adapt
        config: LoRA configuration dict with r, alpha, dropout
        target_modules: List of module name patterns to inject LoRA into
        freeze_base: Whether to freeze base model parameters
    
    Returns:
        Model with LoRA layers injected
    """
    r = config.get('r', 8)
    alpha = config.get('alpha', 16)
    dropout = config.get('dropout', 0.0)
    
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 
                         'fc1', 'fc2', 'out_proj']
    
    # Freeze base model if specified
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    
    # Inject LoRA into target modules
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        should_inject = any(target in name for target in target_modules)
        
        if should_inject:
            if isinstance(module, nn.Linear):
                # Replace with LinearWithLoRA
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                
                lora_module = LinearWithLoRA(
                    module, r=r, alpha=alpha, dropout=dropout
                )
                setattr(parent, child_name, lora_module)
                
            elif isinstance(module, nn.Conv2d):
                # Replace with Conv2dWithLoRA
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                
                lora_module = Conv2dWithLoRA(
                    module, r=r, alpha=alpha, dropout=dropout
                )
                setattr(parent, child_name, lora_module)
    
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get only LoRA parameters from a model
    
    Args:
        model: Model with LoRA layers
    
    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count LoRA parameters vs total parameters
    
    Args:
        model: Model with LoRA layers
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'lora_params': lora_params,
        'trainable_params': trainable_params,
        'trainable_percentage': 100 * trainable_params / total_params,
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base weights for efficient inference
    
    Args:
        model: Model with LoRA layers
    
    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, (LinearWithLoRA, Conv2dWithLoRA)):
            if hasattr(module, 'merge'):
                module.merge()
    return model


def save_lora_weights(model: nn.Module, path: str):
    """
    Save only LoRA weights
    
    Args:
        model: Model with LoRA layers
        path: Path to save weights
    """
    lora_state_dict = {
        name: param for name, param in model.named_parameters()
        if 'lora_' in name
    }
    torch.save(lora_state_dict, path)


def load_lora_weights(model: nn.Module, path: str):
    """
    Load LoRA weights
    
    Args:
        model: Model with LoRA layers
        path: Path to load weights from
    """
    lora_state_dict = torch.load(path)
    
    # Load only LoRA parameters
    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = param
    
    model.load_state_dict(model_state_dict, strict=False)
    return model
