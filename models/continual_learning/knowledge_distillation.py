"""
Knowledge Distillation for Continual Learning
Transfer knowledge from previous model to prevent catastrophic forgetting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class KnowledgeDistillation(nn.Module):
    """
    Knowledge distillation module for continual learning
    Preserves knowledge from previous tasks by distilling from old model
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        distillation_type: str = "logit",
        feature_layers: Optional[List[str]] = None,
    ):
        """
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1-alpha for task loss)
            distillation_type: Type of distillation (logit, feature, attention)
            feature_layers: Layers to use for feature distillation
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_type = distillation_type
        self.feature_layers = feature_layers or []
        
        # Feature adaptation layers
        self.feature_adapters = nn.ModuleDict()
    
    def add_feature_adapter(self, layer_name: str, in_dim: int, out_dim: int):
        """
        Add adaptation layer for feature distillation
        
        Args:
            layer_name: Name of the layer
            in_dim: Input dimension
            out_dim: Output dimension
        """
        if in_dim != out_dim:
            self.feature_adapters[layer_name] = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
            )
        else:
            self.feature_adapters[layer_name] = nn.Identity()
    
    def compute_logit_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute logit-based distillation loss using KL divergence
        
        Args:
            student_logits: Logits from student model [N, C]
            teacher_logits: Logits from teacher model [N, C]
            mask: Optional mask for selective distillation [N, C]
        
        Returns:
            Distillation loss scalar
        """
        # Soften probabilities
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Apply mask if provided (e.g., only distill on old classes)
        if mask is not None:
            student_soft = student_soft * mask
            teacher_soft = teacher_soft * mask
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return distillation_loss
    
    def compute_feature_distillation_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature-based distillation loss
        
        Args:
            student_features: Dictionary of feature maps from student
            teacher_features: Dictionary of feature maps from teacher
        
        Returns:
            Feature distillation loss
        """
        total_loss = 0.0
        count = 0
        
        for layer_name in self.feature_layers:
            if layer_name not in student_features or layer_name not in teacher_features:
                continue
            
            student_feat = student_features[layer_name]
            teacher_feat = teacher_features[layer_name]
            
            # Adapt feature dimensions if needed
            if layer_name in self.feature_adapters:
                student_feat = self.feature_adapters[layer_name](student_feat)
            
            # L2 distance loss
            loss = F.mse_loss(student_feat, teacher_feat)
            total_loss += loss
            count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss
    
    def compute_attention_distillation_loss(
        self,
        student_attention: Dict[str, torch.Tensor],
        teacher_attention: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute attention-based distillation loss
        Transfer attention patterns from teacher to student
        
        Args:
            student_attention: Dictionary of attention maps from student
            teacher_attention: Dictionary of attention maps from teacher
        
        Returns:
            Attention distillation loss
        """
        total_loss = 0.0
        count = 0
        
        for layer_name in self.feature_layers:
            if layer_name not in student_attention or layer_name not in teacher_attention:
                continue
            
            student_attn = student_attention[layer_name]
            teacher_attn = teacher_attention[layer_name]
            
            # Normalize attention maps
            student_attn = student_attn / (student_attn.sum(dim=-1, keepdim=True) + 1e-8)
            teacher_attn = teacher_attn / (teacher_attn.sum(dim=-1, keepdim=True) + 1e-8)
            
            # KL divergence on attention
            loss = F.kl_div(
                torch.log(student_attn + 1e-8),
                teacher_attn,
                reduction='batchmean'
            )
            total_loss += loss
            count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss
    
    def compute_distillation_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute overall distillation loss
        
        Args:
            student_outputs: Dictionary containing student model outputs
            teacher_outputs: Dictionary containing teacher model outputs
            mask: Optional mask for selective distillation
        
        Returns:
            Total distillation loss
        """
        loss = 0.0
        
        # Logit distillation
        if self.distillation_type in ["logit", "all"]:
            if "logits" in student_outputs and "logits" in teacher_outputs:
                logit_loss = self.compute_logit_distillation_loss(
                    student_outputs["logits"],
                    teacher_outputs["logits"],
                    mask=mask,
                )
                loss += logit_loss
        
        # Feature distillation
        if self.distillation_type in ["feature", "all"]:
            if "features" in student_outputs and "features" in teacher_outputs:
                feature_loss = self.compute_feature_distillation_loss(
                    student_outputs["features"],
                    teacher_outputs["features"],
                )
                loss += feature_loss
        
        # Attention distillation
        if self.distillation_type in ["attention", "all"]:
            if "attention" in student_outputs and "attention" in teacher_outputs:
                attention_loss = self.compute_attention_distillation_loss(
                    student_outputs["attention"],
                    teacher_outputs["attention"],
                )
                loss += attention_loss
        
        return loss
    
    def forward(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_loss: torch.Tensor,
        old_class_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with knowledge distillation
        
        Args:
            student_model: Current model being trained
            teacher_model: Previous model (frozen)
            inputs: Input data
            targets: Ground truth targets
            task_loss: Loss on current task
            old_class_mask: Mask for old classes
        
        Returns:
            Combined loss and loss dictionary
        """
        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        
        # Get student predictions
        student_outputs = student_model(inputs)
        
        # Compute distillation loss
        distillation_loss = self.compute_distillation_loss(
            student_outputs,
            teacher_outputs,
            mask=old_class_mask,
        )
        
        # Combine losses
        total_loss = (
            self.alpha * distillation_loss +
            (1 - self.alpha) * task_loss
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distillation_loss': distillation_loss,
        }
        
        return total_loss, loss_dict


class LwF(KnowledgeDistillation):
    """
    Learning without Forgetting (LwF)
    A specific implementation of knowledge distillation for continual learning
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        super().__init__(
            temperature=temperature,
            alpha=alpha,
            distillation_type="logit",
        )
    
    def forward(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_loss: torch.Tensor,
        num_old_classes: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        LwF forward pass
        Only distill on old classes, learn new classes normally
        
        Args:
            student_model: Current model
            teacher_model: Previous model
            inputs: Input data
            targets: Ground truth
            task_loss: Current task loss
            num_old_classes: Number of classes in previous tasks
        
        Returns:
            Combined loss and loss dictionary
        """
        # Create mask for old classes
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
            teacher_logits = teacher_outputs.get("logits", None)
        
        if teacher_logits is None:
            return task_loss, {'total_loss': task_loss, 'task_loss': task_loss}
        
        # Mask: only consider old classes for distillation
        old_class_mask = torch.zeros_like(teacher_logits)
        old_class_mask[:, :num_old_classes] = 1.0
        
        student_outputs = student_model(inputs)
        student_logits = student_outputs.get("logits", None)
        
        if student_logits is None:
            return task_loss, {'total_loss': task_loss, 'task_loss': task_loss}
        
        # Compute distillation loss on old classes
        distillation_loss = self.compute_logit_distillation_loss(
            student_logits,
            teacher_logits,
            mask=old_class_mask,
        )
        
        # Combine losses
        total_loss = (
            self.alpha * distillation_loss +
            (1 - self.alpha) * task_loss
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distillation_loss': distillation_loss,
        }
        
        return total_loss, loss_dict


def build_knowledge_distillation(cfg):
    """Build knowledge distillation module from config"""
    if cfg.CONTINUAL_LEARNING.DISTILLATION.ENABLED:
        return KnowledgeDistillation(
            temperature=cfg.CONTINUAL_LEARNING.DISTILLATION.TEMPERATURE,
            alpha=cfg.CONTINUAL_LEARNING.DISTILLATION.ALPHA,
            distillation_type=cfg.CONTINUAL_LEARNING.DISTILLATION.TYPE,
            feature_layers=cfg.CONTINUAL_LEARNING.DISTILLATION.FEATURE_LAYERS,
        )
    return None
