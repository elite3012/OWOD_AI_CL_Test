"""
Memory Replay for Continual Learning
Implements exemplar selection and replay strategies to prevent catastrophic forgetting
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import random


class MemoryBuffer:
    """
    Memory buffer for storing exemplars from previous tasks
    """
    
    def __init__(
        self,
        memory_size: int,
        selection_strategy: str = "herding",
        update_strategy: str = "reservoir",
    ):
        """
        Args:
            memory_size: Total number of exemplars to store
            selection_strategy: How to select exemplars (herding, random, entropy, forgetting)
            update_strategy: How to update buffer (reservoir, ring_buffer)
        """
        self.memory_size = memory_size
        self.selection_strategy = selection_strategy
        self.update_strategy = update_strategy
        
        # Storage
        self.images = []
        self.targets = []
        self.features = []
        self.class_counts = defaultdict(int)
        self.seen_samples = 0
    
    def update(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
    ):
        """
        Update memory buffer with new samples
        
        Args:
            images: Image tensors [N, C, H, W]
            targets: Class labels [N]
            features: Feature representations [N, D]
            scores: Sample importance scores [N]
        """
        batch_size = images.size(0)
        
        for i in range(batch_size):
            img = images[i].cpu()
            target = targets[i].cpu()
            feat = features[i].cpu() if features is not None else None
            score = scores[i].cpu() if scores is not None else None
            
            if self.update_strategy == "reservoir":
                self._reservoir_update(img, target, feat, score)
            elif self.update_strategy == "ring_buffer":
                self._ring_buffer_update(img, target, feat, score)
            else:
                self._basic_update(img, target, feat, score)
            
            self.seen_samples += 1
    
    def _reservoir_update(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        feature: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
    ):
        """
        Reservoir sampling for memory update
        Ensures uniform distribution over all seen samples
        """
        if len(self.images) < self.memory_size:
            # Buffer not full, add directly
            self.images.append(image)
            self.targets.append(target)
            if feature is not None:
                self.features.append(feature)
            self.class_counts[target.item()] += 1
        else:
            # Replace with probability memory_size / seen_samples
            j = random.randint(0, self.seen_samples)
            if j < self.memory_size:
                # Remove old class count
                old_target = self.targets[j].item()
                self.class_counts[old_target] -= 1
                
                # Replace
                self.images[j] = image
                self.targets[j] = target
                if feature is not None and len(self.features) > j:
                    self.features[j] = feature
                
                # Update new class count
                self.class_counts[target.item()] += 1
    
    def _ring_buffer_update(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        feature: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
    ):
        """
        Ring buffer (FIFO) update strategy
        """
        if len(self.images) < self.memory_size:
            self.images.append(image)
            self.targets.append(target)
            if feature is not None:
                self.features.append(feature)
        else:
            # Remove oldest, add newest
            old_target = self.targets[0].item()
            self.class_counts[old_target] -= 1
            
            self.images.pop(0)
            self.targets.pop(0)
            if len(self.features) > 0:
                self.features.pop(0)
            
            self.images.append(image)
            self.targets.append(target)
            if feature is not None:
                self.features.append(feature)
        
        self.class_counts[target.item()] += 1
    
    def _basic_update(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        feature: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
    ):
        """Basic update - add if space available"""
        if len(self.images) < self.memory_size:
            self.images.append(image)
            self.targets.append(target)
            if feature is not None:
                self.features.append(feature)
            self.class_counts[target.item()] += 1
    
    def select_exemplars(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        num_exemplars: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select exemplars using specified strategy
        
        Args:
            images: All available images [N, C, H, W]
            targets: All targets [N]
            features: All features [N, D]
            num_exemplars: Number of exemplars to select
        
        Returns:
            Selected images and targets
        """
        if self.selection_strategy == "herding":
            return self._herding_selection(images, targets, features, num_exemplars)
        elif self.selection_strategy == "random":
            return self._random_selection(images, targets, num_exemplars)
        elif self.selection_strategy == "entropy":
            return self._entropy_selection(images, targets, features, num_exemplars)
        else:
            return self._random_selection(images, targets, num_exemplars)
    
    def _herding_selection(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        num_exemplars: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Herding selection: choose samples closest to class mean
        """
        # Compute class mean
        class_mean = features.mean(dim=0, keepdim=True)
        
        selected_indices = []
        selected_sum = torch.zeros_like(class_mean)
        
        features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        class_mean_norm = class_mean / (class_mean.norm() + 1e-8)
        
        for k in range(min(num_exemplars, len(features))):
            # Find sample that minimizes distance to class mean
            current_mean = (selected_sum + features_norm) / (k + 1)
            distances = (class_mean_norm - current_mean).norm(dim=1)
            
            # Exclude already selected
            for idx in selected_indices:
                distances[idx] = float('inf')
            
            selected_idx = distances.argmin().item()
            selected_indices.append(selected_idx)
            selected_sum += features_norm[selected_idx]
        
        return images[selected_indices], targets[selected_indices]
    
    def _random_selection(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        num_exemplars: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random exemplar selection"""
        num_samples = min(num_exemplars, len(images))
        indices = torch.randperm(len(images))[:num_samples]
        return images[indices], targets[indices]
    
    def _entropy_selection(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        num_exemplars: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select samples with highest prediction entropy
        (assumes features are logits or probabilities)
        """
        # Compute entropy
        probs = torch.softmax(features, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        # Select top-k by entropy
        num_samples = min(num_exemplars, len(images))
        _, indices = torch.topk(entropy, num_samples)
        
        return images[indices], targets[indices]
    
    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch from memory buffer
        
        Args:
            batch_size: Number of samples to return
            device: Device to place tensors on
        
        Returns:
            Batch of images and targets
        """
        if len(self.images) == 0:
            return None, None
        
        # Sample indices
        num_samples = min(batch_size, len(self.images))
        indices = random.sample(range(len(self.images)), num_samples)
        
        # Gather samples
        images = torch.stack([self.images[i] for i in indices]).to(device)
        targets = torch.stack([self.targets[i] for i in indices]).to(device)
        
        return images, targets
    
    def sample_balanced(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a class-balanced batch from memory buffer
        
        Args:
            batch_size: Number of samples to return
            device: Device to place tensors on
        
        Returns:
            Class-balanced batch of images and targets
        """
        if len(self.images) == 0:
            return None, None
        
        # Group indices by class
        class_indices = defaultdict(list)
        for i, target in enumerate(self.targets):
            class_indices[target.item()].append(i)
        
        # Sample from each class
        num_classes = len(class_indices)
        samples_per_class = batch_size // num_classes
        
        selected_indices = []
        for class_id, indices in class_indices.items():
            n_samples = min(samples_per_class, len(indices))
            selected = random.sample(indices, n_samples)
            selected_indices.extend(selected)
        
        # Gather samples
        images = torch.stack([self.images[i] for i in selected_indices]).to(device)
        targets = torch.stack([self.targets[i] for i in selected_indices]).to(device)
        
        return images, targets
    
    def get_all(
        self,
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all samples in memory buffer
        
        Returns:
            All images and targets
        """
        if len(self.images) == 0:
            return None, None
        
        images = torch.stack(self.images).to(device)
        targets = torch.stack(self.targets).to(device)
        
        return images, targets
    
    def size(self) -> int:
        """Return current buffer size"""
        return len(self.images)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.images) == 0
    
    def clear(self):
        """Clear all samples from buffer"""
        self.images = []
        self.targets = []
        self.features = []
        self.class_counts = defaultdict(int)
        self.seen_samples = 0
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in buffer"""
        return dict(self.class_counts)
    
    def save(self, path: str):
        """Save buffer to disk"""
        state = {
            'images': self.images,
            'targets': self.targets,
            'features': self.features,
            'class_counts': dict(self.class_counts),
            'seen_samples': self.seen_samples,
            'memory_size': self.memory_size,
            'selection_strategy': self.selection_strategy,
            'update_strategy': self.update_strategy,
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load buffer from disk"""
        state = torch.load(path)
        self.images = state['images']
        self.targets = state['targets']
        self.features = state.get('features', [])
        self.class_counts = defaultdict(int, state['class_counts'])
        self.seen_samples = state['seen_samples']
        self.memory_size = state['memory_size']
        self.selection_strategy = state['selection_strategy']
        self.update_strategy = state['update_strategy']


def build_memory_buffer(cfg):
    """Build memory buffer from config"""
    return MemoryBuffer(
        memory_size=cfg.CONTINUAL_LEARNING.REPLAY.MEMORY_SIZE,
        selection_strategy=cfg.CONTINUAL_LEARNING.REPLAY.SELECTION_STRATEGY,
        update_strategy=cfg.CONTINUAL_LEARNING.REPLAY.UPDATE_STRATEGY,
    )
