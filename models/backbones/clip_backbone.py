"""
CLIP Backbone for Enhanced OWOD
Integrates OpenAI's CLIP model as a backbone for open world object detection
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import open_clip


class CLIPBackbone(nn.Module):
    """
    CLIP Vision Transformer backbone with optional LoRA adaptation
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Load CLIP model
        model_name = cfg.FOUNDATION_MODEL.CLIP.MODEL_NAME
        pretrained = cfg.FOUNDATION_MODEL.CLIP.PRETRAINED
        
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        
        # Get vision encoder
        self.visual = self.clip_model.visual
        self.embed_dim = cfg.FOUNDATION_MODEL.CLIP.EMBED_DIM
        
        # Freeze parameters if specified
        if cfg.FOUNDATION_MODEL.CLIP.FREEZE_VISION:
            for param in self.visual.parameters():
                param.requires_grad = False
        
        # Add projection layers for detection
        self._build_detection_heads()
        
        # Apply PEFT if enabled
        if cfg.PEFT.ENABLED and cfg.PEFT.METHOD == "lora":
            self._apply_lora()
    
    def _build_detection_heads(self):
        """Build projection heads for multi-scale feature maps"""
        # Feature pyramid dimensions
        if "ViT" in self.cfg.FOUNDATION_MODEL.CLIP.MODEL_NAME:
            # For Vision Transformer
            self.feature_dims = {
                'layer1': self.embed_dim,
                'layer2': self.embed_dim,
                'layer3': self.embed_dim,
                'layer4': self.embed_dim,
            }
        else:
            # For ResNet-based CLIP
            self.feature_dims = {
                'layer1': 256,
                'layer2': 512,
                'layer3': 1024,
                'layer4': 2048,
            }
        
        # Create projection layers for FPN-like structure
        self.projections = nn.ModuleDict()
        target_dim = 256  # Standard FPN dimension
        
        for name, dim in self.feature_dims.items():
            self.projections[name] = nn.Sequential(
                nn.Conv2d(dim, target_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(target_dim),
                nn.ReLU(inplace=True)
            )
    
    def _apply_lora(self):
        """Apply LoRA adaptation to vision encoder"""
        from ..peft.lora import inject_lora
        
        lora_config = {
            'r': self.cfg.PEFT.LORA.R,
            'alpha': self.cfg.PEFT.LORA.ALPHA,
            'dropout': self.cfg.PEFT.LORA.DROPOUT,
            'target_modules': self.cfg.PEFT.LORA.TARGET_MODULES,
        }
        
        self.visual = inject_lora(
            self.visual, 
            lora_config, 
            freeze_base=self.cfg.FOUNDATION_MODEL.CLIP.FREEZE_VISION
        )
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLIP backbone
        
        Args:
            images: Input images tensor [B, 3, H, W]
        
        Returns:
            Dictionary of multi-scale feature maps
        """
        # Get intermediate features from vision encoder
        features = self._extract_features(images)
        
        # Project to common dimension
        outputs = {}
        for name, feat in features.items():
            if name in self.projections:
                outputs[name] = self.projections[name](feat)
        
        return outputs
    
    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from CLIP vision encoder
        
        For ViT-based models, we extract features at different transformer blocks
        For ResNet-based models, we extract features from different residual stages
        """
        features = {}
        
        if "ViT" in self.cfg.FOUNDATION_MODEL.CLIP.MODEL_NAME:
            # Vision Transformer feature extraction
            x = self.visual.conv1(x)  # Patch embedding
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, H*W]
            x = x.permute(0, 2, 1)  # [B, H*W, C]
            x = torch.cat([self.visual.class_embedding.to(x.dtype) + 
                          torch.zeros(x.shape[0], 1, x.shape[-1], 
                          dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.visual.positional_embedding.to(x.dtype)
            x = self.visual.ln_pre(x)
            
            # Extract features from different transformer layers
            x = x.permute(1, 0, 2)  # [HW+1, B, C]
            
            num_layers = len(self.visual.transformer.resblocks)
            layer_indices = {
                'layer1': num_layers // 4,
                'layer2': num_layers // 2,
                'layer3': 3 * num_layers // 4,
                'layer4': num_layers,
            }
            
            for i, block in enumerate(self.visual.transformer.resblocks):
                x = block(x)
                for name, idx in layer_indices.items():
                    if i + 1 == idx:
                        # Reshape to spatial feature map
                        feat = x[1:].permute(1, 2, 0)  # [B, C, HW]
                        hw = int(feat.shape[-1] ** 0.5)
                        feat = feat.reshape(feat.shape[0], feat.shape[1], hw, hw)
                        features[name] = feat
        else:
            # ResNet feature extraction
            x = self.visual.act1(self.visual.bn1(self.visual.conv1(x)))
            x = self.visual.act2(self.visual.bn2(self.visual.conv2(x)))
            x = self.visual.act3(self.visual.bn3(self.visual.conv3(x)))
            x = self.visual.avgpool(x)
            
            # Extract from residual layers
            features['layer1'] = self.visual.layer1(x)
            features['layer2'] = self.visual.layer2(features['layer1'])
            features['layer3'] = self.visual.layer3(features['layer2'])
            features['layer4'] = self.visual.layer4(features['layer3'])
        
        return features
    
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode text using CLIP text encoder
        
        Args:
            text: List of text strings
        
        Returns:
            Text embeddings [N, D]
        """
        if self.cfg.FOUNDATION_MODEL.CLIP.FREEZE_TEXT:
            with torch.no_grad():
                text_tokens = open_clip.tokenize(text).to(next(self.parameters()).device)
                text_features = self.clip_model.encode_text(text_tokens)
        else:
            text_tokens = open_clip.tokenize(text).to(next(self.parameters()).device)
            text_features = self.clip_model.encode_text(text_tokens)
        
        return text_features
    
    def get_text_embeddings_for_classes(self, class_names: List[str]) -> torch.Tensor:
        """
        Get text embeddings for all classes using templates
        
        Args:
            class_names: List of class names
        
        Returns:
            Class text embeddings [N_classes, D]
        """
        template = self.cfg.MULTIMODAL.TEXT_GUIDED.PROMPT_TEMPLATE
        
        texts = []
        for class_name in class_names:
            if template:
                texts.append(template.format(class_name))
            else:
                texts.append(class_name)
        
        text_embeddings = self.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings
    
    def output_shape(self):
        """Return output feature shapes"""
        return {
            name: {"channels": 256, "stride": 2 ** (i + 2)}
            for i, name in enumerate(self.feature_dims.keys())
        }


def build_clip_backbone(cfg):
    """
    Build CLIP backbone from config
    """
    return CLIPBackbone(cfg)
