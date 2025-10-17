"""
Backbones module - Foundation model integrations
"""

from .clip_backbone import build_clip_backbone, CLIPBackbone

__all__ = ['build_clip_backbone', 'CLIPBackbone']
