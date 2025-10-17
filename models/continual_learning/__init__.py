"""
Continual Learning module
"""

from .memory_replay import MemoryBuffer, build_memory_buffer
from .knowledge_distillation import (
    KnowledgeDistillation,
    LwF,
    build_knowledge_distillation,
)

__all__ = [
    'MemoryBuffer',
    'build_memory_buffer',
    'KnowledgeDistillation',
    'LwF',
    'build_knowledge_distillation',
]
