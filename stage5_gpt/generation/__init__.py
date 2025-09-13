"""
Stage 5: GPT Generation
=======================

GPT文本生成模块，包含：
- 多种生成策略
- 配置管理
- 批量生成
"""

from .generator import (
    GenerationConfig,
    BaseGenerator,
    GreedyGenerator,
    SamplingGenerator,
    BeamSearchGenerator,
    ContrastiveSearchGenerator,
    BatchGenerator,
    create_generator,
    generate_text
)

__all__ = [
    # 配置
    'GenerationConfig',
    
    # 生成器
    'BaseGenerator',
    'GreedyGenerator',
    'SamplingGenerator',
    'BeamSearchGenerator',
    'ContrastiveSearchGenerator',
    'BatchGenerator',
    
    # 便捷函数
    'create_generator',
    'generate_text'
]