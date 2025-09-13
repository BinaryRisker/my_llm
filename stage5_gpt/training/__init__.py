"""
Stage 5: GPT Training
====================

GPT训练模块，包含：
- 训练器实现
- 数据集处理
- 评估工具
"""

from .trainer import (
    GPTTrainingConfig,
    LanguageModelingDataset,
    GPTTrainer,
    compute_metrics
)

__all__ = [
    # 配置
    'GPTTrainingConfig',
    
    # 数据集
    'LanguageModelingDataset',
    
    # 训练器
    'GPTTrainer',
    
    # 评估
    'compute_metrics'
]