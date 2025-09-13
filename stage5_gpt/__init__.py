"""
Stage 5: GPT Architecture Implementation
=======================================

自回归语言模型实现，包含：
- GPT-Mini模型架构
- 因果掩码机制
- 预训练和微调
- 多种文本生成策略

主要模块：
- models: GPT模型实现
- training: 训练和微调脚本
- generation: 文本生成工具
- utils: 工具函数
"""

__version__ = "1.0.0"
__author__ = "GPT Learning Project"

# 模块导入
try:
    from .models import *
    from .training import *
    from .generation import *
    from .utils import *
except ImportError:
    # 在模块尚未完全构建时提供友好提示
    print("GPT modules are being initialized...")

__all__ = [
    # 核心模型
    'GPT', 'GPTMini', 'GPTBlock',
    'CausalSelfAttention', 'GPTConfig',
    
    # 训练相关
    'GPTTrainer', 'LanguageModelingDataset',
    'GPTForClassification', 'GPTForGeneration',
    
    # 生成策略
    'GreedyGenerator', 'SamplingGenerator', 
    'BeamSearchGenerator', 'TopKTopPGenerator',
    
    # 工具函数
    'create_causal_mask', 'apply_causal_mask',
    'autoregressive_loss', 'compute_perplexity',
    'prepare_training_data', 'tokenize_batch',
]