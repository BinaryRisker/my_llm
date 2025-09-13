"""
Stage 5: GPT Utils
==================

GPT工具函数模块，包含：
- 掩码处理
- 文本预处理
- 评估指标
- 模型工具
"""

from .helpers import (
    # 掩码相关
    create_causal_mask,
    apply_causal_mask,
    create_padding_mask,
    
    # 损失函数
    autoregressive_loss,
    compute_perplexity,
    
    # 文本处理
    prepare_training_data,
    clean_text,
    tokenize_batch,
    
    # 评估指标
    compute_token_accuracy,
    compute_sequence_accuracy,
    compute_bleu_score,
    
    # 生成评估
    evaluate_generation_diversity,
    evaluate_repetition,
    
    # 模型工具
    count_parameters,
    get_model_size_mb,
    save_model_summary,
    
    # 分词器
    SimpleTokenizer
)

__all__ = [
    # 掩码相关
    'create_causal_mask',
    'apply_causal_mask', 
    'create_padding_mask',
    
    # 损失函数
    'autoregressive_loss',
    'compute_perplexity',
    
    # 文本处理
    'prepare_training_data',
    'clean_text',
    'tokenize_batch',
    
    # 评估指标
    'compute_token_accuracy',
    'compute_sequence_accuracy',
    'compute_bleu_score',
    
    # 生成评估
    'evaluate_generation_diversity',
    'evaluate_repetition',
    
    # 模型工具
    'count_parameters',
    'get_model_size_mb',
    'save_model_summary',
    
    # 分词器
    'SimpleTokenizer'
]