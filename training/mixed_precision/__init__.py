"""
混合精度训练优化模块

提供高效的FP16/BF16混合精度训练功能，包括：
- 自动混合精度训练 (AMP)
- 动态损失缩放
- 梯度裁剪改进
- 精度转换优化
- 训练稳定性保证

支持的精度类型：
- FP16: Half precision (16位浮点数)
- BF16: Brain float 16 (Google的16位浮点格式)
- FP32: Full precision (32位浮点数，用作备选)
"""

from .amp_trainer import AMPTrainer, AMPConfig
from .loss_scaler import DynamicLossScaler, StaticLossScaler, LossScaler
from .gradient_clipper import GradientClipper, AdaptiveGradientClipper
from .precision_utils import (
    PrecisionManager, 
    convert_model_to_precision,
    check_precision_support,
    get_optimal_precision
)
# from .mixed_precision_trainer import MixedPrecisionTrainer  # 暂时注释掉

__all__ = [
    # 核心训练器
    'AMPTrainer',
    'AMPConfig',
    # 'MixedPrecisionTrainer',  # 暂时注释掉
    
    # 损失缩放
    'DynamicLossScaler',
    'StaticLossScaler', 
    'LossScaler',
    
    # 梯度裁剪
    'GradientClipper',
    'AdaptiveGradientClipper',
    
    # 精度管理
    'PrecisionManager',
    'convert_model_to_precision',
    'check_precision_support',
    'get_optimal_precision'
]

# 支持的精度类型
SUPPORTED_PRECISIONS = ['fp16', 'bf16', 'fp32']

# 默认配置
DEFAULT_AMP_CONFIG = {
    'precision': 'fp16',
    'loss_scale': 'dynamic',
    'initial_scale': 65536.0,
    'growth_factor': 2.0,
    'backoff_factor': 0.5,
    'growth_interval': 2000,
    'enabled': True
}