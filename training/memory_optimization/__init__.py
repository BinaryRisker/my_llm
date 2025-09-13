"""
内存优化模块

提供各种内存优化技术，包括：
- 梯度累积 (Gradient Accumulation)
- 激活检查点 (Activation Checkpointing)  
- 内存映射数据加载
- 动态批量大小调整
- 内存监控和管理
"""

from .gradient_accumulation import GradientAccumulator, AccumulationConfig
# 暂时注释掉未实现的模块
# from .activation_checkpointing import ActivationCheckpointing, CheckpointConfig
# from .memory_monitor import MemoryMonitor, MemoryStats
# from .dynamic_batching import DynamicBatchSizer, BatchConfig
# from .memory_mapper import MemoryMappedDataLoader

__all__ = [
    # 梯度累积
    'GradientAccumulator',
    'AccumulationConfig',
    
    # 暂时注释掉未实现的功能
    # 'ActivationCheckpointing', 
    # 'CheckpointConfig',
    # 'MemoryMonitor',
    # 'MemoryStats',
    # 'DynamicBatchSizer',
    # 'BatchConfig',
    # 'MemoryMappedDataLoader'
]

# 默认配置
DEFAULT_ACCUMULATION_CONFIG = {
    'accumulation_steps': 4,
    'sync_gradients': True,
    'average_gradients': True
}

DEFAULT_CHECKPOINT_CONFIG = {
    'enabled': True,
    'preserve_rng_state': True,
    'pack_sequences': True
}