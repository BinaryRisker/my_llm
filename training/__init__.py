"""
分布式训练支持模块
================

提供完整的分布式训练支持，包括：
- 数据并行训练（DDP）
- 模型并行训练
- 管道并行训练
- 混合精度训练
- 分布式数据加载
- 梯度同步和通信

组件说明:
- distributed_training: 核心分布式训练实现
- training_utils: 训练工具和辅助函数
- optimization: 优化器和学习率调度

使用方法:
    from training import DistributedTrainer, DistributedConfig
    
    # 配置分布式训练
    config = DistributedConfig(world_size=4, batch_size=32)
    
    # 创建分布式训练器
    trainer = DistributedTrainer(model, config, strategy='ddp')
    trainer.setup()
    
    # 训练
    trainer.train_step(batch, optimizer, criterion)

版本: 1.0.0
"""

# 分布式训练导入
try:
    from .distributed_training import (
        DistributedTrainer,
        DistributedConfig,
        DistributedDataLoader,
        BaseDistributedStrategy,
        DataParallelStrategy,
        ModelParallelStrategy,
        PipelineParallelStrategy,
        demo as distributed_demo
    )
    DISTRIBUTED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入分布式训练模块: {e}")
    DistributedTrainer = DistributedConfig = DistributedDataLoader = None
    BaseDistributedStrategy = DataParallelStrategy = None
    ModelParallelStrategy = PipelineParallelStrategy = None
    distributed_demo = None
    DISTRIBUTED_AVAILABLE = False

# 版本信息
__version__ = "1.0.0"
__author__ = "LLM Implementation Team"
__description__ = "Distributed training support for large language models"

# 导出的公共接口
__all__ = [
    # 分布式训练
    'DistributedTrainer',
    'DistributedConfig',
    'DistributedDataLoader',
    
    # 分布式策略
    'BaseDistributedStrategy',
    'DataParallelStrategy',
    'ModelParallelStrategy',
    'PipelineParallelStrategy',
    
    # 演示函数
    'distributed_demo',
    'demo',
    
    # 可用性标志
    'DISTRIBUTED_AVAILABLE',
]


def get_training_info():
    """获取训练系统信息"""
    return {
        "name": "Distributed Training System",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "components": {
            "distributed_training": DISTRIBUTED_AVAILABLE,
        },
        "supported_strategies": [
            "ddp", "data_parallel", "model_parallel", "pipeline_parallel"
        ],
        "supported_backends": [
            "nccl", "gloo", "mpi"
        ],
        "features": [
            "Multi-GPU training", "Multi-node training",
            "Mixed precision", "Gradient accumulation",
            "Model checkpointing", "Distributed data loading"
        ]
    }


def list_available_strategies():
    """列出所有可用的分布式策略"""
    strategies = []
    
    if DISTRIBUTED_AVAILABLE:
        strategies.extend([
            {
                "name": "ddp",
                "description": "分布式数据并行（推荐）",
                "type": "data_parallel",
                "gpu_requirement": "多GPU",
                "memory_efficiency": "高",
                "communication_overhead": "低",
                "available": True
            },
            {
                "name": "data_parallel",
                "description": "数据并行训练",
                "type": "data_parallel", 
                "gpu_requirement": "多GPU",
                "memory_efficiency": "高",
                "communication_overhead": "低",
                "available": True
            },
            {
                "name": "model_parallel",
                "description": "模型并行训练",
                "type": "model_parallel",
                "gpu_requirement": "多GPU",
                "memory_efficiency": "中等",
                "communication_overhead": "高",
                "available": True
            },
            {
                "name": "pipeline_parallel",
                "description": "管道并行训练",
                "type": "pipeline_parallel",
                "gpu_requirement": "多GPU/多节点",
                "memory_efficiency": "高",
                "communication_overhead": "中等",
                "available": True
            }
        ])
    
    return strategies


def create_distributed_trainer(
    model,
    strategy: str = "ddp",
    world_size: int = 1,
    rank: int = 0,
    batch_size: int = 32,
    **kwargs
):
    """
    创建分布式训练器的便捷函数
    
    Args:
        model: PyTorch模型
        strategy: 分布式策略 ("ddp", "model_parallel", "pipeline_parallel")
        world_size: 世界大小（进程总数）
        rank: 当前进程的rank
        batch_size: 批次大小
        **kwargs: 其他配置参数
        
    Returns:
        配置好的分布式训练器
        
    Example:
        >>> trainer = create_distributed_trainer(
        ...     model, strategy="ddp", world_size=4, batch_size=32
        ... )
        >>> trainer.setup()
        >>> trainer.train_step(batch, optimizer, criterion)
    """
    if not DISTRIBUTED_AVAILABLE:
        raise ImportError("分布式训练模块不可用")
    
    # 创建配置
    config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        batch_size=batch_size,
        **kwargs
    )
    
    # 创建训练器
    trainer = DistributedTrainer(model, config, strategy=strategy)
    
    return trainer


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    **kwargs
):
    """
    创建分布式数据加载器的便捷函数
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        world_size: 世界大小
        rank: 当前进程rank
        **kwargs: 其他参数
        
    Returns:
        分布式数据加载器
        
    Example:
        >>> dataloader = create_distributed_dataloader(
        ...     dataset, batch_size=32, world_size=4
        ... )
        >>> loader = dataloader.setup()
    """
    if not DISTRIBUTED_AVAILABLE:
        raise ImportError("分布式训练模块不可用")
    
    config = DistributedConfig(world_size=world_size, rank=rank)
    
    return DistributedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        config=config,
        **kwargs
    )


def demo():
    """运行完整的分布式训练演示"""
    print("🚀 分布式训练系统完整演示")
    print("=" * 60)
    
    # 显示系统信息
    info = get_training_info()
    print("📋 系统信息:")
    print(f"  名称: {info['name']}")
    print(f"  版本: {info['version']}")
    print(f"  描述: {info['description']}")
    
    print(f"\n🧩 组件状态:")
    for component, available in info['components'].items():
        status = "✅ 可用" if available else "❌ 不可用"
        print(f"  {component}: {status}")
    
    # 显示支持的策略
    print(f"\n🔧 支持的分布式策略:")
    strategies = list_available_strategies()
    for strategy in strategies:
        print(f"  • {strategy['name']}: {strategy['description']}")
        print(f"    类型: {strategy['type']}, GPU要求: {strategy['gpu_requirement']}")
        print(f"    内存效率: {strategy['memory_efficiency']}, 通信开销: {strategy['communication_overhead']}")
    
    # 显示支持的特性
    print(f"\n🎯 支持的特性:")
    for feature in info['features']:
        print(f"  • {feature}")
    
    # 运行分布式训练演示
    if DISTRIBUTED_AVAILABLE and distributed_demo:
        print(f"\n" + "="*60)
        print("🔧 分布式训练演示")
        print("="*60)
        try:
            distributed_demo()
        except Exception as e:
            print(f"❌ 分布式训练演示失败: {e}")
    
    print(f"\n" + "="*60)
    print("🎉 分布式训练系统演示完成!")
    print("="*60)


if __name__ == "__main__":
    demo()