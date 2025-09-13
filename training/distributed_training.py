"""
分布式训练支持模块
================

实现分布式训练的核心功能，包括：
- 数据并行（Data Parallelism）
- 模型并行（Model Parallelism）
- 分布式数据加载
- 多GPU/多节点训练支持
- 梯度同步和通信

支持的分布式策略：
- PyTorch DDP (DistributedDataParallel)
- 模型分片 (Model Sharding)
- 管道并行 (Pipeline Parallelism)
- 混合精度训练

使用方法:
    from training.distributed_training import DistributedTrainer
    
    trainer = DistributedTrainer(model, strategy='ddp')
    trainer.setup_distributed()
    trainer.train(train_loader)
"""

import os
import logging
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = dist = DDP = DataLoader = DistributedSampler = None
    
    # 创建模拟类用于类型检查
    class MockModule:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def __getattr__(self, name):
            return MockModule()
    
    class MockTorch:
        class nn:
            Module = MockModule
        class optim:
            Optimizer = MockModule
    
    if torch is None:
        torch = MockTorch()
        nn = torch.nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """分布式训练配置"""
    backend: str = "nccl"  # 通信后端：nccl, gloo, mpi
    world_size: int = 1    # 总进程数
    rank: int = 0          # 当前进程的rank
    local_rank: int = 0    # 本地GPU的rank
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # 训练配置
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 优化配置
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    
    # 调试配置
    debug: bool = False
    profile: bool = False


class BaseDistributedStrategy(ABC):
    """基础分布式策略类"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    def setup(self) -> None:
        """初始化分布式环境"""
        pass
    
    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """包装模型以支持分布式训练"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理分布式环境"""
        pass
    
    @abstractmethod
    def is_main_process(self) -> bool:
        """检查是否为主进程"""
        pass
    
    @abstractmethod
    def synchronize(self) -> None:
        """同步所有进程"""
        pass


class DataParallelStrategy(BaseDistributedStrategy):
    """数据并行策略（DDP）"""
    
    def __init__(self, config: DistributedConfig):
        super().__init__(config)
        self.name = "data_parallel"
    
    def setup(self) -> None:
        """初始化DDP环境"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用，无法使用分布式训练")
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
        
        # 设置当前设备
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
        
        self.is_initialized = True
        logger.info(f"DDP初始化完成 - Rank: {self.config.rank}, World Size: {self.config.world_size}")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """用DDP包装模型"""
        if not self.is_initialized:
            raise RuntimeError("分布式环境未初始化")
        
        # 将模型移到正确的设备
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.local_rank}')
            model = model.to(device)
        
        # 包装为DDP
        ddp_model = DDP(
            model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
            bucket_cap_mb=self.config.bucket_cap_mb
        )
        
        logger.info(f"模型已包装为DDP - Device: {device if torch.cuda.is_available() else 'cpu'}")
        return ddp_model
    
    def cleanup(self) -> None:
        """清理DDP环境"""
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info("DDP环境已清理")
    
    def is_main_process(self) -> bool:
        """检查是否为主进程（rank 0）"""
        return self.config.rank == 0
    
    def synchronize(self) -> None:
        """同步所有进程"""
        if dist.is_initialized():
            dist.barrier()


class ModelParallelStrategy(BaseDistributedStrategy):
    """模型并行策略"""
    
    def __init__(self, config: DistributedConfig):
        super().__init__(config)
        self.name = "model_parallel"
        self.device_map = {}
    
    def setup(self) -> None:
        """初始化模型并行环境"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用，无法使用分布式训练")
        
        # 检查可用GPU数量
        if not torch.cuda.is_available():
            raise RuntimeError("模型并行需要CUDA支持")
        
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            logger.warning("模型并行建议使用多个GPU")
        
        self.is_initialized = True
        logger.info(f"模型并行初始化完成 - 可用GPU数: {gpu_count}")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """为模型并行分割模型"""
        if not self.is_initialized:
            raise RuntimeError("分布式环境未初始化")
        
        # 简化的模型并行实现
        # 实际实现中需要根据模型结构进行更复杂的分割
        gpu_count = torch.cuda.device_count()
        
        if hasattr(model, 'layers') and hasattr(model.layers, '__len__'):
            # 假设模型有layers属性
            layers_per_gpu = len(model.layers) // gpu_count
            
            for i, layer in enumerate(model.layers):
                device_id = min(i // layers_per_gpu, gpu_count - 1)
                layer.to(f'cuda:{device_id}')
                self.device_map[f'layer_{i}'] = device_id
            
            logger.info(f"模型已分割到{gpu_count}个GPU上")
        else:
            # 简单地将整个模型放到第一个GPU
            model.to('cuda:0')
            logger.info("模型放置在单个GPU上（未进行层级分割）")
        
        return model
    
    def cleanup(self) -> None:
        """清理模型并行环境"""
        self.device_map.clear()
        logger.info("模型并行环境已清理")
    
    def is_main_process(self) -> bool:
        """模型并行中总是返回True"""
        return True
    
    def synchronize(self) -> None:
        """模型并行不需要进程同步"""
        pass


class PipelineParallelStrategy(BaseDistributedStrategy):
    """管道并行策略"""
    
    def __init__(self, config: DistributedConfig, num_stages: int = 2):
        super().__init__(config)
        self.name = "pipeline_parallel"
        self.num_stages = num_stages
        self.stage_id = config.rank % num_stages
    
    def setup(self) -> None:
        """初始化管道并行环境"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用，无法使用分布式训练")
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
        
        self.is_initialized = True
        logger.info(f"管道并行初始化完成 - Stage: {self.stage_id}/{self.num_stages}")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """为管道并行分割模型"""
        if not self.is_initialized:
            raise RuntimeError("分布式环境未初始化")
        
        # 简化的管道并行实现
        # 实际实现需要更复杂的模型分割和中间激活传递
        
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.local_rank}')
            model = model.to(device)
        
        # 这里应该实现实际的管道分割逻辑
        # 目前只是简单地将整个模型放在指定设备上
        
        logger.info(f"管道阶段{self.stage_id}的模型已准备")
        return model
    
    def cleanup(self) -> None:
        """清理管道并行环境"""
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info("管道并行环境已清理")
    
    def is_main_process(self) -> bool:
        """检查是否为主进程"""
        return self.config.rank == 0
    
    def synchronize(self) -> None:
        """同步所有进程"""
        if dist.is_initialized():
            dist.barrier()


class DistributedDataLoader:
    """分布式数据加载器"""
    
    def __init__(self, 
                 dataset,
                 batch_size: int,
                 config: DistributedConfig,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.config = config
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.sampler = None
        self.dataloader = None
    
    def setup(self) -> DataLoader:
        """创建分布式数据加载器"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用")
        
        # 创建分布式采样器
        if self.config.world_size > 1:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=self.shuffle
            )
            shuffle = False  # 使用采样器时不能同时shuffle
        else:
            self.sampler = None
            shuffle = self.shuffle
        
        # 创建数据加载器
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            drop_last=True  # 确保所有进程的batch数量相同
        )
        
        logger.info(f"分布式数据加载器已创建 - Batch Size: {self.batch_size}, Workers: {self.num_workers}")
        return self.dataloader
    
    def set_epoch(self, epoch: int) -> None:
        """设置epoch（用于分布式采样器）"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)


class DistributedTrainer:
    """分布式训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: DistributedConfig,
                 strategy: str = "ddp"):
        self.model = model
        self.config = config
        self.strategy_name = strategy
        
        # 创建分布式策略
        self.strategy = self._create_strategy(strategy, config)
        self.wrapped_model = None
        
        # 训练状态
        self.is_setup = False
    
    def _create_strategy(self, strategy: str, config: DistributedConfig) -> BaseDistributedStrategy:
        """创建分布式策略"""
        strategies = {
            "ddp": DataParallelStrategy,
            "data_parallel": DataParallelStrategy,
            "model_parallel": ModelParallelStrategy,
            "pipeline_parallel": PipelineParallelStrategy,
        }
        
        if strategy not in strategies:
            raise ValueError(f"不支持的分布式策略: {strategy}. 可用策略: {list(strategies.keys())}")
        
        return strategies[strategy](config)
    
    def setup(self) -> None:
        """设置分布式训练环境"""
        # 初始化分布式策略
        self.strategy.setup()
        
        # 包装模型
        self.wrapped_model = self.strategy.wrap_model(self.model)
        
        self.is_setup = True
        logger.info(f"分布式训练器已设置 - 策略: {self.strategy.name}")
    
    def train_step(self, 
                   batch: Dict[str, Any],
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   scaler: Optional[Any] = None) -> Dict[str, float]:
        """单步训练"""
        if not self.is_setup:
            raise RuntimeError("分布式训练器未设置")
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用")
        
        self.wrapped_model.train()
        
        # 前向传播
        if scaler is not None:
            # 混合精度训练
            with torch.cuda.amp.autocast():
                outputs = self.wrapped_model(**batch)
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    # 假设第一个输出是logits
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    labels = batch.get('labels')
                    loss = criterion(logits, labels) if labels is not None else logits.mean()
            
            # 缩放损失
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            # 梯度裁剪
            if self.config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.wrapped_model.parameters(), self.config.max_grad_norm)
            
            # 优化器步骤
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练
            outputs = self.wrapped_model(**batch)
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                labels = batch.get('labels')
                loss = criterion(logits, labels) if labels is not None else logits.mean()
            
            loss.backward()
            
            # 梯度裁剪
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.wrapped_model.parameters(), self.config.max_grad_norm)
            
            optimizer.step()
        
        optimizer.zero_grad()
        
        return {"loss": loss.item()}
    
    def evaluate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """单步评估"""
        if not self.is_setup:
            raise RuntimeError("分布式训练器未设置")
        
        self.wrapped_model.eval()
        
        with torch.no_grad():
            outputs = self.wrapped_model(**batch)
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = loss.mean()  # 简化处理
        
        return {"loss": loss.item()}
    
    def save_checkpoint(self, filepath: str, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        """保存检查点"""
        if not self.strategy.is_main_process():
            return  # 只有主进程保存
        
        if not TORCH_AVAILABLE:
            return
        
        # 获取原始模型状态（去除DDP包装）
        model_state = self.wrapped_model.module.state_dict() \
                     if hasattr(self.wrapped_model, 'module') \
                     else self.wrapped_model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str, optimizer: torch.optim.Optimizer) -> int:
        """加载检查点"""
        if not TORCH_AVAILABLE:
            return 0
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 加载模型状态
        if hasattr(self.wrapped_model, 'module'):
            self.wrapped_model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.wrapped_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"检查点已加载: {filepath}, Epoch: {epoch}")
        
        return epoch
    
    def cleanup(self) -> None:
        """清理分布式环境"""
        if self.strategy:
            self.strategy.cleanup()
        logger.info("分布式训练器已清理")


def demo():
    """分布式训练演示"""
    print("🚀 分布式训练系统演示")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch未安装，无法运行演示")
        return
    
    # 创建配置
    config = DistributedConfig(
        world_size=1,  # 单进程演示
        rank=0,
        batch_size=16,
        gradient_accumulation_steps=2
    )
    
    print("📋 分布式配置:")
    print(f"  后端: {config.backend}")
    print(f"  世界大小: {config.world_size}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  梯度累积步数: {config.gradient_accumulation_steps}")
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    print(f"\n🧮 模型信息:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数数: {total_params:,}")
    
    # 演示不同的分布式策略
    strategies = ["ddp", "model_parallel", "pipeline_parallel"]
    
    for strategy_name in strategies:
        print(f"\n🔧 {strategy_name.upper()} 策略演示:")
        print("-" * 30)
        
        try:
            # 创建训练器
            trainer = DistributedTrainer(model, config, strategy=strategy_name)
            
            # 由于是单进程演示，跳过实际的分布式设置
            print(f"  ✅ {strategy_name} 训练器已创建")
            print(f"  策略名称: {trainer.strategy.name}")
            print(f"  是否为主进程: {trainer.strategy.is_main_process()}")
            
            # 清理
            trainer.cleanup()
            
        except Exception as e:
            print(f"  ⚠️ {strategy_name} 演示失败: {e}")
    
    # 演示分布式数据加载器
    print(f"\n📊 分布式数据加载器演示:")
    print("-" * 30)
    
    # 创建模拟数据集
    class MockDataset:
        def __init__(self, size=1000):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randn(50),
                'labels': torch.randint(0, 10, (1,)).item()
            }
    
    dataset = MockDataset(1000)
    
    try:
        dist_loader = DistributedDataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            config=config,
            num_workers=0  # 避免Windows上的多进程问题
        )
        
        print(f"  ✅ 分布式数据加载器已创建")
        print(f"  数据集大小: {len(dataset)}")
        print(f"  批次大小: {config.batch_size}")
        print(f"  世界大小: {config.world_size}")
        
        # 实际创建数据加载器（在单进程环境下）
        dataloader = dist_loader.setup()
        print(f"  批次数量: {len(dataloader)}")
        
    except Exception as e:
        print(f"  ⚠️ 数据加载器演示失败: {e}")
    
    print(f"\n🎯 分布式训练优势:")
    print("  • 数据并行：提高训练吞吐量")
    print("  • 模型并行：支持超大模型训练")
    print("  • 管道并行：减少内存占用")
    print("  • 混合精度：加速训练过程")
    print("  • 梯度同步：保证训练一致性")
    
    print(f"\n🎉 分布式训练演示完成!")


if __name__ == "__main__":
    demo()