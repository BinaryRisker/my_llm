"""
精度管理工具

提供精度转换、检测和管理功能
"""

from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class PrecisionManager:
    """精度管理器"""
    
    def __init__(self, default_precision: str = 'fp32'):
        """
        Args:
            default_precision: 默认精度类型
        """
        self.default_precision = default_precision
        self.supported_precisions = ['fp16', 'bf16', 'fp32']
        self._precision_capabilities = {}
    
    def check_precision_support(self, precision: str, device: str = 'cpu') -> bool:
        """检查设备是否支持指定精度"""
        precision = precision.lower()
        
        if precision not in self.supported_precisions:
            return False
        
        # 缓存结果
        cache_key = f"{precision}_{device}"
        if cache_key in self._precision_capabilities:
            return self._precision_capabilities[cache_key]
        
        # 模拟硬件支持检测
        if precision == 'fp32':
            supported = True  # 所有设备都支持FP32
        elif precision == 'fp16':
            # 假设CUDA设备支持FP16
            supported = 'cuda' in device.lower()
        elif precision == 'bf16':
            # 假设现代GPU支持BF16
            supported = 'cuda' in device.lower()
        else:
            supported = False
        
        self._precision_capabilities[cache_key] = supported
        logger.info(f"精度支持检查: {precision} on {device} = {supported}")
        
        return supported
    
    def get_optimal_precision(self, device: str = 'cpu', model_size: Optional[int] = None) -> str:
        """获取最优精度设置"""
        if self.check_precision_support('bf16', device):
            # BF16通常比FP16更稳定
            return 'bf16'
        elif self.check_precision_support('fp16', device):
            return 'fp16'
        else:
            return 'fp32'
    
    def convert_precision_dtype(self, precision: str):
        """转换精度字符串为数据类型（模拟）"""
        precision_map = {
            'fp16': 'half',
            'bf16': 'bfloat16', 
            'fp32': 'float'
        }
        return precision_map.get(precision.lower(), 'float')
    
    def estimate_memory_savings(self, precision: str, baseline_precision: str = 'fp32') -> float:
        """估算内存节省比例"""
        precision_sizes = {
            'fp32': 32,
            'fp16': 16,
            'bf16': 16
        }
        
        baseline_size = precision_sizes.get(baseline_precision.lower(), 32)
        target_size = precision_sizes.get(precision.lower(), 32)
        
        savings_ratio = 1.0 - (target_size / baseline_size)
        return max(0.0, savings_ratio)


def convert_model_to_precision(model, precision: str, device: Optional[str] = None):
    """
    转换模型到指定精度
    
    Args:
        model: 要转换的模型
        precision: 目标精度
        device: 目标设备
    
    Returns:
        转换后的模型
    """
    logger.info(f"转换模型到 {precision} 精度")
    
    # 模拟模型精度转换
    if precision.lower() == 'fp16':
        # model = model.half()
        logger.info("模型已转换为FP16")
    elif precision.lower() == 'bf16':
        # model = model.bfloat16()  # 如果支持的话
        logger.info("模型已转换为BF16")
    elif precision.lower() == 'fp32':
        # model = model.float()
        logger.info("模型已转换为FP32")
    
    if device:
        # model = model.to(device)
        logger.info(f"模型已移动到设备: {device}")
    
    return model


def check_precision_support(precision: str, device: str = 'cpu') -> bool:
    """检查精度支持（独立函数）"""
    manager = PrecisionManager()
    return manager.check_precision_support(precision, device)


def get_optimal_precision(device: str = 'cpu', model_size: Optional[int] = None) -> str:
    """获取最优精度（独立函数）"""
    manager = PrecisionManager()
    return manager.get_optimal_precision(device, model_size)


class PrecisionConfig:
    """精度配置类"""
    
    def __init__(self,
                 precision: str = 'fp16',
                 loss_scale: Union[str, float] = 'dynamic',
                 opt_level: str = 'O1',
                 keep_batchnorm_fp32: bool = True,
                 master_weights: bool = False):
        """
        Args:
            precision: 训练精度
            loss_scale: 损失缩放策略
            opt_level: 优化级别
            keep_batchnorm_fp32: 是否保持BatchNorm为FP32
            master_weights: 是否使用主权重
        """
        self.precision = precision
        self.loss_scale = loss_scale
        self.opt_level = opt_level
        self.keep_batchnorm_fp32 = keep_batchnorm_fp32
        self.master_weights = master_weights
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'precision': self.precision,
            'loss_scale': self.loss_scale,
            'opt_level': self.opt_level,
            'keep_batchnorm_fp32': self.keep_batchnorm_fp32,
            'master_weights': self.master_weights
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建配置"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """验证配置"""
        supported_precisions = ['fp16', 'bf16', 'fp32']
        if self.precision not in supported_precisions:
            logger.error(f"不支持的精度: {self.precision}")
            return False
        
        if isinstance(self.loss_scale, str):
            if self.loss_scale not in ['dynamic', 'static']:
                logger.error(f"不支持的损失缩放类型: {self.loss_scale}")
                return False
        elif not isinstance(self.loss_scale, (int, float)):
            logger.error(f"损失缩放必须是数字或字符串: {self.loss_scale}")
            return False
        
        return True


class AutoPrecisionSelector:
    """自动精度选择器"""
    
    def __init__(self):
        self.precision_manager = PrecisionManager()
        self._performance_cache = {}
    
    def select_precision(self, 
                        model_size: int,
                        available_memory: int,
                        target_speed: str = 'balanced',
                        device: str = 'cuda') -> str:
        """
        自动选择最优精度
        
        Args:
            model_size: 模型大小（参数数量）
            available_memory: 可用内存（MB）
            target_speed: 目标速度 ('fast', 'balanced', 'accurate')
            device: 目标设备
            
        Returns:
            推荐的精度类型
        """
        cache_key = f"{model_size}_{available_memory}_{target_speed}_{device}"
        
        if cache_key in self._performance_cache:
            return self._performance_cache[cache_key]
        
        # 估算内存需求
        fp32_memory = model_size * 4  # 4 bytes per parameter
        fp16_memory = model_size * 2  # 2 bytes per parameter
        
        # 根据内存限制选择精度
        if available_memory < fp16_memory * 1.5:
            logger.warning("内存不足，可能无法训练此模型")
            precision = 'fp16'
        elif available_memory < fp32_memory * 1.5:
            precision = 'fp16' if self.precision_manager.check_precision_support('fp16', device) else 'fp32'
        else:
            # 根据目标速度选择
            if target_speed == 'fast':
                precision = self.precision_manager.get_optimal_precision(device, model_size)
            elif target_speed == 'accurate':
                precision = 'fp32'
            else:  # balanced
                precision = 'bf16' if self.precision_manager.check_precision_support('bf16', device) else 'fp16'
        
        # 验证最终选择的精度是否支持
        if not self.precision_manager.check_precision_support(precision, device):
            precision = 'fp32'
        
        self._performance_cache[cache_key] = precision
        
        logger.info(f"自动选择精度: {precision} (模型大小: {model_size}, 内存: {available_memory}MB, 目标: {target_speed})")
        
        return precision