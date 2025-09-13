"""
自动混合精度训练器

提供完整的AMP训练功能，包括配置管理和训练循环优化
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import logging

from .loss_scaler import LossScalerManager, create_loss_scaler
from .gradient_clipper import GradientClipper
from .precision_utils import PrecisionConfig, PrecisionManager

logger = logging.getLogger(__name__)


@dataclass
class AMPConfig:
    """AMP配置类"""
    
    # 基础精度设置
    precision: str = 'fp16'  # 'fp16', 'bf16', 'fp32'
    enabled: bool = True
    
    # 损失缩放设置
    loss_scale: str = 'dynamic'  # 'dynamic', 'static', 'adaptive'
    initial_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    # 梯度裁剪设置
    max_grad_norm: Optional[float] = 1.0
    grad_clip_type: str = 'norm'  # 'norm', 'adaptive', 'layerwise'
    
    # 模型设置
    keep_batchnorm_fp32: bool = True
    master_weights: bool = False
    
    # 优化设置
    opt_level: str = 'O1'  # O0, O1, O2, O3
    
    def __post_init__(self):
        """验证配置"""
        if self.precision not in ['fp16', 'bf16', 'fp32']:
            raise ValueError(f"不支持的精度: {self.precision}")
        
        if self.loss_scale not in ['dynamic', 'static', 'adaptive']:
            raise ValueError(f"不支持的损失缩放类型: {self.loss_scale}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'precision': self.precision,
            'enabled': self.enabled,
            'loss_scale': self.loss_scale,
            'initial_scale': self.initial_scale,
            'growth_factor': self.growth_factor,
            'backoff_factor': self.backoff_factor,
            'growth_interval': self.growth_interval,
            'max_grad_norm': self.max_grad_norm,
            'grad_clip_type': self.grad_clip_type,
            'keep_batchnorm_fp32': self.keep_batchnorm_fp32,
            'master_weights': self.master_weights,
            'opt_level': self.opt_level
        }


class AMPTrainer:
    """自动混合精度训练器"""
    
    def __init__(self, 
                 config: Optional[AMPConfig] = None,
                 model=None,
                 optimizer=None):
        """
        Args:
            config: AMP配置
            model: 训练模型
            optimizer: 优化器
        """
        self.config = config or AMPConfig()
        self.model = model
        self.optimizer = optimizer
        
        # 初始化组件
        self._initialize_components()
        
        # 训练统计
        self.step_count = 0
        self.overflow_count = 0
        self.stats = {
            'successful_steps': 0,
            'overflow_steps': 0,
            'gradient_norms': [],
            'loss_scales': []
        }
    
    def _initialize_components(self):
        """初始化AMP组件"""
        # 损失缩放器
        scaler_config = {
            'initial_scale': self.config.initial_scale,
            'growth_factor': self.config.growth_factor,
            'backoff_factor': self.config.backoff_factor,
            'growth_interval': self.config.growth_interval
        }
        
        scaler = create_loss_scaler(self.config.loss_scale, **scaler_config)
        self.loss_scaler = LossScalerManager(scaler)
        
        # 梯度裁剪器
        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
            if self.config.grad_clip_type == 'adaptive':
                from .gradient_clipper import AdaptiveGradientClipper
                self.grad_clipper = AdaptiveGradientClipper(
                    initial_max_norm=self.config.max_grad_norm
                )
            else:
                self.grad_clipper = GradientClipper(
                    max_norm=self.config.max_grad_norm
                )
        else:
            self.grad_clipper = None
        
        # 精度管理器
        self.precision_manager = PrecisionManager(self.config.precision)
        
        logger.info(f"AMP训练器初始化: {self.config.precision}, 损失缩放: {self.config.loss_scale}")
    
    def prepare_model_and_optimizer(self, model, optimizer):
        """准备模型和优化器"""
        self.model = model
        self.optimizer = optimizer
        
        if not self.config.enabled:
            return model, optimizer
        
        # 转换模型精度
        if self.config.precision != 'fp32':
            # 在实际使用中会调用模型的精度转换方法
            logger.info(f"模型已准备为 {self.config.precision} 精度")
        
        return model, optimizer
    
    def training_step(self, loss, retain_graph: bool = False) -> bool:
        """
        执行训练步骤
        
        Args:
            loss: 训练损失
            retain_graph: 是否保留计算图
            
        Returns:
            是否成功执行步骤
        """
        if not self.config.enabled:
            # 标准训练步骤
            loss.backward(retain_graph=retain_graph)
            if self.grad_clipper and self.model:
                self.grad_clipper.clip_gradients(self.model.parameters())
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True
        
        # AMP训练步骤
        self.step_count += 1
        
        # 1. 反向传播和梯度反缩放
        success = self.loss_scaler.backward_and_unscale(loss, self.optimizer)
        
        if not success:
            # 发生溢出
            self.overflow_count += 1
            self.stats['overflow_steps'] += 1
            self.optimizer.zero_grad()
            logger.debug(f"Step {self.step_count}: 梯度溢出，跳过更新")
            return False
        
        # 2. 梯度裁剪
        if self.grad_clipper and self.model:
            grad_norm = self.grad_clipper.clip_gradients(self.model.parameters())
            self.stats['gradient_norms'].append(grad_norm)
        
        # 3. 优化器步骤和损失缩放更新
        step_success = self.loss_scaler.step_and_update(self.optimizer)
        
        # 4. 清零梯度
        self.optimizer.zero_grad()
        
        # 5. 更新统计信息
        if step_success:
            self.stats['successful_steps'] += 1
        
        scale = self.loss_scaler.get_scale()
        self.stats['loss_scales'].append(scale)
        
        return step_success
    
    def scale_loss(self, loss):
        """缩放损失"""
        if not self.config.enabled:
            return loss
        return self.loss_scaler.scale_loss(loss)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        stats = self.stats.copy()
        
        # 添加汇总信息
        stats.update({
            'total_steps': self.step_count,
            'overflow_rate': self.overflow_count / max(self.step_count, 1),
            'current_scale': self.loss_scaler.get_scale(),
            'config': self.config.to_dict()
        })
        
        # 梯度统计
        if self.stats['gradient_norms']:
            stats['avg_grad_norm'] = sum(self.stats['gradient_norms']) / len(self.stats['gradient_norms'])
            stats['max_grad_norm'] = max(self.stats['gradient_norms'])
        
        # 损失缩放统计
        if self.stats['loss_scales']:
            stats['avg_loss_scale'] = sum(self.stats['loss_scales']) / len(self.stats['loss_scales'])
        
        # 梯度裁剪统计
        if self.grad_clipper:
            stats['gradient_clip_stats'] = self.grad_clipper.get_statistics()
        
        return stats
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = {
            'config': self.config.to_dict(),
            'step_count': self.step_count,
            'overflow_count': self.overflow_count,
            'stats': self.stats,
            'loss_scaler': self.loss_scaler.state_dict()
        }
        
        if self.grad_clipper and hasattr(self.grad_clipper, 'state_dict'):
            state['grad_clipper'] = self.grad_clipper.__dict__
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.step_count = state_dict.get('step_count', 0)
        self.overflow_count = state_dict.get('overflow_count', 0)
        self.stats = state_dict.get('stats', {
            'successful_steps': 0,
            'overflow_steps': 0,
            'gradient_norms': [],
            'loss_scales': []
        })
        
        if 'loss_scaler' in state_dict:
            self.loss_scaler.load_state_dict(state_dict['loss_scaler'])
        
        if 'grad_clipper' in state_dict and self.grad_clipper:
            self.grad_clipper.__dict__.update(state_dict['grad_clipper'])
    
    def enable(self):
        """启用AMP训练"""
        self.config.enabled = True
        self.loss_scaler.enabled = True
        logger.info("AMP训练已启用")
    
    def disable(self):
        """禁用AMP训练"""
        self.config.enabled = False
        self.loss_scaler.enabled = False
        logger.info("AMP训练已禁用")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.step_count = 0
        self.overflow_count = 0
        self.stats = {
            'successful_steps': 0,
            'overflow_steps': 0,
            'gradient_norms': [],
            'loss_scales': []
        }
        logger.info("AMP统计信息已重置")