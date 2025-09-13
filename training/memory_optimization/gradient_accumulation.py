"""
梯度累积实现

允许使用更大的有效批量大小而不增加内存使用
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class AccumulationConfig:
    """梯度累积配置"""
    
    accumulation_steps: int = 4  # 累积步数
    sync_gradients: bool = True  # 是否同步梯度
    average_gradients: bool = True  # 是否平均梯度
    clip_gradients: bool = True  # 是否裁剪梯度
    max_grad_norm: float = 1.0  # 最大梯度范数
    
    def __post_init__(self):
        if self.accumulation_steps <= 0:
            raise ValueError(f"accumulation_steps必须大于0，得到: {self.accumulation_steps}")


class GradientAccumulator:
    """梯度累积器"""
    
    def __init__(self, 
                 config: Optional[AccumulationConfig] = None,
                 model=None,
                 optimizer=None):
        """
        Args:
            config: 累积配置
            model: 训练模型
            optimizer: 优化器
        """
        self.config = config or AccumulationConfig()
        self.model = model
        self.optimizer = optimizer
        
        # 累积状态
        self.current_step = 0
        self.accumulated_steps = 0
        self.total_loss = 0.0
        self.step_losses = []
        
        # 统计信息
        self.stats = {
            'total_accumulation_cycles': 0,
            'average_loss_per_cycle': 0.0,
            'gradient_norms': [],
            'effective_batch_size': 0
        }
        
        logger.info(f"梯度累积器初始化: 累积步数={self.config.accumulation_steps}")
    
    def accumulate_gradients(self, 
                           loss,
                           backward_kwargs: Optional[Dict] = None) -> bool:
        """
        累积梯度
        
        Args:
            loss: 当前步的损失
            backward_kwargs: 反向传播参数
            
        Returns:
            是否应该执行优化器步骤
        """
        backward_kwargs = backward_kwargs or {}
        
        # 累积损失
        self.step_losses.append(float(loss))
        self.total_loss += float(loss)
        self.current_step += 1
        
        # 标准化损失（除以累积步数）
        if self.config.average_gradients:
            scaled_loss = loss / self.config.accumulation_steps
        else:
            scaled_loss = loss
        
        # 反向传播
        # 注意：在最后一步之前保留计算图
        retain_graph = self.current_step < self.config.accumulation_steps
        scaled_loss.backward(retain_graph=retain_graph, **backward_kwargs)
        
        # 检查是否达到累积步数
        if self.current_step >= self.config.accumulation_steps:
            return True
        
        return False
    
    def should_sync_gradients(self) -> bool:
        """检查是否应该同步梯度"""
        return (self.config.sync_gradients and 
                self.current_step >= self.config.accumulation_steps)
    
    def step_optimizer(self, 
                      grad_scaler=None,
                      scheduler=None) -> Dict[str, Any]:
        """
        执行优化器步骤
        
        Args:
            grad_scaler: 梯度缩放器（用于混合精度训练）
            scheduler: 学习率调度器
            
        Returns:
            步骤统计信息
        """
        if self.current_step < self.config.accumulation_steps:
            logger.warning("尚未达到累积步数，不应执行优化器步骤")
            return {}
        
        # 梯度裁剪
        grad_norm = None
        if self.config.clip_gradients and self.model:
            grad_norm = self._clip_gradients()
        
        # 优化器步骤
        if grad_scaler:
            # 混合精度训练
            grad_scaler.step(self.optimizer)
            grad_scaler.update()
        else:
            # 标准训练
            self.optimizer.step()
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 计算统计信息
        avg_loss = self.total_loss / self.config.accumulation_steps if self.step_losses else 0.0
        
        step_info = {
            'accumulated_steps': self.current_step,
            'average_loss': avg_loss,
            'total_loss': self.total_loss,
            'step_losses': self.step_losses.copy(),
            'gradient_norm': grad_norm,
            'effective_batch_size': self.stats['effective_batch_size']
        }
        
        # 更新统计信息
        self._update_statistics(avg_loss, grad_norm)
        
        # 重置累积状态
        self._reset_accumulation()
        
        return step_info
    
    def _clip_gradients(self) -> float:
        """裁剪梯度并返回梯度范数"""
        if not self.model:
            return 0.0
        
        # 收集所有梯度
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        
        if not parameters:
            return 0.0
        
        # 计算梯度范数（简化实现）
        total_norm = 0.0
        for param in parameters:
            # 简化的范数计算
            param_norm = sum(x * x for x in param.grad.data.view(-1)) ** 0.5
            total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        # 裁剪梯度
        if total_norm > self.config.max_grad_norm:
            clip_coef = self.config.max_grad_norm / total_norm
            for param in parameters:
                param.grad.data.mul_(clip_coef)
            
            logger.debug(f"梯度裁剪: {total_norm:.4f} -> {self.config.max_grad_norm:.4f}")
        
        return float(total_norm)
    
    def _update_statistics(self, avg_loss: float, grad_norm: Optional[float]):
        """更新统计信息"""
        self.stats['total_accumulation_cycles'] += 1
        
        # 更新平均损失
        total_cycles = self.stats['total_accumulation_cycles']
        current_avg = self.stats['average_loss_per_cycle']
        self.stats['average_loss_per_cycle'] = (
            (current_avg * (total_cycles - 1) + avg_loss) / total_cycles
        )
        
        # 记录梯度范数
        if grad_norm is not None:
            self.stats['gradient_norms'].append(grad_norm)
            
            # 保持最近1000个记录
            if len(self.stats['gradient_norms']) > 1000:
                self.stats['gradient_norms'] = self.stats['gradient_norms'][-1000:]
    
    def _reset_accumulation(self):
        """重置累积状态"""
        self.current_step = 0
        self.total_loss = 0.0
        self.step_losses = []
    
    def set_effective_batch_size(self, batch_size: int):
        """设置有效批量大小"""
        self.stats['effective_batch_size'] = batch_size * self.config.accumulation_steps
        logger.info(f"有效批量大小: {self.stats['effective_batch_size']}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 添加梯度范数统计
        if self.stats['gradient_norms']:
            stats['avg_gradient_norm'] = sum(self.stats['gradient_norms']) / len(self.stats['gradient_norms'])
            stats['max_gradient_norm'] = max(self.stats['gradient_norms'])
            stats['min_gradient_norm'] = min(self.stats['gradient_norms'])
        
        # 添加当前状态
        stats.update({
            'current_accumulation_step': self.current_step,
            'is_accumulation_complete': self.current_step >= self.config.accumulation_steps,
            'config': {
                'accumulation_steps': self.config.accumulation_steps,
                'sync_gradients': self.config.sync_gradients,
                'average_gradients': self.config.average_gradients,
                'clip_gradients': self.config.clip_gradients,
                'max_grad_norm': self.config.max_grad_norm
            }
        })
        
        return stats
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'config': {
                'accumulation_steps': self.config.accumulation_steps,
                'sync_gradients': self.config.sync_gradients,
                'average_gradients': self.config.average_gradients,
                'clip_gradients': self.config.clip_gradients,
                'max_grad_norm': self.config.max_grad_norm
            },
            'current_step': self.current_step,
            'total_loss': self.total_loss,
            'step_losses': self.step_losses,
            'stats': self.stats
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        # 加载配置
        if 'config' in state_dict:
            config_dict = state_dict['config']
            self.config = AccumulationConfig(**config_dict)
        
        # 加载状态
        self.current_step = state_dict.get('current_step', 0)
        self.total_loss = state_dict.get('total_loss', 0.0)
        self.step_losses = state_dict.get('step_losses', [])
        self.stats = state_dict.get('stats', {
            'total_accumulation_cycles': 0,
            'average_loss_per_cycle': 0.0,
            'gradient_norms': [],
            'effective_batch_size': 0
        })
    
    def reset(self):
        """重置累积器"""
        self._reset_accumulation()
        self.stats = {
            'total_accumulation_cycles': 0,
            'average_loss_per_cycle': 0.0,
            'gradient_norms': [],
            'effective_batch_size': 0
        }
        logger.info("梯度累积器已重置")


class AdaptiveGradientAccumulator(GradientAccumulator):
    """自适应梯度累积器"""
    
    def __init__(self,
                 config: Optional[AccumulationConfig] = None,
                 model=None,
                 optimizer=None,
                 target_memory_usage: float = 0.8):
        """
        Args:
            target_memory_usage: 目标内存使用率 (0.0-1.0)
        """
        super().__init__(config, model, optimizer)
        self.target_memory_usage = target_memory_usage
        self.memory_monitor = None  # 将在实际使用中初始化
        
        # 自适应参数
        self.min_accumulation_steps = 1
        self.max_accumulation_steps = 16
        self.adaptation_history = []
    
    def adapt_accumulation_steps(self, current_memory_usage: float):
        """根据内存使用情况调整累积步数"""
        if current_memory_usage > self.target_memory_usage:
            # 内存使用过高，增加累积步数（减少批量大小）
            new_steps = min(
                self.config.accumulation_steps + 1,
                self.max_accumulation_steps
            )
        elif current_memory_usage < self.target_memory_usage * 0.7:
            # 内存使用较低，减少累积步数（增加批量大小）
            new_steps = max(
                self.config.accumulation_steps - 1,
                self.min_accumulation_steps
            )
        else:
            # 内存使用在合理范围内
            return
        
        if new_steps != self.config.accumulation_steps:
            logger.info(f"自适应调整累积步数: {self.config.accumulation_steps} -> {new_steps}")
            self.config.accumulation_steps = new_steps
            self.adaptation_history.append({
                'old_steps': self.config.accumulation_steps,
                'new_steps': new_steps,
                'memory_usage': current_memory_usage
            })