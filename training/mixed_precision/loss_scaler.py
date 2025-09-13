"""
损失缩放器实现

提供动态和静态损失缩放功能，用于FP16训练中的数值稳定性
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class LossScaler(ABC):
    """损失缩放器基类"""
    
    def __init__(self):
        self.enabled = True
        
    @abstractmethod
    def scale_loss(self, loss) -> Any:
        """缩放损失"""
        pass
    
    @abstractmethod
    def unscale_gradients(self, optimizer) -> None:
        """反缩放梯度"""
        pass
    
    @abstractmethod
    def step(self, optimizer) -> bool:
        """执行优化器步骤"""
        pass
    
    @abstractmethod
    def update(self) -> None:
        """更新缩放因子"""
        pass
    
    @abstractmethod
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {'enabled': self.enabled}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        self.enabled = state_dict.get('enabled', True)


class StaticLossScaler(LossScaler):
    """静态损失缩放器"""
    
    def __init__(self, scale: float = 65536.0):
        """
        Args:
            scale: 固定的损失缩放因子
        """
        super().__init__()
        self.scale_value = scale
        self._has_inf_or_nan = False
    
    def scale_loss(self, loss):
        """缩放损失"""
        if not self.enabled:
            return loss
        return loss * self.scale_value
    
    def unscale_gradients(self, optimizer) -> None:
        """反缩放梯度"""
        if not self.enabled:
            return
        
        self._has_inf_or_nan = False
        
        # 模拟检查无穷大和NaN
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # 反缩放梯度
                    param.grad.data.div_(self.scale_value)
                    
                    # 检查数值异常（简化版本）
                    if hasattr(param.grad, 'isnan') and hasattr(param.grad, 'isinf'):
                        if param.grad.isnan().any() or param.grad.isinf().any():
                            self._has_inf_or_nan = True
                            break
            if self._has_inf_or_nan:
                break
    
    def step(self, optimizer) -> bool:
        """执行优化器步骤"""
        if not self.enabled:
            optimizer.step()
            return True
        
        if not self._has_inf_or_nan:
            optimizer.step()
            return True
        else:
            logger.warning("跳过优化器步骤：检测到inf或nan梯度")
            return False
    
    def update(self) -> None:
        """更新缩放因子（静态缩放器不需要更新）"""
        pass
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        return self.scale_value
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = super().state_dict()
        state.update({
            'scale': self.scale_value,
            'has_inf_or_nan': self._has_inf_or_nan
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        super().load_state_dict(state_dict)
        self.scale_value = state_dict.get('scale', 65536.0)
        self._has_inf_or_nan = state_dict.get('has_inf_or_nan', False)


class DynamicLossScaler(LossScaler):
    """动态损失缩放器"""
    
    def __init__(self, 
                 initial_scale: float = 65536.0,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 min_scale: float = 1e-4,
                 max_scale: float = 65536.0):
        """
        Args:
            initial_scale: 初始缩放因子
            growth_factor: 增长因子
            backoff_factor: 回退因子
            growth_interval: 增长间隔（成功步数）
            min_scale: 最小缩放因子
            max_scale: 最大缩放因子
        """
        super().__init__()
        self.scale_value = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self._growth_tracker = 0
        self._has_inf_or_nan = False
        self._last_overflow_step = -1
        self._step_count = 0
    
    def scale_loss(self, loss):
        """缩放损失"""
        if not self.enabled:
            return loss
        return loss * self.scale_value
    
    def unscale_gradients(self, optimizer) -> None:
        """反缩放梯度"""
        if not self.enabled:
            return
        
        self._has_inf_or_nan = False
        
        # 检查和反缩放梯度
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # 反缩放梯度
                    param.grad.data.div_(self.scale_value)
                    
                    # 检查数值异常（简化版本）
                    if hasattr(param.grad, 'isnan') and hasattr(param.grad, 'isinf'):
                        if param.grad.isnan().any() or param.grad.isinf().any():
                            self._has_inf_or_nan = True
                            break
            if self._has_inf_or_nan:
                break
    
    def step(self, optimizer) -> bool:
        """执行优化器步骤"""
        if not self.enabled:
            optimizer.step()
            return True
        
        self._step_count += 1
        
        if not self._has_inf_or_nan:
            optimizer.step()
            self._growth_tracker += 1
            return True
        else:
            logger.warning(f"Step {self._step_count}: 跳过优化器步骤，检测到inf或nan梯度")
            self._last_overflow_step = self._step_count
            self._growth_tracker = 0
            return False
    
    def update(self) -> None:
        """更新缩放因子"""
        if not self.enabled:
            return
        
        if self._has_inf_or_nan:
            # 发生溢出，减小缩放因子
            self.scale_value = max(
                self.scale_value * self.backoff_factor,
                self.min_scale
            )
            logger.info(f"损失缩放因子减小到: {self.scale_value}")
        elif self._growth_tracker >= self.growth_interval:
            # 成功训练足够步数，增大缩放因子
            self.scale_value = min(
                self.scale_value * self.growth_factor,
                self.max_scale
            )
            self._growth_tracker = 0
            logger.info(f"损失缩放因子增大到: {self.scale_value}")
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        return self.scale_value
    
    def get_growth_tracker(self) -> int:
        """获取增长跟踪器值"""
        return self._growth_tracker
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = super().state_dict()
        state.update({
            'scale': self.scale_value,
            'growth_factor': self.growth_factor,
            'backoff_factor': self.backoff_factor,
            'growth_interval': self.growth_interval,
            'min_scale': self.min_scale,
            'max_scale': self.max_scale,
            'growth_tracker': self._growth_tracker,
            'has_inf_or_nan': self._has_inf_or_nan,
            'last_overflow_step': self._last_overflow_step,
            'step_count': self._step_count
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        super().load_state_dict(state_dict)
        self.scale_value = state_dict.get('scale', 65536.0)
        self.growth_factor = state_dict.get('growth_factor', 2.0)
        self.backoff_factor = state_dict.get('backoff_factor', 0.5)
        self.growth_interval = state_dict.get('growth_interval', 2000)
        self.min_scale = state_dict.get('min_scale', 1e-4)
        self.max_scale = state_dict.get('max_scale', 65536.0)
        self._growth_tracker = state_dict.get('growth_tracker', 0)
        self._has_inf_or_nan = state_dict.get('has_inf_or_nan', False)
        self._last_overflow_step = state_dict.get('last_overflow_step', -1)
        self._step_count = state_dict.get('step_count', 0)


class AdaptiveLossScaler(DynamicLossScaler):
    """自适应损失缩放器"""
    
    def __init__(self, 
                 initial_scale: float = 65536.0,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 min_scale: float = 1e-4,
                 max_scale: float = 65536.0,
                 patience: int = 10):
        """
        Args:
            patience: 连续溢出次数阈值，超过后会更激进地减小缩放因子
        """
        super().__init__(
            initial_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            min_scale=min_scale,
            max_scale=max_scale
        )
        self.patience = patience
        self._consecutive_overflows = 0
    
    def update(self) -> None:
        """更新缩放因子（带自适应逻辑）"""
        if not self.enabled:
            return
        
        if self._has_inf_or_nan:
            self._consecutive_overflows += 1
            
            # 根据连续溢出次数调整回退策略
            if self._consecutive_overflows > self.patience:
                # 更激进的回退
                adaptive_backoff = self.backoff_factor ** 2
                logger.warning(f"连续溢出 {self._consecutive_overflows} 次，使用更激进的回退因子: {adaptive_backoff}")
            else:
                adaptive_backoff = self.backoff_factor
            
            self.scale_value = max(
                self.scale_value * adaptive_backoff,
                self.min_scale
            )
            logger.info(f"损失缩放因子减小到: {self.scale_value}")
            
        elif self._growth_tracker >= self.growth_interval:
            # 成功训练，重置连续溢出计数
            self._consecutive_overflows = 0
            
            # 根据历史表现调整增长因子
            if self._consecutive_overflows == 0:
                adaptive_growth = self.growth_factor
            else:
                # 如果最近有溢出，保守增长
                adaptive_growth = (self.growth_factor + 1.0) / 2.0
            
            self.scale_value = min(
                self.scale_value * adaptive_growth,
                self.max_scale
            )
            self._growth_tracker = 0
            logger.info(f"损失缩放因子增大到: {self.scale_value}")
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = super().state_dict()
        state.update({
            'patience': self.patience,
            'consecutive_overflows': self._consecutive_overflows
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        super().load_state_dict(state_dict)
        self.patience = state_dict.get('patience', 10)
        self._consecutive_overflows = state_dict.get('consecutive_overflows', 0)


def create_loss_scaler(scaler_type: str, **kwargs) -> LossScaler:
    """创建损失缩放器的工厂函数"""
    if scaler_type.lower() == 'static':
        return StaticLossScaler(**kwargs)
    elif scaler_type.lower() == 'dynamic':
        return DynamicLossScaler(**kwargs)
    elif scaler_type.lower() == 'adaptive':
        return AdaptiveLossScaler(**kwargs)
    else:
        raise ValueError(f"未知的损失缩放器类型: {scaler_type}")


class LossScalerManager:
    """损失缩放器管理器"""
    
    def __init__(self, scaler: Optional[LossScaler] = None):
        self.scaler = scaler or DynamicLossScaler()
        self.enabled = True
    
    def scale_loss(self, loss):
        """缩放损失"""
        if not self.enabled or not self.scaler:
            return loss
        return self.scaler.scale_loss(loss)
    
    def backward_and_unscale(self, loss, optimizer) -> bool:
        """反向传播并反缩放梯度"""
        if self.enabled and self.scaler:
            scaled_loss = self.scaler.scale_loss(loss)
            scaled_loss.backward()
            self.scaler.unscale_gradients(optimizer)
            return not self.scaler._has_inf_or_nan
        else:
            loss.backward()
            return True
    
    def step_and_update(self, optimizer) -> bool:
        """执行优化器步骤并更新缩放器"""
        if self.enabled and self.scaler:
            success = self.scaler.step(optimizer)
            self.scaler.update()
            return success
        else:
            optimizer.step()
            return True
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self.scaler:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'enabled': self.enabled,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        self.enabled = state_dict.get('enabled', True)
        if self.scaler and 'scaler' in state_dict and state_dict['scaler']:
            self.scaler.load_state_dict(state_dict['scaler'])