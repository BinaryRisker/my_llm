"""
梯度裁剪器实现

提供各种梯度裁剪策略，用于训练稳定性和性能优化
"""

from typing import Optional, Dict, Any, List, Union
import math
import logging

logger = logging.getLogger(__name__)


class GradientClipper:
    """基础梯度裁剪器"""
    
    def __init__(self, 
                 max_norm: float = 1.0,
                 norm_type: float = 2.0,
                 error_if_nonfinite: bool = False):
        """
        Args:
            max_norm: 梯度范数上限
            norm_type: 范数类型 (1.0, 2.0, inf等)
            error_if_nonfinite: 遇到非有限梯度时是否报错
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
        
        # 统计信息
        self._clip_count = 0
        self._total_norm_history = []
        self._max_history_length = 1000
    
    def clip_gradients(self, parameters, model=None) -> float:
        """
        裁剪梯度
        
        Args:
            parameters: 模型参数
            model: 模型对象（可选）
            
        Returns:
            total_norm: 裁剪前的总梯度范数
        """
        # 收集所有梯度
        grads = []
        for param in parameters:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        if not grads:
            return 0.0
        
        # 模拟梯度范数计算
        total_norm = 0.0
        for grad in grads:
            # 简化的范数计算（实际应使用torch.norm）
            if self.norm_type == 2.0:
                param_norm = sum(x * x for x in grad.view(-1)) ** 0.5
            elif self.norm_type == 1.0:
                param_norm = sum(abs(x) for x in grad.view(-1))
            elif self.norm_type == float('inf'):
                param_norm = max(abs(x) for x in grad.view(-1))
            else:
                param_norm = sum(abs(x) ** self.norm_type for x in grad.view(-1)) ** (1.0 / self.norm_type)
            
            total_norm += param_norm ** self.norm_type
        
        total_norm = total_norm ** (1.0 / self.norm_type)
        
        # 记录统计信息
        self._total_norm_history.append(float(total_norm))
        if len(self._total_norm_history) > self._max_history_length:
            self._total_norm_history.pop(0)
        
        # 检查是否需要裁剪
        if total_norm > self.max_norm:
            clip_coef = self.max_norm / total_norm
            
            # 裁剪梯度
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
            
            self._clip_count += 1
            logger.debug(f"梯度裁剪: {total_norm:.4f} -> {self.max_norm:.4f}")
        
        return float(total_norm)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取裁剪统计信息"""
        if not self._total_norm_history:
            return {
                'clip_count': self._clip_count,
                'avg_norm': 0.0,
                'max_norm_seen': 0.0,
                'clip_rate': 0.0
            }
        
        return {
            'clip_count': self._clip_count,
            'avg_norm': sum(self._total_norm_history) / len(self._total_norm_history),
            'max_norm_seen': max(self._total_norm_history),
            'clip_rate': self._clip_count / len(self._total_norm_history) if self._total_norm_history else 0.0,
            'total_steps': len(self._total_norm_history)
        }


class AdaptiveGradientClipper(GradientClipper):
    """自适应梯度裁剪器"""
    
    def __init__(self,
                 initial_max_norm: float = 1.0,
                 norm_type: float = 2.0,
                 adaptation_rate: float = 0.01,
                 target_clip_rate: float = 0.1,
                 min_max_norm: float = 0.1,
                 max_max_norm: float = 10.0,
                 update_interval: int = 100):
        """
        Args:
            initial_max_norm: 初始最大范数
            adaptation_rate: 自适应学习率
            target_clip_rate: 目标裁剪率
            min_max_norm: 最小范数上限
            max_max_norm: 最大范数上限
            update_interval: 更新间隔
        """
        super().__init__(initial_max_norm, norm_type)
        self.adaptation_rate = adaptation_rate
        self.target_clip_rate = target_clip_rate
        self.min_max_norm = min_max_norm
        self.max_max_norm = max_max_norm
        self.update_interval = update_interval
        
        self._step_count = 0
        self._recent_clip_count = 0
    
    def clip_gradients(self, parameters, model=None) -> float:
        """自适应梯度裁剪"""
        total_norm = super().clip_gradients(parameters, model)
        
        self._step_count += 1
        if total_norm > self.max_norm:
            self._recent_clip_count += 1
        
        # 定期更新max_norm
        if self._step_count % self.update_interval == 0:
            current_clip_rate = self._recent_clip_count / self.update_interval
            
            # 根据裁剪率调整max_norm
            if current_clip_rate > self.target_clip_rate:
                # 裁剪率过高，增加max_norm
                self.max_norm = min(
                    self.max_norm * (1 + self.adaptation_rate),
                    self.max_max_norm
                )
            elif current_clip_rate < self.target_clip_rate * 0.5:
                # 裁剪率过低，减少max_norm
                self.max_norm = max(
                    self.max_norm * (1 - self.adaptation_rate),
                    self.min_max_norm
                )
            
            logger.info(f"自适应梯度裁剪: clip_rate={current_clip_rate:.3f}, new_max_norm={self.max_norm:.3f}")
            self._recent_clip_count = 0
        
        return total_norm


class LayerWiseGradientClipper:
    """逐层梯度裁剪器"""
    
    def __init__(self,
                 max_norm_per_layer: float = 1.0,
                 norm_type: float = 2.0):
        self.max_norm_per_layer = max_norm_per_layer
        self.norm_type = norm_type
        self._layer_statistics = {}
    
    def clip_gradients(self, parameters, model=None) -> Dict[str, float]:
        """逐层裁剪梯度"""
        layer_norms = {}
        
        # 如果提供了模型，按层组织参数
        if model is not None:
            for name, module in model.named_modules():
                layer_params = list(module.parameters())
                if layer_params:
                    layer_norm = self._clip_layer_gradients(layer_params, name)
                    if layer_norm > 0:
                        layer_norms[name] = layer_norm
        else:
            # 简单地将所有参数作为一层处理
            all_params = list(parameters)
            layer_norm = self._clip_layer_gradients(all_params, "all_params")
            layer_norms["all_params"] = layer_norm
        
        return layer_norms
    
    def _clip_layer_gradients(self, parameters, layer_name: str) -> float:
        """裁剪单层梯度"""
        grads = []
        for param in parameters:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        if not grads:
            return 0.0
        
        # 计算层的梯度范数
        layer_norm = 0.0
        for grad in grads:
            # 简化的范数计算
            if self.norm_type == 2.0:
                param_norm = sum(x * x for x in grad) ** 0.5
            else:
                param_norm = sum(abs(x) ** self.norm_type for x in grad) ** (1.0 / self.norm_type)
            layer_norm += param_norm ** self.norm_type
        
        layer_norm = layer_norm ** (1.0 / self.norm_type)
        
        # 裁剪该层梯度
        if layer_norm > self.max_norm_per_layer:
            clip_coef = self.max_norm_per_layer / layer_norm
            
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
            
            logger.debug(f"层 {layer_name} 梯度裁剪: {layer_norm:.4f} -> {self.max_norm_per_layer:.4f}")
        
        # 更新统计信息
        if layer_name not in self._layer_statistics:
            self._layer_statistics[layer_name] = []
        self._layer_statistics[layer_name].append(float(layer_norm))
        
        return float(layer_norm)