"""
实验跟踪模块
=========

包含各种实验跟踪和管理工具：
- MLflowTracker: MLflow实验跟踪器
- 基础跟踪器接口
- 实验管理工具

使用方法:
    from utils.experiment_tracking import MLflowTracker
"""

try:
    from .mlflow_tracker import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
    class MLflowTracker:
        def __init__(self, *args, **kwargs):
            raise ImportError("MLflow未安装。请运行: pip install mlflow")

__all__ = ['MLflowTracker', 'MLFLOW_AVAILABLE']