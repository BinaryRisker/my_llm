"""
MLflow实验跟踪器
===============

集成MLflow进行实验跟踪和管理：
- 自动记录训练参数、指标和模型
- 支持实验比较和可视化
- 模型版本管理和部署
- 与现有训练流程无缝集成

使用方法:
    from utils.experiment_tracking.mlflow_tracker import MLflowTracker
    
    tracker = MLflowTracker(experiment_name="my_experiment")
    
    with tracker.start_run():
        tracker.log_params({"lr": 0.001, "batch_size": 32})
        tracker.log_metric("loss", 0.5, step=100)
        tracker.log_model(model, "transformer_model")
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
import contextlib
from datetime import datetime

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.entities import RunStatus
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow实验跟踪器"""
    
    def __init__(self,
                 experiment_name: str = "default_experiment",
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 artifact_location: Optional[str] = None,
                 run_name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None,
                 auto_log: bool = True):
        """
        初始化MLflow跟踪器
        
        Args:
            experiment_name: 实验名称
            tracking_uri: MLflow tracking server URI
            registry_uri: 模型注册URI
            artifact_location: artifact存储位置
            run_name: 运行名称
            tags: 运行标签
            auto_log: 是否开启自动日志记录
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow未安装。请运行: pip install mlflow")
        
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.auto_log = auto_log
        
        # 设置MLflow配置
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        # 创建或获取实验
        try:
            self.experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
            logger.info(f"创建新实验: {experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"使用现有实验: {experiment_name}")
        
        # MLflow客户端
        self.client = MlflowClient()
        
        # 当前运行
        self.current_run = None
        self.run_id = None
        
        # 启用自动日志记录
        if self.auto_log:
            self._setup_auto_logging()
    
    def _setup_auto_logging(self):
        """设置自动日志记录"""
        try:
            if TORCH_AVAILABLE:
                mlflow.pytorch.autolog()
            mlflow.sklearn.autolog()
            logger.info("已启用MLflow自动日志记录")
        except Exception as e:
            logger.warning(f"设置自动日志记录失败: {e}")
    
    @contextlib.contextmanager
    def start_run(self, 
                  run_name: Optional[str] = None,
                  nested: bool = False,
                  tags: Optional[Dict[str, str]] = None):
        """
        开始新的MLflow运行
        
        Args:
            run_name: 运行名称
            nested: 是否嵌套运行
            tags: 运行标签
        """
        run_name = run_name or self.run_name
        run_tags = {**self.tags, **(tags or {})}
        
        # 添加默认标签
        run_tags.update({
            "start_time": datetime.now().isoformat(),
            "python_version": str(os.sys.version_info[:2]),
        })
        
        try:
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested,
                tags=run_tags
            ) as run:
                self.current_run = run
                self.run_id = run.info.run_id
                
                logger.info(f"开始MLflow运行: {run.info.run_id}")
                
                yield self
                
        finally:
            self.current_run = None
            self.run_id = None
            logger.info("MLflow运行已结束")
    
    def log_param(self, key: str, value: Any) -> None:
        """记录参数"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """批量记录参数"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.log_params(params)
        logger.info(f"记录参数: {list(params.keys())}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """记录指标"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.log_metric(key, value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """批量记录指标"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.log_metrics(metrics, step)
        logger.info(f"记录指标: {list(metrics.keys())}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """记录文件artifact"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"记录artifact: {local_path}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """记录目录artifacts"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.info(f"记录artifacts目录: {local_dir}")
    
    def log_text(self, text: str, artifact_file: str) -> None:
        """记录文本内容"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.log_text(text, artifact_file)
        logger.info(f"记录文本: {artifact_file}")
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """记录字典内容"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.log_dict(dictionary, artifact_file)
        logger.info(f"记录字典: {artifact_file}")
    
    def log_figure(self, figure, artifact_file: str) -> None:
        """记录图表"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.log_figure(figure, artifact_file)
        logger.info(f"记录图表: {artifact_file}")
    
    def log_model(self, 
                  model,
                  artifact_path: str,
                  conda_env: Optional[str] = None,
                  signature: Optional[Any] = None,
                  input_example: Optional[Any] = None,
                  registered_model_name: Optional[str] = None) -> None:
        """
        记录模型
        
        Args:
            model: 模型对象
            artifact_path: artifact路径
            conda_env: Conda环境文件
            signature: 模型签名
            input_example: 输入样例
            registered_model_name: 注册模型名称
        """
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        try:
            if TORCH_AVAILABLE and hasattr(model, 'state_dict'):
                # PyTorch模型
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    conda_env=conda_env,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            else:
                # 其他类型模型，使用通用方法
                mlflow.log_artifact(model, artifact_path)
            
            logger.info(f"记录模型: {artifact_path}")
            
        except Exception as e:
            logger.error(f"记录模型失败: {e}")
            raise
    
    def log_training_progress(self, 
                            epoch: int,
                            train_loss: float,
                            val_loss: Optional[float] = None,
                            train_metrics: Optional[Dict[str, float]] = None,
                            val_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        记录训练进度
        
        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            val_loss: 验证损失
            train_metrics: 训练指标
            val_metrics: 验证指标
        """
        # 记录损失
        self.log_metric("train_loss", train_loss, step=epoch)
        if val_loss is not None:
            self.log_metric("val_loss", val_loss, step=epoch)
        
        # 记录训练指标
        if train_metrics:
            for metric_name, value in train_metrics.items():
                self.log_metric(f"train_{metric_name}", value, step=epoch)
        
        # 记录验证指标
        if val_metrics:
            for metric_name, value in val_metrics.items():
                self.log_metric(f"val_{metric_name}", value, step=epoch)
    
    def set_tag(self, key: str, value: str) -> None:
        """设置标签"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.set_tag(key, value)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """批量设置标签"""
        if not self.current_run:
            raise RuntimeError("没有活跃的MLflow运行，请使用start_run()上下文管理器")
        
        mlflow.set_tags(tags)
        logger.info(f"设置标签: {list(tags.keys())}")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """结束当前运行"""
        if self.current_run:
            mlflow.end_run(status)
            logger.info(f"结束运行: {self.run_id}")
            self.current_run = None
            self.run_id = None
    
    def get_experiment_runs(self, 
                           max_results: int = 100,
                           order_by: Optional[List[str]] = None) -> List[Any]:
        """获取实验的所有运行"""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=max_results,
            order_by=order_by or ["start_time DESC"]
        )
        return runs
    
    def get_best_run(self, metric: str, ascending: bool = False) -> Optional[Any]:
        """获取最佳运行"""
        runs = self.get_experiment_runs()
        
        if not runs:
            return None
        
        # 过滤有指定指标的运行
        runs_with_metric = [run for run in runs if metric in run.data.metrics]
        
        if not runs_with_metric:
            return None
        
        # 按指标排序
        best_run = min(runs_with_metric, 
                      key=lambda r: r.data.metrics[metric] if not ascending 
                      else -r.data.metrics[metric])
        
        return best_run
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """比较多个运行"""
        runs_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            runs_data.append({
                'run_id': run_id,
                'params': run.data.params,
                'metrics': run.data.metrics,
                'tags': run.data.tags,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time
            })
        
        return {
            'runs': runs_data,
            'comparison_time': datetime.now().isoformat()
        }
    
    def load_model(self, 
                   run_id: str, 
                   model_path: str = "model") -> Any:
        """加载模型"""
        model_uri = f"runs:/{run_id}/{model_path}"
        
        try:
            if TORCH_AVAILABLE:
                return mlflow.pytorch.load_model(model_uri)
            else:
                return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def register_model(self, 
                      model_uri: str,
                      name: str,
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> Any:
        """注册模型到模型注册表"""
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=name,
                tags=tags
            )
            
            if description:
                self.client.update_model_version(
                    name=name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"模型已注册: {name} v{model_version.version}")
            return model_version
            
        except Exception as e:
            logger.error(f"注册模型失败: {e}")
            raise
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """获取实验信息"""
        experiment = self.client.get_experiment(self.experiment_id)
        runs = self.get_experiment_runs()
        
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': experiment.name,
            'artifact_location': experiment.artifact_location,
            'lifecycle_stage': experiment.lifecycle_stage,
            'total_runs': len(runs),
            'active_runs': len([r for r in runs if r.info.status == RunStatus.RUNNING]),
            'completed_runs': len([r for r in runs if r.info.status == RunStatus.FINISHED]),
            'failed_runs': len([r for r in runs if r.info.status == RunStatus.FAILED])
        }


def demo():
    """演示MLflow跟踪器的使用"""
    if not MLFLOW_AVAILABLE:
        print("❌ MLflow未安装，无法运行演示")
        print("请运行: pip install mlflow")
        return
    
    print("🚀 MLflow跟踪器演示")
    print("=" * 50)
    
    # 创建跟踪器
    tracker = MLflowTracker(
        experiment_name="demo_experiment",
        auto_log=False  # 关闭自动日志以便演示
    )
    
    print(f"📊 实验信息:")
    exp_info = tracker.get_experiment_info()
    for key, value in exp_info.items():
        print(f"  {key}: {value}")
    
    # 模拟训练过程
    print("\n🔧 模拟训练过程...")
    
    with tracker.start_run(run_name="demo_training") as t:
        # 记录超参数
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model_type": "transformer",
            "optimizer": "adam"
        }
        t.log_params(params)
        print("✅ 已记录训练参数")
        
        # 设置标签
        tags = {
            "model_family": "transformer",
            "dataset": "demo_data",
            "stage": "stage4"
        }
        t.set_tags(tags)
        print("✅ 已设置运行标签")
        
        # 模拟训练循环
        import random
        import time
        
        for epoch in range(5):  # 简化为5个epoch
            # 模拟训练指标
            train_loss = 2.0 - epoch * 0.3 + random.uniform(-0.1, 0.1)
            val_loss = 2.2 - epoch * 0.25 + random.uniform(-0.1, 0.1)
            
            train_metrics = {
                "accuracy": 0.5 + epoch * 0.08 + random.uniform(-0.02, 0.02),
                "bleu": 0.2 + epoch * 0.05 + random.uniform(-0.01, 0.01)
            }
            
            val_metrics = {
                "accuracy": 0.45 + epoch * 0.07 + random.uniform(-0.02, 0.02),
                "bleu": 0.18 + epoch * 0.04 + random.uniform(-0.01, 0.01)
            }
            
            # 记录训练进度
            t.log_training_progress(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics
            )
            
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            time.sleep(0.1)  # 模拟训练时间
        
        # 记录文本信息
        training_log = f\"\"\"训练完成摘要:\n- 总epochs: 5\n- 最终训练损失: {train_loss:.4f}\n- 最终验证损失: {val_loss:.4f}\n- 最终训练准确率: {train_metrics['accuracy']:.4f}\n- 最终验证准确率: {val_metrics['accuracy']:.4f}\n\"\"\"
        
        t.log_text(training_log, "training_summary.txt")
        
        # 记录配置字典
        config_dict = {
            "model_config": {
                "d_model": 512,
                "num_heads": 8,
                "num_layers": 6,
                "d_ff": 2048
            },
            "training_config": params,
            "final_metrics": {
                "train": train_metrics,
                "val": val_metrics
            }
        }
        t.log_dict(config_dict, "experiment_config.json")
        
        print("✅ 训练完成，所有数据已记录到MLflow")
    
    # 获取实验运行
    print("\n📋 获取实验运行...")
    runs = tracker.get_experiment_runs(max_results=5)
    
    print(f"找到 {len(runs)} 个运行:")
    for run in runs[:3]:  # 显示前3个
        print(f"  - Run ID: {run.info.run_id}")
        print(f"    状态: {run.info.status}")
        print(f"    开始时间: {run.info.start_time}")
        if run.data.metrics:
            print(f"    指标: {list(run.data.metrics.keys())}")
    
    # 获取最佳运行
    print("\n🏆 寻找最佳运行...")
    best_run = tracker.get_best_run("val_loss", ascending=True)
    if best_run:
        print(f"最佳运行 (最低验证损失):")
        print(f"  - Run ID: {best_run.info.run_id}")
        print(f"  - 验证损失: {best_run.data.metrics['val_loss']:.4f}")
        print(f"  - 验证准确率: {best_run.data.metrics.get('val_accuracy', 'N/A')}")
    
    print("\n🎉 演示完成!")
    print("💡 启动MLflow UI查看结果: mlflow ui")


if __name__ == "__main__":
    demo()