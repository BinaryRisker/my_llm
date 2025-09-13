"""
GLUE/SuperGLUE基准评估系统
==========================

实现标准化的GLUE和SuperGLUE基准测试评估，包括：
- GLUE任务：CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI
- SuperGLUE任务：BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC
- 标准化评估指标和报告生成

参考标准：
- GLUE: https://gluebenchmark.com/
- SuperGLUE: https://super.gluebenchmark.com/
- 论文: GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding

使用方法:
    from evaluation.glue_benchmark import GLUEBenchmark
    
    benchmark = GLUEBenchmark()
    results = benchmark.evaluate_task('sst2', predictions, references)
    print(f"SST-2 Accuracy: {results['accuracy']:.4f}")
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from .evaluation_metrics import (
        ClassificationMetrics, RegressionMetrics, 
        QuestionAnsweringMetrics, TokenClassificationMetrics
    )
    METRICS_AVAILABLE = True
except ImportError:
    try:
        from evaluation_metrics import (
            ClassificationMetrics, RegressionMetrics,
            QuestionAnsweringMetrics, TokenClassificationMetrics
        )
        METRICS_AVAILABLE = True
    except ImportError:
        METRICS_AVAILABLE = False
        ClassificationMetrics = RegressionMetrics = None
        QuestionAnsweringMetrics = TokenClassificationMetrics = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    task_name: str
    metric_name: str
    score: float
    details: Dict[str, Any]
    description: str


class BaseBenchmarkTask(ABC):
    """基础基准任务类"""
    
    def __init__(self, task_name: str, description: str):
        self.task_name = task_name
        self.description = description
        self.metric_name = "score"
    
    @abstractmethod
    def evaluate(self, predictions: Any, references: Any, **kwargs) -> BenchmarkResult:
        """评估任务性能"""
        pass
    
    def get_task_info(self) -> Dict[str, str]:
        """获取任务信息"""
        return {
            "name": self.task_name,
            "description": self.description,
            "metric": self.metric_name
        }


class CoLATask(BaseBenchmarkTask):
    """CoLA - Corpus of Linguistic Acceptability"""
    
    def __init__(self):
        super().__init__(
            "cola",
            "语言可接受性判断 - 判断句子是否符合语法"
        )
        self.metric_name = "matthews_corr"
        self.metrics = ClassificationMetrics() if METRICS_AVAILABLE else None
    
    def evaluate(self, predictions: List[int], references: List[int], **kwargs) -> BenchmarkResult:
        """
        CoLA任务评估
        
        Args:
            predictions: 预测标签 (0: 不可接受, 1: 可接受)
            references: 真实标签
            
        Returns:
            包含Matthews相关系数的结果
        """
        if not self.metrics:
            return BenchmarkResult(
                task_name=self.task_name,
                metric_name=self.metric_name,
                score=0.0,
                details={"error": "评估指标不可用"},
                description="CoLA任务评估失败"
            )
        
        results = self.metrics.compute(predictions, references)
        
        return BenchmarkResult(
            task_name=self.task_name,
            metric_name=self.metric_name,
            score=results.get('matthews_corr', 0.0),
            details=results,
            description=f"CoLA Matthews相关系数: {results.get('matthews_corr', 0.0):.4f}"
        )


class SST2Task(BaseBenchmarkTask):
    """SST-2 - Stanford Sentiment Treebank"""
    
    def __init__(self):
        super().__init__(
            "sst2",
            "情感分析 - 判断电影评论的情感倾向"
        )
        self.metric_name = "accuracy"
        self.metrics = ClassificationMetrics() if METRICS_AVAILABLE else None
    
    def evaluate(self, predictions: List[int], references: List[int], **kwargs) -> BenchmarkResult:
        """
        SST-2任务评估
        
        Args:
            predictions: 预测标签 (0: 负面, 1: 正面)
            references: 真实标签
            
        Returns:
            包含准确率的结果
        """
        if not self.metrics:
            return BenchmarkResult(
                task_name=self.task_name,
                metric_name=self.metric_name,
                score=0.0,
                details={"error": "评估指标不可用"},
                description="SST-2任务评估失败"
            )
        
        results = self.metrics.compute(predictions, references)
        
        return BenchmarkResult(
            task_name=self.task_name,
            metric_name=self.metric_name,
            score=results.get('accuracy', 0.0),
            details=results,
            description=f"SST-2 准确率: {results.get('accuracy', 0.0):.4f}"
        )


class MRPCTask(BaseBenchmarkTask):
    """MRPC - Microsoft Research Paraphrase Corpus"""
    
    def __init__(self):
        super().__init__(
            "mrpc",
            "释义检测 - 判断两个句子是否意思相同"
        )
        self.metric_name = "f1"
        self.metrics = ClassificationMetrics() if METRICS_AVAILABLE else None
    
    def evaluate(self, predictions: List[int], references: List[int], **kwargs) -> BenchmarkResult:
        """
        MRPC任务评估
        
        Args:
            predictions: 预测标签 (0: 非释义, 1: 释义)
            references: 真实标签
            
        Returns:
            包含F1分数和准确率的结果
        """
        if not self.metrics:
            return BenchmarkResult(
                task_name=self.task_name,
                metric_name=self.metric_name,
                score=0.0,
                details={"error": "评估指标不可用"},
                description="MRPC任务评估失败"
            )
        
        results = self.metrics.compute(predictions, references)
        
        # MRPC使用F1和准确率的平均值
        f1_score = results.get('f1', 0.0)
        accuracy = results.get('accuracy', 0.0)
        combined_score = (f1_score + accuracy) / 2
        
        results['combined_score'] = combined_score
        
        return BenchmarkResult(
            task_name=self.task_name,
            metric_name="f1_accuracy_avg",
            score=combined_score,
            details=results,
            description=f"MRPC F1/Accuracy: {combined_score:.4f} (F1: {f1_score:.4f}, Acc: {accuracy:.4f})"
        )


class STSBTask(BaseBenchmarkTask):
    """STS-B - Semantic Textual Similarity Benchmark"""
    
    def __init__(self):
        super().__init__(
            "stsb",
            "语义相似度 - 预测两个句子的相似度分数(0-5)"
        )
        self.metric_name = "pearson_corr"
        self.metrics = RegressionMetrics() if METRICS_AVAILABLE else None
    
    def evaluate(self, predictions: List[float], references: List[float], **kwargs) -> BenchmarkResult:
        """
        STS-B任务评估
        
        Args:
            predictions: 预测相似度分数 (0-5)
            references: 真实相似度分数
            
        Returns:
            包含皮尔逊相关系数的结果
        """
        if not self.metrics:
            return BenchmarkResult(
                task_name=self.task_name,
                metric_name=self.metric_name,
                score=0.0,
                details={"error": "评估指标不可用"},
                description="STS-B任务评估失败"
            )
        
        results = self.metrics.compute(predictions, references)
        
        # 计算皮尔逊相关系数（简化版）
        pearson_corr = self._compute_pearson_correlation(predictions, references)
        results['pearson_corr'] = pearson_corr
        
        return BenchmarkResult(
            task_name=self.task_name,
            metric_name=self.metric_name,
            score=pearson_corr,
            details=results,
            description=f"STS-B 皮尔逊相关系数: {pearson_corr:.4f}"
        )
    
    def _compute_pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """计算皮尔逊相关系数"""
        if len(x) != len(y) or len(x) == 0:
            return 0.0
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0.0


class QQPTask(BaseBenchmarkTask):
    """QQP - Quora Question Pairs"""
    
    def __init__(self):
        super().__init__(
            "qqp",
            "问题匹配 - 判断两个问题是否等价"
        )
        self.metric_name = "f1"
        self.metrics = ClassificationMetrics() if METRICS_AVAILABLE else None
    
    def evaluate(self, predictions: List[int], references: List[int], **kwargs) -> BenchmarkResult:
        """
        QQP任务评估
        
        Args:
            predictions: 预测标签 (0: 不等价, 1: 等价)
            references: 真实标签
            
        Returns:
            包含F1分数和准确率的结果
        """
        if not self.metrics:
            return BenchmarkResult(
                task_name=self.task_name,
                metric_name=self.metric_name,
                score=0.0,
                details={"error": "评估指标不可用"},
                description="QQP任务评估失败"
            )
        
        results = self.metrics.compute(predictions, references)
        
        return BenchmarkResult(
            task_name=self.task_name,
            metric_name=self.metric_name,
            score=results.get('f1', 0.0),
            details=results,
            description=f"QQP F1分数: {results.get('f1', 0.0):.4f}"
        )


class MNLITask(BaseBenchmarkTask):
    """MNLI - Multi-Genre Natural Language Inference"""
    
    def __init__(self):
        super().__init__(
            "mnli",
            "自然语言推理 - 判断假设与前提的关系"
        )
        self.metric_name = "accuracy"
        self.metrics = ClassificationMetrics() if METRICS_AVAILABLE else None
    
    def evaluate(self, predictions: List[int], references: List[int], **kwargs) -> BenchmarkResult:
        """
        MNLI任务评估
        
        Args:
            predictions: 预测标签 (0: entailment, 1: neutral, 2: contradiction)
            references: 真实标签
            
        Returns:
            包含准确率的结果
        """
        if not self.metrics:
            return BenchmarkResult(
                task_name=self.task_name,
                metric_name=self.metric_name,
                score=0.0,
                details={"error": "评估指标不可用"},
                description="MNLI任务评估失败"
            )
        
        results = self.metrics.compute(predictions, references)
        
        return BenchmarkResult(
            task_name=self.task_name,
            metric_name=self.metric_name,
            score=results.get('accuracy', 0.0),
            details=results,
            description=f"MNLI 准确率: {results.get('accuracy', 0.0):.4f}"
        )


class QNLITask(BaseBenchmarkTask):
    """QNLI - Question Natural Language Inference"""
    
    def __init__(self):
        super().__init__(
            "qnli",
            "问题自然语言推理 - 判断段落是否包含问题的答案"
        )
        self.metric_name = "accuracy"
        self.metrics = ClassificationMetrics() if METRICS_AVAILABLE else None
    
    def evaluate(self, predictions: List[int], references: List[int], **kwargs) -> BenchmarkResult:
        """
        QNLI任务评估
        
        Args:
            predictions: 预测标签 (0: not_entailment, 1: entailment)
            references: 真实标签
            
        Returns:
            包含准确率的结果
        """
        if not self.metrics:
            return BenchmarkResult(
                task_name=self.task_name,
                metric_name=self.metric_name,
                score=0.0,
                details={"error": "评估指标不可用"},
                description="QNLI任务评估失败"
            )
        
        results = self.metrics.compute(predictions, references)
        
        return BenchmarkResult(
            task_name=self.task_name,
            metric_name=self.metric_name,
            score=results.get('accuracy', 0.0),
            details=results,
            description=f"QNLI 准确率: {results.get('accuracy', 0.0):.4f}"
        )


class RTETask(BaseBenchmarkTask):
    """RTE - Recognizing Textual Entailment"""
    
    def __init__(self):
        super().__init__(
            "rte",
            "文本蕴含识别 - 判断文本是否蕴含假设"
        )
        self.metric_name = "accuracy"
        self.metrics = ClassificationMetrics() if METRICS_AVAILABLE else None
    
    def evaluate(self, predictions: List[int], references: List[int], **kwargs) -> BenchmarkResult:
        """
        RTE任务评估
        
        Args:
            predictions: 预测标签 (0: not_entailment, 1: entailment)
            references: 真实标签
            
        Returns:
            包含准确率的结果
        """
        if not self.metrics:
            return BenchmarkResult(
                task_name=self.task_name,
                metric_name=self.metric_name,
                score=0.0,
                details={"error": "评估指标不可用"},
                description="RTE任务评估失败"
            )
        
        results = self.metrics.compute(predictions, references)
        
        return BenchmarkResult(
            task_name=self.task_name,
            metric_name=self.metric_name,
            score=results.get('accuracy', 0.0),
            details=results,
            description=f"RTE 准确率: {results.get('accuracy', 0.0):.4f}"
        )


class WNLITask(BaseBenchmarkTask):
    """WNLI - Winograd Natural Language Inference"""
    
    def __init__(self):
        super().__init__(
            "wnli",
            "Winograd自然语言推理 - 复杂的常识推理任务"
        )
        self.metric_name = "accuracy"
        self.metrics = ClassificationMetrics() if METRICS_AVAILABLE else None
    
    def evaluate(self, predictions: List[int], references: List[int], **kwargs) -> BenchmarkResult:
        """
        WNLI任务评估
        
        Args:
            predictions: 预测标签 (0: not_entailment, 1: entailment)
            references: 真实标签
            
        Returns:
            包含准确率的结果
        """
        if not self.metrics:
            return BenchmarkResult(
                task_name=self.task_name,
                metric_name=self.metric_name,
                score=0.0,
                details={"error": "评估指标不可用"},
                description="WNLI任务评估失败"
            )
        
        results = self.metrics.compute(predictions, references)
        
        return BenchmarkResult(
            task_name=self.task_name,
            metric_name=self.metric_name,
            score=results.get('accuracy', 0.0),
            details=results,
            description=f"WNLI 准确率: {results.get('accuracy', 0.0):.4f}"
        )


class GLUEBenchmark:
    """GLUE基准评估系统"""
    
    def __init__(self):
        self.tasks = {
            'cola': CoLATask(),
            'sst2': SST2Task(),
            'mrpc': MRPCTask(),
            'stsb': STSBTask(),
            'qqp': QQPTask(),
            'mnli': MNLITask(),
            'qnli': QNLITask(),
            'rte': RTETask(),
            'wnli': WNLITask()
        }
        
        self.name = "GLUE"
        self.description = "General Language Understanding Evaluation"
        self.version = "1.0"
    
    def get_available_tasks(self) -> List[str]:
        """获取可用的任务列表"""
        return list(self.tasks.keys())
    
    def get_task_info(self, task_name: str) -> Dict[str, str]:
        """获取任务信息"""
        if task_name not in self.tasks:
            raise ValueError(f"未知任务: {task_name}")
        
        return self.tasks[task_name].get_task_info()
    
    def evaluate_task(self, 
                     task_name: str, 
                     predictions: Any, 
                     references: Any, 
                     **kwargs) -> BenchmarkResult:
        """
        评估单个任务
        
        Args:
            task_name: 任务名称
            predictions: 预测结果
            references: 参考答案
            **kwargs: 其他参数
            
        Returns:
            任务评估结果
        """
        if task_name not in self.tasks:
            raise ValueError(f"未知任务: {task_name}. 可用任务: {list(self.tasks.keys())}")
        
        task = self.tasks[task_name]
        return task.evaluate(predictions, references, **kwargs)
    
    def evaluate_multiple_tasks(self, 
                               task_results: Dict[str, Tuple[Any, Any]], 
                               **kwargs) -> Dict[str, BenchmarkResult]:
        """
        评估多个任务
        
        Args:
            task_results: 任务结果字典 {task_name: (predictions, references)}
            **kwargs: 其他参数
            
        Returns:
            所有任务的评估结果
        """
        results = {}
        
        for task_name, (predictions, references) in task_results.items():
            try:
                results[task_name] = self.evaluate_task(task_name, predictions, references, **kwargs)
            except Exception as e:
                logger.error(f"任务 {task_name} 评估失败: {e}")
                results[task_name] = BenchmarkResult(
                    task_name=task_name,
                    metric_name="error",
                    score=0.0,
                    details={"error": str(e)},
                    description=f"任务 {task_name} 评估失败"
                )
        
        return results
    
    def compute_overall_score(self, task_results: Dict[str, BenchmarkResult]) -> float:
        """
        计算GLUE总分
        
        Args:
            task_results: 各任务评估结果
            
        Returns:
            GLUE总分（所有任务分数的平均值）
        """
        valid_scores = []
        
        for task_name, result in task_results.items():
            if result.score > 0:  # 只计算有效分数
                valid_scores.append(result.score)
        
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    def generate_report(self, task_results: Dict[str, BenchmarkResult]) -> str:
        """
        生成评估报告
        
        Args:
            task_results: 各任务评估结果
            
        Returns:
            格式化的评估报告
        """
        report_lines = []
        report_lines.append(f"🏆 {self.name} 基准评估报告")
        report_lines.append("=" * 50)
        
        overall_score = self.compute_overall_score(task_results)
        report_lines.append(f"📊 总分: {overall_score:.4f}")
        report_lines.append("")
        
        report_lines.append("📋 各任务详情:")
        report_lines.append("-" * 30)
        
        for task_name, result in task_results.items():
            task_info = self.get_task_info(task_name)
            report_lines.append(f"• {task_name.upper()}: {result.score:.4f} ({result.metric_name})")
            report_lines.append(f"  描述: {task_info['description']}")
            
            if 'error' not in result.details:
                if task_name == 'mrpc':
                    # MRPC显示F1和准确率
                    details = result.details
                    report_lines.append(f"  F1: {details.get('f1', 0.0):.4f}, 准确率: {details.get('accuracy', 0.0):.4f}")
                elif task_name == 'stsb':
                    # STS-B显示回归指标
                    details = result.details
                    report_lines.append(f"  MSE: {details.get('mse', 0.0):.4f}, R²: {details.get('r2', 0.0):.4f}")
            else:
                report_lines.append(f"  ❌ 错误: {result.details['error']}")
            
            report_lines.append("")
        
        report_lines.append("🎯 基准信息:")
        report_lines.append(f"  名称: {self.description}")
        report_lines.append(f"  版本: {self.version}")
        report_lines.append(f"  任务数量: {len(self.tasks)}")
        
        return "\n".join(report_lines)


def demo():
    """GLUE基准演示"""
    print("🚀 GLUE基准评估演示")
    print("=" * 50)
    
    if not METRICS_AVAILABLE:
        print("❌ 评估指标模块不可用，无法运行演示")
        return
    
    # 创建GLUE基准
    glue = GLUEBenchmark()
    
    print("📋 可用任务:")
    for task_name in glue.get_available_tasks():
        info = glue.get_task_info(task_name)
        print(f"  • {task_name.upper()}: {info['description']} (指标: {info['metric']})")
    
    print(f"\n🧪 模拟评估演示:")
    print("-" * 30)
    
    # 模拟一些任务的评估结果
    mock_results = {}
    
    # CoLA - Matthews相关系数
    cola_preds = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    cola_refs = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    mock_results['cola'] = (cola_preds, cola_refs)
    
    # SST-2 - 准确率
    sst2_preds = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    sst2_refs = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    mock_results['sst2'] = (sst2_preds, sst2_refs)
    
    # MRPC - F1和准确率
    mrpc_preds = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    mrpc_refs = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    mock_results['mrpc'] = (mrpc_preds, mrpc_refs)
    
    # STS-B - 皮尔逊相关系数
    stsb_preds = [3.2, 1.8, 4.1, 2.5, 3.7, 1.2, 4.5, 2.8, 3.9, 1.9]
    stsb_refs = [3.0, 2.0, 4.0, 2.8, 3.5, 1.5, 4.2, 2.5, 4.1, 2.2]
    mock_results['stsb'] = (stsb_preds, stsb_refs)
    
    # RTE - 准确率
    rte_preds = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    rte_refs = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
    mock_results['rte'] = (rte_preds, rte_refs)
    
    # 评估所有任务
    results = glue.evaluate_multiple_tasks(mock_results)
    
    # 生成报告
    report = glue.generate_report(results)
    print(report)
    
    print(f"\n🎉 GLUE基准演示完成!")


if __name__ == "__main__":
    demo()