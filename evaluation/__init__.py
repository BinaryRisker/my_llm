"""
标准化评估系统
==============

完整的NLP任务评估系统，包括：
- 标准评估指标（分类、回归、生成、问答等）
- GLUE/SuperGLUE基准测试
- 自定义评估管道
- 结果分析和报告生成

组件说明:
- evaluation_metrics: 核心评估指标实现
- glue_benchmark: GLUE/SuperGLUE基准测试系统
- custom_evaluator: 自定义评估器

使用方法:
    from evaluation import ClassificationMetrics, GLUEBenchmark
    
    # 分类任务评估
    metrics = ClassificationMetrics()
    results = metrics.compute(predictions, references)
    
    # GLUE基准评估
    glue = GLUEBenchmark()
    benchmark_results = glue.evaluate_task('sst2', pred, ref)

版本: 1.0.0
"""

# 评估指标导入
try:
    from .evaluation_metrics import (
        BaseMetrics,
        ClassificationMetrics,
        RegressionMetrics,
        TokenClassificationMetrics,
        QuestionAnsweringMetrics,
        TextGenerationMetrics,
        demo as metrics_demo
    )
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入评估指标模块: {e}")
    BaseMetrics = ClassificationMetrics = RegressionMetrics = None
    TokenClassificationMetrics = QuestionAnsweringMetrics = TextGenerationMetrics = None
    metrics_demo = None
    METRICS_AVAILABLE = False

# GLUE基准导入
try:
    from .glue_benchmark import (
        GLUEBenchmark,
        BenchmarkResult,
        BaseBenchmarkTask,
        CoLATask, SST2Task, MRPCTask, STSBTask,
        QQPTask, MNLITask, QNLITask, RTETask, WNLITask,
        demo as glue_demo
    )
    GLUE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入GLUE基准模块: {e}")
    GLUEBenchmark = BenchmarkResult = BaseBenchmarkTask = None
    CoLATask = SST2Task = MRPCTask = STSBTask = None
    QQPTask = MNLITask = QNLITask = RTETask = WNLITask = None
    glue_demo = None
    GLUE_AVAILABLE = False

# 版本信息
__version__ = "1.0.0"
__author__ = "LLM Implementation Team"
__description__ = "Comprehensive NLP evaluation system"

# 导出的公共接口
__all__ = [
    # 基础评估指标
    'BaseMetrics',
    'ClassificationMetrics',
    'RegressionMetrics', 
    'TokenClassificationMetrics',
    'QuestionAnsweringMetrics',
    'TextGenerationMetrics',
    
    # GLUE基准
    'GLUEBenchmark',
    'BenchmarkResult',
    'BaseBenchmarkTask',
    'CoLATask', 'SST2Task', 'MRPCTask', 'STSBTask',
    'QQPTask', 'MNLITask', 'QNLITask', 'RTETask', 'WNLITask',
    
    # 演示函数
    'metrics_demo',
    'glue_demo',
    'demo',
    
    # 可用性标志
    'METRICS_AVAILABLE',
    'GLUE_AVAILABLE',
]


def get_evaluation_info():
    """获取评估系统信息"""
    return {
        "name": "Evaluation System",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "components": {
            "metrics": METRICS_AVAILABLE,
            "glue_benchmark": GLUE_AVAILABLE,
        },
        "supported_metrics": [
            "accuracy", "precision", "recall", "f1",
            "mse", "mae", "rmse", "r2", "pearson_corr",
            "exact_match", "entity_f1", "token_accuracy",
            "bleu", "rouge", "matthews_corr"
        ],
        "supported_benchmarks": [
            "GLUE", "CoLA", "SST-2", "MRPC", "STS-B",
            "QQP", "MNLI", "QNLI", "RTE", "WNLI"
        ]
    }


def list_available_metrics():
    """列出所有可用的评估指标"""
    metrics = []
    
    if METRICS_AVAILABLE:
        metrics.extend([
            {
                "name": "ClassificationMetrics",
                "description": "分类任务评估指标",
                "metrics": ["accuracy", "precision", "recall", "f1", "macro_f1", "matthews_corr"],
                "available": True
            },
            {
                "name": "RegressionMetrics", 
                "description": "回归任务评估指标",
                "metrics": ["mse", "mae", "rmse", "r2"],
                "available": True
            },
            {
                "name": "TokenClassificationMetrics",
                "description": "token分类任务评估指标（NER等）",
                "metrics": ["token_accuracy", "entity_precision", "entity_recall", "entity_f1"],
                "available": True
            },
            {
                "name": "QuestionAnsweringMetrics",
                "description": "问答任务评估指标",
                "metrics": ["exact_match", "f1"],
                "available": True
            },
            {
                "name": "TextGenerationMetrics",
                "description": "文本生成任务评估指标",
                "metrics": ["bleu", "rouge-1", "rouge-2", "rouge-l"],
                "available": True
            }
        ])
    
    return metrics


def list_available_benchmarks():
    """列出所有可用的基准测试"""
    benchmarks = []
    
    if GLUE_AVAILABLE:
        benchmarks.append({
            "name": "GLUE",
            "description": "General Language Understanding Evaluation",
            "tasks": [
                "CoLA", "SST-2", "MRPC", "STS-B", "QQP",
                "MNLI", "QNLI", "RTE", "WNLI"
            ],
            "available": True
        })
    
    return benchmarks


def create_evaluator(task_type: str, **kwargs):
    """
    创建评估器的便捷函数
    
    Args:
        task_type: 任务类型 ("classification", "regression", "token_classification", 
                           "question_answering", "text_generation")
        **kwargs: 其他参数
        
    Returns:
        相应的评估器实例
        
    Example:
        >>> evaluator = create_evaluator("classification")
        >>> results = evaluator.compute(predictions, references)
    """
    if not METRICS_AVAILABLE:
        raise ImportError("评估指标模块不可用")
    
    evaluators = {
        "classification": ClassificationMetrics,
        "regression": RegressionMetrics,
        "token_classification": TokenClassificationMetrics,
        "question_answering": QuestionAnsweringMetrics,
        "text_generation": TextGenerationMetrics
    }
    
    if task_type not in evaluators:
        raise ValueError(f"不支持的任务类型: {task_type}. 支持的类型: {list(evaluators.keys())}")
    
    return evaluators[task_type](**kwargs)


def create_benchmark(benchmark_name: str, **kwargs):
    """
    创建基准测试的便捷函数
    
    Args:
        benchmark_name: 基准名称 ("glue")
        **kwargs: 其他参数
        
    Returns:
        相应的基准测试实例
        
    Example:
        >>> benchmark = create_benchmark("glue")
        >>> results = benchmark.evaluate_task("sst2", pred, ref)
    """
    if not GLUE_AVAILABLE:
        raise ImportError("GLUE基准模块不可用")
    
    benchmarks = {
        "glue": GLUEBenchmark,
    }
    
    if benchmark_name not in benchmarks:
        raise ValueError(f"不支持的基准: {benchmark_name}. 支持的基准: {list(benchmarks.keys())}")
    
    return benchmarks[benchmark_name](**kwargs)


def demo():
    """运行完整的评估系统演示"""
    print("🚀 标准化评估系统完整演示")
    print("=" * 60)
    
    # 显示系统信息
    info = get_evaluation_info()
    print("📋 系统信息:")
    print(f"  名称: {info['name']}")
    print(f"  版本: {info['version']}")
    print(f"  描述: {info['description']}")
    
    print(f"\n🧩 组件状态:")
    for component, available in info['components'].items():
        status = "✅ 可用" if available else "❌ 不可用"
        print(f"  {component}: {status}")
    
    # 显示可用指标
    print(f"\n📊 可用评估指标:")
    metrics = list_available_metrics()
    for metric in metrics:
        print(f"  • {metric['name']}: {metric['description']}")
        print(f"    指标: {', '.join(metric['metrics'])}")
    
    # 显示可用基准
    print(f"\n🏆 可用基准测试:")
    benchmarks = list_available_benchmarks()
    for benchmark in benchmarks:
        print(f"  • {benchmark['name']}: {benchmark['description']}")
        print(f"    任务: {', '.join(benchmark['tasks'])}")
    
    # 运行各组件演示
    if METRICS_AVAILABLE and metrics_demo:
        print(f"\n" + "="*60)
        print("📊 评估指标演示")
        print("="*60)
        try:
            metrics_demo()
        except Exception as e:
            print(f"❌ 评估指标演示失败: {e}")
    
    if GLUE_AVAILABLE and glue_demo:
        print(f"\n" + "="*60)
        print("🏆 GLUE基准演示")
        print("="*60)
        try:
            glue_demo()
        except Exception as e:
            print(f"❌ GLUE基准演示失败: {e}")
    
    print(f"\n" + "="*60)
    print("🎉 评估系统演示完成!")
    print("="*60)


if __name__ == "__main__":
    demo()