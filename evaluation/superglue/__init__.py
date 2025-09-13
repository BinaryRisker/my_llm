"""
SuperGLUE基准测试模块

SuperGLUE (Super General Language Understanding Evaluation) 是一个更具挑战性的
自然语言理解评估基准，包含9个任务：

1. BoolQ - 布尔问题回答
2. CB - CommitmentBank 蕴含关系
3. COPA - Choice of Plausible Alternatives 因果推理
4. MultiRC - Multi-Sentence Reading Comprehension 多句阅读理解
5. ReCoRD - Reading Comprehension with Commonsense Reasoning Dataset
6. RTE - Recognizing Textual Entailment 文本蕴含识别
7. WiC - Words in Context 上下文中的词汇
8. WSC - Winograd Schema Challenge 代词消解
9. AX-b - Broad Coverage Diagnostic 广覆盖诊断任务

本模块提供了完整的数据加载、预处理、评估和指标计算功能。
"""

from .tasks import (
    BoolQTask,
    CBTask, 
    COPATask,
    MultiRCTask,
    ReCoRDTask,
    RTETask,
    WiCTask,
    WSCTask,
    AXbTask
)

from .evaluator import SuperGLUEEvaluator
from .metrics import SuperGLUEMetrics
from .data_loader import SuperGLUEDataLoader
from .processor import SuperGLUEProcessor

__all__ = [
    # 任务类
    'BoolQTask',
    'CBTask',
    'COPATask', 
    'MultiRCTask',
    'ReCoRDTask',
    'RTETask',
    'WiCTask',
    'WSCTask',
    'AXbTask',
    
    # 核心组件
    'SuperGLUEEvaluator',
    'SuperGLUEMetrics',
    'SuperGLUEDataLoader',
    'SuperGLUEProcessor'
]

# SuperGLUE任务配置
SUPERGLUE_TASKS = {
    'boolq': {
        'name': 'BoolQ',
        'description': 'Boolean Questions - 布尔问题回答',
        'task_type': 'classification',
        'num_classes': 2,
        'metric': 'accuracy'
    },
    'cb': {
        'name': 'CommitmentBank',
        'description': 'CommitmentBank - 蕴含关系判断',
        'task_type': 'classification',
        'num_classes': 3,
        'metric': 'f1_macro'
    },
    'copa': {
        'name': 'COPA',
        'description': 'Choice of Plausible Alternatives - 因果推理',
        'task_type': 'classification',
        'num_classes': 2,
        'metric': 'accuracy'
    },
    'multirc': {
        'name': 'MultiRC',
        'description': 'Multi-Sentence Reading Comprehension - 多句阅读理解',
        'task_type': 'classification',
        'num_classes': 2,
        'metric': 'f1_macro'
    },
    'record': {
        'name': 'ReCoRD',
        'description': 'Reading Comprehension with Commonsense Reasoning',
        'task_type': 'generation',
        'num_classes': None,
        'metric': 'em_f1'
    },
    'rte': {
        'name': 'RTE',
        'description': 'Recognizing Textual Entailment - 文本蕴含识别',
        'task_type': 'classification', 
        'num_classes': 2,
        'metric': 'accuracy'
    },
    'wic': {
        'name': 'WiC',
        'description': 'Words in Context - 上下文中的词汇',
        'task_type': 'classification',
        'num_classes': 2,
        'metric': 'accuracy'
    },
    'wsc': {
        'name': 'WSC',
        'description': 'Winograd Schema Challenge - 代词消解',
        'task_type': 'classification',
        'num_classes': 2, 
        'metric': 'accuracy'
    },
    'axb': {
        'name': 'AX-b',
        'description': 'Broad Coverage Diagnostic - 广覆盖诊断',
        'task_type': 'classification',
        'num_classes': 3,
        'metric': 'matthews_correlation'
    }
}

def get_task_info(task_name):
    """获取任务信息"""
    return SUPERGLUE_TASKS.get(task_name.lower(), None)

def list_all_tasks():
    """列出所有SuperGLUE任务"""
    return list(SUPERGLUE_TASKS.keys())