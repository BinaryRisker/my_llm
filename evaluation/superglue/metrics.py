"""
SuperGLUE指标计算模块

提供各种SuperGLUE任务的评估指标计算，包括：
- 准确率 (Accuracy)
- F1分数 (F1 Score)
- 精确匹配和F1 (Exact Match & F1)
- 马修斯相关系数 (Matthews Correlation Coefficient)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import re
import string


def normalize_answer(s: str) -> str:
    """标准化答案文本"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """计算精确匹配分数"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score_single(prediction: str, ground_truth: str) -> float:
    """计算单个样本的F1分数"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    """计算准确率"""
    if len(predictions) != len(labels):
        raise ValueError("预测和标签长度不匹配")
    
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)


def compute_f1_macro(predictions: List[int], labels: List[int], num_classes: int) -> float:
    """计算宏平均F1分数"""
    if len(predictions) != len(labels):
        raise ValueError("预测和标签长度不匹配")
    
    f1_scores = []
    
    for class_idx in range(num_classes):
        # 计算每个类别的精确率和召回率
        tp = sum(1 for p, l in zip(predictions, labels) if p == class_idx and l == class_idx)
        fp = sum(1 for p, l in zip(predictions, labels) if p == class_idx and l != class_idx)
        fn = sum(1 for p, l in zip(predictions, labels) if p != class_idx and l == class_idx)
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def compute_matthews_correlation(predictions: List[int], labels: List[int]) -> float:
    """计算马修斯相关系数 (MCC)"""
    if len(predictions) != len(labels):
        raise ValueError("预测和标签长度不匹配")
    
    # 转换为numpy数组进行计算
    y_true = np.array(labels)
    y_pred = np.array(predictions)
    
    # 获取所有类别
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    if n_classes == 2:
        # 二分类MCC
        tn = np.sum((y_true == 0) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    else:
        # 多分类MCC
        C = np.zeros((n_classes, n_classes))
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                C[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        t_k = np.sum(C, axis=1)  # 每个真实类别的总数
        p_k = np.sum(C, axis=0)  # 每个预测类别的总数
        c = np.trace(C)  # 正确预测的总数
        s = np.sum(C)  # 总样本数
        
        numerator = c * s - np.sum(t_k * p_k)
        denominator = np.sqrt((s**2 - np.sum(p_k**2)) * (s**2 - np.sum(t_k**2)))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


def compute_em_f1(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算精确匹配和F1分数"""
    em_scores = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        em_scores.append(exact_match_score(pred, ref))
        f1_scores.append(f1_score_single(pred, ref))
    
    return {
        'exact_match': np.mean(em_scores),
        'f1': np.mean(f1_scores)
    }


def compute_multirc_metrics(predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
    """计算MultiRC特定指标"""
    question_level_scores = []
    answer_level_scores = []
    
    # 按问题ID分组
    question_groups = {}
    for pred, ref in zip(predictions, references):
        q_id = ref['question_id']
        if q_id not in question_groups:
            question_groups[q_id] = {'preds': [], 'refs': []}
        question_groups[q_id]['preds'].append(pred['label'])
        question_groups[q_id]['refs'].append(ref['label'])
    
    # 计算问题级别的指标
    for q_id, group in question_groups.items():
        # 问题级别：所有答案都正确才算正确
        question_correct = all(p == r for p, r in zip(group['preds'], group['refs']))
        question_level_scores.append(int(question_correct))
        
        # 答案级别：计算F1
        f1 = compute_f1_macro(group['preds'], group['refs'], num_classes=2)
        answer_level_scores.append(f1)
    
    return {
        'question_accuracy': np.mean(question_level_scores),
        'answer_f1': np.mean(answer_level_scores)
    }


class SuperGLUEMetrics:
    """SuperGLUE指标计算器"""
    
    @staticmethod
    def compute_task_metrics(task_name: str, predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        """计算特定任务的指标"""
        task_name = task_name.lower()
        
        if task_name == 'boolq':
            return {'accuracy': compute_accuracy(predictions, references)}
        
        elif task_name == 'cb':
            return {'f1_macro': compute_f1_macro(predictions, references, num_classes=3)}
        
        elif task_name == 'copa':
            return {'accuracy': compute_accuracy(predictions, references)}
        
        elif task_name == 'multirc':
            return compute_multirc_metrics(predictions, references)
        
        elif task_name == 'record':
            return compute_em_f1(predictions, references)
        
        elif task_name == 'rte':
            return {'accuracy': compute_accuracy(predictions, references)}
        
        elif task_name == 'wic':
            return {'accuracy': compute_accuracy(predictions, references)}
        
        elif task_name == 'wsc':
            return {'accuracy': compute_accuracy(predictions, references)}
        
        elif task_name == 'axb':
            return {'matthews_correlation': compute_matthews_correlation(predictions, references)}
        
        else:
            raise ValueError(f"未知的任务: {task_name}")
    
    @staticmethod
    def compute_superglue_score(task_results: Dict[str, Dict[str, float]]) -> float:
        """计算总体SuperGLUE分数"""
        # 任务权重（官方权重）
        task_weights = {
            'boolq': 1.0,
            'cb': 1.0,
            'copa': 1.0,
            'multirc': 1.0,
            'record': 1.0,
            'rte': 1.0,
            'wic': 1.0,
            'wsc': 1.0,
            'axb': 0.0  # AX-b不参与最终分数计算
        }
        
        # 获取每个任务的主要指标
        task_scores = []
        total_weight = 0
        
        for task_name, results in task_results.items():
            if task_name not in task_weights or task_weights[task_name] == 0:
                continue
            
            # 根据任务类型选择主要指标
            if task_name == 'boolq':
                score = results['accuracy']
            elif task_name == 'cb':
                score = results['f1_macro'] 
            elif task_name == 'copa':
                score = results['accuracy']
            elif task_name == 'multirc':
                score = results['question_accuracy']  # 使用问题级别准确率
            elif task_name == 'record':
                score = results['f1']  # 使用F1分数
            elif task_name == 'rte':
                score = results['accuracy']
            elif task_name == 'wic':
                score = results['accuracy']
            elif task_name == 'wsc':
                score = results['accuracy']
            else:
                continue
            
            task_scores.append(score * task_weights[task_name])
            total_weight += task_weights[task_name]
        
        if total_weight == 0:
            return 0.0
        
        return sum(task_scores) / total_weight
    
    @staticmethod 
    def format_results(task_results: Dict[str, Dict[str, float]], 
                      overall_score: Optional[float] = None) -> str:
        """格式化结果输出"""
        lines = ["SuperGLUE评估结果", "=" * 50]
        
        for task_name, results in task_results.items():
            task_info = f"{task_name.upper()}:"
            lines.append(task_info)
            
            for metric_name, score in results.items():
                metric_line = f"  {metric_name}: {score:.4f}"
                lines.append(metric_line)
            
            lines.append("")
        
        if overall_score is not None:
            lines.extend([
                "=" * 50,
                f"总体SuperGLUE分数: {overall_score:.4f}",
                "=" * 50
            ])
        
        return "\n".join(lines)