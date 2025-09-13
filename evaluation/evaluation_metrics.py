"""
标准化评估系统 - 评估指标模块
================================

实现各种NLP任务的标准评估指标，支持：
- 分类任务指标（准确率、精确率、召回率、F1分数）
- 序列标注任务指标（实体级F1、token级准确率）
- 生成任务指标（BLEU、ROUGE、METEOR）
- 问答任务指标（EM、F1）
- GLUE/SuperGLUE基准指标

参考标准：
- GLUE: https://gluebenchmark.com/
- SuperGLUE: https://super.gluebenchmark.com/
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/

使用方法:
    from evaluation.evaluation_metrics import ClassificationMetrics
    
    metrics = ClassificationMetrics()
    scores = metrics.compute(predictions, labels)
    print(f"Accuracy: {scores['accuracy']:.4f}")
"""

import math
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
import re

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseMetrics:
    """基础评估指标类"""
    
    def __init__(self):
        self.name = "base_metrics"
        self.description = "Base metrics class"
    
    def compute(self, predictions: Any, references: Any, **kwargs) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions: 模型预测结果
            references: 参考标准答案
            **kwargs: 其他参数
            
        Returns:
            评估结果字典
        """
        raise NotImplementedError("子类需要实现compute方法")
    
    def _safe_division(self, numerator: float, denominator: float) -> float:
        """安全除法，避免除零错误"""
        return numerator / denominator if denominator != 0 else 0.0


class ClassificationMetrics(BaseMetrics):
    """分类任务评估指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "classification_metrics"
        self.description = "Classification task evaluation metrics"
    
    def compute(self, predictions: List[int], references: List[int], **kwargs) -> Dict[str, float]:
        """
        计算分类任务指标
        
        Args:
            predictions: 预测标签列表
            references: 真实标签列表
            
        Returns:
            包含accuracy, precision, recall, f1等指标的字典
        """
        if len(predictions) != len(references):
            raise ValueError(f"预测结果长度({len(predictions)})与参考答案长度({len(references)})不匹配")
        
        results = {}
        
        # 基础准确率
        correct = sum(p == r for p, r in zip(predictions, references))
        results['accuracy'] = correct / len(predictions)
        
        if SKLEARN_AVAILABLE:
            # 使用sklearn计算更详细的指标
            results['accuracy'] = accuracy_score(references, predictions)
            
            # 计算精确率、召回率、F1分数
            precision, recall, f1, _ = precision_recall_fscore_support(
                references, predictions, average='weighted', zero_division=0
            )
            results['precision'] = precision
            results['recall'] = recall
            results['f1'] = f1
            
            # 宏平均F1
            _, _, macro_f1, _ = precision_recall_fscore_support(
                references, predictions, average='macro', zero_division=0
            )
            results['macro_f1'] = macro_f1
            
            # Matthews相关系数（适用于不平衡数据集）
            try:
                results['matthews_corr'] = matthews_corrcoef(references, predictions)
            except:
                results['matthews_corr'] = 0.0
        else:
            # 简化版本的指标计算
            results.update(self._compute_basic_metrics(predictions, references))
        
        return results
    
    def _compute_basic_metrics(self, predictions: List[int], references: List[int]) -> Dict[str, float]:
        """不依赖sklearn的基础指标计算"""
        # 获取所有类别
        all_labels = set(predictions + references)
        
        # 计算每个类别的TP, FP, FN
        per_class_metrics = {}
        for label in all_labels:
            tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
            fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
            fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)
            
            precision = self._safe_division(tp, tp + fp)
            recall = self._safe_division(tp, tp + fn)
            f1 = self._safe_division(2 * precision * recall, precision + recall)
            
            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # 计算加权平均
        total_samples = len(references)
        weighted_precision = sum(
            per_class_metrics[label]['precision'] * references.count(label) / total_samples
            for label in all_labels
        )
        weighted_recall = sum(
            per_class_metrics[label]['recall'] * references.count(label) / total_samples
            for label in all_labels
        )
        weighted_f1 = sum(
            per_class_metrics[label]['f1'] * references.count(label) / total_samples
            for label in all_labels
        )
        
        # 计算宏平均
        macro_precision = sum(per_class_metrics[label]['precision'] for label in all_labels) / len(all_labels)
        macro_recall = sum(per_class_metrics[label]['recall'] for label in all_labels) / len(all_labels)
        macro_f1 = sum(per_class_metrics[label]['f1'] for label in all_labels) / len(all_labels)
        
        return {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1,
            'macro_f1': macro_f1,
            'matthews_corr': 0.0  # 简化版不计算
        }


class RegressionMetrics(BaseMetrics):
    """回归任务评估指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "regression_metrics"
        self.description = "Regression task evaluation metrics"
    
    def compute(self, predictions: List[float], references: List[float], **kwargs) -> Dict[str, float]:
        """
        计算回归任务指标
        
        Args:
            predictions: 预测值列表
            references: 真实值列表
            
        Returns:
            包含MSE, MAE, RMSE, R²等指标的字典
        """
        if len(predictions) != len(references):
            raise ValueError(f"预测结果长度({len(predictions)})与参考答案长度({len(references)})不匹配")
        
        results = {}
        
        if SKLEARN_AVAILABLE:
            results['mse'] = mean_squared_error(references, predictions)
            results['mae'] = mean_absolute_error(references, predictions)
        else:
            # 手动计算
            results['mse'] = sum((p - r) ** 2 for p, r in zip(predictions, references)) / len(predictions)
            results['mae'] = sum(abs(p - r) for p, r in zip(predictions, references)) / len(predictions)
        
        results['rmse'] = math.sqrt(results['mse'])
        
        # 计算R²（决定系数）
        mean_ref = sum(references) / len(references)
        ss_res = sum((r - p) ** 2 for r, p in zip(references, predictions))
        ss_tot = sum((r - mean_ref) ** 2 for r in references)
        results['r2'] = 1 - self._safe_division(ss_res, ss_tot)
        
        return results


class TokenClassificationMetrics(BaseMetrics):
    """token分类任务评估指标（适用于NER等）"""
    
    def __init__(self):
        super().__init__()
        self.name = "token_classification_metrics"
        self.description = "Token classification evaluation metrics (NER, POS tagging)"
    
    def compute(self, 
                predictions: List[List[str]], 
                references: List[List[str]], 
                scheme: str = "IOB2",
                **kwargs) -> Dict[str, float]:
        """
        计算token分类指标
        
        Args:
            predictions: 预测标签序列列表
            references: 真实标签序列列表
            scheme: 标注方案 ("IOB2", "IOBES", "BILOU")
            
        Returns:
            包含token级和实体级准确率、F1等指标的字典
        """
        if len(predictions) != len(references):
            raise ValueError(f"预测结果数量({len(predictions)})与参考答案数量({len(references)})不匹配")
        
        results = {}
        
        # Token级别准确率
        total_tokens = 0
        correct_tokens = 0
        
        for pred_seq, ref_seq in zip(predictions, references):
            if len(pred_seq) != len(ref_seq):
                logger.warning(f"序列长度不匹配: pred={len(pred_seq)}, ref={len(ref_seq)}")
                min_len = min(len(pred_seq), len(ref_seq))
                pred_seq, ref_seq = pred_seq[:min_len], ref_seq[:min_len]
            
            total_tokens += len(ref_seq)
            correct_tokens += sum(p == r for p, r in zip(pred_seq, ref_seq))
        
        results['token_accuracy'] = self._safe_division(correct_tokens, total_tokens)
        
        # 实体级别评估
        pred_entities = self._extract_entities(predictions, scheme)
        ref_entities = self._extract_entities(references, scheme)
        
        entity_metrics = self._compute_entity_metrics(pred_entities, ref_entities)
        results.update(entity_metrics)
        
        return results
    
    def _extract_entities(self, sequences: List[List[str]], scheme: str) -> List[List[Tuple[int, int, str]]]:
        """从标注序列中提取实体"""
        all_entities = []
        
        for seq in sequences:
            entities = []
            current_entity = None
            
            for i, label in enumerate(seq):
                if scheme == "IOB2":
                    if label.startswith('B-'):
                        # 开始新实体
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = [i, i, label[2:]]
                    elif label.startswith('I-') and current_entity and label[2:] == current_entity[2]:
                        # 继续当前实体
                        current_entity[1] = i
                    else:
                        # 结束当前实体
                        if current_entity:
                            entities.append(tuple(current_entity))
                            current_entity = None
                
                elif scheme == "BILOU":
                    if label.startswith('B-'):
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = [i, i, label[2:]]
                    elif label.startswith('I-') and current_entity and label[2:] == current_entity[2]:
                        current_entity[1] = i
                    elif label.startswith('L-') and current_entity and label[2:] == current_entity[2]:
                        current_entity[1] = i
                        entities.append(tuple(current_entity))
                        current_entity = None
                    elif label.startswith('U-'):
                        if current_entity:
                            entities.append(current_entity)
                        entities.append((i, i, label[2:]))
                        current_entity = None
                    else:
                        if current_entity:
                            entities.append(tuple(current_entity))
                            current_entity = None
            
            # 处理序列末尾的实体
            if current_entity:
                entities.append(tuple(current_entity))
            
            all_entities.append(entities)
        
        return all_entities
    
    def _compute_entity_metrics(self, pred_entities: List[List[Tuple]], ref_entities: List[List[Tuple]]) -> Dict[str, float]:
        """计算实体级别指标"""
        pred_flat = []
        ref_flat = []
        
        # 为每个实体添加序列索引，确保可以hash
        for seq_idx, seq in enumerate(pred_entities):
            for entity in seq:
                if isinstance(entity, (list, tuple)) and len(entity) >= 3:
                    pred_flat.append((seq_idx, entity[0], entity[1], entity[2]))
        
        for seq_idx, seq in enumerate(ref_entities):
            for entity in seq:
                if isinstance(entity, (list, tuple)) and len(entity) >= 3:
                    ref_flat.append((seq_idx, entity[0], entity[1], entity[2]))
        
        pred_set = set(pred_flat)
        ref_set = set(ref_flat)
        
        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)
        
        precision = self._safe_division(tp, tp + fp)
        recall = self._safe_division(tp, tp + fn)
        f1 = self._safe_division(2 * precision * recall, precision + recall)
        
        return {
            'entity_precision': precision,
            'entity_recall': recall,
            'entity_f1': f1,
            'entity_count_pred': len(pred_flat),
            'entity_count_ref': len(ref_flat),
            'entity_count_correct': tp
        }


class QuestionAnsweringMetrics(BaseMetrics):
    """问答任务评估指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "question_answering_metrics"
        self.description = "Question answering evaluation metrics (SQuAD style)"
    
    def compute(self, 
                predictions: List[str], 
                references: List[str], 
                **kwargs) -> Dict[str, float]:
        """
        计算问答任务指标
        
        Args:
            predictions: 预测答案列表
            references: 参考答案列表
            
        Returns:
            包含EM（精确匹配）和F1分数的字典
        """
        if len(predictions) != len(references):
            raise ValueError(f"预测结果数量({len(predictions)})与参考答案数量({len(references)})不匹配")
        
        em_scores = []
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # 精确匹配
            em_scores.append(1.0 if self._normalize_answer(pred) == self._normalize_answer(ref) else 0.0)
            
            # F1分数
            f1_scores.append(self._compute_answer_f1(pred, ref))
        
        return {
            'exact_match': sum(em_scores) / len(em_scores),
            'f1': sum(f1_scores) / len(f1_scores)
        }
    
    def _normalize_answer(self, answer: str) -> str:
        """标准化答案文本"""
        # 转小写
        answer = answer.lower()
        
        # 移除标点符号
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # 移除多余空格
        answer = ' '.join(answer.split())
        
        # 移除冠词
        articles = {'a', 'an', 'the'}
        tokens = answer.split()
        tokens = [token for token in tokens if token not in articles]
        
        return ' '.join(tokens)
    
    def _compute_answer_f1(self, pred: str, ref: str) -> float:
        """计算答案F1分数"""
        pred_tokens = self._normalize_answer(pred).split()
        ref_tokens = self._normalize_answer(ref).split()
        
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        common = pred_counter & ref_counter
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        
        return self._safe_division(2 * precision * recall, precision + recall)


class TextGenerationMetrics(BaseMetrics):
    """文本生成任务评估指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "text_generation_metrics"
        self.description = "Text generation evaluation metrics (BLEU, ROUGE)"
    
    def compute(self, 
                predictions: List[str], 
                references: List[List[str]], 
                **kwargs) -> Dict[str, float]:
        """
        计算文本生成指标
        
        Args:
            predictions: 生成文本列表
            references: 参考文本列表（每个样本可以有多个参考）
            
        Returns:
            包含BLEU和ROUGE分数的字典
        """
        if len(predictions) != len(references):
            raise ValueError(f"预测结果数量({len(predictions)})与参考答案数量({len(references)})不匹配")
        
        results = {}
        
        # 计算BLEU分数
        bleu_scores = []
        for pred, refs in zip(predictions, references):
            bleu_scores.append(self._compute_bleu(pred, refs))
        
        results['bleu'] = sum(bleu_scores) / len(bleu_scores)
        
        # 计算ROUGE分数
        rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
        for pred, refs in zip(predictions, references):
            rouge = self._compute_rouge(pred, refs[0] if refs else "")  # 简化版只用第一个参考
            for key in rouge_scores:
                rouge_scores[key].append(rouge.get(key, 0.0))
        
        for key in rouge_scores:
            results[key] = sum(rouge_scores[key]) / len(rouge_scores[key])
        
        return results
    
    def _compute_bleu(self, prediction: str, references: List[str], max_n: int = 4) -> float:
        """计算BLEU分数（简化版）"""
        if not prediction or not references:
            return 0.0
        
        pred_tokens = prediction.split()
        ref_tokens_list = [ref.split() for ref in references]
        
        if not pred_tokens:
            return 0.0
        
        # 计算n-gram精确度
        precisions = []
        for n in range(1, max_n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams_all = []
            for ref_tokens in ref_tokens_list:
                ref_ngrams_all.extend(self._get_ngrams(ref_tokens, n))
            
            if not pred_ngrams:
                precisions.append(0.0)
                continue
            
            ref_counter = Counter(ref_ngrams_all)
            pred_counter = Counter(pred_ngrams)
            
            common = pred_counter & ref_counter
            precision = sum(common.values()) / len(pred_ngrams)
            precisions.append(precision)
        
        if all(p == 0 for p in precisions):
            return 0.0
        
        # 几何平均
        bleu = math.exp(sum(math.log(p) if p > 0 else float('-inf') for p in precisions) / len(precisions))
        
        # 简化版不计算brevity penalty
        return bleu
    
    def _compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """计算ROUGE分数（简化版）"""
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        
        results = {}
        
        # ROUGE-1 (unigram)
        results['rouge-1'] = self._compute_rouge_n(pred_tokens, ref_tokens, 1)
        
        # ROUGE-2 (bigram)
        results['rouge-2'] = self._compute_rouge_n(pred_tokens, ref_tokens, 2)
        
        # ROUGE-L (longest common subsequence)
        results['rouge-l'] = self._compute_rouge_l(pred_tokens, ref_tokens)
        
        return results
    
    def _compute_rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """计算ROUGE-n分数"""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if not ref_ngrams:
            return 0.0
        
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        common = pred_counter & ref_counter
        overlap = sum(common.values())
        
        return overlap / len(ref_ngrams)
    
    def _compute_rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """计算ROUGE-L分数（基于LCS）"""
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """获取n-gram列表"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


def demo():
    """评估指标演示"""
    print("🚀 标准化评估系统演示")
    print("=" * 50)
    
    # 1. 分类任务演示
    print("\n📊 1. 分类任务评估")
    print("-" * 30)
    
    classification_metrics = ClassificationMetrics()
    pred_labels = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1]
    true_labels = [0, 1, 2, 2, 0, 1, 1, 0, 2, 1]
    
    class_results = classification_metrics.compute(pred_labels, true_labels)
    print(f"  准确率: {class_results['accuracy']:.4f}")
    print(f"  精确率: {class_results['precision']:.4f}")
    print(f"  召回率: {class_results['recall']:.4f}")
    print(f"  F1分数: {class_results['f1']:.4f}")
    print(f"  宏平均F1: {class_results['macro_f1']:.4f}")
    
    # 2. 回归任务演示
    print("\n📈 2. 回归任务评估")
    print("-" * 30)
    
    regression_metrics = RegressionMetrics()
    pred_values = [2.5, 3.1, 1.8, 4.2, 2.9, 3.7, 2.1, 3.5, 4.1, 2.8]
    true_values = [2.3, 3.2, 1.9, 4.0, 3.1, 3.6, 2.0, 3.4, 4.3, 2.7]
    
    reg_results = regression_metrics.compute(pred_values, true_values)
    print(f"  MSE: {reg_results['mse']:.4f}")
    print(f"  MAE: {reg_results['mae']:.4f}")
    print(f"  RMSE: {reg_results['rmse']:.4f}")
    print(f"  R²: {reg_results['r2']:.4f}")
    
    # 3. Token分类任务演示
    print("\n🏷️ 3. Token分类任务评估 (NER)")
    print("-" * 30)
    
    token_metrics = TokenClassificationMetrics()
    pred_sequences = [
        ["B-PER", "I-PER", "O", "B-LOC", "O"],
        ["O", "B-ORG", "I-ORG", "O", "B-PER"]
    ]
    true_sequences = [
        ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"],
        ["O", "B-ORG", "I-ORG", "I-ORG", "B-PER"]
    ]
    
    token_results = token_metrics.compute(pred_sequences, true_sequences)
    print(f"  Token准确率: {token_results['token_accuracy']:.4f}")
    print(f"  实体精确率: {token_results['entity_precision']:.4f}")
    print(f"  实体召回率: {token_results['entity_recall']:.4f}")
    print(f"  实体F1: {token_results['entity_f1']:.4f}")
    print(f"  预测实体数: {token_results['entity_count_pred']}")
    print(f"  参考实体数: {token_results['entity_count_ref']}")
    
    # 4. 问答任务演示
    print("\n❓ 4. 问答任务评估")
    print("-" * 30)
    
    qa_metrics = QuestionAnsweringMetrics()
    pred_answers = ["Paris", "The capital of France", "France capital"]
    true_answers = ["Paris", "Paris", "Paris"]
    
    qa_results = qa_metrics.compute(pred_answers, true_answers)
    print(f"  精确匹配: {qa_results['exact_match']:.4f}")
    print(f"  F1分数: {qa_results['f1']:.4f}")
    
    # 5. 文本生成任务演示
    print("\n📝 5. 文本生成任务评估")
    print("-" * 30)
    
    generation_metrics = TextGenerationMetrics()
    pred_texts = [
        "The cat sat on the mat",
        "It is raining today"
    ]
    ref_texts = [
        ["A cat was sitting on the mat", "The cat is on the mat"],
        ["Today it is raining", "It rains today"]
    ]
    
    gen_results = generation_metrics.compute(pred_texts, ref_texts)
    print(f"  BLEU分数: {gen_results['bleu']:.4f}")
    print(f"  ROUGE-1: {gen_results['rouge-1']:.4f}")
    print(f"  ROUGE-2: {gen_results['rouge-2']:.4f}")
    print(f"  ROUGE-L: {gen_results['rouge-l']:.4f}")
    
    print(f"\n🎉 评估指标演示完成!")


if __name__ == "__main__":
    demo()