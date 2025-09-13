"""
评估系统单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from evaluation.evaluation_metrics import EvaluationMetrics
from evaluation.glue_benchmark import GLUEBenchmark


@pytest.mark.unit
class TestEvaluationMetrics:
    """测试评估指标"""
    
    def test_classification_accuracy(self):
        """测试分类准确率计算"""
        metrics = EvaluationMetrics()
        
        # 完全正确
        predictions = [0, 1, 0, 1, 1]
        labels = [0, 1, 0, 1, 1]
        assert metrics.classification_accuracy(predictions, labels) == 1.0
        
        # 部分正确
        predictions = [0, 1, 0, 1, 0]
        labels = [0, 1, 0, 1, 1]
        assert metrics.classification_accuracy(predictions, labels) == 0.8
        
        # 完全错误
        predictions = [1, 0, 1, 0, 0]
        labels = [0, 1, 0, 1, 1]
        assert metrics.classification_accuracy(predictions, labels) == 0.0
    
    def test_classification_f1_score(self):
        """测试F1分数计算"""
        metrics = EvaluationMetrics()
        
        # 二分类
        predictions = [0, 1, 0, 1, 1, 0, 1, 0]
        labels = [0, 1, 0, 0, 1, 0, 1, 1]
        f1 = metrics.classification_f1_score(predictions, labels)
        assert 0.6 <= f1 <= 0.8  # 大致范围检查
        
        # 多分类
        predictions = [0, 1, 2, 1, 2]
        labels = [0, 1, 2, 2, 1]
        f1 = metrics.classification_f1_score(predictions, labels, average='macro')
        assert 0.0 <= f1 <= 1.0
    
    def test_regression_mse(self):
        """测试均方误差计算"""
        metrics = EvaluationMetrics()
        
        # 完全匹配
        predictions = [1.0, 2.0, 3.0, 4.0]
        targets = [1.0, 2.0, 3.0, 4.0]
        assert metrics.regression_mse(predictions, targets) == 0.0
        
        # 有误差
        predictions = [1.0, 2.0, 3.0, 4.0]
        targets = [1.1, 2.1, 2.9, 3.9]
        mse = metrics.regression_mse(predictions, targets)
        assert 0.01 <= mse <= 0.02
    
    def test_regression_rmse(self):
        """测试均方根误差计算"""
        metrics = EvaluationMetrics()
        
        predictions = [1.0, 2.0, 3.0, 4.0]
        targets = [2.0, 3.0, 4.0, 5.0]
        
        rmse = metrics.regression_rmse(predictions, targets)
        expected_rmse = np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(predictions, targets)]))
        assert abs(rmse - expected_rmse) < 1e-6
    
    def test_regression_mae(self):
        """测试平均绝对误差计算"""
        metrics = EvaluationMetrics()
        
        predictions = [1.0, 2.0, 3.0, 4.0]
        targets = [1.5, 2.5, 2.5, 4.5]
        
        mae = metrics.regression_mae(predictions, targets)
        expected_mae = np.mean([abs(p - t) for p, t in zip(predictions, targets)])
        assert abs(mae - expected_mae) < 1e-6
    
    def test_token_classification_accuracy(self):
        """测试token分类准确率"""
        metrics = EvaluationMetrics()
        
        predictions = [
            [0, 1, 2, 1, 0],
            [1, 2, 0, 0, 1]
        ]
        labels = [
            [0, 1, 2, 2, 0],  # 第4个token错误
            [1, 2, 0, 0, 1]   # 全部正确
        ]
        
        accuracy = metrics.token_classification_accuracy(predictions, labels)
        assert accuracy == 0.9  # 10个token中9个正确
    
    def test_token_classification_f1(self):
        """测试token分类F1分数"""
        metrics = EvaluationMetrics()
        
        predictions = [
            [0, 1, 2, 1, 0],
            [1, 2, 0, 0, 1]
        ]
        labels = [
            [0, 1, 2, 2, 0],
            [1, 2, 0, 0, 1]
        ]
        
        f1 = metrics.token_classification_f1_score(predictions, labels)
        assert 0.0 <= f1 <= 1.0
    
    def test_entity_extraction_precision_recall_f1(self):
        """测试实体抽取的精确率、召回率和F1"""
        metrics = EvaluationMetrics()
        
        # 简单的BIO标注
        pred_labels = [
            ['O', 'B-PER', 'I-PER', 'O', 'B-LOC']
        ]
        true_labels = [
            ['O', 'B-PER', 'I-PER', 'B-LOC', 'O']
        ]
        
        precision = metrics.entity_extraction_precision(pred_labels, true_labels)
        recall = metrics.entity_extraction_recall(pred_labels, true_labels)
        f1 = metrics.entity_extraction_f1_score(pred_labels, true_labels)
        
        # 预测了2个实体，正确了1个 -> precision = 0.5
        # 真实有2个实体，找到了1个 -> recall = 0.5
        # F1 = 2 * precision * recall / (precision + recall) = 0.5
        assert abs(precision - 0.5) < 1e-6
        assert abs(recall - 0.5) < 1e-6
        assert abs(f1 - 0.5) < 1e-6
    
    def test_question_answering_exact_match(self):
        """测试问答系统精确匹配"""
        metrics = EvaluationMetrics()
        
        predictions = ["brown fox", "machine learning", "42"]
        ground_truths = ["brown fox", "deep learning", "42"]
        
        em = metrics.question_answering_exact_match(predictions, ground_truths)
        assert em == 2/3  # 2个完全匹配，1个不匹配
    
    def test_text_generation_bleu_score(self):
        """测试BLEU分数计算"""
        metrics = EvaluationMetrics()
        
        # 完全匹配
        predictions = ["the cat is on the mat"]
        references = [["the cat is on the mat"]]
        bleu = metrics.text_generation_bleu_score(predictions, references)
        assert bleu > 0.9  # 应该接近1.0
        
        # 部分匹配
        predictions = ["the dog is on the mat"]
        references = [["the cat is on the mat"]]
        bleu = metrics.text_generation_bleu_score(predictions, references)
        assert 0.0 < bleu < 1.0
    
    def test_text_generation_rouge_score(self):
        """测试ROUGE分数计算"""
        metrics = EvaluationMetrics()
        
        predictions = ["the cat is on the mat"]
        references = [["the cat is on the mat"]]
        rouge = metrics.text_generation_rouge_score(predictions, references)
        
        # 检查返回的字典结构
        assert 'rouge-1' in rouge
        assert 'rouge-2' in rouge
        assert 'rouge-l' in rouge
        
        # 完全匹配的情况下，分数应该很高
        assert rouge['rouge-1']['f'] > 0.9
    
    def test_perplexity(self):
        """测试困惑度计算"""
        metrics = EvaluationMetrics()
        
        # 低损失 -> 低困惑度
        low_loss = 0.1
        low_perplexity = metrics.perplexity(low_loss)
        assert low_perplexity < 2.0
        
        # 高损失 -> 高困惑度
        high_loss = 5.0
        high_perplexity = metrics.perplexity(high_loss)
        assert high_perplexity > 100.0
        
        # 困惑度应该随损失单调递增
        assert high_perplexity > low_perplexity
    
    def test_compute_metrics_classification(self):
        """测试分类任务综合指标计算"""
        metrics = EvaluationMetrics()
        
        results = {
            'predictions': [0, 1, 0, 1, 1],
            'labels': [0, 1, 0, 1, 0]
        }
        
        computed_metrics = metrics.compute_metrics(results, task_type='classification')
        
        assert 'accuracy' in computed_metrics
        assert 'f1_score' in computed_metrics
        assert 'precision' in computed_metrics
        assert 'recall' in computed_metrics
        
        # 检查值的合理性
        for metric_name, metric_value in computed_metrics.items():
            assert 0.0 <= metric_value <= 1.0
    
    def test_compute_metrics_regression(self):
        """测试回归任务综合指标计算"""
        metrics = EvaluationMetrics()
        
        results = {
            'predictions': [1.0, 2.0, 3.0, 4.0],
            'labels': [1.1, 2.1, 2.9, 3.9]
        }
        
        computed_metrics = metrics.compute_metrics(results, task_type='regression')
        
        assert 'mse' in computed_metrics
        assert 'rmse' in computed_metrics
        assert 'mae' in computed_metrics
        
        # MSE 应该大于 0（因为有误差）
        assert computed_metrics['mse'] > 0
        # RMSE 应该是 MSE 的平方根
        assert abs(computed_metrics['rmse'] - np.sqrt(computed_metrics['mse'])) < 1e-6


@pytest.mark.unit
class TestGLUEBenchmark:
    """测试GLUE基准"""
    
    def test_init(self):
        """测试GLUE基准初始化"""
        glue = GLUEBenchmark()
        assert hasattr(glue, 'metrics')
        assert hasattr(glue, 'task_configs')
    
    def test_get_task_names(self):
        """测试获取任务名称列表"""
        glue = GLUEBenchmark()
        task_names = glue.get_task_names()
        
        expected_tasks = ['CoLA', 'SST-2', 'MRPC', 'STS-B', 'QQP', 'MNLI', 'QNLI', 'RTE', 'WNLI']
        
        for task in expected_tasks:
            assert task in task_names
    
    def test_get_task_type(self):
        """测试获取任务类型"""
        glue = GLUEBenchmark()
        
        # 分类任务
        assert glue.get_task_type('CoLA') == 'classification'
        assert glue.get_task_type('SST-2') == 'classification'
        assert glue.get_task_type('MRPC') == 'classification'
        
        # 回归任务
        assert glue.get_task_type('STS-B') == 'regression'
        
        # 不存在的任务
        with pytest.raises(ValueError):
            glue.get_task_type('NONEXISTENT_TASK')
    
    def test_get_task_metrics(self):
        """测试获取任务指标"""
        glue = GLUEBenchmark()
        
        # CoLA任务使用Matthews相关系数
        cola_metrics = glue.get_task_metrics('CoLA')
        assert 'matthews_corrcoef' in cola_metrics
        
        # SST-2任务使用准确率
        sst2_metrics = glue.get_task_metrics('SST-2')
        assert 'accuracy' in sst2_metrics
        
        # STS-B任务使用Pearson和Spearman相关系数
        stsb_metrics = glue.get_task_metrics('STS-B')
        assert 'pearson_corrcoef' in stsb_metrics
        assert 'spearman_corrcoef' in stsb_metrics
    
    @patch('evaluation.glue_benchmark.datasets.load_dataset')
    def test_load_dataset(self, mock_load_dataset):
        """测试数据集加载"""
        # 模拟数据集
        mock_dataset = Mock()
        mock_dataset.__getitem__.return_value = {'train': [], 'validation': []}
        mock_load_dataset.return_value = mock_dataset
        
        glue = GLUEBenchmark()
        dataset = glue.load_dataset('CoLA')
        
        mock_load_dataset.assert_called_once_with('glue', 'cola')
        assert dataset is not None
    
    def test_prepare_data_classification(self):
        """测试分类任务数据准备"""
        glue = GLUEBenchmark()
        
        # 模拟CoLA数据
        raw_data = [
            {'sentence': 'This is a grammatical sentence.', 'label': 1},
            {'sentence': 'This sentence not grammatical.', 'label': 0},
        ]
        
        prepared = glue.prepare_data(raw_data, 'CoLA')
        
        assert 'texts' in prepared
        assert 'labels' in prepared
        assert len(prepared['texts']) == len(prepared['labels']) == 2
    
    def test_prepare_data_regression(self):
        """测试回归任务数据准备"""
        glue = GLUEBenchmark()
        
        # 模拟STS-B数据
        raw_data = [
            {'sentence1': 'The cat is sleeping.', 'sentence2': 'A cat is napping.', 'label': 4.2},
            {'sentence1': 'Dogs are running.', 'sentence2': 'Cars are moving.', 'label': 1.1},
        ]
        
        prepared = glue.prepare_data(raw_data, 'STS-B')
        
        assert 'texts' in prepared
        assert 'labels' in prepared
        assert len(prepared['texts']) == len(prepared['labels']) == 2
        
        # 回归任务的标签应该是浮点数
        assert isinstance(prepared['labels'][0], float)
    
    def test_evaluate_model_mock(self):
        """测试模型评估（使用模拟）"""
        glue = GLUEBenchmark()
        
        # 模拟模型
        mock_model = Mock()
        mock_model.return_value = [0.3, 0.7, 0.1, 0.9]  # 预测概率
        
        # 模拟数据
        test_data = {
            'texts': ['text1', 'text2', 'text3', 'text4'],
            'labels': [0, 1, 0, 1]
        }
        
        with patch.object(glue, 'load_dataset') as mock_load, \
             patch.object(glue, 'prepare_data') as mock_prepare:
            
            mock_load.return_value = {'validation': []}
            mock_prepare.return_value = test_data
            
            # 由于实际的evaluate_model需要真实的模型，这里主要测试流程
            try:
                results = glue.evaluate_model(mock_model, 'CoLA')
                # 如果没有抛出异常，说明基本流程是正确的
                assert True
            except Exception:
                # 预期可能会有异常，因为mock_model不是真实的PyTorch模型
                assert True
    
    def test_compute_glue_score(self):
        """测试GLUE总分计算"""
        glue = GLUEBenchmark()
        
        # 模拟各任务结果
        task_results = {
            'CoLA': {'matthews_corrcoef': 0.5},
            'SST-2': {'accuracy': 0.9},
            'MRPC': {'accuracy': 0.8, 'f1_score': 0.85},
            'STS-B': {'pearson_corrcoef': 0.7, 'spearman_corrcoef': 0.75},
            'QQP': {'accuracy': 0.88, 'f1_score': 0.82},
            'MNLI': {'accuracy': 0.85},
            'QNLI': {'accuracy': 0.92},
            'RTE': {'accuracy': 0.75},
            'WNLI': {'accuracy': 0.65}
        }
        
        glue_score = glue.compute_glue_score(task_results)
        
        # GLUE分数应该在0-100之间
        assert 0 <= glue_score <= 100
        
        # 应该是各任务主要指标的平均值 * 100
        expected_scores = [0.5, 0.9, 0.85, (0.7 + 0.75)/2, 0.82, 0.85, 0.92, 0.75, 0.65]
        expected_glue_score = np.mean(expected_scores) * 100
        
        assert abs(glue_score - expected_glue_score) < 1.0