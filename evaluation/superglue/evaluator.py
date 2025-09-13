"""
SuperGLUE评估器

统一的SuperGLUE评估框架，支持：
- 多任务并行评估
- T5模型集成
- 详细的结果分析和报告
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .tasks import get_task, convert_examples_to_text2text, extract_predictions
from .metrics import SuperGLUEMetrics
from . import SUPERGLUE_TASKS, get_task_info, list_all_tasks


class SuperGLUEEvaluator:
    """SuperGLUE评估器"""
    
    def __init__(self, 
                 model=None, 
                 tokenizer=None,
                 device='cpu',
                 max_length: int = 512,
                 batch_size: int = 8,
                 num_beams: int = 4,
                 cache_dir: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            model: T5模型实例
            tokenizer: T5分词器
            device: 计算设备
            max_length: 最大序列长度
            batch_size: 批处理大小
            num_beams: 束搜索宽度
            cache_dir: 缓存目录
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_beams = num_beams
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        # 支持的任务
        self.supported_tasks = list_all_tasks()
        
        # 结果存储
        self.results = {}
    
    def load_data(self, task_name: str, split: str = 'validation') -> List[Dict[str, Any]]:
        """
        加载任务数据
        
        Args:
            task_name: 任务名称
            split: 数据分割 ('train', 'validation', 'test')
            
        Returns:
            数据样本列表
        """
        # 这里是模拟数据加载，实际使用时应该对接真实的数据源
        # 比如 Hugging Face datasets 或本地文件
        
        if task_name not in self.supported_tasks:
            raise ValueError(f"不支持的任务: {task_name}")
        
        # 模拟数据结构
        mock_data = self._generate_mock_data(task_name, split)
        return mock_data
    
    def _generate_mock_data(self, task_name: str, split: str) -> List[Dict[str, Any]]:
        """生成模拟数据用于测试"""
        np.random.seed(42)  # 固定随机种子
        
        # 每个任务的样本数量
        num_samples = 10 if split == 'validation' else 5
        
        if task_name == 'boolq':
            return [
                {
                    'question': f'Is this statement {i} true?',
                    'passage': f'This is a test passage {i} that contains relevant information.',
                    'label': np.random.randint(0, 2)
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'cb':
            return [
                {
                    'premise': f'This is premise {i}.',
                    'hypothesis': f'This is hypothesis {i}.',
                    'label': np.random.randint(0, 3)
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'copa':
            return [
                {
                    'premise': f'The premise {i} happened.',
                    'choice1': f'Choice 1 for {i}',
                    'choice2': f'Choice 2 for {i}',
                    'question': 'cause' if i % 2 == 0 else 'effect',
                    'label': np.random.randint(0, 2)
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'multirc':
            return [
                {
                    'passage': {'text': f'This is passage {i} with multiple sentences.'},
                    'question': f'Question {i}?',
                    'answer': f'Answer option {i}',
                    'label': np.random.randint(0, 2)
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'record':
            return [
                {
                    'passage': {'text': f'Passage {i} @highlight Something happened to PLACEHOLDER.'},
                    'query': f'What happened to @placeholder in passage {i}?',
                    'answers': [f'answer_{i}']
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'rte':
            return [
                {
                    'premise': f'Premise {i}.',
                    'hypothesis': f'Hypothesis {i}.',
                    'label': np.random.randint(0, 2)
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'wic':
            return [
                {
                    'word': f'word{i}',
                    'sentence1': f'This is sentence 1 with word{i}.',
                    'sentence2': f'This is sentence 2 with word{i}.',
                    'label': np.random.randint(0, 2)
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'wsc':
            return [
                {
                    'text': f'Text {i} contains a pronoun.',
                    'span1_text': f'entity{i}',
                    'span2_text': f'pronoun{i}',
                    'label': np.random.randint(0, 2)
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'axb':
            return [
                {
                    'sentence1': f'Sentence 1 number {i}.',
                    'sentence2': f'Sentence 2 number {i}.',
                    'label': np.random.randint(0, 3)
                }
                for i in range(num_samples)
            ]
        
        else:
            return []
    
    def predict_batch(self, input_texts: List[str]) -> List[str]:
        """
        批量预测
        
        Args:
            input_texts: 输入文本列表
            
        Returns:
            生成的文本列表
        """
        if not self.model or not self.tokenizer:
            # 模拟预测，返回随机结果
            return [f"mock_prediction_{i}" for i in range(len(input_texts))]
        
        predictions = []
        
        for i in range(0, len(input_texts), self.batch_size):
            batch_texts = input_texts[i:i + self.batch_size]
            
            # 分词
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
            
            # 解码
            batch_predictions = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            predictions.extend(batch_predictions)
        
        return predictions
    
    def evaluate_task(self, 
                     task_name: str, 
                     split: str = 'validation',
                     save_predictions: bool = False) -> Dict[str, Any]:
        """
        评估单个任务
        
        Args:
            task_name: 任务名称
            split: 数据分割
            save_predictions: 是否保存预测结果
            
        Returns:
            评估结果字典
        """
        print(f"\n开始评估任务: {task_name}")
        
        # 加载数据
        data = self.load_data(task_name, split)
        if not data:
            return {'error': f'无法加载任务数据: {task_name}'}
        
        print(f"加载了 {len(data)} 个样本")
        
        # 转换为Text-to-Text格式
        converted_data = convert_examples_to_text2text(task_name, data)
        input_texts = [item['input_text'] for item in converted_data]
        target_texts = [item['target_text'] for item in converted_data]
        
        # 获取模型预测
        print("生成预测...")
        generated_texts = self.predict_batch(input_texts)
        
        # 提取结构化预测
        predictions = extract_predictions(task_name, generated_texts)
        
        # 提取真实标签
        if task_name == 'record':
            # ReCoRD任务使用字符串比较
            references = target_texts
        else:
            # 其他任务提取标签
            references = []
            for example in data:
                if 'label' in example:
                    references.append(example['label'])
                else:
                    # 处理没有label的情况
                    references.append(0)
        
        # 计算指标
        task_metrics = SuperGLUEMetrics.compute_task_metrics(
            task_name, predictions, references
        )
        
        # 保存预测结果
        result = {
            'task_name': task_name,
            'split': split,
            'num_samples': len(data),
            'metrics': task_metrics,
            'predictions': predictions if save_predictions else None,
            'references': references if save_predictions else None
        }
        
        if save_predictions and self.cache_dir:
            pred_file = self.cache_dir / f"{task_name}_{split}_predictions.json"
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'predictions': predictions,
                    'references': references,
                    'generated_texts': generated_texts,
                    'input_texts': input_texts
                }, f, indent=2, ensure_ascii=False)
        
        # 打印结果
        print(f"\n{task_name} 评估结果:")
        for metric_name, score in task_metrics.items():
            print(f"  {metric_name}: {score:.4f}")
        
        return result
    
    def evaluate_all(self, 
                    tasks: Optional[List[str]] = None,
                    split: str = 'validation',
                    save_predictions: bool = False) -> Dict[str, Any]:
        """评估所有或指定任务"""
        if tasks is None:
            tasks = self.supported_tasks
        
        all_results = {}
        task_metrics = {}
        
        print(f"开始评估 {len(tasks)} 个SuperGLUE任务")
        print("=" * 60)
        
        for task_name in tasks:
            try:
                result = self.evaluate_task(task_name, split, save_predictions)
                all_results[task_name] = result
                
                if 'metrics' in result:
                    task_metrics[task_name] = result['metrics']
                    
            except Exception as e:
                print(f"评估任务 {task_name} 时出错: {e}")
                all_results[task_name] = {'error': str(e)}
        
        # 计算总体分数
        if task_metrics:
            overall_score = SuperGLUEMetrics.compute_superglue_score(task_metrics)
        else:
            overall_score = 0.0
        
        # 汇总结果
        summary = {
            'overall_score': overall_score,
            'task_results': task_metrics,
            'detailed_results': all_results,
            'num_tasks_completed': len([r for r in all_results.values() if 'error' not in r]),
            'num_tasks_failed': len([r for r in all_results.values() if 'error' in r])
        }
        
        # 保存汇总结果
        if self.cache_dir:
            summary_file = self.cache_dir / f"superglue_summary_{split}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                # 移除不能序列化的对象
                serializable_summary = {
                    k: v for k, v in summary.items() 
                    if k != 'detailed_results'
                }
                serializable_summary['task_names'] = list(task_metrics.keys())
                json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
        
        # 打印最终结果
        print("\n" + "=" * 60)
        print(SuperGLUEMetrics.format_results(task_metrics, overall_score))
        
        return summary
    
    def get_task_info(self, task_name: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        return get_task_info(task_name)
    
    def list_supported_tasks(self) -> List[str]:
        """列出支持的任务"""
        return self.supported_tasks.copy()
    
    def create_evaluation_report(self, results: Dict[str, Any], output_file: str):
        """创建详细的评估报告"""
        report_lines = [
            "# SuperGLUE评估报告",
            "",
            f"总体分数: {results['overall_score']:.4f}",
            f"完成任务数: {results['num_tasks_completed']}",
            f"失败任务数: {results['num_tasks_failed']}",
            "",
            "## 各任务详细结果",
            ""
        ]
        
        for task_name, metrics in results['task_results'].items():
            task_info = self.get_task_info(task_name)
            report_lines.extend([
                f"### {task_name.upper()}",
                f"描述: {task_info['description'] if task_info else 'N/A'}",
                f"任务类型: {task_info['task_type'] if task_info else 'N/A'}",
                "",
                "指标:"
            ])
            
            for metric_name, score in metrics.items():
                report_lines.append(f"- {metric_name}: {score:.4f}")
            
            report_lines.append("")
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        print(f"评估报告已保存至: {output_file}")
