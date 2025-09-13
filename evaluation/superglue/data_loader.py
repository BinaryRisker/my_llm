"""
SuperGLUE数据加载器

负责从各种数据源加载SuperGLUE任务数据，支持：
- Hugging Face datasets
- 本地JSON文件
- CSV文件
- 数据预处理和格式标准化
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SuperGLUEDataLoader:
    """SuperGLUE数据加载器"""
    
    def __init__(self, data_dir: Optional[str] = None, use_cache: bool = True):
        """
        Args:
            data_dir: 数据目录路径
            use_cache: 是否使用缓存
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.use_cache = use_cache
        self.cache = {}
    
    def load_task_data(self, 
                      task_name: str, 
                      split: str = 'validation',
                      data_source: str = 'local',
                      max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        加载指定任务的数据
        
        Args:
            task_name: 任务名称
            split: 数据分割 ('train', 'validation', 'test')
            data_source: 数据源类型 ('local', 'huggingface', 'mock')
            max_samples: 最大样本数量
            
        Returns:
            数据样本列表
        """
        cache_key = f"{task_name}_{split}_{data_source}"
        
        # 检查缓存
        if self.use_cache and cache_key in self.cache:
            data = self.cache[cache_key]
        else:
            # 加载数据
            if data_source == 'huggingface':
                data = self._load_from_huggingface(task_name, split)
            elif data_source == 'local':
                data = self._load_from_local(task_name, split)
            elif data_source == 'mock':
                data = self._generate_mock_data(task_name, split)
            else:
                raise ValueError(f"未知的数据源: {data_source}")
            
            # 缓存数据
            if self.use_cache:
                self.cache[cache_key] = data
        
        # 限制样本数量
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]
        
        logger.info(f"加载了 {len(data)} 个 {task_name} {split} 样本")
        return data
    
    def _load_from_huggingface(self, task_name: str, split: str) -> List[Dict[str, Any]]:
        """从Hugging Face datasets加载数据"""
        try:
            from datasets import load_dataset
            
            # SuperGLUE数据集映射
            dataset_mapping = {
                'boolq': ('super_glue', 'boolq'),
                'cb': ('super_glue', 'cb'),
                'copa': ('super_glue', 'copa'),
                'multirc': ('super_glue', 'multirc'),
                'record': ('super_glue', 'record'),
                'rte': ('super_glue', 'rte'),
                'wic': ('super_glue', 'wic'),
                'wsc': ('super_glue', 'wsc'),
                'axb': ('super_glue', 'axb')
            }
            
            if task_name not in dataset_mapping:
                raise ValueError(f"未知任务: {task_name}")
            
            dataset_name, config_name = dataset_mapping[task_name]
            dataset = load_dataset(dataset_name, config_name, split=split)
            
            return [dict(example) for example in dataset]
            
        except ImportError:
            logger.warning("未安装datasets库，无法从Hugging Face加载数据")
            return self._generate_mock_data(task_name, split)
        except Exception as e:
            logger.error(f"从Hugging Face加载 {task_name} 数据失败: {e}")
            return self._generate_mock_data(task_name, split)
    
    def _load_from_local(self, task_name: str, split: str) -> List[Dict[str, Any]]:
        """从本地文件加载数据"""
        if not self.data_dir or not self.data_dir.exists():
            logger.warning(f"数据目录不存在: {self.data_dir}")
            return self._generate_mock_data(task_name, split)
        
        # 尝试不同的文件格式
        file_patterns = [
            f"{task_name}_{split}.json",
            f"{task_name}_{split}.jsonl",
            f"{task_name}_{split}.csv",
            f"{task_name}/{split}.json",
            f"{task_name}/{split}.jsonl",
            f"{task_name}/{split}.csv"
        ]
        
        for pattern in file_patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                return self._load_file(file_path)
        
        logger.warning(f"未找到 {task_name} {split} 数据文件")
        return self._generate_mock_data(task_name, split)
    
    def _load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载单个文件"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'data' in data:
                        return data['data']
                    else:
                        return [data]
            
            elif file_path.suffix == '.jsonl':
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                return data
            
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                return df.to_dict('records')
            
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {e}")
            return []
    
    def _generate_mock_data(self, task_name: str, split: str) -> List[Dict[str, Any]]:
        """生成模拟数据"""
        import numpy as np
        np.random.seed(42)
        
        num_samples = {'train': 100, 'validation': 20, 'test': 10}.get(split, 10)
        
        if task_name == 'boolq':
            return [
                {
                    'question': f'Is statement {i} about topic {i%5} correct?',
                    'passage': f'This is a detailed passage {i} discussing various aspects of topic {i%5}. ' +
                              f'The passage contains {np.random.randint(50, 200)} words of relevant information.',
                    'label': int(np.random.choice([0, 1]))
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'cb':
            return [
                {
                    'premise': f'Premise {i}: Something happened in situation {i%3}.',
                    'hypothesis': f'Hypothesis {i}: This implies a specific outcome for case {i%3}.',
                    'label': int(np.random.choice([0, 1, 2]))  # entailment, contradiction, neutral
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'copa':
            return [
                {
                    'premise': f'Event {i} occurred in context {i%4}.',
                    'choice1': f'This resulted in outcome A{i}.',
                    'choice2': f'This resulted in outcome B{i}.',
                    'question': np.random.choice(['cause', 'effect']),
                    'label': int(np.random.choice([0, 1]))
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'multirc':
            return [
                {
                    'passage': {
                        'text': f'Passage {i}: This is a comprehensive text about subject {i%6}. ' +
                               f'It discusses multiple aspects and contains several key points. ' +
                               f'The main focus is on understanding concept {i%6} thoroughly.'
                    },
                    'question': f'What is the main point about subject {i%6}?',
                    'answer': f'The main point is aspect {np.random.randint(1, 4)} of subject {i%6}.',
                    'label': int(np.random.choice([0, 1]))
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'record':
            return [
                {
                    'passage': {
                        'text': f'News article {i}: @highlight Entity{i%8} was involved in event {i}. ' +
                               f'@highlight The situation involved multiple parties. ' +
                               f'@highlight @placeholder played a crucial role in the outcome.'
                    },
                    'query': f'What role did @placeholder play in event {i}?',
                    'answers': [f'Entity{i%8}', f'Person{i%8}', f'Organization{i%8}'][:np.random.randint(1, 4)]
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'rte':
            return [
                {
                    'premise': f'Premise {i}: Situation {i%7} involves multiple factors.',
                    'hypothesis': f'Hypothesis {i}: Factor A leads to outcome B in situation {i%7}.',
                    'label': int(np.random.choice([0, 1]))  # entailment, not_entailment
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'wic':
            words = ['bank', 'chip', 'match', 'wave', 'draw', 'run', 'cut', 'play']
            return [
                {
                    'word': words[i % len(words)],
                    'sentence1': f'First sentence {i} uses {words[i % len(words)]} in context A.',
                    'sentence2': f'Second sentence {i} uses {words[i % len(words)]} in context B.',
                    'label': int(np.random.choice([0, 1]))
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'wsc':
            return [
                {
                    'text': f'In story {i}, the character went to the store. They bought something.',
                    'span1_text': 'character',
                    'span2_text': 'They',
                    'label': int(np.random.choice([0, 1]))
                }
                for i in range(num_samples)
            ]
        
        elif task_name == 'axb':
            return [
                {
                    'sentence1': f'Statement {i}: Condition X holds in scenario {i%5}.',
                    'sentence2': f'Statement {i}: Result Y follows from condition X.',
                    'label': int(np.random.choice([0, 1, 2]))
                }
                for i in range(num_samples)
            ]
        
        else:
            return []
    
    def preprocess_data(self, 
                       task_name: str, 
                       data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """预处理数据，标准化格式"""
        processed_data = []
        
        for example in data:
            try:
                processed_example = self._preprocess_example(task_name, example)
                if processed_example:
                    processed_data.append(processed_example)
            except Exception as e:
                logger.warning(f"预处理样本失败 {task_name}: {e}")
                continue
        
        logger.info(f"预处理完成，保留 {len(processed_data)} 个有效样本")
        return processed_data
    
    def _preprocess_example(self, 
                           task_name: str, 
                           example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """预处理单个样本"""
        if task_name == 'boolq':
            return {
                'question': str(example.get('question', '')),
                'passage': str(example.get('passage', '')),
                'label': int(example.get('label', 0))
            }
        
        elif task_name == 'cb':
            return {
                'premise': str(example.get('premise', '')),
                'hypothesis': str(example.get('hypothesis', '')),
                'label': int(example.get('label', 2))  # 默认为neutral
            }
        
        elif task_name == 'copa':
            return {
                'premise': str(example.get('premise', '')),
                'choice1': str(example.get('choice1', '')),
                'choice2': str(example.get('choice2', '')),
                'question': str(example.get('question', 'effect')),
                'label': int(example.get('label', 0))
            }
        
        # 其他任务的预处理逻辑...
        # 这里简化处理，实际使用时需要根据具体数据格式完善
        
        return example
    
    def get_dataset_info(self, task_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            'boolq': {
                'description': 'Boolean Questions dataset',
                'num_classes': 2,
                'input_fields': ['question', 'passage'],
                'target_field': 'label'
            },
            'cb': {
                'description': 'CommitmentBank dataset', 
                'num_classes': 3,
                'input_fields': ['premise', 'hypothesis'],
                'target_field': 'label'
            },
            'copa': {
                'description': 'Choice of Plausible Alternatives',
                'num_classes': 2,
                'input_fields': ['premise', 'choice1', 'choice2', 'question'],
                'target_field': 'label'
            }
            # 可以添加更多任务信息...
        }
        
        return info.get(task_name, {
            'description': f'SuperGLUE task: {task_name}',
            'num_classes': 2,
            'input_fields': ['text'],
            'target_field': 'label'
        })
    
    def validate_data(self, 
                     task_name: str, 
                     data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """验证数据格式"""
        dataset_info = self.get_dataset_info(task_name)
        required_fields = dataset_info.get('input_fields', []) + [dataset_info.get('target_field', 'label')]
        
        errors = []
        
        if not data:
            errors.append("数据为空")
            return False, errors
        
        for i, example in enumerate(data[:10]):  # 只检查前10个样本
            for field in required_fields:
                if field not in example:
                    errors.append(f"样本 {i} 缺少字段: {field}")
        
        return len(errors) == 0, errors