"""
数据预处理管道
==============

提供完整的数据预处理管道，支持：
- 多种数据格式的统一处理
- 文本清洗和标准化
- 多语言数据处理
- 数据验证和统计分析
- 管道式处理流程

使用方法:
    from utils.data_processing.preprocessing_pipeline import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.add_step('clean_text')
    preprocessor.add_step('tokenize')
    processed_data = preprocessor.process(raw_data)
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import unicodedata

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataStats:
    """数据统计信息"""
    total_samples: int = 0
    total_tokens: int = 0
    avg_tokens_per_sample: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0
    unique_tokens: int = 0
    vocabulary_size: int = 0
    language_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.language_distribution is None:
            self.language_distribution = {}


class TextCleaner:
    """文本清洗工具"""
    
    def __init__(self):
        # 编译正则表达式模式
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'(\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'),
            'whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]'),
            'numbers': re.compile(r'\d+'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),
            'html_tags': re.compile(r'<[^<]+?>')
        }
    
    def clean_text(self, text: str, 
                   remove_emails: bool = True,
                   remove_urls: bool = True,
                   remove_phones: bool = True,
                   normalize_whitespace: bool = True,
                   remove_html: bool = True,
                   normalize_unicode: bool = True,
                   lowercase: bool = False,
                   remove_special_chars: bool = False,
                   remove_numbers: bool = False,
                   fix_repeated_chars: bool = True) -> str:
        """
        清洗文本
        
        Args:
            text: 输入文本
            remove_emails: 移除邮箱地址
            remove_urls: 移除URL
            remove_phones: 移除电话号码
            normalize_whitespace: 标准化空白字符
            remove_html: 移除HTML标签
            normalize_unicode: Unicode标准化
            lowercase: 转换为小写
            remove_special_chars: 移除特殊字符
            remove_numbers: 移除数字
            fix_repeated_chars: 修复重复字符
            
        Returns:
            清洗后的文本
        """
        if not text:
            return text
        
        # Unicode标准化
        if normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # 移除HTML标签
        if remove_html:
            text = self.patterns['html_tags'].sub('', text)
        
        # 移除邮箱地址
        if remove_emails:
            text = self.patterns['email'].sub('[EMAIL]', text)
        
        # 移除URL
        if remove_urls:
            text = self.patterns['url'].sub('[URL]', text)
        
        # 移除电话号码
        if remove_phones:
            text = self.patterns['phone'].sub('[PHONE]', text)
        
        # 修复重复字符
        if fix_repeated_chars:
            text = self.patterns['repeated_chars'].sub(r'\1\1', text)
        
        # 移除数字
        if remove_numbers:
            text = self.patterns['numbers'].sub('[NUM]', text)
        
        # 移除特殊字符 (保留中日韩文字)
        if remove_special_chars:
            text = self.patterns['special_chars'].sub(' ', text)
        
        # 标准化空白字符
        if normalize_whitespace:
            text = self.patterns['whitespace'].sub(' ', text)
        
        # 转换为小写
        if lowercase:
            text = text.lower()
        
        return text.strip()


class LanguageDetector:
    """语言检测工具"""
    
    def __init__(self):
        # 简单的基于Unicode范围的语言检测
        self.language_patterns = {
            'en': re.compile(r'[a-zA-Z]'),
            'zh': re.compile(r'[\u4e00-\u9fff]'),
            'ja': re.compile(r'[\u3040-\u309f\u30a0-\u30ff]'),
            'ko': re.compile(r'[\uac00-\ud7af]'),
            'ar': re.compile(r'[\u0600-\u06ff]'),
            'ru': re.compile(r'[\u0400-\u04ff]'),
            'de': re.compile(r'[äöüßÄÖÜ]'),
            'fr': re.compile(r'[àâäéèêëïîôöùûüÿç]'),
            'es': re.compile(r'[áéíóúñü¡¿]'),
            'pt': re.compile(r'[áàâãéêíóôõúç]')
        }
    
    def detect_language(self, text: str, threshold: float = 0.1) -> str:
        """
        检测文本语言
        
        Args:
            text: 输入文本
            threshold: 字符比例阈值
            
        Returns:
            检测到的语言代码
        """
        if not text:
            return 'unknown'
        
        text_length = len(text)
        language_scores = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = pattern.findall(text)
            score = len(''.join(matches)) / text_length if text_length > 0 else 0
            language_scores[lang] = score
        
        # 找到最高分的语言
        best_lang = max(language_scores.items(), key=lambda x: x[1])
        
        if best_lang[1] >= threshold:
            return best_lang[0]
        else:
            return 'mixed' if len([s for s in language_scores.values() if s > threshold/2]) > 1 else 'unknown'


class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        self.validation_rules = []
    
    def add_rule(self, rule: Callable[[str], bool], name: str, description: str):
        """添加验证规则"""
        self.validation_rules.append({
            'rule': rule,
            'name': name,
            'description': description
        })
    
    def validate_text(self, text: str) -> Dict[str, Any]:
        """验证单个文本"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not text or not text.strip():
            results['is_valid'] = False
            results['errors'].append('Empty or whitespace-only text')
            return results
        
        # 应用所有验证规则
        for rule_info in self.validation_rules:
            try:
                if not rule_info['rule'](text):
                    results['warnings'].append(f"Failed rule: {rule_info['name']} - {rule_info['description']}")
            except Exception as e:
                results['errors'].append(f"Error in rule {rule_info['name']}: {str(e)}")
        
        return results
    
    def validate_dataset(self, texts: List[str]) -> Dict[str, Any]:
        """验证整个数据集"""
        results = {
            'total_samples': len(texts),
            'valid_samples': 0,
            'invalid_samples': 0,
            'warnings_count': 0,
            'errors_count': 0,
            'sample_results': []
        }
        
        for i, text in enumerate(texts):
            sample_result = self.validate_text(text)
            sample_result['index'] = i
            results['sample_results'].append(sample_result)
            
            if sample_result['is_valid']:
                results['valid_samples'] += 1
            else:
                results['invalid_samples'] += 1
            
            results['warnings_count'] += len(sample_result['warnings'])
            results['errors_count'] += len(sample_result['errors'])
        
        return results


class DataPreprocessor:
    """数据预处理器主类"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.language_detector = LanguageDetector()
        self.data_validator = DataValidator()
        
        # 处理步骤管道
        self.pipeline_steps = []
        
        # 内置处理步骤
        self.builtin_steps = {
            'clean_text': self._clean_text_step,
            'detect_language': self._detect_language_step,
            'validate_data': self._validate_data_step,
            'compute_stats': self._compute_stats_step,
            'filter_by_length': self._filter_by_length_step,
            'filter_by_language': self._filter_by_language_step,
            'deduplicate': self._deduplicate_step
        }
    
    def add_step(self, step_name: str, **kwargs):
        """添加处理步骤"""
        if step_name in self.builtin_steps:
            self.pipeline_steps.append((step_name, kwargs))
        else:
            raise ValueError(f"Unknown step: {step_name}")
    
    def add_custom_step(self, step_func: Callable, step_name: str, **kwargs):
        """添加自定义处理步骤"""
        self.pipeline_steps.append((step_func, kwargs))
    
    def clear_steps(self):
        """清空处理步骤"""
        self.pipeline_steps.clear()
    
    def process(self, data: Union[List[str], List[Dict[str, Any]]], 
                return_stats: bool = True) -> Tuple[List[Any], Optional[Dict[str, Any]]]:
        """
        执行数据预处理管道
        
        Args:
            data: 输入数据
            return_stats: 是否返回统计信息
            
        Returns:
            处理后的数据和统计信息
        """
        logger.info(f"开始数据预处理，输入样本数: {len(data)}")
        
        processed_data = data.copy()
        processing_stats = {
            'original_count': len(data),
            'final_count': 0,
            'step_results': {}
        }
        
        # 执行每个处理步骤
        for step_info in self.pipeline_steps:
            if isinstance(step_info[0], str):
                # 内置步骤
                step_name, kwargs = step_info
                step_func = self.builtin_steps[step_name]
                logger.info(f"执行步骤: {step_name}")
            else:
                # 自定义步骤
                step_func, kwargs = step_info
                step_name = getattr(step_func, '__name__', 'custom_step')
                logger.info(f"执行自定义步骤: {step_name}")
            
            try:
                processed_data, step_stats = step_func(processed_data, **kwargs)
                processing_stats['step_results'][step_name] = step_stats
                logger.info(f"步骤 {step_name} 完成，剩余样本数: {len(processed_data)}")
            except Exception as e:
                logger.error(f"步骤 {step_name} 执行失败: {e}")
                raise
        
        processing_stats['final_count'] = len(processed_data)
        
        if return_stats:
            return processed_data, processing_stats
        else:
            return processed_data, None
    
    def _clean_text_step(self, data: List[Any], **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """文本清洗步骤"""
        cleaned_data = []
        clean_stats = {'cleaned_count': 0}
        
        for item in data:
            if isinstance(item, str):
                cleaned_text = self.text_cleaner.clean_text(item, **kwargs)
                cleaned_data.append(cleaned_text)
                if cleaned_text != item:
                    clean_stats['cleaned_count'] += 1
            elif isinstance(item, dict) and 'text' in item:
                cleaned_text = self.text_cleaner.clean_text(item['text'], **kwargs)
                item_copy = item.copy()
                item_copy['text'] = cleaned_text
                cleaned_data.append(item_copy)
                if cleaned_text != item['text']:
                    clean_stats['cleaned_count'] += 1
            else:
                cleaned_data.append(item)
        
        return cleaned_data, clean_stats
    
    def _detect_language_step(self, data: List[Any], **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """语言检测步骤"""
        language_stats = defaultdict(int)
        
        for item in data:
            text = item if isinstance(item, str) else item.get('text', '')
            language = self.language_detector.detect_language(text, **kwargs)
            language_stats[language] += 1
            
            if isinstance(item, dict):
                item['detected_language'] = language
        
        return data, {'language_distribution': dict(language_stats)}
    
    def _validate_data_step(self, data: List[Any], **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """数据验证步骤"""
        texts = [item if isinstance(item, str) else item.get('text', '') for item in data]
        validation_results = self.data_validator.validate_dataset(texts)
        
        return data, validation_results
    
    def _compute_stats_step(self, data: List[Any], **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """计算统计信息步骤"""
        texts = [item if isinstance(item, str) else item.get('text', '') for item in data]
        
        token_counts = [len(text.split()) for text in texts]
        all_tokens = []
        for text in texts:
            all_tokens.extend(text.split())
        
        stats = DataStats(
            total_samples=len(texts),
            total_tokens=len(all_tokens),
            avg_tokens_per_sample=sum(token_counts) / len(token_counts) if token_counts else 0,
            max_tokens=max(token_counts) if token_counts else 0,
            min_tokens=min(token_counts) if token_counts else 0,
            unique_tokens=len(set(all_tokens)),
            vocabulary_size=len(set(all_tokens))
        )
        
        return data, stats.__dict__
    
    def _filter_by_length_step(self, data: List[Any], 
                              min_length: int = 1, 
                              max_length: int = 10000, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """按长度过滤步骤"""
        filtered_data = []
        filter_stats = {'filtered_out': 0, 'too_short': 0, 'too_long': 0}
        
        for item in data:
            text = item if isinstance(item, str) else item.get('text', '')
            text_length = len(text.split())
            
            if text_length < min_length:
                filter_stats['filtered_out'] += 1
                filter_stats['too_short'] += 1
            elif text_length > max_length:
                filter_stats['filtered_out'] += 1
                filter_stats['too_long'] += 1
            else:
                filtered_data.append(item)
        
        return filtered_data, filter_stats
    
    def _filter_by_language_step(self, data: List[Any], 
                                allowed_languages: List[str], **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """按语言过滤步骤"""
        filtered_data = []
        filter_stats = {'filtered_out': 0, 'language_distribution': defaultdict(int)}
        
        for item in data:
            if isinstance(item, dict) and 'detected_language' in item:
                language = item['detected_language']
            else:
                text = item if isinstance(item, str) else item.get('text', '')
                language = self.language_detector.detect_language(text)
            
            filter_stats['language_distribution'][language] += 1
            
            if language in allowed_languages:
                filtered_data.append(item)
            else:
                filter_stats['filtered_out'] += 1
        
        filter_stats['language_distribution'] = dict(filter_stats['language_distribution'])
        return filtered_data, filter_stats
    
    def _deduplicate_step(self, data: List[Any], **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """去重步骤"""
        seen_texts = set()
        deduplicated_data = []
        dedup_stats = {'duplicates_removed': 0}
        
        for item in data:
            text = item if isinstance(item, str) else item.get('text', '')
            text_hash = hash(text)
            
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                deduplicated_data.append(item)
            else:
                dedup_stats['duplicates_removed'] += 1
        
        return deduplicated_data, dedup_stats
    
    def setup_default_pipeline(self, 
                              clean_text: bool = True,
                              detect_language: bool = True,
                              validate_data: bool = True,
                              filter_by_length: bool = True,
                              min_length: int = 5,
                              max_length: int = 512,
                              deduplicate: bool = True,
                              compute_stats: bool = True):
        """设置默认的预处理管道"""
        self.clear_steps()
        
        if clean_text:
            self.add_step('clean_text', 
                         normalize_whitespace=True,
                         remove_html=True,
                         normalize_unicode=True)
        
        if detect_language:
            self.add_step('detect_language')
        
        if validate_data:
            self.add_step('validate_data')
        
        if filter_by_length:
            self.add_step('filter_by_length', 
                         min_length=min_length, 
                         max_length=max_length)
        
        if deduplicate:
            self.add_step('deduplicate')
        
        if compute_stats:
            self.add_step('compute_stats')


def demo():
    """演示数据预处理管道的使用"""
    print("🚀 数据预处理管道演示")
    print("=" * 50)
    
    # 创建示例数据
    sample_data = [
        "Hello world! This is a great example.",
        "这是一个中文示例文本。包含一些内容。",
        "   Extra   whitespace   everywhere   ",
        "Contact us at test@example.com or visit https://example.com",
        "<p>HTML content with <b>bold</b> text</p>",
        "Short",
        "This is a duplicate text for testing.",
        "This is a duplicate text for testing.",  # 重复文本
        "日本語のテキストサンプルです。",
        "Very very very long repeated characters!!!!!!!",
        "",  # 空文本
        "Another normal English sentence with good length."
    ]
    
    print(f"📝 原始数据 ({len(sample_data)} 样本):")
    for i, text in enumerate(sample_data[:5]):
        print(f"  {i+1}. '{text}'")
    print(f"  ... 共 {len(sample_data)} 条")
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 设置默认管道
    preprocessor.setup_default_pipeline(
        clean_text=True,
        detect_language=True,
        validate_data=True,
        filter_by_length=True,
        min_length=3,
        max_length=100,
        deduplicate=True,
        compute_stats=True
    )
    
    print(f"\n🔧 处理管道步骤:")
    for i, (step_name, kwargs) in enumerate(preprocessor.pipeline_steps):
        print(f"  {i+1}. {step_name} {kwargs}")
    
    # 执行预处理
    print(f"\n⚙️ 执行预处理...")
    processed_data, stats = preprocessor.process(sample_data, return_stats=True)
    
    print(f"\n✅ 预处理完成!")
    print(f"📊 处理结果:")
    print(f"  原始样本数: {stats['original_count']}")
    print(f"  最终样本数: {stats['final_count']}")
    print(f"  过滤率: {(1 - stats['final_count'] / stats['original_count']) * 100:.1f}%")
    
    # 显示各步骤统计
    print(f"\n📋 各步骤统计:")
    for step_name, step_stats in stats['step_results'].items():
        print(f"  {step_name}:")
        if isinstance(step_stats, dict):
            for key, value in step_stats.items():
                if isinstance(value, dict):
                    print(f"    {key}: {len(value)} 项")
                else:
                    print(f"    {key}: {value}")
        else:
            print(f"    结果: {step_stats}")
    
    # 显示处理后的数据示例
    print(f"\n📄 处理后的数据示例:")
    for i, text in enumerate(processed_data[:5]):
        display_text = text if isinstance(text, str) else text.get('text', str(text))
        print(f"  {i+1}. '{display_text}'")
    
    print(f"\n🎉 演示完成!")


if __name__ == "__main__":
    demo()