"""
pytest配置文件

定义全局fixtures和测试配置
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock

from config import get_config


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """获取测试环境配置"""
    return get_config(env='testing')


@pytest.fixture(scope="session") 
def device():
    """获取测试设备 (优先使用CPU以确保测试稳定性)"""
    return torch.device('cpu')


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_texts():
    """样本文本数据"""
    return [
        "This is a positive example.",
        "This is a negative example.",
        "The weather is beautiful today.",
        "I love machine learning and deep learning.",
        "Natural language processing is fascinating."
    ]


@pytest.fixture
def sample_labels():
    """样本标签数据"""
    return [1, 0, 1, 1, 1]


@pytest.fixture 
def sample_tokens():
    """样本token数据"""
    return [
        ["This", "is", "a", "test", "sentence", "."],
        ["Another", "test", "sentence", "here", "."],
        ["Machine", "learning", "is", "great", "!"]
    ]


@pytest.fixture
def sample_token_labels():
    """样本token标签数据 (BIO标注)"""
    return [
        ["O", "O", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "O"],
        ["B-TECH", "I-TECH", "O", "O", "O"]
    ]


@pytest.fixture
def mock_tokenizer():
    """模拟分词器"""
    tokenizer = Mock()
    tokenizer.vocab_size = 1000
    tokenizer.pad_token_id = 0
    tokenizer.unk_token_id = 1
    tokenizer.cls_token_id = 2
    tokenizer.sep_token_id = 3
    
    def encode_mock(text):
        # 简单的模拟编码：返回固定长度的token ids
        return list(range(len(text.split())))
    
    tokenizer.encode = encode_mock
    return tokenizer


@pytest.fixture
def mock_model():
    """模拟模型"""
    model = Mock()
    model.eval = Mock()
    model.train = Mock()
    
    def forward_mock(*args, **kwargs):
        batch_size = args[0].shape[0] if args else 1
        return torch.randn(batch_size, 2)  # 二分类输出
    
    model.forward = forward_mock
    model.__call__ = forward_mock
    return model


@pytest.fixture
def sample_qa_data():
    """样本问答数据"""
    return [
        {
            'context': "The quick brown fox jumps over the lazy dog.",
            'question': "What color is the fox?",
            'answer': "brown",
            'answer_start': 10
        },
        {
            'context': "Machine learning is a subset of artificial intelligence.",
            'question': "What is machine learning?",
            'answer': "a subset of artificial intelligence",
            'answer_start': 22
        }
    ]


@pytest.fixture
def small_vocab():
    """小型词汇表用于测试"""
    return {
        '<pad>': 0,
        '<unk>': 1, 
        '<cls>': 2,
        '<sep>': 3,
        'the': 4,
        'is': 5,
        'a': 6,
        'test': 7,
        'this': 8,
        'that': 9,
        'example': 10,
        'machine': 11,
        'learning': 12,
    }


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """自动设置测试环境"""
    # 设置环境变量
    monkeypatch.setenv("LLM_ENV", "testing")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # 禁用CUDA
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def classification_dataset():
    """分类任务测试数据集"""
    return {
        'texts': [
            "I love this product!",
            "This is terrible.",
            "Not bad, could be better.",
            "Excellent quality and service.",
            "Worst purchase ever."
        ],
        'labels': [1, 0, 0, 1, 0]  # 1=positive, 0=negative
    }


@pytest.fixture
def regression_dataset():
    """回归任务测试数据集"""
    return {
        'texts': [
            "This movie is amazing!",
            "Okay movie, nothing special.",
            "Terrible movie, waste of time.",
            "One of the best movies ever!",
            "Average movie with good acting."
        ],
        'scores': [4.8, 3.0, 1.2, 4.9, 3.5]
    }


# 标记定义
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual functions"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interactions"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that take more time to run"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "model: Tests that involve model loading/training"
    )