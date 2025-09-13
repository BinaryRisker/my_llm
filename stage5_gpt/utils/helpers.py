"""
GPT工具函数集合
================

包含：
- 掩码相关功能
- 文本预处理
- 评估指标
- 可视化工具
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import re
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict


# ==================== 掩码相关 ====================

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    创建因果掩码
    
    Args:
        seq_len: 序列长度
        device: 设备
        
    Returns:
        因果掩码 [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0  # 下三角为True


def apply_causal_mask(attention_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    应用因果掩码到注意力分数
    
    Args:
        attention_scores: 注意力分数 [batch, heads, seq_len, seq_len]
        mask: 掩码 [seq_len, seq_len]
        
    Returns:
        掩码后的注意力分数
    """
    return attention_scores.masked_fill(~mask, -torch.inf)


def create_padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    创建padding掩码
    
    Args:
        input_ids: 输入token IDs [batch, seq_len]
        pad_token_id: padding token ID
        
    Returns:
        padding掩码 [batch, seq_len]
    """
    return input_ids != pad_token_id


# ==================== 损失函数 ====================

def autoregressive_loss(logits: torch.Tensor, 
                       targets: torch.Tensor, 
                       ignore_index: int = -100) -> torch.Tensor:
    """
    计算自回归语言模型损失
    
    Args:
        logits: 模型输出 [batch_size, seq_len, vocab_size]
        targets: 目标tokens [batch_size, seq_len]
        ignore_index: 忽略的token索引
        
    Returns:
        损失值
    """
    # 移位：预测下一个token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    
    # 计算交叉熵损失
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    
    return loss


def compute_perplexity(loss: float) -> float:
    """从损失计算困惑度"""
    return math.exp(loss)


# ==================== 文本预处理 ====================

def prepare_training_data(texts: List[str], 
                         tokenizer,
                         max_length: int = 512,
                         stride: int = None,
                         min_length: int = 10) -> List[List[int]]:
    """
    准备GPT预训练数据
    
    Args:
        texts: 原始文本列表
        tokenizer: 分词器
        max_length: 最大序列长度
        stride: 滑动窗口步长
        min_length: 最小序列长度
        
    Returns:
        token序列列表
    """
    if stride is None:
        stride = max_length // 2
    
    sequences = []
    
    for text in texts:
        # 基本清洗
        text = clean_text(text)
        if len(text.strip()) < min_length:
            continue
            
        # 分词
        tokens = tokenizer.encode(text)
        
        # 滑动窗口切分
        for i in range(0, len(tokens) - max_length + 1, stride):
            chunk = tokens[i:i + max_length]
            if len(chunk) == max_length:
                sequences.append(chunk)
    
    return sequences


def clean_text(text: str) -> str:
    """基本的文本清洗"""
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text


def tokenize_batch(texts: List[str], 
                  tokenizer,
                  max_length: int = 512,
                  padding: bool = True,
                  truncation: bool = True) -> Dict[str, torch.Tensor]:
    """
    批量分词
    
    Args:
        texts: 文本列表
        tokenizer: 分词器
        max_length: 最大长度
        padding: 是否padding
        truncation: 是否截断
        
    Returns:
        包含input_ids和attention_mask的字典
    """
    input_ids_list = []
    attention_masks = []
    
    for text in texts:
        # 分词
        tokens = tokenizer.encode(text)
        
        # 截断
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Padding
        if padding:
            pad_length = max_length - len(tokens)
            attention_mask = [1] * len(tokens) + [0] * pad_length
            tokens = tokens + [tokenizer.pad_token_id or 0] * pad_length
        else:
            attention_mask = [1] * len(tokens)
        
        input_ids_list.append(tokens)
        attention_masks.append(attention_mask)
    
    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
    }


# ==================== 评估指标 ====================

def compute_token_accuracy(logits: torch.Tensor, 
                          targets: torch.Tensor,
                          ignore_index: int = -100) -> float:
    """
    计算token级别准确率
    
    Args:
        logits: 模型输出 [batch_size, seq_len, vocab_size]
        targets: 目标tokens [batch_size, seq_len]
        ignore_index: 忽略的索引
        
    Returns:
        准确率
    """
    # 移位预测
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    
    # 预测
    predictions = shift_logits.argmax(dim=-1)
    
    # 计算准确率 (忽略指定索引)
    mask = shift_labels != ignore_index
    correct = (predictions == shift_labels) & mask
    
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def compute_sequence_accuracy(predictions: List[str], 
                            references: List[str]) -> float:
    """计算序列级别准确率"""
    if len(predictions) != len(references):
        raise ValueError("预测和参考序列数量不匹配")
    
    correct = sum(pred == ref for pred, ref in zip(predictions, references))
    return correct / len(predictions)


def compute_bleu_score(predictions: List[str], 
                      references: List[str],
                      n_gram: int = 4) -> float:
    """
    计算BLEU分数 (简化版本)
    
    Args:
        predictions: 预测序列列表
        references: 参考序列列表  
        n_gram: n-gram的n
        
    Returns:
        BLEU分数
    """
    def get_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return dict(ngrams)
    
    def compute_precision(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if not pred_ngrams:
            return 0.0
        
        matches = 0
        total = 0
        
        for ngram, count in pred_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
            total += count
        
        return matches / total if total > 0 else 0.0
    
    # 计算各阶n-gram精度
    precisions = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        ngram_precisions = []
        for n in range(1, n_gram + 1):
            precision = compute_precision(pred_tokens, ref_tokens, n)
            ngram_precisions.append(precision)
        
        precisions.append(ngram_precisions)
    
    # 平均精度
    avg_precisions = [
        sum(p[i] for p in precisions) / len(precisions)
        for i in range(n_gram)
    ]
    
    # 几何平均
    if all(p > 0 for p in avg_precisions):
        bleu = math.exp(sum(math.log(p) for p in avg_precisions) / len(avg_precisions))
    else:
        bleu = 0.0
    
    return bleu


# ==================== 生成评估 ====================

def evaluate_generation_diversity(texts: List[str]) -> Dict[str, float]:
    """
    评估生成文本的多样性
    
    Args:
        texts: 生成的文本列表
        
    Returns:
        多样性指标字典
    """
    # 收集所有n-gram
    unigrams = set()
    bigrams = set()
    trigrams = set()
    
    total_tokens = 0
    
    for text in texts:
        tokens = text.split()
        total_tokens += len(tokens)
        
        # Unigrams
        unigrams.update(tokens)
        
        # Bigrams  
        for i in range(len(tokens) - 1):
            bigrams.add((tokens[i], tokens[i+1]))
        
        # Trigrams
        for i in range(len(tokens) - 2):
            trigrams.add((tokens[i], tokens[i+1], tokens[i+2]))
    
    return {
        'unique_unigrams': len(unigrams),
        'unique_bigrams': len(bigrams), 
        'unique_trigrams': len(trigrams),
        'total_tokens': total_tokens,
        'unigram_diversity': len(unigrams) / total_tokens if total_tokens > 0 else 0,
        'bigram_diversity': len(bigrams) / max(1, total_tokens - len(texts)),
        'trigram_diversity': len(trigrams) / max(1, total_tokens - 2 * len(texts))
    }


def evaluate_repetition(text: str, n: int = 3) -> float:
    """
    评估文本中的n-gram重复率
    
    Args:
        text: 输入文本
        n: n-gram的n
        
    Returns:
        重复率 (0-1)
    """
    tokens = text.split()
    
    if len(tokens) < n:
        return 0.0
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    
    repetition_rate = 1.0 - (unique_ngrams / total_ngrams)
    return repetition_rate


# ==================== 模型工具 ====================

def count_parameters(model) -> Dict[str, int]:
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size_mb(model) -> float:
    """获取模型大小 (MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def save_model_summary(model, save_path: str):
    """保存模型摘要信息"""
    summary = {
        'model_class': model.__class__.__name__,
        'parameters': count_parameters(model),
        'size_mb': get_model_size_mb(model),
    }
    
    # 如果有config属性
    if hasattr(model, 'config'):
        summary['config'] = model.config.__dict__
    
    import json
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ==================== 分词器相关 ====================

class SimpleTokenizer:
    """简单的字符级分词器 (用于演示和测试)"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # 特殊token
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
        self.bos_token = '<bos>'
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 3
    
    def fit(self, texts: List[str]):
        """从文本构建词汇表"""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # 添加特殊token
        vocab = [self.pad_token, self.unk_token, self.eos_token, self.bos_token]
        vocab.extend(sorted(chars))
        
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        self.vocab_size = len(vocab)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """编码文本"""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.unk_token_id))
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码token序列"""
        chars = []
        special_ids = {self.pad_token_id, self.unk_token_id, self.eos_token_id, self.bos_token_id}
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            char = self.idx_to_char.get(token_id, self.unk_token)
            chars.append(char)
        
        return ''.join(chars)


if __name__ == "__main__":
    # 测试工具函数
    print("🧪 测试GPT工具函数")
    
    # 测试因果掩码
    seq_len = 4
    device = torch.device('cpu')
    mask = create_causal_mask(seq_len, device)
    print(f"因果掩码:\n{mask}")
    
    # 测试简单分词器
    texts = ["Hello world!", "GPT is great!", "Deep learning rocks!"]
    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)
    
    print(f"\n分词器词汇表大小: {tokenizer.vocab_size}")
    
    # 编码解码测试
    test_text = "Hello GPT!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
    
    # 测试多样性评估
    sample_texts = [
        "The cat sat on the mat",
        "A dog ran in the park", 
        "Birds fly in the sky"
    ]
    diversity = evaluate_generation_diversity(sample_texts)
    print(f"\n文本多样性: {diversity}")
    
    print("\n✅ 工具函数测试完成!")