#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段3：注意力机制 Seq2Seq 训练脚本

支持英法翻译任务训练，包含：
- 数据预处理和词汇表构建
- 多种注意力机制对比
- BLEU评估和可视化
- 模型保存和恢复训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
import sys
sys.path.append(str(Path(__file__).parent))

from models.encoder import LSTMEncoder
from models.decoder import AttentionDecoder  
from models.attention import create_attention

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TranslationDataset(Dataset):
    """翻译数据集类"""
    
    def __init__(self, source_texts, target_texts, source_vocab, target_vocab, max_length=50):
        """
        Args:
            source_texts (list): 源语言文本列表
            target_texts (list): 目标语言文本列表  
            source_vocab (dict): 源语言词汇表
            target_vocab (dict): 目标语言词汇表
            max_length (int): 最大序列长度
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_length = max_length
        
        # 特殊标记
        self.SOS_TOKEN = 1  # Start of sequence
        self.EOS_TOKEN = 2  # End of sequence
        self.PAD_TOKEN = 0  # Padding
        self.UNK_TOKEN = 3  # Unknown
        
        # 处理数据
        self.source_sequences = []
        self.target_sequences = []
        self.source_lengths = []
        self.target_lengths = []
        
        self._process_data()
    
    def _process_data(self):
        """处理原始文本数据"""
        for src_text, tgt_text in zip(self.source_texts, self.target_texts):
            # 分词
            src_words = src_text.lower().split()
            tgt_words = tgt_text.lower().split()
            
            # 限制长度
            src_words = src_words[:self.max_length-1]
            tgt_words = tgt_words[:self.max_length-2]  # 为SOS和EOS预留空间
            
            # 转换为索引
            src_indices = [self.source_vocab.get(word, self.UNK_TOKEN) for word in src_words]
            tgt_indices = [self.SOS_TOKEN] + [self.target_vocab.get(word, self.UNK_TOKEN) for word in tgt_words] + [self.EOS_TOKEN]
            
            # 添加填充
            src_length = len(src_indices)
            tgt_length = len(tgt_indices)
            
            src_indices += [self.PAD_TOKEN] * (self.max_length - len(src_indices))
            tgt_indices += [self.PAD_TOKEN] * (self.max_length - len(tgt_indices))
            
            self.source_sequences.append(src_indices)
            self.target_sequences.append(tgt_indices)
            self.source_lengths.append(src_length)
            self.target_lengths.append(tgt_length)
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        return {
            'source': torch.tensor(self.source_sequences[idx], dtype=torch.long),
            'target': torch.tensor(self.target_sequences[idx], dtype=torch.long),
            'source_length': self.source_lengths[idx],
            'target_length': self.target_lengths[idx]
        }


class Seq2SeqModel(nn.Module):
    """完整的Seq2Seq模型"""
    
    def __init__(self, source_vocab_size, target_vocab_size, embedding_dim, 
                 hidden_size, attention_type, num_layers=1, dropout=0.1):
        """
        Args:
            source_vocab_size (int): 源语言词汇表大小
            target_vocab_size (int): 目标语言词汇表大小
            embedding_dim (int): 嵌入维度
            hidden_size (int): 隐藏状态大小
            attention_type (str): 注意力机制类型
            num_layers (int): LSTM层数
            dropout (float): Dropout率
        """
        super(Seq2SeqModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 编码器
        self.encoder = LSTMEncoder(
            vocab_size=source_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout
        )
        
        # 解码器
        encoder_output_size = hidden_size * 2  # 双向LSTM
        self.decoder = AttentionDecoder(
            vocab_size=target_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            encoder_output_size=encoder_output_size,
            attention_type=attention_type,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 初始化解码器隐藏状态的线性层
        self.hidden_projection = nn.Linear(encoder_output_size, hidden_size)
        self.cell_projection = nn.Linear(encoder_output_size, hidden_size)
    
    def forward(self, source, target, source_lengths, teacher_forcing_ratio=1.0):
        """
        前向传播
        
        Args:
            source (Tensor): 源序列 [batch_size, src_len]
            target (Tensor): 目标序列 [batch_size, tgt_len]  
            source_lengths (list): 源序列长度
            teacher_forcing_ratio (float): Teacher forcing比例
            
        Returns:
            output_logits (Tensor): 输出概率 [batch_size, tgt_len-1, target_vocab_size]
            attention_weights (Tensor): 注意力权重 [batch_size, tgt_len-1, src_len]
        """
        # 编码
        encoder_outputs, (encoder_hidden, encoder_cell), encoder_mask = self.encoder(source, source_lengths)
        
        # 初始化解码器隐藏状态
        # 使用双向LSTM的最后一层来初始化解码器
        batch_size = source.size(0)
        
        # 合并前向和后向的最终隐藏状态
        if encoder_hidden.size(0) == self.num_layers * 2:  # 双向
            # 重新整形: [num_layers*2, batch_size, hidden_size] -> [num_layers, batch_size, hidden_size*2]
            encoder_hidden = encoder_hidden.view(self.num_layers, 2, batch_size, -1)
            encoder_cell = encoder_cell.view(self.num_layers, 2, batch_size, -1)
            
            # 连接前向和后向状态
            encoder_hidden = torch.cat([encoder_hidden[:, 0], encoder_hidden[:, 1]], dim=2)
            encoder_cell = torch.cat([encoder_cell[:, 0], encoder_cell[:, 1]], dim=2)
        
        # 投影到解码器隐藏维度
        decoder_hidden = torch.tanh(self.hidden_projection(encoder_hidden))
        decoder_cell = torch.tanh(self.cell_projection(encoder_cell))
        
        # 解码
        output_logits, attention_weights = self.decoder.forward_train(
            target, encoder_outputs, encoder_mask, teacher_forcing_ratio
        )
        
        return output_logits, attention_weights
    
    def translate(self, source, source_lengths, max_length=50, start_token=1, end_token=2):
        """
        翻译函数
        
        Args:
            source (Tensor): 源序列 [batch_size, src_len]
            source_lengths (list): 源序列长度
            max_length (int): 最大翻译长度
            start_token (int): 开始标记
            end_token (int): 结束标记
            
        Returns:
            translations (Tensor): 翻译结果 [batch_size, translation_len]
            attention_weights (Tensor): 注意力权重
        """
        self.eval()
        
        with torch.no_grad():
            # 编码
            encoder_outputs, (encoder_hidden, encoder_cell), encoder_mask = self.encoder(source, source_lengths)
            
            # 初始化解码器隐藏状态
            batch_size = source.size(0)
            
            if encoder_hidden.size(0) == self.num_layers * 2:
                encoder_hidden = encoder_hidden.view(self.num_layers, 2, batch_size, -1)
                encoder_cell = encoder_cell.view(self.num_layers, 2, batch_size, -1)
                encoder_hidden = torch.cat([encoder_hidden[:, 0], encoder_hidden[:, 1]], dim=2)
                encoder_cell = torch.cat([encoder_cell[:, 0], encoder_cell[:, 1]], dim=2)
            
            decoder_hidden = torch.tanh(self.hidden_projection(encoder_hidden))
            decoder_cell = torch.tanh(self.cell_projection(encoder_cell))
            
            # 设置初始隐藏状态
            self.decoder._init_hidden_state = lambda bs, dev: (decoder_hidden, decoder_cell)
            
            # 生成翻译
            translations, attention_weights = self.decoder.generate(
                encoder_outputs, encoder_mask, max_length, start_token, end_token
            )
            
        return translations, attention_weights


def build_vocabulary(texts, max_vocab_size=10000, min_freq=2):
    """构建词汇表"""
    from collections import Counter
    
    # 统计词频
    word_freq = Counter()
    for text in texts:
        words = text.lower().split()
        word_freq.update(words)
    
    # 构建词汇表
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    
    # 按频率排序，添加高频词
    sorted_words = word_freq.most_common(max_vocab_size - 4)
    for word, freq in sorted_words:
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab


def load_data(data_path, max_samples=None):
    """加载翻译数据"""
    if not os.path.exists(data_path):
        # 创建示例数据
        logger.info("创建示例英法翻译数据...")
        english_texts = [
            "hello world",
            "how are you",
            "good morning",
            "thank you very much",
            "i love programming",
            "this is a test",
            "artificial intelligence is amazing",
            "deep learning models",
            "natural language processing",
            "machine translation system"
        ] * 100  # 复制100次作为示例数据
        
        french_texts = [
            "salut le monde",
            "comment allez vous",
            "bonjour",
            "merci beaucoup",
            "j'aime la programmation", 
            "c'est un test",
            "l'intelligence artificielle est incroyable",
            "modèles d'apprentissage profond",
            "traitement du langage naturel",
            "système de traduction automatique"
        ] * 100
        
        # 保存示例数据
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data = {
            'english': english_texts,
            'french': french_texts
        }
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    english_texts = data['english']
    french_texts = data['french']
    
    if max_samples:
        english_texts = english_texts[:max_samples]
        french_texts = french_texts[:max_samples]
    
    logger.info(f"加载了 {len(english_texts)} 个翻译对")
    return english_texts, french_texts


def calculate_bleu(references, candidates):
    """计算BLEU分数"""
    from collections import Counter
    import math
    
    def get_ngrams(tokens, n):
        """获取n-gram"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams
    
    def calculate_precision(ref_ngrams, cand_ngrams):
        """计算精确度"""
        if not cand_ngrams:
            return 0
        
        ref_counter = Counter(ref_ngrams)
        cand_counter = Counter(cand_ngrams)
        
        overlap = 0
        for ngram in cand_counter:
            overlap += min(cand_counter[ngram], ref_counter[ngram])
        
        return overlap / len(cand_ngrams)
    
    # 计算1-4gram精确度
    precisions = []
    for n in range(1, 5):
        total_precision = 0
        valid_sentences = 0
        
        for ref, cand in zip(references, candidates):
            ref_tokens = ref.split()
            cand_tokens = cand.split()
            
            if len(cand_tokens) >= n:
                ref_ngrams = get_ngrams(ref_tokens, n)
                cand_ngrams = get_ngrams(cand_tokens, n)
                precision = calculate_precision(ref_ngrams, cand_ngrams)
                total_precision += precision
                valid_sentences += 1
        
        if valid_sentences > 0:
            precisions.append(total_precision / valid_sentences)
        else:
            precisions.append(0)
    
    # 计算简洁性惩罚
    ref_length = sum(len(ref.split()) for ref in references)
    cand_length = sum(len(cand.split()) for cand in candidates)
    
    if cand_length > ref_length:
        bp = 1
    else:
        bp = math.exp(1 - ref_length / cand_length) if cand_length > 0 else 0
    
    # 计算BLEU分数
    if all(p > 0 for p in precisions):
        log_precision_sum = sum(math.log(p) for p in precisions) / 4
        bleu = bp * math.exp(log_precision_sum)
    else:
        bleu = 0
    
    return bleu, precisions


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        source_lengths = batch['source_length']
        
        optimizer.zero_grad()
        
        # 前向传播
        output_logits, attention_weights = model(
            source, target, source_lengths, args.teacher_forcing_ratio
        )
        
        # 计算损失
        target_output = target[:, 1:]  # 移除SOS标记
        target_flat = target_output.contiguous().view(-1)
        output_flat = output_logits.contiguous().view(-1, output_logits.size(-1))
        
        loss = criterion(output_flat, target_flat)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{avg_loss:.4f}'
        })
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device, source_vocab, target_vocab, args):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    references = []
    candidates = []
    
    # 创建反向词汇表
    idx2word_target = {idx: word for word, idx in target_vocab.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            source_lengths = batch['source_length']
            
            # 计算损失
            output_logits, _ = model(source, target, source_lengths, teacher_forcing_ratio=1.0)
            target_output = target[:, 1:]
            target_flat = target_output.contiguous().view(-1)
            output_flat = output_logits.contiguous().view(-1, output_logits.size(-1))
            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()
            
            # 生成翻译用于BLEU计算
            translations, _ = model.translate(source, source_lengths, max_length=50)
            
            for i in range(source.size(0)):
                # 获取参考翻译
                ref_tokens = []
                for idx in target[i, 1:]:  # 跳过SOS
                    if idx.item() == 2:  # EOS
                        break
                    if idx.item() not in [0, 1, 2]:  # 跳过PAD, SOS, EOS
                        word = idx2word_target.get(idx.item(), '<unk>')
                        ref_tokens.append(word)
                
                # 获取候选翻译
                cand_tokens = []
                for idx in translations[i]:
                    if idx.item() == 2:  # EOS
                        break
                    if idx.item() not in [0, 1, 2]:  # 跳过PAD, SOS, EOS
                        word = idx2word_target.get(idx.item(), '<unk>')
                        cand_tokens.append(word)
                
                references.append(' '.join(ref_tokens))
                candidates.append(' '.join(cand_tokens))
    
    avg_loss = total_loss / num_batches
    bleu_score, precisions = calculate_bleu(references, candidates)
    
    return avg_loss, bleu_score, precisions, references[:5], candidates[:5]


def save_checkpoint(model, optimizer, epoch, loss, bleu, args, filename):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'bleu': bleu,
        'args': vars(args)
    }
    torch.save(checkpoint, filename)
    logger.info(f"检查点已保存: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Seq2Seq with Attention Training')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/translation_data.json',
                       help='翻译数据路径')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='最大样本数量') 
    parser.add_argument('--max_length', type=int, default=50,
                       help='最大序列长度')
    parser.add_argument('--max_vocab_size', type=int, default=5000,
                       help='最大词汇表大小')
    parser.add_argument('--min_freq', type=int, default=2,
                       help='词汇最小频率')
    
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='嵌入维度')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='隐藏状态大小')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='LSTM层数')
    parser.add_argument('--attention_type', type=str, default='bahdanau',
                       choices=['bahdanau', 'luong_general', 'luong_dot', 'luong_concat'],
                       help='注意力机制类型')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                       help='Teacher forcing比例')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪阈值')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='日志打印间隔')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='模型保存间隔')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    english_texts, french_texts = load_data(args.data_path, args.max_samples)
    
    # 构建词汇表
    logger.info("构建词汇表...")
    source_vocab = build_vocabulary(english_texts, args.max_vocab_size, args.min_freq)
    target_vocab = build_vocabulary(french_texts, args.max_vocab_size, args.min_freq)
    
    logger.info(f"源语言词汇表大小: {len(source_vocab)}")
    logger.info(f"目标语言词汇表大小: {len(target_vocab)}")
    
    # 保存词汇表
    with open(os.path.join(args.save_dir, 'source_vocab.pkl'), 'wb') as f:
        pickle.dump(source_vocab, f)
    with open(os.path.join(args.save_dir, 'target_vocab.pkl'), 'wb') as f:
        pickle.dump(target_vocab, f)
    
    # 划分训练和验证集
    split_idx = int(len(english_texts) * 0.8)
    train_en, val_en = english_texts[:split_idx], english_texts[split_idx:]
    train_fr, val_fr = french_texts[:split_idx], french_texts[split_idx:]
    
    # 创建数据集
    train_dataset = TranslationDataset(train_en, train_fr, source_vocab, target_vocab, args.max_length)
    val_dataset = TranslationDataset(val_en, val_fr, source_vocab, target_vocab, args.max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = Seq2SeqModel(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        attention_type=args.attention_type,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标记
    
    # 训练历史
    train_losses = []
    val_losses = []
    bleu_scores = []
    
    logger.info("开始训练...")
    best_bleu = 0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args)
        train_losses.append(train_loss)
        
        # 验证
        val_loss, bleu_score, precisions, sample_refs, sample_cands = evaluate(
            model, val_loader, criterion, device, source_vocab, target_vocab, args
        )
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)
        
        # 打印结果
        logger.info(f'Epoch {epoch}/{args.epochs}:')
        logger.info(f'  Train Loss: {train_loss:.4f}')
        logger.info(f'  Val Loss: {val_loss:.4f}')
        logger.info(f'  BLEU Score: {bleu_score:.4f}')
        logger.info(f'  BLEU-1/2/3/4: {precisions}')
        
        # 显示翻译示例
        logger.info("翻译示例:")
        for i in range(min(3, len(sample_refs))):
            logger.info(f"  参考: {sample_refs[i]}")
            logger.info(f"  翻译: {sample_cands[i]}")
            logger.info("")
        
        # 保存最佳模型
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            save_checkpoint(
                model, optimizer, epoch, val_loss, bleu_score, args,
                os.path.join(args.save_dir, f'best_{args.attention_type}_model.pt')
            )
        
        # 定期保存检查点
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, bleu_score, args,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
            )
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'bleu_scores': bleu_scores,
        'best_bleu': best_bleu
    }
    
    with open(os.path.join(args.save_dir, f'{args.attention_type}_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(bleu_scores, 'g-', label='BLEU Score')
    plt.title('BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.plot(epochs, [s*10 for s in bleu_scores], 'g-', label='BLEU Score x10')
    plt.title('Training Overview')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / BLEU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f'{args.attention_type}_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("训练完成!")
    logger.info(f"最佳BLEU分数: {best_bleu:.4f}")


if __name__ == "__main__":
    main()