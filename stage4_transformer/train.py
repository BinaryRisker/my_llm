"""
阶段4: Transformer模型训练脚本
==============================

用于机器翻译任务的Transformer模型训练，包含：
- 英法翻译数据处理
- Transformer模型训练
- BLEU评估
- 与LSTM Seq2Seq对比
"""

import os
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import random
from tqdm import tqdm
import numpy as np
from collections import Counter

# 导入模型
from models.transformer import Transformer, TransformerConfig, create_transformer_small


class TranslationDataset(Dataset):
    """机器翻译数据集"""
    
    def __init__(self, 
                 src_sentences: List[str],
                 tgt_sentences: List[str], 
                 src_vocab: Dict[str, int],
                 tgt_vocab: Dict[str, int],
                 max_length: int = 100):
        
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        # 特殊token
        self.src_pad_idx = src_vocab.get('<pad>', 0)
        self.src_unk_idx = src_vocab.get('<unk>', 1)
        self.src_sos_idx = src_vocab.get('<sos>', 2)
        self.src_eos_idx = src_vocab.get('<eos>', 3)
        
        self.tgt_pad_idx = tgt_vocab.get('<pad>', 0)
        self.tgt_unk_idx = tgt_vocab.get('<unk>', 1)
        self.tgt_sos_idx = tgt_vocab.get('<sos>', 2)
        self.tgt_eos_idx = tgt_vocab.get('<eos>', 3)
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_tokens = self._tokenize_and_encode(
            self.src_sentences[idx], 
            self.src_vocab, 
            self.src_sos_idx, 
            self.src_eos_idx,
            self.src_unk_idx
        )
        
        tgt_tokens = self._tokenize_and_encode(
            self.tgt_sentences[idx], 
            self.tgt_vocab, 
            self.tgt_sos_idx, 
            self.tgt_eos_idx,
            self.tgt_unk_idx
        )
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens, dtype=torch.long)
        }
    
    def _tokenize_and_encode(self, sentence: str, vocab: Dict[str, int], 
                            sos_idx: int, eos_idx: int, unk_idx: int) -> List[int]:
        """分词并编码"""
        # 简单分词（实际应用中应使用更复杂的分词器）
        tokens = sentence.lower().split()[:self.max_length-2]
        
        # 添加特殊token并编码
        encoded = [sos_idx]
        for token in tokens:
            encoded.append(vocab.get(token, unk_idx))
        encoded.append(eos_idx)
        
        return encoded


def build_vocab(sentences: List[str], min_freq: int = 2) -> Dict[str, int]:
    """构建词汇表"""
    # 统计词频
    counter = Counter()
    for sentence in sentences:
        tokens = sentence.lower().split()
        counter.update(tokens)
    
    # 构建词汇表
    vocab = {
        '<pad>': 0,
        '<unk>': 1, 
        '<sos>': 2,
        '<eos>': 3
    }
    
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    
    return vocab


def create_sample_data(num_samples: int = 1000) -> Tuple[List[str], List[str]]:
    """创建示例英法翻译数据"""
    
    # 简单的英法对照词典
    en_fr_dict = {
        'hello': 'bonjour',
        'world': 'monde', 
        'good': 'bon',
        'morning': 'matin',
        'evening': 'soir',
        'thank': 'merci',
        'you': 'vous',
        'please': 'sil vous plait',
        'yes': 'oui',
        'no': 'non',
        'cat': 'chat',
        'dog': 'chien',
        'house': 'maison',
        'car': 'voiture',
        'book': 'livre',
        'water': 'eau',
        'food': 'nourriture',
        'love': 'amour',
        'life': 'vie',
        'time': 'temps',
        'day': 'jour',
        'night': 'nuit',
        'sun': 'soleil',
        'moon': 'lune',
        'red': 'rouge',
        'blue': 'bleu',
        'green': 'vert',
        'big': 'grand',
        'small': 'petit',
        'beautiful': 'beau',
        'happy': 'heureux',
        'sad': 'triste'
    }
    
    # 模式
    patterns = [
        ("hello world", "bonjour monde"),
        ("good morning", "bon matin"),
        ("good evening", "bon soir"), 
        ("thank you", "merci vous"),
        ("the cat is beautiful", "le chat est beau"),
        ("the dog is big", "le chien est grand"),
        ("the house is small", "la maison est petit"),
        ("i love you", "je vous aime"),
        ("life is beautiful", "la vie est beau"),
        ("the sun is red", "le soleil est rouge"),
        ("the moon is blue", "la lune est bleu"),
        ("water is good", "l'eau est bon")
    ]
    
    en_sentences = []
    fr_sentences = []
    
    # 生成基础模式数据
    for _ in range(num_samples // 2):
        en_pattern, fr_pattern = random.choice(patterns)
        en_sentences.append(en_pattern)
        fr_sentences.append(fr_pattern)
    
    # 生成随机组合数据
    words = list(en_fr_dict.keys())
    for _ in range(num_samples - len(en_sentences)):
        # 随机选择1-4个词
        num_words = random.randint(1, 4)
        en_words = random.sample(words, num_words)
        fr_words = [en_fr_dict[w] for w in en_words]
        
        en_sentences.append(' '.join(en_words))
        fr_sentences.append(' '.join(fr_words))
    
    return en_sentences, fr_sentences


def collate_fn(batch):
    """批量数据处理函数"""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    # Padding
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return {
        'src': src_batch,
        'tgt': tgt_batch
    }


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    
    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        return self.criterion(x, true_dist)


class TransformerTrainer:
    """Transformer训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 src_vocab: Dict[str, int],
                 tgt_vocab: Dict[str, int],
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 label_smoothing: float = 0.1):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        
        # 反向词典
        self.tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.98), 
            eps=1e-9
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.95
        )
        
        # 损失函数
        self.criterion = LabelSmoothingLoss(
            size=len(tgt_vocab),
            padding_idx=tgt_vocab['<pad>'],
            smoothing=label_smoothing
        )
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
        
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="训练中")
        
        for batch in progress_bar:
            src = batch['src'].to(self.device)  # [batch, src_len]
            tgt = batch['tgt'].to(self.device)  # [batch, tgt_len]
            
            # 准备decoder输入和标签
            tgt_input = tgt[:, :-1]  # 去掉最后一个token作为输入
            tgt_output = tgt[:, 1:]  # 去掉第一个token作为标签
            
            # 前向传播
            self.optimizer.zero_grad()
            
            logits = self.model(src, tgt_input)  # [batch, tgt_len-1, vocab_size]
            
            # 计算损失
            loss = self.criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_output.contiguous().view(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (progress_bar.n + 1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        translations = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中"):
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # 计算损失
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_output.contiguous().view(-1)
                )
                
                total_loss += loss.item()
                
                # 生成翻译用于BLEU计算
                for i in range(min(5, src.size(0))):  # 只计算前5个样本的BLEU
                    src_seq = src[i:i+1]
                    translated = self.translate(src_seq)
                    reference = self._decode_sequence(tgt[i])
                    
                    translations.append(translated)
                    references.append(reference)
        
        avg_loss = total_loss / num_batches
        bleu_score = self.compute_bleu(references, translations)
        
        return avg_loss, bleu_score
    
    def translate(self, src: torch.Tensor, max_length: int = 50) -> str:
        """翻译单个序列"""
        self.model.eval()
        
        with torch.no_grad():
            batch_size = src.size(0)
            
            # 初始化decoder输入
            tgt_tokens = [self.tgt_vocab['<sos>']]
            
            for _ in range(max_length):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long, device=self.device)
                
                # 前向传播
                logits = self.model(src, tgt_tensor)  # [1, cur_len, vocab_size]
                
                # 获取下一个token
                next_token = logits[0, -1, :].argmax(dim=-1).item()
                
                tgt_tokens.append(next_token)
                
                # 检查结束条件
                if next_token == self.tgt_vocab['<eos>']:
                    break
            
            # 解码
            return self._decode_sequence(torch.tensor(tgt_tokens))
    
    def _decode_sequence(self, sequence: torch.Tensor) -> str:
        """将token序列解码为文本"""
        tokens = []
        for token_id in sequence:
            token_id = token_id.item() if hasattr(token_id, 'item') else token_id
            if token_id in [self.tgt_vocab['<sos>'], self.tgt_vocab['<eos>'], self.tgt_vocab['<pad>']]:
                continue
            token = self.tgt_idx2word.get(token_id, '<unk>')
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def compute_bleu(self, references: List[str], candidates: List[str]) -> float:
        """计算BLEU分数（简化版）"""
        if len(references) != len(candidates):
            return 0.0
        
        total_score = 0
        count = 0
        
        for ref, cand in zip(references, candidates):
            ref_tokens = ref.split()
            cand_tokens = cand.split()
            
            if len(cand_tokens) == 0:
                continue
            
            # 计算1-gram精度
            common_tokens = set(ref_tokens) & set(cand_tokens)
            precision = len(common_tokens) / len(cand_tokens) if cand_tokens else 0
            
            # 简化的BLEU（只考虑1-gram）
            total_score += precision
            count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints'):
        """完整训练过程"""
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        best_bleu = 0
        
        print("🚀 开始训练Transformer...")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, bleu_score = self.validate()
            self.val_losses.append(val_loss)
            self.bleu_scores.append(bleu_score)
            
            # 更新学习率
            self.scheduler.step()
            
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"BLEU分数: {bleu_score:.4f}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'bleu_score': bleu_score,
                    'src_vocab': self.src_vocab,
                    'tgt_vocab': self.tgt_vocab
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"💾 保存最佳模型 (BLEU: {best_bleu:.4f})")
            
            # 示例翻译
            if (epoch + 1) % 5 == 0:
                self.show_examples()
        
        print(f"\n🎉 训练完成! 最佳BLEU: {best_bleu:.4f}")
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'bleu_scores': self.bleu_scores
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def show_examples(self, num_examples: int = 3):
        """显示翻译示例"""
        print("\n🎯 翻译示例:")
        
        self.model.eval()
        examples_shown = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if examples_shown >= num_examples:
                    break
                
                src = batch['src'].to(self.device)
                tgt = batch['tgt']
                
                for i in range(min(num_examples - examples_shown, src.size(0))):
                    src_seq = src[i:i+1]
                    src_text = self._decode_sequence(batch['src'][i])
                    tgt_text = self._decode_sequence(tgt[i])
                    translated = self.translate(src_seq)
                    
                    print(f"  输入: {src_text}")
                    print(f"  参考: {tgt_text}")
                    print(f"  翻译: {translated}")
                    print()
                    
                    examples_shown += 1
                    
                if examples_shown >= num_examples:
                    break


def main():
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 1. 准备数据
    print("📚 准备数据...")
    en_sentences, fr_sentences = create_sample_data(2000)
    
    # 划分训练/验证集
    split_idx = int(0.8 * len(en_sentences))
    train_en, val_en = en_sentences[:split_idx], en_sentences[split_idx:]
    train_fr, val_fr = fr_sentences[:split_idx], fr_sentences[split_idx:]
    
    # 构建词汇表
    print("🔤 构建词汇表...")
    src_vocab = build_vocab(train_en, min_freq=1)
    tgt_vocab = build_vocab(train_fr, min_freq=1)
    
    print(f"英语词汇量: {len(src_vocab)}")
    print(f"法语词汇量: {len(tgt_vocab)}")
    
    # 创建数据集
    train_dataset = TranslationDataset(train_en, train_fr, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_en, val_fr, src_vocab, tgt_vocab)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 2. 创建模型
    print("🏗️ 创建Transformer模型...")
    
    config = TransformerConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=100,
        dropout=0.1
    )
    
    model = Transformer(config).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 训练
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        learning_rate=1e-4
    )
    
    trainer.train(num_epochs=20, save_dir='./transformer_checkpoints')
    
    print("✅ 训练完成!")


if __name__ == "__main__":
    main()