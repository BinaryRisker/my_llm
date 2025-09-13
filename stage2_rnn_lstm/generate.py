#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本生成脚本 - 使用训练好的RNN/LSTM模型生成文本

支持多种文本生成策略：
- 贪心解码
- 随机采样  
- 温度采样
- Top-k采样
- Top-p (nucleus) 采样
"""

import torch
import torch.nn.functional as F
import argparse
import pickle
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from models.rnn import create_rnn_model
from models.lstm import SimpleLSTM, BiLSTM, LayerNormLSTMCell
from utils.text_data import CharacterVocabulary, WordVocabulary


def load_model_and_vocab(checkpoint_path, vocab_path):
    """
    加载训练好的模型和词汇表
    
    Args:
        checkpoint_path (str): 模型检查点路径
        vocab_path (str): 词汇表路径
        
    Returns:
        tuple: (model, vocabulary, device)
    """
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")
    
    # 加载设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检查点
    print(f"正在加载模型从 {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取模型参数
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', 'lstm')
    vocab_type = model_config.get('vocab_type', 'char')
    
    # 加载词汇表
    print(f"正在加载词汇表从 {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # 创建模型
    vocab_size = len(vocab)
    hidden_size = model_config.get('hidden_size', 256)
    num_layers = model_config.get('num_layers', 2)
    embedding_dim = model_config.get('embedding_dim', 128)
    dropout = model_config.get('dropout', 0.3)
    
    if model_type in ['rnn', 'gru']:
        model = create_rnn_model(
            model_type=model_type,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    else:  # lstm
        model = SimpleLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型加载成功! 设备: {device}")
    print(f"模型类型: {model_type}, 词汇类型: {vocab_type}")
    print(f"词汇表大小: {vocab_size}, 隐藏维度: {hidden_size}")
    
    return model, vocab, device


def greedy_decode(model, vocab, device, start_text="", max_length=200, end_token=None):
    """
    贪心解码 - 每次选择概率最高的词
    
    Args:
        model: 训练好的模型
        vocab: 词汇表
        device: 计算设备
        start_text (str): 起始文本
        max_length (int): 最大生成长度
        end_token (str): 结束符号
        
    Returns:
        str: 生成的文本
    """
    model.eval()
    
    # 准备起始序列
    if start_text:
        if isinstance(vocab, CharacterVocabulary):
            tokens = vocab.text_to_indices(start_text)
        else:
            tokens = vocab.text_to_indices(start_text.split())
    else:
        tokens = [vocab.get_start_token()]
    
    with torch.no_grad():
        for _ in range(max_length):
            # 准备输入
            input_seq = torch.tensor(tokens[-100:], dtype=torch.long, device=device).unsqueeze(0)
            
            # 前向传播
            logits = model(input_seq)
            
            # 贪心选择
            next_token = torch.argmax(logits[0, -1]).item()
            tokens.append(next_token)
            
            # 检查结束条件
            if end_token and vocab.idx_to_token(next_token) == end_token:
                break
    
    # 转换回文本
    if isinstance(vocab, CharacterVocabulary):
        return vocab.indices_to_text(tokens)
    else:
        return ' '.join([vocab.idx_to_token(idx) for idx in tokens])


def temperature_sampling(model, vocab, device, start_text="", max_length=200, 
                        temperature=1.0, end_token=None):
    """
    温度采样 - 使用温度参数控制随机性
    
    Args:
        temperature (float): 温度参数
            - < 1.0: 更保守的选择
            - = 1.0: 标准softmax  
            - > 1.0: 更随机的选择
    """
    model.eval()
    
    # 准备起始序列
    if start_text:
        if isinstance(vocab, CharacterVocabulary):
            tokens = vocab.text_to_indices(start_text)
        else:
            tokens = vocab.text_to_indices(start_text.split())
    else:
        tokens = [vocab.get_start_token()]
    
    with torch.no_grad():
        for _ in range(max_length):
            # 准备输入
            input_seq = torch.tensor(tokens[-100:], dtype=torch.long, device=device).unsqueeze(0)
            
            # 前向传播
            logits = model(input_seq)
            
            # 温度缩放
            scaled_logits = logits[0, -1] / temperature
            probabilities = F.softmax(scaled_logits, dim=-1)
            
            # 采样
            next_token = torch.multinomial(probabilities, 1).item()
            tokens.append(next_token)
            
            # 检查结束条件
            if end_token and vocab.idx_to_token(next_token) == end_token:
                break
    
    # 转换回文本
    if isinstance(vocab, CharacterVocabulary):
        return vocab.indices_to_text(tokens)
    else:
        return ' '.join([vocab.idx_to_token(idx) for idx in tokens])


def top_k_sampling(model, vocab, device, start_text="", max_length=200, 
                   k=50, temperature=1.0, end_token=None):
    """
    Top-k采样 - 只考虑概率最高的k个词
    
    Args:
        k (int): 考虑的候选词数量
        temperature (float): 温度参数
    """
    model.eval()
    
    # 准备起始序列
    if start_text:
        if isinstance(vocab, CharacterVocabulary):
            tokens = vocab.text_to_indices(start_text)
        else:
            tokens = vocab.text_to_indices(start_text.split())
    else:
        tokens = [vocab.get_start_token()]
    
    with torch.no_grad():
        for _ in range(max_length):
            # 准备输入
            input_seq = torch.tensor(tokens[-100:], dtype=torch.long, device=device).unsqueeze(0)
            
            # 前向传播
            logits = model(input_seq)
            
            # 温度缩放
            scaled_logits = logits[0, -1] / temperature
            
            # Top-k过滤
            top_k_logits, top_k_indices = torch.topk(scaled_logits, k)
            probabilities = F.softmax(top_k_logits, dim=-1)
            
            # 采样
            sampled_index = torch.multinomial(probabilities, 1).item()
            next_token = top_k_indices[sampled_index].item()
            tokens.append(next_token)
            
            # 检查结束条件
            if end_token and vocab.idx_to_token(next_token) == end_token:
                break
    
    # 转换回文本
    if isinstance(vocab, CharacterVocabulary):
        return vocab.indices_to_text(tokens)
    else:
        return ' '.join([vocab.idx_to_token(idx) for idx in tokens])


def top_p_sampling(model, vocab, device, start_text="", max_length=200, 
                   p=0.9, temperature=1.0, end_token=None):
    """
    Top-p (nucleus) 采样 - 选择累积概率达到p的最小词集
    
    Args:
        p (float): 累积概率阈值 (0 < p <= 1)
        temperature (float): 温度参数
    """
    model.eval()
    
    # 准备起始序列
    if start_text:
        if isinstance(vocab, CharacterVocabulary):
            tokens = vocab.text_to_indices(start_text)
        else:
            tokens = vocab.text_to_indices(start_text.split())
    else:
        tokens = [vocab.get_start_token()]
    
    with torch.no_grad():
        for _ in range(max_length):
            # 准备输入
            input_seq = torch.tensor(tokens[-100:], dtype=torch.long, device=device).unsqueeze(0)
            
            # 前向传播
            logits = model(input_seq)
            
            # 温度缩放
            scaled_logits = logits[0, -1] / temperature
            probabilities = F.softmax(scaled_logits, dim=-1)
            
            # Top-p过滤
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 找到累积概率超过p的位置
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            # 过滤概率
            filtered_probs = sorted_probs.clone()
            filtered_probs[sorted_indices_to_remove] = 0
            filtered_probs = filtered_probs / filtered_probs.sum()
            
            # 采样
            sampled_index = torch.multinomial(filtered_probs, 1).item()
            next_token = sorted_indices[sampled_index].item()
            tokens.append(next_token)
            
            # 检查结束条件
            if end_token and vocab.idx_to_token(next_token) == end_token:
                break
    
    # 转换回文本
    if isinstance(vocab, CharacterVocabulary):
        return vocab.indices_to_text(tokens)
    else:
        return ' '.join([vocab.idx_to_token(idx) for idx in tokens])


def interactive_generation(model, vocab, device):
    """
    交互式文本生成 - 用户可以输入起始文本并选择生成策略
    """
    print("\n" + "="*60)
    print("🎭 交互式文本生成")
    print("="*60)
    
    while True:
        print("\n请选择生成策略:")
        print("1. 贪心解码 (Greedy)")
        print("2. 温度采样 (Temperature)")
        print("3. Top-k 采样")
        print("4. Top-p (Nucleus) 采样")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-4): ").strip()
        
        if choice == '0':
            break
        
        if choice not in ['1', '2', '3', '4']:
            print("❌ 无效选择，请重新输入")
            continue
        
        # 获取参数
        start_text = input("\n请输入起始文本 (留空使用默认): ").strip()
        try:
            max_length = int(input("生成长度 (默认200): ") or "200")
        except ValueError:
            max_length = 200
        
        print(f"\n🚀 正在生成文本...")
        print("-" * 60)
        
        try:
            if choice == '1':
                # 贪心解码
                generated_text = greedy_decode(
                    model, vocab, device, start_text, max_length
                )
                print("📝 贪心解码结果:")
                
            elif choice == '2':
                # 温度采样
                try:
                    temperature = float(input("温度参数 (默认0.8): ") or "0.8")
                except ValueError:
                    temperature = 0.8
                
                generated_text = temperature_sampling(
                    model, vocab, device, start_text, max_length, temperature
                )
                print(f"📝 温度采样结果 (T={temperature}):")
                
            elif choice == '3':
                # Top-k采样
                try:
                    k = int(input("k值 (默认50): ") or "50")
                    temperature = float(input("温度参数 (默认0.8): ") or "0.8")
                except ValueError:
                    k = 50
                    temperature = 0.8
                
                generated_text = top_k_sampling(
                    model, vocab, device, start_text, max_length, k, temperature
                )
                print(f"📝 Top-k采样结果 (k={k}, T={temperature}):")
                
            elif choice == '4':
                # Top-p采样
                try:
                    p = float(input("p值 (默认0.9): ") or "0.9")
                    temperature = float(input("温度参数 (默认0.8): ") or "0.8")
                except ValueError:
                    p = 0.9
                    temperature = 0.8
                
                generated_text = top_p_sampling(
                    model, vocab, device, start_text, max_length, p, temperature
                )
                print(f"📝 Top-p采样结果 (p={p}, T={temperature}):")
            
            print(f"\n{generated_text}")
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='RNN/LSTM文本生成脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--vocab', type=str, required=True,
                       help='词汇表文件路径')
    parser.add_argument('--strategy', type=str, default='temperature',
                       choices=['greedy', 'temperature', 'top_k', 'top_p'],
                       help='生成策略')
    parser.add_argument('--start_text', type=str, default='',
                       help='起始文本')
    parser.add_argument('--max_length', type=int, default=200,
                       help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='温度参数 (用于temperature, top_k, top_p)')
    parser.add_argument('--k', type=int, default=50,
                       help='Top-k采样的k值')
    parser.add_argument('--p', type=float, default=0.9,
                       help='Top-p采样的p值')
    parser.add_argument('--interactive', action='store_true',
                       help='启用交互模式')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='生成样本数量')
    
    args = parser.parse_args()
    
    try:
        # 加载模型和词汇表
        model, vocab, device = load_model_and_vocab(args.checkpoint, args.vocab)
        
        if args.interactive:
            # 交互式生成
            interactive_generation(model, vocab, device)
        else:
            # 批量生成
            print(f"\n🚀 使用 {args.strategy} 策略生成 {args.num_samples} 个样本")
            print("="*60)
            
            for i in range(args.num_samples):
                print(f"\n📝 样本 {i+1}:")
                print("-" * 40)
                
                if args.strategy == 'greedy':
                    generated_text = greedy_decode(
                        model, vocab, device, args.start_text, args.max_length
                    )
                elif args.strategy == 'temperature':
                    generated_text = temperature_sampling(
                        model, vocab, device, args.start_text, args.max_length, 
                        args.temperature
                    )
                elif args.strategy == 'top_k':
                    generated_text = top_k_sampling(
                        model, vocab, device, args.start_text, args.max_length,
                        args.k, args.temperature
                    )
                elif args.strategy == 'top_p':
                    generated_text = top_p_sampling(
                        model, vocab, device, args.start_text, args.max_length,
                        args.p, args.temperature
                    )
                
                print(generated_text)
        
        print(f"\n✅ 生成完成!")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())