"""
阶段4: Transformer vs LSTM Seq2Seq 模型对比评估
================================================

比较Transformer和LSTM Seq2Seq在机器翻译任务上的性能：
- BLEU分数对比
- 推理速度对比  
- 翻译质量分析
- 可视化对比结果
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm

# 导入模型
from models.transformer import Transformer, TransformerConfig
from train import TranslationDataset, build_vocab, create_sample_data, collate_fn
from torch.utils.data import DataLoader


class SimpleLSTMSeq2Seq(nn.Module):
    """简化的LSTM Seq2Seq模型用于对比"""
    
    def __init__(self, 
                 src_vocab_size: int,
                 tgt_vocab_size: int, 
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.encoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, 
            dropout=dropout, batch_first=True
        )
        
        # Decoder
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True
        )
        
        # 输出层
        self.output_projection = nn.Linear(hidden_size, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        # Encoder
        src_emb = self.dropout(self.encoder_embedding(src))
        encoder_output, (hidden, cell) = self.encoder_lstm(src_emb)
        
        # Decoder
        tgt_emb = self.dropout(self.decoder_embedding(tgt))
        decoder_output, _ = self.decoder_lstm(tgt_emb, (hidden, cell))
        
        # 输出
        logits = self.output_projection(decoder_output)
        
        return logits


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results = defaultdict(dict)
    
    def load_transformer(self, checkpoint_path: str, config: TransformerConfig) -> Transformer:
        """加载Transformer模型"""
        model = Transformer(config).to(self.device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 加载Transformer模型: {checkpoint_path}")
        else:
            print(f"⚠️ Transformer模型文件不存在，使用随机初始化")
        
        return model
    
    def create_lstm_baseline(self, src_vocab_size: int, tgt_vocab_size: int) -> SimpleLSTMSeq2Seq:
        """创建LSTM Seq2Seq基线模型"""
        model = SimpleLSTMSeq2Seq(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            hidden_size=256,
            num_layers=2,
            dropout=0.1
        ).to(self.device)
        
        print("✅ 创建LSTM Seq2Seq基线模型")
        return model
    
    def compute_bleu_score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """计算详细的BLEU分数"""
        if len(references) != len(candidates):
            return {'bleu-1': 0.0, 'bleu-2': 0.0, 'bleu-3': 0.0, 'bleu-4': 0.0}
        
        def get_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
            ngrams = defaultdict(int)
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams[ngram] += 1
            return dict(ngrams)
        
        def compute_precision(ref_tokens: List[str], cand_tokens: List[str], n: int) -> float:
            if len(cand_tokens) < n:
                return 0.0
            
            ref_ngrams = get_ngrams(ref_tokens, n)
            cand_ngrams = get_ngrams(cand_tokens, n)
            
            matches = 0
            total = 0
            
            for ngram, count in cand_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
                total += count
            
            return matches / total if total > 0 else 0.0
        
        # 计算各阶n-gram BLEU
        bleu_scores = {'bleu-1': 0.0, 'bleu-2': 0.0, 'bleu-3': 0.0, 'bleu-4': 0.0}
        
        for n in range(1, 5):
            total_precision = 0
            valid_count = 0
            
            for ref, cand in zip(references, candidates):
                ref_tokens = ref.split()
                cand_tokens = cand.split()
                
                precision = compute_precision(ref_tokens, cand_tokens, n)
                total_precision += precision
                valid_count += 1
            
            bleu_scores[f'bleu-{n}'] = total_precision / valid_count if valid_count > 0 else 0.0
        
        return bleu_scores
    
    def translate_with_transformer(self, 
                                  model: Transformer, 
                                  src: torch.Tensor, 
                                  tgt_vocab: Dict[str, int],
                                  max_length: int = 50) -> str:
        """使用Transformer翻译"""
        model.eval()
        
        with torch.no_grad():
            # 初始化decoder输入
            tgt_tokens = [tgt_vocab['<sos>']]
            
            for _ in range(max_length):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long, device=self.device)
                
                # 前向传播
                logits = model(src, tgt_tensor)
                
                # 获取下一个token
                next_token = logits[0, -1, :].argmax(dim=-1).item()
                tgt_tokens.append(next_token)
                
                # 检查结束条件
                if next_token == tgt_vocab['<eos>']:
                    break
            
            return tgt_tokens[1:-1]  # 去掉<sos>和<eos>
    
    def translate_with_lstm(self, 
                           model: SimpleLSTMSeq2Seq,
                           src: torch.Tensor,
                           tgt_vocab: Dict[str, int],
                           max_length: int = 50) -> str:
        """使用LSTM Seq2Seq翻译"""
        model.eval()
        
        with torch.no_grad():
            # Encoder
            src_emb = model.encoder_embedding(src)
            encoder_output, (hidden, cell) = model.encoder_lstm(src_emb)
            
            # Decoder
            tgt_tokens = [tgt_vocab['<sos>']]
            
            for _ in range(max_length):
                tgt_input = torch.tensor([[tgt_tokens[-1]]], dtype=torch.long, device=self.device)
                tgt_emb = model.decoder_embedding(tgt_input)
                
                decoder_output, (hidden, cell) = model.decoder_lstm(tgt_emb, (hidden, cell))
                logits = model.output_projection(decoder_output)
                
                next_token = logits[0, -1, :].argmax(dim=-1).item()
                tgt_tokens.append(next_token)
                
                if next_token == tgt_vocab['<eos>']:
                    break
            
            return tgt_tokens[1:-1]  # 去掉<sos>和<eos>
    
    def evaluate_model(self, 
                      model, 
                      model_name: str,
                      test_loader: DataLoader, 
                      src_vocab: Dict[str, int],
                      tgt_vocab: Dict[str, int],
                      is_transformer: bool = True) -> Dict[str, float]:
        """评估单个模型"""
        
        print(f"\n🔍 评估{model_name}模型...")
        
        # 反向词典
        tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}
        
        def decode_sequence(sequence):
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.tolist()
            tokens = []
            for token_id in sequence:
                if token_id in [tgt_vocab['<sos>'], tgt_vocab['<eos>'], tgt_vocab['<pad>']]:
                    continue
                token = tgt_idx2word.get(token_id, '<unk>')
                tokens.append(token)
            return ' '.join(tokens)
        
        references = []
        candidates = []
        inference_times = []
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"评估{model_name}")):
                if batch_idx >= 50:  # 只评估前50个batch以节省时间
                    break
                
                src = batch['src'].to(self.device)
                tgt = batch['tgt']
                
                batch_size = src.size(0)
                
                for i in range(min(5, batch_size)):  # 每个batch只取前5个样本
                    src_seq = src[i:i+1]
                    
                    # 记录推理时间
                    start_time = time.time()
                    
                    if is_transformer:
                        translated_tokens = self.translate_with_transformer(
                            model, src_seq, tgt_vocab, max_length=50
                        )
                    else:
                        translated_tokens = self.translate_with_lstm(
                            model, src_seq, tgt_vocab, max_length=50
                        )
                    
                    end_time = time.time()
                    inference_times.append(end_time - start_time)
                    
                    # 解码
                    reference = decode_sequence(tgt[i])
                    candidate = decode_sequence(translated_tokens)
                    
                    references.append(reference)
                    candidates.append(candidate)
        
        # 计算指标
        bleu_scores = self.compute_bleu_score(references, candidates)
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        
        results = {
            'bleu_scores': bleu_scores,
            'avg_inference_time': avg_inference_time,
            'num_params': sum(p.numel() for p in model.parameters()),
            'sample_translations': list(zip(references[:5], candidates[:5]))
        }
        
        self.results[model_name] = results
        
        return results
    
    def compare_models(self,
                      transformer_checkpoint: str,
                      test_loader: DataLoader,
                      src_vocab: Dict[str, int], 
                      tgt_vocab: Dict[str, int]):
        """对比两个模型"""
        
        print("🆚 开始模型对比评估...")
        
        # 1. 加载/创建模型
        transformer_config = TransformerConfig(
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
        
        transformer = self.load_transformer(transformer_checkpoint, transformer_config)
        lstm_seq2seq = self.create_lstm_baseline(len(src_vocab), len(tgt_vocab))
        
        # 2. 评估模型
        transformer_results = self.evaluate_model(
            transformer, "Transformer", test_loader, src_vocab, tgt_vocab, is_transformer=True
        )
        
        lstm_results = self.evaluate_model(
            lstm_seq2seq, "LSTM Seq2Seq", test_loader, src_vocab, tgt_vocab, is_transformer=False
        )
        
        # 3. 打印对比结果
        self.print_comparison()
        
        # 4. 可视化
        self.visualize_comparison()
        
        return self.results
    
    def print_comparison(self):
        """打印对比结果"""
        print("\n" + "="*60)
        print("📊 模型对比结果")
        print("="*60)
        
        for model_name, results in self.results.items():
            print(f"\n🔹 {model_name}:")
            print(f"   参数量: {results['num_params']:,}")
            print(f"   推理时间: {results['avg_inference_time']:.4f}s")
            
            bleu = results['bleu_scores']
            print(f"   BLEU-1: {bleu['bleu-1']:.4f}")
            print(f"   BLEU-2: {bleu['bleu-2']:.4f}")  
            print(f"   BLEU-3: {bleu['bleu-3']:.4f}")
            print(f"   BLEU-4: {bleu['bleu-4']:.4f}")
        
        # 对比分析
        if len(self.results) == 2:
            models = list(self.results.keys())
            transformer_results = self.results[models[0]] if 'Transformer' in models[0] else self.results[models[1]]
            lstm_results = self.results[models[1]] if 'LSTM' in models[1] else self.results[models[0]]
            
            print(f"\n🔍 性能对比:")
            
            bleu_improvement = (
                transformer_results['bleu_scores']['bleu-4'] - 
                lstm_results['bleu_scores']['bleu-4']
            ) * 100
            
            speed_ratio = lstm_results['avg_inference_time'] / transformer_results['avg_inference_time']
            
            print(f"   BLEU-4提升: {bleu_improvement:+.2f}%")
            print(f"   速度对比: Transformer是LSTM的{speed_ratio:.2f}倍")
        
        print("\n🎯 翻译示例对比:")
        for model_name, results in self.results.items():
            print(f"\n{model_name}翻译示例:")
            for i, (ref, cand) in enumerate(results['sample_translations'][:3]):
                print(f"  示例{i+1}:")
                print(f"    参考: {ref}")
                print(f"    翻译: {cand}")
    
    def visualize_comparison(self):
        """可视化对比结果"""
        if len(self.results) < 2:
            return
        
        # 设置图形样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Transformer vs LSTM Seq2Seq 对比分析', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
        
        # 1. BLEU分数对比
        ax1 = axes[0, 0]
        bleu_metrics = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']
        x = np.arange(len(bleu_metrics))
        width = 0.35
        
        for i, model_name in enumerate(models):
            bleu_values = [self.results[model_name]['bleu_scores'][metric] for metric in bleu_metrics]
            ax1.bar(x + i*width, bleu_values, width, label=model_name, alpha=0.8)
        
        ax1.set_xlabel('BLEU指标')
        ax1.set_ylabel('分数')
        ax1.set_title('BLEU分数对比')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(bleu_metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 参数量对比
        ax2 = axes[0, 1]
        param_counts = [self.results[model]['num_params'] for model in models]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax2.bar(models, param_counts, color=colors, alpha=0.8)
        ax2.set_ylabel('参数量')
        ax2.set_title('模型参数量对比')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, param_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom')
        
        # 3. 推理时间对比
        ax3 = axes[1, 0]
        inference_times = [self.results[model]['avg_inference_time'] for model in models]
        
        bars = ax3.bar(models, inference_times, color=['lightgreen', 'lightsalmon'], alpha=0.8)
        ax3.set_ylabel('平均推理时间 (秒)')
        ax3.set_title('推理速度对比')
        ax3.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, time_val in zip(bars, inference_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.4f}s', ha='center', va='bottom')
        
        # 4. 综合性能雷达图
        ax4 = axes[1, 1]
        
        # 归一化指标 (0-1)
        def normalize_metric(values, higher_better=True):
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [0.5] * len(values)
            if higher_better:
                return [(v - min_val) / (max_val - min_val) for v in values]
            else:  # lower is better
                return [(max_val - v) / (max_val - min_val) for v in values]
        
        metrics = ['BLEU-4', '速度', '效率']
        
        bleu4_scores = [self.results[model]['bleu_scores']['bleu-4'] for model in models]
        speed_scores = normalize_metric([1/self.results[model]['avg_inference_time'] for model in models])
        efficiency_scores = normalize_metric([
            self.results[model]['bleu_scores']['bleu-4'] / (self.results[model]['num_params'] / 1e6)
            for model in models
        ])
        
        bleu4_normalized = normalize_metric(bleu4_scores)
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # 闭合
        
        for i, model in enumerate(models):
            values = [bleu4_normalized[i], speed_scores[i], efficiency_scores[i]]
            values = np.concatenate((values, [values[0]]))  # 闭合
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=model)
            ax4.fill(angles, values, alpha=0.25)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('综合性能对比')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('./evaluation_results', exist_ok=True)
        plt.savefig('./evaluation_results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 对比图已保存到 ./evaluation_results/model_comparison.png")
    
    def save_results(self, save_path: str = './evaluation_results/comparison_results.json'):
        """保存评估结果"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 处理不可序列化的对象
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {
                'bleu_scores': results['bleu_scores'],
                'avg_inference_time': results['avg_inference_time'],
                'num_params': results['num_params'],
                'sample_translations': results['sample_translations'][:3]  # 只保存前3个示例
            }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 评估结果已保存到: {save_path}")


def main():
    """主评估函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 准备测试数据
    print("📚 准备测试数据...")
    en_sentences, fr_sentences = create_sample_data(500)  # 更少的数据用于快速评估
    
    # 构建词汇表
    src_vocab = build_vocab(en_sentences, min_freq=1)
    tgt_vocab = build_vocab(fr_sentences, min_freq=1)
    
    # 创建测试数据集
    test_dataset = TranslationDataset(en_sentences, fr_sentences, src_vocab, tgt_vocab)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # 创建评估器
    evaluator = ModelEvaluator(device)
    
    # 对比评估
    transformer_checkpoint = './transformer_checkpoints/best_model.pt'
    
    results = evaluator.compare_models(
        transformer_checkpoint=transformer_checkpoint,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab
    )
    
    # 保存结果
    evaluator.save_results()
    
    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()