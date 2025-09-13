"""
GPT文本生成器实现
==================

包含多种生成策略：
- 贪心解码
- 随机采样 (Top-K, Top-P)
- 束搜索
- 对比搜索
- 批量生成
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Union, Tuple
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """文本生成配置"""
    max_new_tokens: int = 50          # 最大生成token数
    temperature: float = 1.0          # 采样温度
    top_k: Optional[int] = None       # Top-K采样
    top_p: Optional[float] = None     # Top-P (nucleus)采样
    repetition_penalty: float = 1.0   # 重复惩罚
    length_penalty: float = 1.0       # 长度惩罚
    no_repeat_ngram_size: int = 0     # 防止n-gram重复
    
    # 束搜索参数
    num_beams: int = 1               # 束搜索大小
    early_stopping: bool = True      # 早停
    
    # 特殊token
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    
    # 输出控制
    do_sample: bool = True           # 是否采样
    num_return_sequences: int = 1    # 返回序列数量
    
    # 批量生成
    batch_size: int = 1             # 批量大小


class BaseGenerator(ABC):
    """生成器基类"""
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        
    @abstractmethod
    def generate(self, 
                 input_ids: torch.Tensor, 
                 config: GenerationConfig) -> torch.Tensor:
        """生成文本"""
        pass
    
    def generate_text(self, 
                     prompt: str, 
                     config: GenerationConfig) -> List[str]:
        """从文本prompt生成"""
        if self.tokenizer is None:
            raise ValueError("需要提供tokenizer来处理文本")
        
        # 编码输入
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        
        # 生成
        outputs = self.generate(input_ids, config)
        
        # 解码输出
        results = []
        for output in outputs:
            text = self.tokenizer.decode(output.tolist())
            results.append(text)
        
        return results


class GreedyGenerator(BaseGenerator):
    """贪心解码生成器"""
    
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor, 
                 config: GenerationConfig) -> torch.Tensor:
        """贪心解码生成"""
        self.model.eval()
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        for _ in range(config.max_new_tokens):
            # 截断到最大长度
            if hasattr(self.model, 'config'):
                max_len = self.model.config.max_seq_len
                if input_ids.size(1) > max_len:
                    input_ids = input_ids[:, -max_len:]
            
            # 前向传播
            outputs = self.model(input_ids)
            logits = outputs["logits"][:, -1, :]  # 最后位置的logits
            
            # 应用重复惩罚
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, input_ids, config.repetition_penalty)
            
            # 贪心选择
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # 添加新token
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查停止条件
            if config.eos_token_id and next_token.item() == config.eos_token_id:
                break
        
        return input_ids
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                 input_ids: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        """应用重复惩罚"""
        if penalty == 1.0:
            return logits
        
        for batch_idx in range(logits.size(0)):
            for token_id in set(input_ids[batch_idx].tolist()):
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= penalty
                else:
                    logits[batch_idx, token_id] /= penalty
        
        return logits


class SamplingGenerator(BaseGenerator):
    """随机采样生成器 (支持Top-K和Top-P)"""
    
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor, 
                 config: GenerationConfig) -> torch.Tensor:
        """随机采样生成"""
        self.model.eval()
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        for _ in range(config.max_new_tokens):
            # 截断到最大长度
            if hasattr(self.model, 'config'):
                max_len = self.model.config.max_seq_len
                if input_ids.size(1) > max_len:
                    input_ids = input_ids[:, -max_len:]
            
            # 前向传播
            outputs = self.model(input_ids)
            logits = outputs["logits"][:, -1, :] / config.temperature
            
            # 应用重复惩罚
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, input_ids, config.repetition_penalty)
            
            # Top-K过滤
            if config.top_k is not None and config.top_k > 0:
                logits = self._top_k_filtering(logits, config.top_k)
            
            # Top-P过滤
            if config.top_p is not None and config.top_p < 1.0:
                logits = self._top_p_filtering(logits, config.top_p)
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加新token
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查停止条件
            if config.eos_token_id and next_token.item() == config.eos_token_id:
                break
        
        return input_ids
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                 input_ids: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        """应用重复惩罚"""
        if penalty == 1.0:
            return logits
        
        for batch_idx in range(logits.size(0)):
            for token_id in set(input_ids[batch_idx].tolist()):
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= penalty
                else:
                    logits[batch_idx, token_id] /= penalty
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Top-K过滤"""
        if k <= 0:
            return logits
        
        top_k_logits, _ = torch.topk(logits, min(k, logits.size(-1)))
        indices_to_remove = logits < top_k_logits[..., [-1]]
        logits = logits.masked_fill(indices_to_remove, -torch.inf)
        
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Top-P (nucleus)过滤"""
        if p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过p的tokens
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 映射回原始索引
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -torch.inf)
        
        return logits


class BeamSearchGenerator(BaseGenerator):
    """束搜索生成器"""
    
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor, 
                 config: GenerationConfig) -> torch.Tensor:
        """束搜索生成"""
        if config.num_beams <= 1:
            # 退化为贪心搜索
            greedy_gen = GreedyGenerator(self.model, self.tokenizer)
            return greedy_gen.generate(input_ids, config)
        
        self.model.eval()
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        batch_size, seq_len = input_ids.size()
        beam_size = config.num_beams
        
        # 扩展输入到beam维度
        input_ids = input_ids.unsqueeze(1).repeat(1, beam_size, 1)
        input_ids = input_ids.view(batch_size * beam_size, seq_len)
        
        # 初始化beam状态
        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_scores[:, 1:] = -1e9  # 只保留第一个beam活跃
        beam_scores = beam_scores.view(-1)  # [batch_size * beam_size]
        
        # 完成的序列
        done = [False] * batch_size
        
        for step in range(config.max_new_tokens):
            # 截断到最大长度
            if hasattr(self.model, 'config'):
                max_len = self.model.config.max_seq_len
                if input_ids.size(1) > max_len:
                    input_ids = input_ids[:, -max_len:]
            
            # 前向传播
            outputs = self.model(input_ids)
            logits = outputs["logits"][:, -1, :]  # [batch_size * beam_size, vocab_size]
            
            vocab_size = logits.size(-1)
            
            # 计算log概率
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 加上之前的分数
            log_probs = log_probs + beam_scores.unsqueeze(-1)
            
            # 重塑为 [batch_size, beam_size * vocab_size]
            log_probs = log_probs.view(batch_size, beam_size * vocab_size)
            
            # 选择top-k candidates
            top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=1)
            
            # 计算beam和token索引
            beam_indices = top_indices // vocab_size  # 哪个beam
            token_indices = top_indices % vocab_size   # 哪个token
            
            # 更新beam_scores
            beam_scores = top_log_probs.view(-1)
            
            # 重新排列input_ids
            beam_indices_expanded = beam_indices.view(-1) + torch.arange(
                0, batch_size * beam_size, beam_size, device=device
            ).unsqueeze(-1).expand(-1, beam_size).reshape(-1)
            
            input_ids = input_ids[beam_indices_expanded]
            
            # 添加新tokens
            new_tokens = token_indices.view(-1, 1)
            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            
            # 检查结束条件
            if config.eos_token_id:
                eos_mask = new_tokens.squeeze(-1) == config.eos_token_id
                if eos_mask.any() and config.early_stopping:
                    break
        
        # 返回最佳序列
        best_beam_indices = beam_scores.view(batch_size, beam_size).argmax(dim=1)
        best_sequences = []
        
        for i in range(batch_size):
            best_idx = i * beam_size + best_beam_indices[i]
            best_sequences.append(input_ids[best_idx])
        
        return torch.stack(best_sequences)


class ContrastiveSearchGenerator(BaseGenerator):
    """对比搜索生成器"""
    
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor, 
                 config: GenerationConfig,
                 alpha: float = 0.6,
                 k: int = 4) -> torch.Tensor:
        """
        对比搜索生成
        
        Args:
            alpha: 对比权重
            k: 候选数量
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        for _ in range(config.max_new_tokens):
            # 截断到最大长度
            if hasattr(self.model, 'config'):
                max_len = self.model.config.max_seq_len
                if input_ids.size(1) > max_len:
                    input_ids = input_ids[:, -max_len:]
            
            # 前向传播获取logits和hidden states
            outputs = self.model(input_ids, output_hidden_states=True)
            logits = outputs["logits"][:, -1, :]
            
            # Top-k候选
            top_k_logits, top_k_indices = torch.topk(logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # 计算对比分数 (这里简化实现)
            # 在实际应用中，需要计算token表示之间的相似性
            contrastive_scores = torch.zeros_like(probs)
            
            # 结合概率和对比分数
            final_scores = alpha * probs + (1 - alpha) * contrastive_scores
            
            # 选择最佳候选
            best_idx = torch.argmax(final_scores, dim=-1)
            next_token = top_k_indices[torch.arange(input_ids.size(0)), best_idx].unsqueeze(-1)
            
            # 添加新token
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查停止条件
            if config.eos_token_id and next_token.item() == config.eos_token_id:
                break
        
        return input_ids


class BatchGenerator:
    """批量生成器 - 支持同时生成多个序列"""
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def generate_batch(self, 
                      prompts: List[str], 
                      config: GenerationConfig) -> List[List[str]]:
        """批量生成文本"""
        if self.tokenizer is None:
            raise ValueError("需要提供tokenizer来处理文本")
        
        # 编码所有prompts
        input_ids_list = []
        max_len = 0
        
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt)
            input_ids_list.append(ids)
            max_len = max(max_len, len(ids))
        
        # Padding到相同长度
        padded_inputs = []
        attention_masks = []
        
        pad_id = config.pad_token_id or 0
        
        for ids in input_ids_list:
            padding_length = max_len - len(ids)
            padded_ids = ids + [pad_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length
            
            padded_inputs.append(padded_ids)
            attention_masks.append(mask)
        
        # 转换为tensor
        input_ids = torch.tensor(padded_inputs, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        
        # 选择生成器
        if config.num_beams > 1:
            generator = BeamSearchGenerator(self.model, self.tokenizer)
        elif config.do_sample:
            generator = SamplingGenerator(self.model, self.tokenizer)
        else:
            generator = GreedyGenerator(self.model, self.tokenizer)
        
        # 批量生成
        results = []
        batch_size = config.batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_inputs = input_ids[i:i+batch_size]
            batch_outputs = generator.generate(batch_inputs, config)
            
            # 解码输出
            batch_results = []
            for output in batch_outputs:
                text = self.tokenizer.decode(output.tolist())
                batch_results.append([text])  # 包装为列表以保持一致性
            
            results.extend(batch_results)
        
        return results


# 便捷函数
def create_generator(strategy: str, model, tokenizer=None):
    """创建指定策略的生成器"""
    generators = {
        'greedy': GreedyGenerator,
        'sampling': SamplingGenerator,
        'beam_search': BeamSearchGenerator,
        'contrastive': ContrastiveSearchGenerator,
    }
    
    if strategy not in generators:
        raise ValueError(f"不支持的生成策略: {strategy}")
    
    return generators[strategy](model, tokenizer)


def generate_text(model, 
                 prompt: str, 
                 tokenizer=None,
                 strategy: str = "sampling",
                 **kwargs) -> str:
    """便捷的文本生成函数"""
    
    # 创建配置
    config = GenerationConfig(**kwargs)
    
    # 创建生成器
    generator = create_generator(strategy, model, tokenizer)
    
    # 生成文本
    results = generator.generate_text(prompt, config)
    
    return results[0] if results else ""


if __name__ == "__main__":
    # 测试生成器
    print("🎯 测试文本生成器")
    
    # 这里需要导入模型进行测试
    # 在实际使用时取消注释相关代码
    
    print("✅ 生成器测试完成!")