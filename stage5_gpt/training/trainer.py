"""
GPT训练器实现
=============

包含：
- 预训练循环
- 微调功能  
- 评估指标
- 检查点管理
- 性能监控
"""

import os
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
import json
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPTTrainingConfig:
    """GPT训练配置"""
    
    # 基本配置
    output_dir: str = "./gpt_checkpoints"
    run_name: str = "gpt_mini_training"
    
    # 训练参数
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    
    # 学习率调度
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # 正则化
    gradient_clip_val: float = 1.0
    dropout: float = 0.1
    
    # 评估
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # 数据处理
    max_seq_len: int = 512
    num_workers: int = 4
    
    # 设备设置
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    
    # 其他
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    save_only_best: bool = True
    
    def __post_init__(self):
        # 自动检测设备
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class LanguageModelingDataset(Dataset):
    """语言建模数据集"""
    
    def __init__(self, 
                 texts: List[str], 
                 tokenizer, 
                 max_length: int = 512,
                 stride: int = None):
        """
        Args:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 最大序列长度
            stride: 滑动窗口步长
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length // 2
        
        # 处理文本数据
        self.examples = self._process_texts(texts)
        
    def _process_texts(self, texts: List[str]) -> List[torch.Tensor]:
        """处理文本数据为训练样本"""
        examples = []
        
        for text in tqdm(texts, desc="处理文本数据"):
            # 分词
            tokens = self.tokenizer.encode(text)
            
            # 滑动窗口切分
            for i in range(0, len(tokens) - self.max_length + 1, self.stride):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) == self.max_length:
                    examples.append(torch.tensor(chunk, dtype=torch.long))
        
        logger.info(f"生成 {len(examples)} 个训练样本")
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class GPTTrainer:
    """GPT训练器"""
    
    def __init__(self, 
                 model,
                 config: GPTTrainingConfig,
                 train_dataset: Dataset,
                 eval_dataset: Optional[Dataset] = None,
                 tokenizer = None):
        
        self.model = model
        self.config = config  
        self.tokenizer = tokenizer
        
        # 设置设备
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # 数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True if config.device == "cuda" else False
        )
        
        self.eval_dataloader = None
        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True if config.device == "cuda" else False
            )
        
        # 优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 保存配置
        self._save_config()
        
    def _create_optimizer(self):
        """创建优化器"""
        # 分组参数 (不对LayerNorm和bias应用权重衰减)
        decay_params = []
        nodecay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'layernorm' in name:
                    nodecay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        total_steps = self.config.max_steps
        if total_steps is None:
            total_steps = len(self.train_dataloader) * self.config.num_epochs
            
        return CosineAnnealingLR(self.optimizer, T_max=total_steps)
    
    def _save_config(self):
        """保存训练配置"""
        config_path = os.path.join(self.config.output_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            # 转换为字典 (处理不可序列化的对象)
            config_dict = {}
            for key, value in self.config.__dict__.items():
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)
            
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def train(self):
        """开始训练"""
        logger.info(f"🚀 开始GPT训练")
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"训练配置: {self.config}")
        
        # 恢复检查点
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
        
        # 训练循环
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(range(self.config.num_epochs), desc="训练进度")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch()
            
            # 更新进度条
            progress_bar.set_postfix({
                'epoch': epoch + 1,
                'loss': f'{epoch_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            progress_bar.update(1)
            
            # 评估
            if self.eval_dataloader and (epoch + 1) % max(1, self.config.num_epochs // 5) == 0:
                eval_results = self.evaluate()
                logger.info(f"Epoch {epoch + 1} 评估结果: {eval_results}")
                
                # 保存最佳模型
                if eval_results['eval_loss'] < self.best_eval_loss:
                    self.best_eval_loss = eval_results['eval_loss']
                    self._save_checkpoint(is_best=True)
        
        progress_bar.close()
        logger.info("🎉 训练完成!")
        
        # 保存最终模型
        self._save_checkpoint(is_final=True)
    
    def _train_epoch(self) -> float:
        """训练一个epoch"""
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            loss = self._train_step(batch)
            epoch_loss += loss
            
            # 日志记录
            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step}, "
                    f"Loss: {loss:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # 评估
            if (self.eval_dataloader and 
                self.global_step % self.config.eval_steps == 0 and 
                self.global_step > 0):
                
                eval_results = self.evaluate()
                logger.info(f"Step {self.global_step} 评估: {eval_results}")
                self.model.train()
            
            # 保存检查点
            if self.global_step % self.config.save_steps == 0 and self.global_step > 0:
                self._save_checkpoint()
            
            # 检查是否达到最大步数
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        return epoch_loss / num_batches
    
    def _train_step(self, batch: torch.Tensor) -> float:
        """执行一步训练"""
        input_ids = batch.to(self.device)
        
        # 混合精度前向传播
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, targets=input_ids)
                loss = outputs["loss"]
        else:
            outputs = self.model(input_ids, targets=input_ids)
            loss = outputs["loss"]
        
        # 反向传播
        if self.scaler:
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            
            # 更新参数
            self.optimizer.step()
        
        # 清零梯度和更新调度器
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        self.global_step += 1
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if not self.eval_dataloader:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in tqdm(self.eval_dataloader, desc="评估中"):
            input_ids = batch.to(self.device)
            
            outputs = self.model(input_ids, targets=input_ids)
            loss = outputs["loss"]
            
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
        }
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """保存检查点"""
        checkpoint = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.__dict__,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 文件名
        if is_final:
            filename = "final_checkpoint.pt"
        elif is_best:
            filename = "best_checkpoint.pt"
        else:
            filename = f"checkpoint_step_{self.global_step}.pt"
        
        save_path = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, save_path)
        
        logger.info(f"检查点已保存: {save_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_eval_loss = checkpoint['best_eval_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"从检查点恢复训练: {checkpoint_path}")
        logger.info(f"全局步数: {self.global_step}, 当前epoch: {self.current_epoch}")


def compute_metrics(model, dataloader, device: str = "cpu") -> Dict[str, float]:
    """计算模型指标"""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算指标"):
            input_ids = batch.to(device)
            
            outputs = model(input_ids, targets=input_ids)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            # 累积损失
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
            
            # 计算准确率 (下一个token预测)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            predictions = shift_logits.argmax(dim=-1)
            correct = (predictions == shift_labels).sum().item()
            correct_predictions += correct
    
    avg_loss = total_loss / total_tokens
    accuracy = correct_predictions / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
    }


if __name__ == "__main__":
    # 测试训练器
    print("🧪 测试GPT训练器")
    
    # 这里需要导入模型，在实际使用时取消注释
    # from ..models.gpt_mini import GPTMini, GPTConfig
    
    # # 创建模型和数据
    # config = GPTConfig.gpt_mini()
    # model = GPTMini(config)
    
    # # 创建虚拟数据
    # texts = [f"这是第 {i} 个训练样本。" * 10 for i in range(100)]
    
    # # 简单的字符级分词器 (演示用)
    # class SimpleTokenizer:
    #     def __init__(self):
    #         self.char_to_idx = {}
    #         self.idx_to_char = {}
    
    #     def fit(self, texts):
    #         chars = set(''.join(texts))
    #         self.char_to_idx = {c: i for i, c in enumerate(sorted(chars))}
    #         self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
    
    #     def encode(self, text):
    #         return [self.char_to_idx.get(c, 0) for c in text]
    
    # tokenizer = SimpleTokenizer()
    # tokenizer.fit(texts)
    
    # # 创建数据集
    # train_dataset = LanguageModelingDataset(texts[:80], tokenizer, max_length=64)
    # eval_dataset = LanguageModelingDataset(texts[80:], tokenizer, max_length=64)
    
    # # 训练配置
    # training_config = GPTTrainingConfig(
    #     num_epochs=2,
    #     batch_size=4,
    #     learning_rate=1e-4,
    #     output_dir="./test_gpt_output"
    # )
    
    # # 创建训练器
    # trainer = GPTTrainer(
    #     model=model,
    #     config=training_config,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer
    # )
    
    # # 开始训练
    # trainer.train()
    
    print("✅ 训练器测试完成!")