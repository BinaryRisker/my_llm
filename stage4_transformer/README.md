# 阶段4: Transformer模型实现与实验

本阶段实现了完整的Transformer模型，用于机器翻译任务，并与LSTM Seq2Seq进行对比评估。

## 🎯 学习目标

- 理解Transformer的完整架构
- 掌握自注意力机制的实现
- 学习机器翻译任务的训练流程
- 对比Transformer与传统序列模型的性能

## 📁 项目结构

```
stage4_transformer/
├── models/                          # 模型实现
│   ├── __init__.py                 # 模块初始化
│   ├── multi_head_attention.py     # 多头注意力机制
│   ├── positional_encoding.py      # 位置编码
│   ├── feed_forward.py             # 前馈网络
│   ├── transformer_layers.py       # Transformer层
│   └── transformer.py              # 完整Transformer模型
├── train.py                        # 训练脚本
├── evaluate.py                     # 评估和对比脚本
├── run_experiment.py              # 快速运行脚本
├── README.md                      # 说明文档
├── data/                          # 数据目录
├── utils/                         # 工具函数
├── transformer_checkpoints/       # 模型检查点 (训练后生成)
├── evaluation_results/            # 评估结果 (评估后生成)
└── experiment_results/            # 实验报告 (完成后生成)
```

## 🚀 快速开始

### 1. 环境要求

```bash
# 必需依赖
pip install torch numpy matplotlib tqdm seaborn

# 可选依赖 (用于更好的可视化)
pip install pandas jupyter
```

### 2. 一键运行实验

```bash
# 完整实验 (训练 + 评估 + 报告)
python run_experiment.py

# 只训练模型
python run_experiment.py --mode train --epochs 15

# 只运行评估
python run_experiment.py --mode eval

# 自定义参数
python run_experiment.py --epochs 20 --batch-size 16
```

### 3. 分步运行

#### 步骤1: 训练Transformer模型

```bash
python train.py
```

这将：
- 创建人工英法翻译数据
- 训练Transformer模型
- 保存最佳模型到 `./transformer_checkpoints/`
- 记录训练历史

#### 步骤2: 评估和对比

```bash
python evaluate.py
```

这将：
- 加载训练好的Transformer模型
- 创建LSTM Seq2Seq基线模型
- 对比两个模型的BLEU分数和推理速度
- 生成可视化对比图表

## 📊 实验结果

### 模型架构对比

| 模型 | 参数量 | 架构特点 | 主要优势 |
|------|--------|----------|----------|
| **Transformer** | ~2.5M | Encoder-Decoder + 自注意力 | 并行训练，长距离依赖建模 |
| **LSTM Seq2Seq** | ~1.8M | Encoder-Decoder + LSTM | 简单有效，内存效率高 |

### 预期性能指标

- **BLEU分数**: Transformer通常比LSTM高5-15%
- **训练速度**: Transformer支持更好的并行化
- **推理速度**: 取决于序列长度和硬件
- **内存使用**: Transformer在长序列上内存使用更大

## 🔧 模型配置

### Transformer配置

```python
TransformerConfig(
    src_vocab_size=vocab_size,      # 源语言词汇量
    tgt_vocab_size=vocab_size,      # 目标语言词汇量  
    d_model=512,                    # 模型维度
    nhead=8,                        # 注意力头数
    num_encoder_layers=6,           # 编码器层数
    num_decoder_layers=6,           # 解码器层数
    dim_feedforward=2048,           # 前馈网络维度
    max_seq_length=100,             # 最大序列长度
    dropout=0.1                     # Dropout概率
)
```

### 训练超参数

- **学习率**: 1e-4 (Adam优化器)
- **批次大小**: 32
- **训练轮数**: 20
- **学习率调度**: StepLR (每10轮衰减0.95)
- **梯度裁剪**: 最大范数1.0
- **标签平滑**: 0.1

## 📈 评估指标

### BLEU分数
- **BLEU-1**: 单词级别匹配精度
- **BLEU-2**: 双词组匹配精度  
- **BLEU-3**: 三词组匹配精度
- **BLEU-4**: 四词组匹配精度

### 推理性能
- **平均推理时间**: 单个样本的推理耗时
- **参数效率**: BLEU分数与参数量的比值
- **内存使用**: 推理时的GPU/CPU内存占用

## 🔍 关键技术点

### 1. 自注意力机制
```python
# 缩放点积注意力
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
attention = F.softmax(scores, dim=-1)
output = torch.matmul(attention, V)
```

### 2. 位置编码
```python
# 正弦余弦位置编码
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 3. 掩码机制
```python
# 因果掩码 (防止看到未来信息)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
scores = scores.masked_fill(~mask, -torch.inf)
```

## 📚 学习资源

### 核心论文
1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Transformer原始论文
   - [链接](https://arxiv.org/abs/1706.03762)

2. **"BLEU: A Method for Automatic Evaluation"** (Papineni et al., 2002)
   - BLEU评估指标
   - [链接](https://aclanthology.org/P02-1040/)

### 相关资源
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer详解博客](https://blog.csdn.net/transformer_tutorial)
- [PyTorch官方Transformer教程](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## 🛠️ 常见问题

### Q1: 训练过程中损失不下降怎么办？
**A**: 
- 检查学习率是否过大或过小
- 确认梯度裁剪设置合理
- 验证数据预处理是否正确
- 尝试调整模型规模

### Q2: 内存不足怎么解决？
**A**:
- 减小批次大小 (`--batch-size`)
- 减少模型层数或维度
- 使用梯度检查点 (gradient checkpointing)
- 使用混合精度训练

### Q3: 如何提高翻译质量？
**A**:
- 增加训练数据量和质量
- 调整标签平滑参数
- 使用束搜索解码
- 增大模型规模

### Q4: 如何加速训练？
**A**:
- 使用GPU加速
- 增大批次大小
- 使用数据并行
- 优化数据加载流程

## 🎯 扩展实验

### 1. 架构改进
- [ ] 实现相对位置编码 (RoPE)
- [ ] 添加Layer Scale
- [ ] 实现Pre-LN vs Post-LN对比

### 2. 训练优化  
- [ ] 实现Warmup学习率调度
- [ ] 添加混合精度训练
- [ ] 实现梯度检查点

### 3. 评估扩展
- [ ] 添加ROUGE评估指标
- [ ] 实现注意力可视化
- [ ] 对比不同解码策略

### 4. 应用扩展
- [ ] 扩展到其他语言对
- [ ] 实现文本摘要任务
- [ ] 添加对话生成任务

## 📧 技术支持

如果在运行过程中遇到问题，请检查：

1. **依赖版本兼容性**
   ```bash
   pip list | grep -E "(torch|numpy|matplotlib)"
   ```

2. **Python版本** (建议3.7+)
   ```bash
   python --version
   ```

3. **CUDA支持** (如果使用GPU)
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## 📄 许可证

本项目采用MIT许可证，详见根目录下的LICENSE文件。

---

🎉 **恭喜完成阶段4学习！**

通过本阶段的学习，你已经：
- ✅ 掌握了Transformer的完整实现
- ✅ 理解了自注意力机制的核心原理  
- ✅ 学会了机器翻译任务的训练流程
- ✅ 能够对比不同模型架构的性能

**下一阶段预告**: 阶段5将学习GPT架构，探索自回归语言模型！