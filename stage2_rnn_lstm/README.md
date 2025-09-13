# 阶段2：RNN/LSTM 序列建模与文本生成

## 📋 概述

在阶段2中，我们将深入探索循环神经网络(RNN)和长短期记忆网络(LSTM)，学习如何处理序列数据并实现文本生成任务。通过这个阶段，您将了解：

- RNN的工作原理和梯度消失问题
- LSTM门控机制的设计理念
- 序列到序列的文本生成
- Teacher Forcing训练策略
- 梯度裁剪和正则化技术

## 🏗️ 项目结构

```
stage2_rnn_lstm/
├── models/
│   ├── rnn.py              # RNN模型实现
│   └── lstm.py             # LSTM模型实现
├── utils/
│   └── text_data.py        # 文本数据处理工具
├── data/                   # 数据存储目录
├── train.py                # 训练脚本
├── generate.py             # 文本生成脚本
├── visualize.ipynb         # 可视化分析
└── README.md              # 本文档
```

## 🚀 快速开始

### 1. 基本训练（字符级LSTM）

```bash
cd stage2_rnn_lstm
python train.py --model_type lstm --vocab_type char --epochs 15
```

### 2. 词级RNN训练

```bash
python train.py --model_type rnn --vocab_type word --epochs 20 --seq_length 32
```

### 3. 自定义训练配置

```bash
python train.py \
    --model_type lstm \
    --vocab_type char \
    --epochs 25 \
    --batch_size 64 \
    --seq_length 128 \
    --hidden_size 512 \
    --num_layers 3 \
    --learning_rate 0.001 \
    --dropout 0.2 \
    --max_grad_norm 5.0
```

### 4. 使用自定义文本数据

```bash
python train.py --data_file path/to/your/text.txt --epochs 30
```

## 📊 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_type` | `lstm` | 模型类型：`rnn`, `lstm`, `gru` |
| `--vocab_type` | `char` | 词汇类型：`char` 或 `word` |
| `--epochs` | `20` | 训练轮数 |
| `--batch_size` | `32` | 批次大小 |
| `--seq_length` | `64` | 序列长度 |
| `--learning_rate` | `0.001` | 学习率 |
| `--hidden_size` | `256` | 隐藏状态维度 |
| `--num_layers` | `2` | 网络层数 |
| `--embedding_dim` | `128` | 嵌入维度 |
| `--dropout` | `0.3` | Dropout率 |
| `--max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `--teacher_forcing_ratio` | `1.0` | Teacher forcing比例 |
| `--save_dir` | `./checkpoints` | 模型保存目录 |

## 🔍 模型架构

### 1. SimpleRNN
```
Input → Embedding → RNN Layers → Linear → Output
                     ↓
              Hidden States (循环连接)
```

**特点：**
- 基础的循环神经网络
- 容易产生梯度消失问题
- 适合短序列任务

### 2. SimpleLSTM
```
Input → Embedding → LSTM Layers → Linear → Output
                      ↓
              Memory Cells + Hidden States
                 (门控机制)
```

**LSTM门控：**
- **遗忘门**：决定丢弃哪些信息
- **输入门**：决定存储哪些新信息  
- **输出门**：控制输出的信息
- **记忆细胞**：长期记忆存储

### 3. BiLSTM（双向LSTM）
```
Forward LSTM  → → → →
Input Sequence ↓ ↓ ↓ ↓ → Classification
Backward LSTM ← ← ← ←
```

**适用场景：**
- 文本分类任务
- 需要完整上下文信息的场景

## 📈 性能指标

### 主要指标
- **损失值 (Loss)**：交叉熵损失
- **困惑度 (Perplexity)**：exp(loss)，越小越好
- **生成质量**：人工评估流畅性和连贯性

### 预期性能
使用默认参数在示例数据上的预期结果：

| 模型 | 词汇类型 | 最终困惑度 | 训练时间 |
|------|----------|------------|----------|
| RNN | 字符级 | 15-25 | 5-10分钟 |
| LSTM | 字符级 | 8-15 | 10-15分钟 |
| LSTM | 词级 | 20-35 | 8-12分钟 |

## 🎨 文本生成策略

### 1. 贪心解码
```python
next_token = torch.argmax(probabilities)
```
- 总是选择概率最高的词
- 生成确定但可能重复的文本

### 2. 随机采样
```python
next_token = torch.multinomial(probabilities, 1)
```
- 按概率分布随机采样
- 生成多样但可能不连贯的文本

### 3. 温度采样
```python
probabilities = torch.softmax(logits / temperature, dim=-1)
```
- `temperature < 1`：更保守的选择
- `temperature > 1`：更随机的选择
- `temperature = 1`：标准softmax

### 4. Top-k采样
```python
top_k_probs, top_k_indices = torch.topk(probabilities, k)
```
- 只考虑概率最高的k个词
- 平衡多样性和质量

## 📚 关键概念解析

### 1. 梯度消失问题
**现象：** RNN训练时，误差信号在反向传播过程中逐渐减弱

**原因：**
```
∂L/∂h_t = (∂L/∂h_{t+1}) * W_hh * σ'(h_t)
```
当权重矩阵的谱范数 < 1时，梯度呈指数衰减

**解决方案：**
- 使用LSTM/GRU门控机制
- 梯度裁剪
- 合适的权重初始化

### 2. Teacher Forcing
**训练时：** 使用真实的目标序列作为下一时刻的输入
**推理时：** 使用模型自己的输出作为下一时刻的输入

**优缺点：**
- ✅ 训练稳定，收敛快
- ❌ 训练与推理存在差异
- ❌ 可能导致误差累积

### 3. 梯度裁剪
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```
- 防止梯度爆炸
- 稳定训练过程
- 常用阈值：1.0-5.0

## 🔧 数据处理

### 字符级处理
```python
char_vocab = CharacterVocabulary()
char_vocab.build_vocabulary(texts)
indices = char_vocab.text_to_indices("Hello World!")
```

**优点：**
- 词汇表小，训练快
- 能处理未见过的词
- 适合生成创造性文本

**缺点：**
- 序列长，计算量大
- 难以捕捉词级语义

### 词级处理
```python
word_vocab = WordVocabulary(max_vocab_size=10000)
word_vocab.build_vocabulary(texts)
indices = word_vocab.text_to_indices("Hello World!")
```

**优点：**
- 序列短，效率高
- 保持词汇语义完整
- 更符合自然语言结构

**缺点：**
- 词汇表大，内存消耗多
- 存在OOV（未登录词）问题

## 🎯 实验建议

### 1. 超参数对比实验
```bash
# 比较不同模型类型
python train.py --model_type rnn --epochs 15
python train.py --model_type lstm --epochs 15  
python train.py --model_type gru --epochs 15

# 比较不同隐藏维度
python train.py --hidden_size 128 --epochs 15
python train.py --hidden_size 256 --epochs 15
python train.py --hidden_size 512 --epochs 15
```

### 2. 序列长度影响
```bash
# 短序列
python train.py --seq_length 32 --epochs 20

# 中等序列  
python train.py --seq_length 64 --epochs 20

# 长序列
python train.py --seq_length 128 --epochs 20
```

### 3. 正则化技术
```bash
# 不同dropout率
python train.py --dropout 0.1 --epochs 15
python train.py --dropout 0.3 --epochs 15
python train.py --dropout 0.5 --epochs 15

# 不同梯度裁剪阈值
python train.py --max_grad_norm 0.5 --epochs 15
python train.py --max_grad_norm 1.0 --epochs 15
python train.py --max_grad_norm 5.0 --epochs 15
```

## ⚠️ 常见问题

### Q1: 训练时损失不下降
**可能原因：**
- 学习率过高或过低
- 梯度消失/爆炸
- 序列长度不合适

**解决方案：**
```bash
# 调整学习率
python train.py --learning_rate 0.01
python train.py --learning_rate 0.0001

# 调整梯度裁剪
python train.py --max_grad_norm 0.25

# 使用LSTM替代RNN
python train.py --model_type lstm
```

### Q2: 生成文本质量差
**可能原因：**
- 训练不充分
- 温度参数不合适
- 数据量太少

**解决方案：**
- 增加训练轮数
- 调整生成温度（0.7-1.2）
- 使用更多训练数据
- 尝试不同采样策略

### Q3: 内存不足
**解决方案：**
```bash
# 减少批次大小
python train.py --batch_size 16

# 减少序列长度
python train.py --seq_length 32

# 减少隐藏维度
python train.py --hidden_size 128
```

## 📝 代码示例

### 快速文本生成
```python
import torch
from models.lstm import SimpleLSTM
from utils.text_data import CharacterVocabulary

# 加载模型和词汇表
model = SimpleLSTM(vocab_size=100, hidden_size=256)
model.load_state_dict(torch.load('checkpoints/best_lstm_char.pt')['model_state_dict'])

vocab = CharacterVocabulary()
vocab.load('checkpoints/char_vocabulary.pkl')

# 生成文本
model.eval()
generated = model.generate(
    start_token=vocab.char2idx['T'],
    max_length=200,
    temperature=0.8
)

generated_text = vocab.indices_to_text(generated)
print(generated_text)
```

### 自定义数据训练
```python
from utils.text_data import CharacterVocabulary, create_data_loaders

# 准备你的文本数据
texts = ["Your text data here...", "More text...", ...]

# 构建词汇表
vocab = CharacterVocabulary()
vocab.build_vocabulary(texts)

# 创建数据加载器
train_loader, val_loader = create_data_loaders(
    train_texts, val_texts, vocab, 
    seq_length=64, batch_size=32
)

# 训练模型...
```

## 🔗 下一步

完成这个阶段后，您可以：

1. **进入阶段3**：学习注意力机制和Seq2Seq模型
2. **深度实验**：尝试更复杂的数据集（如整本小说）
3. **模型优化**：实现更高效的训练策略
4. **应用扩展**：尝试其他序列任务（如情感分析、命名实体识别）

---

## 🎉 恭喜！

完成阶段2后，您已经掌握了：
- RNN/LSTM的工作原理和实现
- 序列建模的核心技术
- 文本生成的各种策略
- 处理序列数据的最佳实践

这些知识为理解更高级的模型（如Transformer）奠定了重要基础！