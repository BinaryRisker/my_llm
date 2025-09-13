# 阶段2：循环神经网络 (RNN/LSTM) 序列建模与文本生成

## 📚 学习目标

在这个阶段，您将学会：
1. 理解循环神经网络的工作原理
2. 掌握序列数据的处理方法
3. 了解梯度消失和爆炸问题
4. 实现LSTM解决长期依赖问题
5. 构建文本生成模型

## 🔄 循环神经网络基础

### 1. 为什么需要RNN？

传统的前馈神经网络（如MLP）无法处理序列数据的时序依赖关系：
- 输入长度固定，无法处理变长序列
- 缺乏"记忆"能力，无法利用历史信息
- 参数不共享，无法学到位置无关的模式

### 2. RNN基本结构

RNN通过引入循环连接来处理序列数据：

```
    h₀     h₁     h₂     h₃
     ↓      ↓      ↓      ↓
[RNN] → [RNN] → [RNN] → [RNN]
     ↑      ↑      ↑      ↑
    x₁     x₂     x₃     x₄
```

#### 数学表示

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

其中：
- `h_t`：时刻t的隐藏状态
- `x_t`：时刻t的输入
- `y_t`：时刻t的输出
- `W_hh`, `W_xh`, `W_hy`：权重矩阵
- `b_h`, `b_y`：偏置向量

### 3. RNN的三种模式

1. **one-to-many**：一个输入，多个输出（如图像描述生成）
2. **many-to-one**：多个输入，一个输出（如情感分类）
3. **many-to-many**：多个输入，多个输出（如机器翻译）

## ⚠️ 梯度消失和爆炸问题

### 梯度消失问题

在反向传播过程中，梯度会随着时间步数的增加而指数性衰减：

```
∂L/∂h_t = (∂L/∂h_{t+1}) * W_hh * tanh'(h_t)
```

当`|W_hh| < 1`且`tanh'() < 1`时，梯度会越来越小，导致：
- 长期依赖信息丢失
- 网络无法学习长距离关系
- 训练效率低下

### 梯度爆炸问题

当`|W_hh| > 1`时，梯度会指数性增长，导致：
- 权重更新过大
- 训练不稳定
- 数值溢出

#### 解决方案

1. **梯度裁剪**：限制梯度的最大范数
2. **合适的激活函数**：如ReLU
3. **权重初始化**：Xavier或He初始化
4. **使用LSTM/GRU**：专门设计的门控机制

## 🚪 长短期记忆网络 (LSTM)

### LSTM设计理念

LSTM通过门控机制解决梯度消失问题：
- **遗忘门**：决定丢弃哪些信息
- **输入门**：决定存储哪些新信息
- **输出门**：决定输出哪些信息
- **记忆细胞**：长期记忆存储

### LSTM数学公式

```
# 遗忘门
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

# 输入门
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

# 更新细胞状态
C_t = f_t * C_{t-1} + i_t * C̃_t

# 输出门
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

其中：
- `σ`：Sigmoid函数，输出[0,1]
- `*`：逐元素乘法
- `C_t`：细胞状态
- `h_t`：隐藏状态

### LSTM门控机制详解

#### 1. 遗忘门 (Forget Gate)
决定从细胞状态中丢弃什么信息：
- 输出0：完全遗忘
- 输出1：完全保留
- 输出0.5：部分保留

#### 2. 输入门 (Input Gate)
分两步决定存储新信息：
- `i_t`：决定更新哪些值
- `C̃_t`：创建新的候选值

#### 3. 细胞状态更新
结合遗忘和输入：
- `f_t * C_{t-1}`：保留的旧信息
- `i_t * C̃_t`：添加的新信息

#### 4. 输出门 (Output Gate)
决定输出细胞状态的哪部分：
- 基于细胞状态输出隐藏状态
- 控制信息流向下一层

## 📊 双向LSTM (BiLSTM)

### 原理

双向LSTM同时处理正向和反向序列：

```
→ LSTM → → LSTM → → LSTM →
x₁        x₂        x₃

← LSTM ← ← LSTM ← ← LSTM ←
```

### 优势

1. **完整上下文**：利用过去和未来信息
2. **更好的表示**：丰富的特征表示
3. **适用场景**：句子分类、命名实体识别等

### 计算过程

```
# 正向LSTM
h_f_t = LSTM_forward(x_t, h_f_{t-1})

# 反向LSTM  
h_b_t = LSTM_backward(x_t, h_b_{t+1})

# 拼接输出
h_t = [h_f_t; h_b_t]
```

## 📝 文本生成

### 1. 语言模型基础

语言模型估计文本序列的概率：

```
P(w₁, w₂, ..., wₙ) = ∏ P(wᵢ | w₁, ..., wᵢ₋₁)
```

### 2. 基于RNN的语言模型

```
# 在时刻t，预测下一个词
P(w_{t+1} | w₁, ..., w_t) = softmax(W * h_t + b)
```

### 3. 训练策略

#### Teacher Forcing
训练时使用真实的前一个词作为输入：
- 优点：训练稳定，收敛快
- 缺点：训练和推理不一致

#### 自由运行
训练时使用模型自己的输出：
- 优点：训练推理一致
- 缺点：训练不稳定，误差累积

#### 混合策略
按概率选择使用真实词或预测词：
```python
if random() < teacher_forcing_ratio:
    input_token = target_token  # 使用真实词
else:
    input_token = predicted_token  # 使用预测词
```

### 4. 文本采样策略

#### 贪心解码
始终选择概率最高的词：
```python
next_word = argmax(probabilities)
```

#### 随机采样
按概率分布随机采样：
```python
next_word = sample(probabilities)
```

#### 温度采样
调节概率分布的"尖锐程度"：
```python
probabilities = softmax(logits / temperature)
```
- 温度 < 1：更确定的选择
- 温度 > 1：更随机的选择

#### Top-k采样
只考虑概率最高的k个词：
```python
top_k_probs, top_k_indices = torch.topk(probabilities, k)
next_word = sample(top_k_probs)
```

## 🎯 实现要点

### 1. 序列填充和掩码

由于批处理需要固定长度，需要进行填充：
```python
# 填充到最大长度
sequences = pad_sequences(texts, maxlen=max_length, padding='post')

# 创建掩码忽略填充位置
mask = sequences != PAD_TOKEN
```

### 2. 梯度裁剪

防止梯度爆炸：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

### 3. 学习率调度

使用学习率衰减：
```python
scheduler = torch.optim.lr_scheduler.ExponentialDecay(optimizer, gamma=0.95)
```

### 4. 正则化技术

#### Dropout
在每个时间步应用：
```python
h_t = dropout(h_t, training=self.training)
```

#### Layer Normalization
稳定训练：
```python
h_t = layer_norm(h_t)
```

## 📈 评估指标

### 1. 困惑度 (Perplexity)

衡量语言模型的质量：
```
PPL = exp(-1/N * Σ log P(w_i))
```
- 越小越好
- 表示模型的"困惑程度"

### 2. BLEU分数

评估生成文本质量：
- 基于n-gram匹配
- 考虑精确率和长度惩罚

### 3. 人工评估

- 流畅性：语法是否正确
- 相关性：内容是否合理
- 多样性：生成是否多样化

## 🔧 优化技巧

### 1. 批处理优化

```python
# 按长度排序减少填充
sorted_indices = sorted(range(len(texts)), key=lambda i: len(texts[i]))

# 动态批处理
def collate_fn(batch):
    max_len = max(len(seq) for seq in batch)
    return pad_sequences(batch, max_len)
```

### 2. 内存优化

```python
# 梯度检查点
torch.utils.checkpoint.checkpoint(layer, input)

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler
```

### 3. 并行化

```python
# 多GPU训练
model = nn.DataParallel(model)

# 分布式训练
torch.distributed.init_process_group()
```

## 🚀 从RNN到Transformer

RNN的局限性导致了Transformer的出现：

### RNN局限性
1. **顺序计算**：无法并行化
2. **长期依赖**：仍存在梯度消失
3. **计算效率**：训练速度慢

### Transformer优势
1. **并行计算**：自注意力机制
2. **长距离依赖**：直接建模关系
3. **计算效率**：更快的训练

## 📊 实验建议

### 1. 超参数调优
- 隐藏维度：128, 256, 512
- 层数：1-4层
- 学习率：1e-3到1e-5
- Dropout率：0.1-0.5

### 2. 架构对比
- RNN vs LSTM vs GRU
- 单向 vs 双向
- 不同层数效果

### 3. 数据分析
- 序列长度分布
- 词汇频率统计
- 生成样本质量

---

## 📚 延伸阅读

1. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
2. [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
3. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
4. [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)