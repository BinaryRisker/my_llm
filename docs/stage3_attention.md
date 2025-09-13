# 阶段3：注意力机制与序列到序列模型 (Seq2Seq)

## 📋 学习目标

在阶段3中，我们将深入探索注意力机制和序列到序列模型，这些技术为后续理解Transformer架构奠定基础。通过本阶段学习，您将掌握：

- 注意力机制的核心思想和计算方法
- Bahdanau注意力与Luong注意力的区别
- Seq2Seq架构的编码器-解码器设计
- 机器翻译任务的实现与评估
- 注意力可视化与解释性分析

## 🧠 理论基础

### 1. 序列到序列模型概述

**传统RNN的局限性：**
- 输入序列被压缩到固定长度的隐藏状态向量
- 长序列容易丢失前期信息（信息瓶颈）
- 无法动态关注输入序列的不同部分

**Seq2Seq模型的优势：**
```
编码器 (Encoder)：
Input: x₁, x₂, ..., xₙ → Hidden States: h₁, h₂, ..., hₙ

解码器 (Decoder)：
Hidden States + Context → Output: y₁, y₂, ..., yₘ
```

### 2. 注意力机制 (Attention Mechanism)

#### 2.1 核心思想

注意力机制允许解码器在生成每个输出词时，动态地"关注"输入序列的不同部分，而不是仅依赖于编码器的最终隐藏状态。

**数学表示：**
```
Context Vector: cᵢ = Σⱼ αᵢⱼ * hⱼ
Attention Weights: αᵢⱼ = softmax(eᵢⱼ)
Attention Energy: eᵢⱼ = f(sᵢ₋₁, hⱼ)
```

其中：
- `sᵢ₋₁`：解码器在时刻i-1的隐藏状态
- `hⱼ`：编码器在时刻j的隐藏状态
- `αᵢⱼ`：注意力权重，表示生成第i个输出时对第j个输入的关注程度
- `cᵢ`：上下文向量，输入信息的加权平均

#### 2.2 Bahdanau注意力 (Additive Attention)

**提出背景：** Bahdanau等人在2014年提出，首次在神经机器翻译中引入注意力机制。

**计算方式：**
```python
# 加性注意力
e_ij = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
alpha_ij = softmax(e_ij)
c_i = sum(alpha_ij * h_j)
```

**特点：**
- 使用前馈神经网络计算注意力得分
- 参数量：`v_a`, `W_a`, `U_a`
- 计算复杂度相对较高，但表达能力强

#### 2.3 Luong注意力 (Multiplicative Attention)

**提出背景：** Luong等人在2015年提出，简化了注意力的计算过程。

**三种计算方式：**

1. **Dot Product（点积）：**
   ```python
   e_ij = s_{i-1}^T * h_j
   ```

2. **General（通用）：**
   ```python
   e_ij = s_{i-1}^T * W_a * h_j
   ```

3. **Concat（连接）：**
   ```python
   e_ij = v_a^T * tanh(W_a * [s_{i-1}; h_j])
   ```

**特点：**
- 计算效率更高
- 参数量相对较少
- 在实践中广泛使用

#### 2.4 注意力对齐 (Attention Alignment)

注意力权重`αᵢⱼ`形成了一个对齐矩阵，可视化地展示输入输出序列之间的对应关系：

```
      Source: "I love machine learning"
Target   I    love  machine  learning
我      0.8   0.1    0.05     0.05
喜欢    0.1   0.7    0.1      0.1  
机器    0.05  0.05   0.8      0.1
学习    0.05  0.05   0.1      0.8
```

### 3. Seq2Seq架构详解

#### 3.1 编码器 (Encoder)

**职责：** 将变长输入序列编码为固定长度的表示

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        output, (hidden, cell) = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, (hidden, cell)
```

**关键技术：**
- **双向LSTM：** 同时捕获前向和后向信息
- **Sequence Packing：** 处理变长序列，提高计算效率
- **多层结构：** 增强表示能力

#### 3.2 解码器 (Decoder)

**职责：** 基于编码器的输出和注意力机制生成目标序列

```python
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size*2, hidden_size, 
                           num_layers, batch_first=True)
        self.attention = attention
        self.output_proj = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        context = self.attention(hidden[0][-1], encoder_outputs)
        lstm_input = torch.cat([embedded, context], dim=-1)
        output, hidden = self.lstm(lstm_input, hidden)
        logits = self.output_proj(output)
        return logits, hidden
```

#### 3.3 Teacher Forcing vs 自回归生成

**训练时（Teacher Forcing）：**
```python
for t in range(target_length):
    output, hidden = decoder(target[:, t:t+1], hidden, encoder_outputs)
    loss += criterion(output, target[:, t+1:t+2])
```

**推理时（自回归）：**
```python
input = start_token
for t in range(max_length):
    output, hidden = decoder(input, hidden, encoder_outputs)
    input = torch.argmax(output, dim=-1)
    if input == end_token:
        break
```

**曝光偏差 (Exposure Bias)：**
训练和推理的差异可能导致错误累积，可通过以下方法缓解：
- **Scheduled Sampling：** 训练时随机使用真实标签或模型预测
- **Curriculum Learning：** 逐步减少teacher forcing比例

### 4. 机器翻译任务

#### 4.1 数据预处理

**分词 (Tokenization)：**
```python
# 英文：基于空格和标点
"Hello, world!" → ["Hello", ",", "world", "!"]

# 法文：处理重音符号和连字符
"C'est très bien." → ["C'", "est", "très", "bien", "."]
```

**词汇表构建：**
```python
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        
    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
```

**序列填充 (Padding)：**
```python
# 批次内序列长度对齐
batch = [
    [1, 2, 3],           # 原始序列
    [4, 5, 6, 7, 8],     # 较长序列
    [9, 10]              # 较短序列
]

padded_batch = [
    [1, 2, 3, 0, 0],     # 填充到最大长度
    [4, 5, 6, 7, 8],
    [9, 10, 0, 0, 0]
]
```

#### 4.2 BLEU评估指标

**BLEU (Bilingual Evaluation Understudy)** 是机器翻译质量的标准评估指标。

**核心思想：** 计算机器翻译与参考翻译之间的n-gram重叠程度

**计算公式：**
```
BLEU = BP × exp(Σᴺₙ₌₁ wₙ × log pₙ)

其中：
- pₙ：n-gram精确度
- wₙ：权重 (通常 w₁=w₂=w₃=w₄=0.25)
- BP：简洁性惩罚 (Brevity Penalty)
```

**简洁性惩罚：**
```python
if len(candidate) > len(reference):
    BP = 1
else:
    BP = exp(1 - len(reference) / len(candidate))
```

**示例计算：**
```python
参考翻译: "The cat is on the mat"
候选翻译: "The cat is on mat"

1-gram: 5/5 = 1.0    (The, cat, is, on, mat)
2-gram: 3/4 = 0.75   (The cat, cat is, is on)
3-gram: 2/3 = 0.67   (The cat is, cat is on)
4-gram: 1/2 = 0.5    (The cat is on)

BP = exp(1 - 6/5) = 0.819
BLEU = 0.819 × exp(0.25×log(1.0) + 0.25×log(0.75) + 0.25×log(0.67) + 0.25×log(0.5))
     ≈ 0.61
```

### 5. 注意力可视化

#### 5.1 注意力矩阵热图

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, source_words, target_words):
    """可视化注意力权重"""
    plt.figure(figsize=(10, 8))
    
    # 创建热图
    sns.heatmap(attention_weights, 
                xticklabels=source_words, 
                yticklabels=target_words,
                cmap='Blues', 
                cbar=True)
    
    plt.xlabel('Source Sequence')
    plt.ylabel('Target Sequence')
    plt.title('Attention Weights Visualization')
    plt.show()
```

#### 5.2 注意力模式分析

**常见的注意力模式：**

1. **单调对齐 (Monotonic Alignment)：**
   - 注意力权重沿对角线分布
   - 适用于语序相近的语言对

2. **倒序对齐 (Inverted Alignment)：**
   - 注意力权重呈反对角线分布
   - 适用于语序相反的语言对

3. **多对一/一对多对齐：**
   - 一个源词对应多个目标词，或反之
   - 处理语言间的结构差异

### 6. 高级技巧与优化

#### 6.1 Coverage机制

**问题：** 标准注意力可能重复关注某些源词，忽略其他重要信息

**解决方案：** 引入覆盖向量 (Coverage Vector)

```python
class CoverageAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.coverage_proj = nn.Linear(1, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, hidden, encoder_outputs, coverage):
        # coverage: 累积的注意力权重
        energy = torch.tanh(
            self.hidden_proj(hidden) +
            self.encoder_proj(encoder_outputs) +
            self.coverage_proj(coverage.unsqueeze(-1))
        )
        energy = torch.matmul(energy, self.v)
        attention = F.softmax(energy, dim=-1)
        
        # 更新覆盖向量
        coverage = coverage + attention
        return attention, coverage
```

#### 6.2 Beam Search解码

**问题：** 贪心解码可能陷入局部最优

**解决方案：** 保持多个候选序列，选择全局最优

```python
def beam_search(model, encoder_outputs, beam_size=5, max_length=50):
    """Beam Search解码"""
    beam = [(0.0, [start_token], initial_hidden)]  # (score, sequence, hidden)
    
    for _ in range(max_length):
        candidates = []
        
        for score, sequence, hidden in beam:
            if sequence[-1] == end_token:
                candidates.append((score, sequence, hidden))
                continue
                
            input_token = sequence[-1]
            logits, new_hidden = model.decoder(input_token, hidden, encoder_outputs)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 扩展候选
            top_tokens = torch.topk(log_probs, beam_size)
            for token_score, token_idx in zip(top_tokens.values, top_tokens.indices):
                new_score = score + token_score.item()
                new_sequence = sequence + [token_idx.item()]
                candidates.append((new_score, new_sequence, new_hidden))
        
        # 选择top-k候选
        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
    
    return beam[0][1]  # 返回最佳序列
```

### 7. 实验设计与评估

#### 7.1 数据集

**推荐数据集：**
- **IWSLT14 (英-德)：** 160K句对，适合快速实验
- **WMT14 (英-法)：** 36M句对，大规模翻译任务
- **Multi30K：** 多语言图像描述，31K句对

#### 7.2 实验配置

**基线模型对比：**
```python
experiments = {
    "Seq2Seq (无注意力)": {
        "encoder": "BiLSTM",
        "decoder": "LSTM", 
        "attention": None
    },
    "Seq2Seq + Bahdanau注意力": {
        "encoder": "BiLSTM",
        "decoder": "LSTM",
        "attention": "Bahdanau"
    },
    "Seq2Seq + Luong注意力": {
        "encoder": "BiLSTM", 
        "decoder": "LSTM",
        "attention": "Luong"
    }
}
```

**评估指标：**
- **BLEU-1, 2, 3, 4：** 不同n-gram的BLEU分数
- **METEOR：** 考虑同义词和词干的评估指标
- **CIDEr：** 基于TF-IDF权重的评估指标
- **人工评估：** 流畅性和准确性评分

#### 7.3 消融研究 (Ablation Study)

**研究问题：**
1. 注意力机制对翻译质量的影响？
2. 不同注意力类型的性能差异？
3. 编码器层数对性能的影响？
4. Teacher forcing比例的影响？

### 8. 当前挑战与发展方向

#### 8.1 局限性

**计算复杂度：**
- 注意力计算的时间复杂度为O(n²)
- 长序列处理效率低下

**对齐质量：**
- 硬注意力vs软注意力的权衡
- 稀疏注意力的探索

#### 8.2 发展趋势

**自注意力 (Self-Attention)：**
- 序列内部的注意力计算
- Transformer的核心组件

**多头注意力 (Multi-Head Attention)：**
- 并行计算多个注意力表示
- 捕获不同类型的依赖关系

**位置编码 (Positional Encoding)：**
- 解决注意力机制缺乏位置信息的问题
- 为Transformer模型奠定基础

## 💡 关键要点总结

1. **注意力机制的本质：** 动态权重分配，解决信息瓶颈问题

2. **Seq2Seq架构：** 编码器-解码器范式，适用于多种序列转换任务

3. **注意力类型：** Bahdanau（加性）vs Luong（乘性），各有优劣

4. **训练策略：** Teacher forcing与曝光偏差的权衡

5. **评估方法：** BLEU等自动指标结合人工评估

6. **可视化分析：** 注意力矩阵揭示模型的对齐模式

7. **技术演进：** 从RNN注意力到Transformer的自注意力

通过掌握注意力机制与Seq2Seq模型，您已经为理解现代深度学习的核心技术——Transformer架构——做好了充分准备！