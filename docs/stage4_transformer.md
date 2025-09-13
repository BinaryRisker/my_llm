# Stage 4: Transformer Architecture - 理论与实践指南 🔀

[![Theory](https://img.shields.io/badge/Level-Advanced-red.svg)](https://github.com/your-username/my_llm)
[![Architecture](https://img.shields.io/badge/Architecture-Transformer-blue.svg)](https://arxiv.org/abs/1706.03762)
[![Applications](https://img.shields.io/badge/Applications-NLP-green.svg)](https://github.com/your-username/my_llm)

## 📖 概述

**Transformer** 是由 Vaswani 等人在 2017 年论文 "Attention Is All You Need" 中提出的革命性架构。它完全抛弃了循环神经网络（RNN）和卷积神经网络（CNN），仅使用注意力机制来处理序列数据，成为了现代大型语言模型的基础架构。

### 🎯 本阶段学习目标

- 理解 Self-Attention 和 Multi-Head Attention 机制
- 掌握 Transformer 的完整架构设计
- 实现位置编码（Positional Encoding）
- 理解层归一化（Layer Normalization）和残差连接
- 掌握 Transformer 的训练和推理过程
- 比较 Transformer 与传统 RNN/CNN 的优劣

## 🏗️ Transformer 架构详解

### 整体架构概览

```
输入序列 → Embedding + Position Encoding
    ↓
Encoder Stack (N=6层)
├── Multi-Head Self-Attention
├── Add & Norm
├── Feed Forward Network  
└── Add & Norm
    ↓
Decoder Stack (N=6层)
├── Masked Multi-Head Self-Attention
├── Add & Norm
├── Multi-Head Cross-Attention (与Encoder连接)
├── Add & Norm
├── Feed Forward Network
└── Add & Norm
    ↓
Linear + Softmax → 输出概率分布
```

### 1. Multi-Head Attention 机制

#### 1.1 Self-Attention 基础

Self-Attention 允许序列中的每个位置关注序列中的所有位置，包括自身。

**数学公式：**
```python
Attention(Q, K, V) = softmax(QK^T / √d_k)V

其中：
- Q (Query): 查询矩阵，形状 [seq_len, d_k]
- K (Key): 键矩阵，形状 [seq_len, d_k] 
- V (Value): 值矩阵，形状 [seq_len, d_v]
- d_k: 键维度，用于缩放点积
```

**直观理解：**
- Query: "我想找什么信息？"
- Key: "我有什么信息可以被找到？"
- Value: "找到后，实际要传递的信息"

#### 1.2 Multi-Head Attention

将注意力机制扩展到多个"头"（heads），每个头学习不同的表示子空间。

```python
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h)W^O

其中每个 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

参数矩阵：
- W_i^Q ∈ R^{d_model × d_k}  # Query投影矩阵
- W_i^K ∈ R^{d_model × d_k}  # Key投影矩阵  
- W_i^V ∈ R^{d_model × d_v}  # Value投影矩阵
- W^O ∈ R^{hd_v × d_model}   # 输出投影矩阵
```

**代码实现示例：**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k or d_model // n_heads
        self.d_v = d_v or d_model // n_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * self.d_v, bias=False)
        self.w_o = nn.Linear(n_heads * self.d_v, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. 线性投影并重塑为多头形式
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # 2. 计算注意力
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 3. 连接多头并投影
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_v
        )
        
        return self.w_o(attention_output), attention_weights
```

### 2. 位置编码 (Positional Encoding)

Transformer 没有循环结构，因此需要显式地为序列中的位置添加信息。

#### 2.1 绝对位置编码

**正弦余弦位置编码：**
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
- pos: 序列中的位置 (0, 1, 2, ...)
- i: 维度索引 (0, 1, 2, ..., d_model/2-1)  
- d_model: 模型维度
```

**代码实现：**
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码表
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # 应用正弦和余弦
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        pe = pe.unsqueeze(0)  # 添加批次维度
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

#### 2.2 位置编码的优势

1. **长度无关**: 可以处理训练时未见过的序列长度
2. **周期性**: 相对位置关系被编码在函数中
3. **可学习性**: 模型可以学习如何利用位置信息

### 3. Layer Normalization & 残差连接

#### 3.1 Layer Normalization

与 Batch Normalization 不同，Layer Normalization 在特征维度上进行归一化：

```python
LayerNorm(x) = γ * (x - μ) / σ + β

其中：
- μ = mean(x, dim=-1)     # 在最后一个维度计算均值
- σ = std(x, dim=-1)      # 在最后一个维度计算标准差
- γ, β: 可学习参数
```

**代码实现：**
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

#### 3.2 残差连接 (Residual Connection)

```python
output = LayerNorm(x + Sublayer(x))
```

这种设计有以下优势：
- 梯度流更顺畅，避免梯度消失
- 训练更稳定
- 允许构建更深的网络

### 4. Feed Forward Network (FFN)

每个 Transformer 层包含一个前馈网络：

```python
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

其中：
- W_1 ∈ R^{d_model × d_ff}  # 第一层权重，通常 d_ff = 4 * d_model
- W_2 ∈ R^{d_ff × d_model}  # 第二层权重  
- ReLU 激活函数
```

**代码实现：**
```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
```

## 🎭 Encoder-Decoder 架构详解

### 1. Transformer Encoder

Encoder 负责理解输入序列，每层包含：
1. Multi-Head Self-Attention
2. Add & Norm
3. Feed Forward Network  
4. Add & Norm

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention + 残差连接 + LayerNorm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward + 残差连接 + LayerNorm  
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 2. Transformer Decoder

Decoder 负责生成输出序列，每层包含：
1. Masked Multi-Head Self-Attention (防止看到未来信息)
2. Add & Norm
3. Multi-Head Cross-Attention (连接 Encoder)
4. Add & Norm
5. Feed Forward Network
6. Add & Norm

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)  
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. Cross-Attention with Encoder
        cross_attn_output, attention_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, attention_weights
```

## 🎭 掩码机制详解

### 1. Padding Mask

防止注意力关注到填充位置：

```python
def create_padding_mask(seq, pad_idx=0):
    # seq: [batch_size, seq_len]
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    # 返回: [batch_size, 1, 1, seq_len]
```

### 2. Look-Ahead Mask (因果掩码)

在 Decoder 中防止关注未来位置：

```python
def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # 下三角为 True，上三角为 False
```

### 3. 组合掩码

```python
def create_masks(src, tgt, src_pad_idx=0, tgt_pad_idx=0):
    src_mask = create_padding_mask(src, src_pad_idx)
    
    tgt_padding_mask = create_padding_mask(tgt, tgt_pad_idx)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1))
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask
    
    return src_mask, tgt_mask
```

## ⚡ 训练与推理

### 1. 训练过程（Teacher Forcing）

```python
def train_step(model, src, tgt, criterion, optimizer):
    model.train()
    
    # 创建掩码
    src_mask, tgt_mask = create_masks(src, tgt[:, :-1])
    
    # 前向传播
    output = model(src, tgt[:, :-1], src_mask, tgt_mask)
    
    # 计算损失（预测下一个词）
    loss = criterion(
        output.reshape(-1, output.size(-1)), 
        tgt[:, 1:].reshape(-1)
    )
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()
```

### 2. 推理过程（自回归生成）

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    model.eval()
    
    # 编码源序列
    encoder_output = model.encoder(src, src_mask)
    
    # 初始化目标序列
    tgt = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len - 1):
        tgt_mask = create_look_ahead_mask(tgt.size(1))
        
        # 解码
        output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # 预测下一个词
        next_word = output[:, -1].argmax(dim=-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_word], dim=1)
        
        if next_word == end_symbol:
            break
    
    return tgt
```

## 📊 Transformer 优势分析

### 1. 相比 RNN 的优势

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| **并行化** | 序列处理，无法并行 | 完全并行化 |
| **长距离依赖** | 梯度消失问题 | 直接连接，O(1)复杂度 |
| **训练速度** | 慢 | 快 |
| **内存效率** | 好 | 需要大量内存 |
| **可解释性** | 较差 | 注意力权重可视化 |

### 2. 计算复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|------------|
| **Self-Attention** | O(n²·d) | O(n²) |
| **RNN** | O(n·d²) | O(d) |
| **CNN** | O(k·n·d²) | O(k·d) |

其中：n=序列长度，d=特征维度，k=卷积核大小

## 🔧 实践技巧与优化

### 1. 学习率调度

**Warmup + Cosine Decay：**
```python
def get_lr_scheduler(optimizer, d_model, warmup_steps=4000):
    def lr_lambda(step):
        if step == 0:
            step = 1
        return min(step ** -0.5, step * warmup_steps ** -1.5) * (d_model ** -0.5)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 2. 标签平滑 (Label Smoothing)

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
    
    def forward(self, x, target):
        # 创建平滑标签分布
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        
        return self.criterion(x, true_dist)
```

### 3. 模型并行化

```python
# 数据并行
model = nn.DataParallel(transformer)

# 模型并行（大模型）
class DistributedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 将不同层放在不同GPU上
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config).to(f'cuda:{i//2}') 
            for i in range(config.num_layers)
        ])
```

## 📈 Transformer 变体

### 1. Encoder-Only 模型
- **BERT**: 双向编码器，用于理解任务
- **应用**: 文本分类、问答、命名实体识别

### 2. Decoder-Only 模型  
- **GPT**: 自回归生成模型
- **应用**: 文本生成、对话系统

### 3. Encoder-Decoder 模型
- **T5**: Text-to-Text Transfer Transformer
- **应用**: 机器翻译、文本摘要

## 🚀 实际应用场景

### 1. 机器翻译
```python
# 英法翻译示例
transformer = Transformer(
    src_vocab_size=30000,
    tgt_vocab_size=30000, 
    d_model=512,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)
```

### 2. 文本摘要
```python
# 抽象摘要任务
summarizer = Transformer(
    src_vocab_size=50000,  # 输入文档词汇表
    tgt_vocab_size=50000,  # 摘要词汇表
    d_model=768,
    n_heads=12
)
```

### 3. 代码生成
```python  
# 自然语言到代码
code_generator = Transformer(
    src_vocab_size=10000,  # 自然语言
    tgt_vocab_size=5000,   # 代码tokens
    d_model=512,
    n_heads=8
)
```

## 🎯 训练建议

### 1. 超参数设置
```python
# 推荐配置
config = {
    'd_model': 512,
    'n_heads': 8,
    'd_ff': 2048,
    'num_layers': 6,
    'dropout': 0.1,
    'warmup_steps': 4000,
    'label_smoothing': 0.1
}
```

### 2. 训练策略
- **梯度累积**: 模拟大批次训练
- **混合精度**: 使用 FP16 加速训练
- **检查点保存**: 定期保存最佳模型
- **早停**: 防止过拟合

### 3. 数据预处理
```python
# 数据增强技巧
def augment_data(src, tgt):
    # 1. 随机删除词汇
    # 2. 同义词替换  
    # 3. 回译增强
    # 4. 噪声注入
    return augmented_src, augmented_tgt
```

## 🔍 调试与分析

### 1. 注意力权重可视化
```python
def visualize_attention(model, src, tgt, layer=0, head=0):
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(src, tgt, return_attention=True)
        
    # 提取特定层和头的注意力权重
    attn = attention_weights[layer][0, head].cpu().numpy()
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, cmap='Blues', cbar=True)
    plt.title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.show()
```

### 2. 梯度分析
```python
def analyze_gradients(model):
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            print(f'{name}: {param_norm:.4f}')
    
    total_norm = total_norm ** (1. / 2)
    print(f'Total gradient norm: {total_norm:.4f}')
```

## 📚 进一步学习

### 推荐论文阅读顺序
1. **Attention Is All You Need** (Vaswani et al., 2017) - 原始论文
2. **BERT** (Devlin et al., 2018) - Encoder-only 架构
3. **GPT** (Radford et al., 2018) - Decoder-only 架构  
4. **T5** (Raffel et al., 2019) - 统一 Text-to-Text 框架
5. **Vision Transformer** (Dosovitskiy et al., 2020) - 视觉领域应用

### 代码实现参考
- **Annotated Transformer**: Harvard NLP 详细注释版本
- **Transformers Library**: Hugging Face 工业级实现
- **FairSeq**: Facebook 研究版本
- **Tensor2Tensor**: Google 官方实现

## 🎉 小结

Transformer 架构的提出是自然语言处理领域的一个重要里程碑：

### 核心创新点
1. **完全基于注意力**: 抛弃了RNN和CNN
2. **并行化训练**: 大幅提升训练效率
3. **长距离建模**: 有效处理长序列依赖
4. **可扩展性**: 支持大规模模型训练

### 影响和意义
- 催生了BERT、GPT等预训练模型浪潮
- 成为大型语言模型的标准架构
- 推动了多模态、跨领域应用发展
- 为现代AI系统奠定了架构基础

### 下一步学习方向
- 实现完整的Transformer模型
- 尝试不同的注意力机制变体
- 探索位置编码的改进方法
- 学习模型压缩和加速技术

---

**下一阶段**: [Stage 5 - GPT架构与预训练](stage5_gpt.md)

通过系统学习Transformer，您将掌握现代深度学习最重要的架构之一，为理解和开发大型语言模型打下坚实基础！