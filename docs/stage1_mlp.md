# 阶段1：多层感知机 (Multi-Layer Perceptron, MLP)

## 📚 学习目标

在这个阶段，您将学会：
1. 理解神经网络的基本组成部分
2. 掌握前向传播和反向传播的数学原理
3. 实现简单的文本分类任务
4. 理解梯度下降优化过程

## 🧠 理论基础

### 1. 感知机 (Perceptron)

感知机是最简单的神经网络模型，由Frank Rosenblatt在1957年提出。它模拟了生物神经元的工作原理。

#### 单个神经元的数学表示

对于输入向量 **x** = [x₁, x₂, ..., xₙ] 和权重向量 **w** = [w₁, w₂, ..., wₙ]，单个神经元的输出为：

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Σ(wᵢxᵢ) + b
y = f(z)
```

其中：
- `z` 是加权和（线性组合）
- `b` 是偏置项 (bias)
- `f()` 是激活函数
- `y` 是最终输出

### 2. 多层感知机 (MLP)

MLP是由多个感知机层级联而成的网络，包含：
- **输入层** (Input Layer)：接收原始数据
- **隐藏层** (Hidden Layers)：进行特征变换
- **输出层** (Output Layer)：产生最终预测

#### 网络结构

```
Input Layer    Hidden Layer 1    Hidden Layer 2    Output Layer
    x₁ ---------> h₁₁ -----------> h₂₁ ------------> y₁
    x₂ ---------> h₁₂ -----------> h₂₂ ------------> y₂
    x₃ ---------> h₁₃ -----------> h₂₃ ------------> y₃
    ...           ...              ...              ...
```

### 3. 激活函数 (Activation Functions)

激活函数引入非线性，使网络能够学习复杂的模式。

#### 常用激活函数

1. **ReLU (Rectified Linear Unit)**
   ```
   f(x) = max(0, x)
   ```
   - 优点：计算简单，缓解梯度消失
   - 缺点：死神经元问题

2. **Sigmoid**
   ```
   f(x) = 1 / (1 + e^(-x))
   ```
   - 优点：输出范围[0,1]，可解释为概率
   - 缺点：梯度消失问题

3. **Tanh**
   ```
   f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   ```
   - 优点：输出范围[-1,1]，零中心化
   - 缺点：仍有梯度消失问题

### 4. 前向传播 (Forward Propagation)

前向传播是数据从输入层流向输出层的过程：

```
# 第一层
z₁ = W₁ᵀx + b₁
h₁ = f(z₁)

# 第二层
z₂ = W₂ᵀh₁ + b₂
h₂ = f(z₂)

# 输出层
z₃ = W₃ᵀh₂ + b₃
ŷ = softmax(z₃)  # 对于分类任务
```

### 5. 损失函数 (Loss Function)

#### 交叉熵损失 (Cross-Entropy Loss)

对于多分类任务，使用交叉熵损失：

```
L(y, ŷ) = -Σ(yᵢ * log(ŷᵢ))
```

其中：
- `y` 是真实标签的one-hot编码
- `ŷ` 是模型预测的概率分布

#### Softmax函数

用于将原始分数转换为概率分布：

```
softmax(zᵢ) = e^zᵢ / Σⱼ(e^zⱼ)
```

### 6. 反向传播 (Backpropagation)

反向传播通过链式法则计算梯度，用于更新网络参数。

#### 链式法则

对于复合函数 f(g(x))，其导数为：
```
∂f(g(x))/∂x = ∂f/∂g * ∂g/∂x
```

#### 梯度计算

从输出层开始，逐层向前计算梯度：

1. **输出层梯度**：
   ```
   ∂L/∂z₃ = ŷ - y  (softmax + cross-entropy的简化形式)
   ```

2. **权重梯度**：
   ```
   ∂L/∂W₃ = h₂ * (∂L/∂z₃)ᵀ
   ∂L/∂b₃ = ∂L/∂z₃
   ```

3. **隐藏层梯度**：
   ```
   ∂L/∂h₂ = W₃ * ∂L/∂z₃
   ∂L/∂z₂ = ∂L/∂h₂ * f'(z₂)
   ```

### 7. 梯度下降优化 (Gradient Descent)

使用计算得到的梯度更新参数：

```
W ← W - η * ∂L/∂W
b ← b - η * ∂L/∂b
```

其中 `η` 是学习率 (learning rate)。

#### 批量梯度下降变体

1. **批量梯度下降** (Batch GD)：使用全部数据
2. **随机梯度下降** (SGD)：每次使用一个样本
3. **小批量梯度下降** (Mini-batch GD)：使用小批量数据

## 🎯 文本分类应用

### 文本预处理

1. **分词** (Tokenization)：将文本分解为词汇
2. **词汇表构建**：创建词汇到索引的映射
3. **向量化**：将文本转换为数值向量
   - Bag of Words (BoW)
   - TF-IDF
   - Word Embeddings

### 模型架构

```
Text Input → Tokenization → Embedding → MLP → Classification
```

## 🔧 实现要点

### 1. 数据处理
- 文本清洗和标准化
- 词汇表构建和索引映射
- 批量数据加载

### 2. 模型实现
- 使用PyTorch的`nn.Linear`层
- 正确的激活函数选择
- 合适的初始化策略

### 3. 训练循环
- 前向传播计算
- 损失函数计算
- 反向传播和参数更新
- 性能监控和记录

## 📊 性能指标

### 分类指标
- **准确率** (Accuracy)：正确分类的比例
- **精确率** (Precision)：预测为正的样本中实际为正的比例
- **召回率** (Recall)：实际为正的样本中被预测为正的比例
- **F1分数**：精确率和召回率的调和平均

### 训练监控
- 训练损失和验证损失
- 训练准确率和验证准确率
- 学习曲线分析

## 🚀 下一步

完成MLP实现后，您将具备：
- 深度学习的基础理解
- PyTorch的基本使用技能
- 文本分类的完整流程

这些知识将为学习更复杂的架构（RNN、LSTM、Transformer）打下坚实基础。

---

## 📚 延伸阅读

1. [Deep Learning Book - Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)
2. [PyTorch官方教程：神经网络](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
3. [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-1/)