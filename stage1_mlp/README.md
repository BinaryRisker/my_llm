# 阶段1：多层感知机 (MLP) 文本分类

## 📋 概述

这是大模型学习路径的第一个阶段，我们将实现一个基础的多层感知机(MLP)来进行文本分类任务。通过这个阶段，您将学会：

- 神经网络的基本组成部分
- 前向传播和反向传播的实现
- 文本数据的预处理和向量化
- 模型训练和评估的完整流程

## 🏗️ 项目结构

```
stage1_mlp/
├── models/
│   └── mlp.py              # MLP模型实现
├── utils/
│   └── data_utils.py       # 数据处理工具
├── data/                   # 数据存储目录
├── train.py                # 训练脚本
├── inference.py            # 推理脚本
├── visualize.ipynb         # 可视化分析
└── README.md              # 本文档
```

## 🚀 快速开始

### 1. 基本训练

使用默认参数训练嵌入式MLP模型：

```bash
cd stage1_mlp
python train.py
```

### 2. 训练Bag-of-Words模型

```bash
python train.py --model_type bow --epochs 15
```

### 3. 自定义参数训练

```bash
python train.py \
    --model_type embedding \
    --epochs 25 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --hidden_dims 512 256 128 \
    --embedding_dim 256 \
    --dropout 0.3 \
    --save_dir ./my_checkpoints
```

## 📊 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_type` | `embedding` | 模型类型：`embedding` 或 `bow` |
| `--epochs` | `20` | 训练轮数 |
| `--batch_size` | `16` | 批次大小 |
| `--learning_rate` | `0.001` | 学习率 |
| `--hidden_dims` | `[256, 128]` | 隐藏层维度列表 |
| `--embedding_dim` | `128` | 词嵌入维度 |
| `--dropout` | `0.5` | Dropout比例 |
| `--max_length` | `64` | 最大序列长度 |
| `--vocab_size` | `5000` | 词汇表大小 |
| `--save_dir` | `./checkpoints` | 模型保存目录 |

## 🔍 模型架构

### SimpleMLP (嵌入式模型)

```
Input Text → Tokenization → Embedding → Mean Pooling → MLP Layers → Classification
```

**组件说明：**
- **嵌入层**：将词汇索引转换为密集向量
- **池化层**：使用平均池化将变长序列转为定长向量
- **MLP层**：多个全连接层进行特征变换
- **输出层**：生成类别概率分布

### MLPWithBagOfWords

```
Input Text → Tokenization → Bag-of-Words Vector → MLP Layers → Classification
```

**特点：**
- 简单的词袋表示，忽略词序
- 输入维度固定为词汇表大小
- 计算更简单，适合小数据集

## 📈 性能监控

训练过程中会自动生成：

1. **训练曲线图** (`training_history.png`)
   - 损失函数变化
   - 准确率变化

2. **训练摘要** (`training_summary.json`)
   - 最终性能指标
   - 训练历史记录

3. **模型检查点** (`best_*_mlp.pt`)
   - 最佳模型权重
   - 优化器状态

## 🎯 预期性能

使用默认参数在样本数据上的预期性能：

| 模型类型 | 训练准确率 | 验证准确率 | 训练时间 |
|----------|------------|------------|----------|
| Embedding MLP | ~85-95% | ~75-85% | 1-2分钟 |
| Bag-of-Words MLP | ~80-90% | ~70-80% | 30秒-1分钟 |

> **注意**: 由于使用的是简化的演示数据，实际大规模数据集的性能可能有所不同。

## 🔧 使用自己的数据

### 数据格式

创建包含文本和标签的数据：

```python
texts = [
    "这是第一个文本样本",
    "这是第二个文本样本",
    # ... 更多文本
]

labels = [0, 1, 2, 3, ...]  # 对应的类别标签
class_names = ["类别1", "类别2", "类别3", "类别4"]
```

### 修改数据加载

在 `train.py` 中替换 `load_ag_news_sample()` 函数调用：

```python
# 替换这一行
texts, labels, class_names = load_ag_news_sample()

# 为你的数据加载函数
texts, labels, class_names = load_your_data()
```

## 📚 理论知识

详细的理论解释请参考：[../docs/stage1_mlp.md](../docs/stage1_mlp.md)

主要概念包括：
- 感知机和多层感知机
- 激活函数（ReLU、Sigmoid、Tanh）
- 损失函数（交叉熵）
- 反向传播算法
- 梯度下降优化

## 🔬 实验建议

### 1. 超参数调优
- 尝试不同的学习率：`[0.1, 0.01, 0.001, 0.0001]`
- 调整隐藏层大小：`[128, 256, 512]`
- 测试不同的Dropout率：`[0.2, 0.3, 0.5, 0.7]`

### 2. 架构对比
- 比较不同层数的效果
- 对比嵌入式模型和词袋模型的性能
- 测试不同激活函数的影响

### 3. 数据分析
- 观察训练曲线，识别过拟合/欠拟合
- 分析混淆矩阵，了解分类错误模式
- 可视化词嵌入的分布

## ⚠️ 常见问题

### Q1: 训练损失不下降
**可能原因：**
- 学习率过高或过低
- 模型架构不合适
- 数据预处理有问题

**解决方案：**
- 调整学习率（尝试0.01, 0.001, 0.0001）
- 简化或复杂化模型架构
- 检查数据加载和预处理流程

### Q2: 验证准确率远低于训练准确率
**可能原因：**
- 过拟合
- 训练/验证数据分布不一致

**解决方案：**
- 增加Dropout率
- 减少模型复杂度
- 使用更多训练数据
- 添加正则化

### Q3: 内存不足
**解决方案：**
- 减少batch_size
- 降低embedding_dim
- 减小vocab_size
- 使用CPU训练（去掉CUDA）

## 📝 代码示例

### 快速推理

```python
from models.mlp import SimpleMLP
from utils.data_utils import TextVocabulary
import torch

# 加载训练好的模型
checkpoint = torch.load('checkpoints/best_embedding_mlp.pt')
model = SimpleMLP(...)
model.load_state_dict(checkpoint['model_state_dict'])

# 预测新文本
text = "这是一个新的文本样本"
vocab = TextVocabulary()
vocab.load('checkpoints/vocabulary.pkl')

# 预处理
indices = vocab.text_to_indices(text, max_length=64)
input_tensor = torch.tensor([indices])

# 预测
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)

print(f"预测类别: {predicted_class.item()}")
print(f"概率分布: {probabilities.numpy()}")
```

## 🔗 下一步

完成这个阶段后，您可以：

1. **深入分析**: 运行 `visualize.ipynb` 进行详细分析
2. **进入阶段2**: 学习RNN/LSTM处理序列数据
3. **扩展实验**: 尝试更复杂的数据集和任务

---

## 🎉 恭喜！

如果您成功完成了这个阶段，您已经掌握了：
- 神经网络的基本工作原理
- PyTorch的基本使用方法
- 文本分类的完整流程
- 模型训练和评估的最佳实践

这些知识为学习更高级的模型架构奠定了坚实的基础！