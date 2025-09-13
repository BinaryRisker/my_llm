# 大模型学习进化之路 (Big Model Learning Path)

本项目提供了一个从零到一学习大模型的完整路径，通过5个逐步递进的阶段，帮助您理解和实现现代大语言模型的核心技术。

## 🎯 学习目标

通过本项目，您将学会：
- 从基础神经网络到现代Transformer的演进历程
- 深入理解注意力机制、位置编码、自回归生成等核心概念
- 实现完整的预训练-微调流程
- 掌握大模型训练的工程技巧和最佳实践

## 📚 学习路径

### 📊 阶段1：多层感知机 (MLP)
**目标**：理解神经网络基础，实现文本分类
- 前向传播与反向传播
- 激活函数与损失函数
- 梯度下降优化

### 🔄 阶段2：循环神经网络 (RNN/LSTM)
**目标**：处理序列数据，实现文本生成
- 时序建模与梯度消失问题
- LSTM门控机制
- 字符级/词级文本生成

### 🎯 阶段3：注意力机制 (Attention)
**目标**：理解注意力，实现序列到序列翻译
- Bahdanau & Luong注意力
- Seq2Seq架构
- 对齐与翻译质量

### ⚡ 阶段4：Transformer架构
**目标**：掌握现代架构基础
- 自注意力机制
- 位置编码与多头注意力
- 编码器-解码器结构

### 🚀 阶段5：GPT预训练语言模型
**目标**：实现完整的大语言模型
- 自回归语言建模
- 大规模预训练
- 下游任务微调

## 🗂️ 项目结构

```
my_llm/
├── stage1_mlp/           # 阶段1：多层感知机
├── stage2_rnn_lstm/      # 阶段2：RNN/LSTM
├── stage3_attention/     # 阶段3：注意力机制
├── stage4_transformer/   # 阶段4：Transformer
├── stage5_gpt/          # 阶段5：GPT模型
├── docs/                # 详细文档和教程
├── scripts/             # 公共工具脚本
├── notebooks/           # Jupyter演示笔记
├── requirements.txt     # 项目依赖
└── README.md           # 项目说明
```

## 🛠️ 环境配置

### 系统要求
- Python 3.8+
- PyTorch 2.0+
- CUDA支持（可选，用于GPU加速）

### 安装依赖
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

1. **克隆项目**：
   ```bash
   git clone <this-repo>
   cd my_llm
   ```

2. **安装环境**：
   ```bash
   pip install -r requirements.txt
   ```

3. **按阶段学习**：
   从 `stage1_mlp/` 开始，每个阶段都包含：
   - `README.md` - 阶段说明
   - `docs/` - 理论文档
   - `models/` - 模型实现
   - `train.py` - 训练脚本
   - `inference.py` - 推理演示
   - `visualize.ipynb` - 可视化分析

## 📖 学习建议

1. **按顺序学习**：每个阶段都基于前一阶段的知识
2. **理论结合实践**：先读文档理解原理，再运行代码验证
3. **动手实验**：尝试修改参数，观察效果变化
4. **记录笔记**：记录学习心得和遇到的问题

## 🤝 贡献指南

欢迎提出问题、建议或贡献代码！请参考 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📜 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 👥 致谢

本项目参考了以下优秀资源：
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)

---

🌟 **开始您的大模型学习之旅吧！** 从 `stage1_mlp/` 目录开始探索。