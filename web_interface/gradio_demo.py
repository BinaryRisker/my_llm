"""
Gradio演示界面
=============

使用Gradio创建交互式Web界面，展示项目的各种功能：
- 文本生成和分析
- 模型对比
- 分词器演示
- 数据预处理展示
- 实验结果可视化

使用方法:
    python web_interface/gradio_demo.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# 导入项目模块
try:
    from utils.tokenizers.bpe_tokenizer import BPETokenizer
    from utils.data_processing.preprocessing_pipeline import DataPreprocessor
    from utils.experiment_tracking.mlflow_tracker import MLflowTracker
    from utils.hyperparameter_optimization.grid_search import GridSearchOptimizer
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Could not import project modules: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储模型和数据
global_state = {
    'bpe_tokenizer': None,
    'preprocessor': None,
    'sample_data': [
        "Hello world! This is a great example of natural language processing.",
        "这是一个中文示例文本，展示多语言处理能力。",
        "Contact us at example@test.com or visit https://example.com for more info.",
        "<p>HTML content with <b>bold</b> tags should be cleaned.</p>",
        "Very very very long repeated characters!!!!!!!",
        "Short text",
        "Another example with    extra    whitespace    everywhere.",
        "日本語のテキストサンプルです。自然言語処理のデモ。",
    ]
}


def check_dependencies():
    """检查依赖项是否可用"""
    missing_deps = []
    
    if not GRADIO_AVAILABLE:
        missing_deps.append("gradio")
    if not PANDAS_AVAILABLE:
        missing_deps.append("pandas") 
    if not NUMPY_AVAILABLE:
        missing_deps.append("numpy")
    if not MODULES_AVAILABLE:
        missing_deps.append("project modules")
    
    return missing_deps


def initialize_components():
    """初始化组件"""
    if not MODULES_AVAILABLE:
        return False
    
    try:
        # 初始化BPE分词器
        global_state['bpe_tokenizer'] = BPETokenizer(vocab_size=100, min_frequency=1)
        
        # 初始化数据预处理器
        global_state['preprocessor'] = DataPreprocessor()
        global_state['preprocessor'].setup_default_pipeline(
            clean_text=True,
            detect_language=True,
            filter_by_length=True,
            min_length=2,
            max_length=100
        )
        
        logger.info("组件初始化完成")
        return True
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        return False


def train_bpe_tokenizer(texts: str, vocab_size: int) -> Tuple[str, str]:
    """训练BPE分词器"""
    try:
        if not texts.strip():
            return "❌ 请输入训练文本", ""
        
        # 解析输入文本
        text_list = [line.strip() for line in texts.split('\n') if line.strip()]
        
        if len(text_list) == 0:
            return "❌ 没有有效的训练文本", ""
        
        # 创建和训练分词器
        tokenizer = BPETokenizer(vocab_size=max(50, min(vocab_size, 1000)), min_frequency=1)
        tokenizer.train(text_list)
        
        # 更新全局状态
        global_state['bpe_tokenizer'] = tokenizer
        
        # 生成报告
        report = f"""✅ BPE分词器训练完成！

📊 训练统计:
- 训练文本数量: {len(text_list)}
- 目标词汇表大小: {vocab_size}
- 实际词汇表大小: {tokenizer.get_vocab_size()}

📖 词汇表示例 (前20个):"""
        
        vocab = tokenizer.get_vocab()
        for i, (token, id_) in enumerate(list(vocab.items())[:20]):
            report += f"\n  {id_:3d}: '{token}'"
        
        return report, "分词器训练完成，可以在下方测试编码功能"
        
    except Exception as e:
        return f"❌ 训练失败: {str(e)}", ""


def test_bpe_encoding(text: str) -> str:
    """测试BPE编码"""
    try:
        if not global_state['bpe_tokenizer']:
            return "❌ 请先训练BPE分词器"
        
        if not text.strip():
            return "❌ 请输入要编码的文本"
        
        tokenizer = global_state['bpe_tokenizer']
        
        # 编码
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        # 解码验证
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # 生成结果
        result = f"""🔤 BPE编码结果:

📝 原文: {text}
🔢 Token IDs: {token_ids}
📄 解码结果: {decoded_text}

📊 统计信息:
- 原文长度: {len(text)} 字符
- Token数量: {len(token_ids)}
- 压缩比: {len(token_ids) / len(text.split()) if text.split() else 0:.2f} tokens/word"""
        
        return result
        
    except Exception as e:
        return f"❌ 编码失败: {str(e)}"


def preprocess_data(texts: str, clean_text: bool, detect_language: bool, 
                   filter_length: bool, min_length: int, max_length: int) -> Tuple[str, str]:
    """数据预处理演示"""
    try:
        if not texts.strip():
            return "❌ 请输入要处理的文本", ""
        
        # 解析输入
        text_list = [line.strip() for line in texts.split('\n') if line.strip()]
        
        if not text_list:
            return "❌ 没有有效的文本数据", ""
        
        # 设置预处理管道
        preprocessor = DataPreprocessor()
        preprocessor.clear_steps()
        
        if clean_text:
            preprocessor.add_step('clean_text')
        if detect_language:
            preprocessor.add_step('detect_language')
        if filter_length:
            preprocessor.add_step('filter_by_length', 
                                min_length=min_length, max_length=max_length)
        preprocessor.add_step('compute_stats')
        
        # 执行预处理
        processed_data, stats = preprocessor.process(text_list, return_stats=True)
        
        # 生成报告
        report = f"""📊 数据预处理结果:

📈 处理统计:
- 原始样本数: {stats['original_count']}
- 最终样本数: {stats['final_count']}
- 过滤率: {(1 - stats['final_count'] / stats['original_count']) * 100:.1f}%

📋 处理步骤详情:"""
        
        for step_name, step_stats in stats['step_results'].items():
            report += f"\n\n🔸 {step_name}:"
            if isinstance(step_stats, dict):
                for key, value in step_stats.items():
                    if isinstance(value, dict) and len(value) <= 10:
                        report += f"\n  {key}: {value}"
                    else:
                        report += f"\n  {key}: {value}"
        
        # 显示处理后的数据
        processed_display = ""
        for i, text in enumerate(processed_data[:10]):
            display_text = text if isinstance(text, str) else text.get('text', str(text))
            processed_display += f"{i+1}. {display_text}\n"
        
        if len(processed_data) > 10:
            processed_display += f"... (共 {len(processed_data)} 条)"
        
        return report, processed_display
        
    except Exception as e:
        return f"❌ 预处理失败: {str(e)}", ""


def compare_models() -> str:
    """模型对比演示"""
    try:
        # 模拟模型对比数据
        models = ['MLP', 'LSTM', 'Attention', 'Transformer', 'GPT']
        metrics = {
            'BLEU Score': [15.2, 23.4, 28.7, 35.2, 38.9],
            'Perplexity': [85.4, 42.1, 28.3, 18.7, 12.4],
            'Training Time (hours)': [0.5, 2.1, 4.5, 8.2, 15.6]
        }
        
        # 创建对比表格
        report = "🏆 模型性能对比\n\n"
        report += "| 模型 | BLEU Score | Perplexity | Training Time (h) |\n"
        report += "|------|------------|------------|-------------------|\n"
        
        for i, model in enumerate(models):
            report += f"| {model} | {metrics['BLEU Score'][i]:.1f} | {metrics['Perplexity'][i]:.1f} | {metrics['Training Time (hours)'][i]:.1f} |\n"
        
        report += "\n📈 性能趋势:\n"
        report += "- BLEU Score: 随着模型复杂度增加而提升\n"
        report += "- Perplexity: 呈现下降趋势，模型表现越来越好\n"
        report += "- Training Time: 复杂模型需要更长训练时间\n"
        
        report += "\n🎯 推荐建议:\n"
        report += "- 快速原型: 选择MLP或LSTM\n"
        report += "- 平衡性能: 选择Attention或Transformer\n"
        report += "- 最佳效果: 选择GPT (如有充足资源)"
        
        return report
        
    except Exception as e:
        return f"❌ 对比生成失败: {str(e)}"


def hyperparameter_optimization_demo(param_ranges: str) -> str:
    """超参数优化演示"""
    try:
        if not param_ranges.strip():
            return "❌ 请输入参数范围，格式：learning_rate=[0.001,0.01,0.1]"
        
        # 解析参数范围
        param_grid = {}
        try:
            for line in param_ranges.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    param_grid[key.strip()] = eval(value.strip())
        except:
            return "❌ 参数格式错误，请使用正确的Python列表格式"
        
        if not param_grid:
            return "❌ 没有有效的参数定义"
        
        # 模拟优化过程
        def mock_objective(params):
            import random
            # 模拟一些参数的影响
            score = 0.8
            for key, value in params.items():
                if 'learning_rate' in key:
                    score -= abs(value - 0.01) * 10
                elif 'batch_size' in key:
                    score -= abs(value - 64) * 0.001
            
            score += random.uniform(-0.1, 0.1)
            return max(0.1, score)
        
        # 创建优化器
        optimizer = GridSearchOptimizer(n_jobs=1, verbose=False)
        
        # 运行搜索
        result = optimizer.search(
            param_grid=param_grid,
            objective_function=mock_objective,
            maximize=True
        )
        
        # 生成报告
        report = f"""🔍 超参数优化结果:

🏆 最佳配置:
"""
        for key, value in result.best_params.items():
            report += f"  {key}: {value}\n"
        
        report += f"""
📊 优化统计:
- 最佳得分: {result.best_score:.4f}
- 总试验次数: {result.total_trials}
- 成功试验: {result.successful_trials}
- 搜索时间: {result.search_time:.2f}秒

🏅 前3名结果:"""
        
        top_results = optimizer.get_top_results(n=3)
        for i, trial in enumerate(top_results):
            report += f"\n  {i+1}. 得分: {trial.score:.4f}, 参数: {trial.params}"
        
        return report
        
    except Exception as e:
        return f"❌ 优化失败: {str(e)}"


def create_interface():
    """创建Gradio界面"""
    
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        def show_error():
            return f"❌ 缺少依赖包: {', '.join(missing_deps)}\n请运行: pip install {' '.join(missing_deps)}"
        
        with gr.Blocks(title="依赖缺失") as demo:
            gr.Markdown("# ❌ 依赖包缺失")
            gr.Textbox(value=show_error(), interactive=False)
        
        return demo
    
    # 初始化组件
    init_success = initialize_components()
    if not init_success:
        def show_init_error():
            return "❌ 组件初始化失败，请检查项目模块是否正确安装"
        
        with gr.Blocks(title="初始化失败") as demo:
            gr.Markdown("# ❌ 初始化失败")  
            gr.Textbox(value=show_init_error(), interactive=False)
        
        return demo
    
    # 创建主界面
    with gr.Blocks(
        title="LLM学习项目演示",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .gr-button {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            border: none;
            color: white;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🚀 大语言模型学习项目 - 交互式演示
        
        欢迎使用我们的LLM学习项目演示！这里包含了项目的各种核心功能。
        """)
        
        with gr.Tabs():
            
            # Tab 1: BPE分词器
            with gr.TabItem("🔤 BPE分词器"):
                gr.Markdown("## BPE (Byte Pair Encoding) 分词器演示")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1. 训练分词器")
                        train_texts = gr.Textbox(
                            label="训练文本 (每行一个句子)",
                            placeholder="Hello world!\n这是中文示例\nMore training text...",
                            lines=8,
                            value="\n".join(global_state['sample_data'][:6])
                        )
                        vocab_size = gr.Slider(
                            minimum=50, maximum=1000, value=200, step=50,
                            label="词汇表大小"
                        )
                        train_btn = gr.Button("训练BPE分词器", variant="primary")
                    
                    with gr.Column():
                        train_output = gr.Textbox(
                            label="训练结果", 
                            lines=15,
                            max_lines=15
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 2. 测试编码")
                        test_text = gr.Textbox(
                            label="测试文本",
                            placeholder="输入要编码的文本...",
                            value="Hello BPE tokenizer! 这是测试文本。"
                        )
                        encode_btn = gr.Button("编码测试")
                    
                    with gr.Column():
                        encode_output = gr.Textbox(
                            label="编码结果",
                            lines=10
                        )
                
                train_btn.click(
                    fn=train_bpe_tokenizer,
                    inputs=[train_texts, vocab_size],
                    outputs=[train_output, gr.Textbox(visible=False)]
                )
                
                encode_btn.click(
                    fn=test_bpe_encoding,
                    inputs=[test_text],
                    outputs=[encode_output]
                )
            
            # Tab 2: 数据预处理
            with gr.TabItem("🧹 数据预处理"):
                gr.Markdown("## 数据预处理管道演示")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 输入数据")
                        input_texts = gr.Textbox(
                            label="原始文本 (每行一个)",
                            lines=8,
                            value="\n".join(global_state['sample_data'])
                        )
                        
                        gr.Markdown("### 预处理选项")
                        clean_text = gr.Checkbox(label="文本清洗", value=True)
                        detect_language = gr.Checkbox(label="语言检测", value=True)
                        filter_length = gr.Checkbox(label="长度过滤", value=True)
                        
                        with gr.Row():
                            min_length = gr.Number(label="最小长度", value=2)
                            max_length = gr.Number(label="最大长度", value=100)
                        
                        process_btn = gr.Button("开始预处理", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### 处理结果")
                        process_stats = gr.Textbox(
                            label="处理统计",
                            lines=15
                        )
                        
                        process_data = gr.Textbox(
                            label="处理后数据",
                            lines=10
                        )
                
                process_btn.click(
                    fn=preprocess_data,
                    inputs=[input_texts, clean_text, detect_language, 
                           filter_length, min_length, max_length],
                    outputs=[process_stats, process_data]
                )
            
            # Tab 3: 模型对比
            with gr.TabItem("📊 模型对比"):
                gr.Markdown("## 模型性能对比分析")
                
                with gr.Row():
                    with gr.Column():
                        compare_btn = gr.Button("生成模型对比", variant="primary")
                        gr.Markdown("""
                        ### 对比说明
                        - **MLP**: 多层感知机，基础神经网络
                        - **LSTM**: 长短期记忆网络，处理序列数据
                        - **Attention**: 注意力机制，动态权重分配
                        - **Transformer**: 自注意力架构，现代NLP基石
                        - **GPT**: 生成式预训练模型，最先进的架构
                        """)
                    
                    with gr.Column():
                        compare_output = gr.Textbox(
                            label="对比结果",
                            lines=20
                        )
                
                compare_btn.click(
                    fn=compare_models,
                    outputs=[compare_output]
                )
            
            # Tab 4: 超参数优化
            with gr.TabItem("🔍 超参数优化"):
                gr.Markdown("## 超参数优化演示")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 参数搜索空间")
                        param_input = gr.Textbox(
                            label="参数定义 (Python格式)",
                            lines=5,
                            value="""learning_rate=[0.001, 0.01, 0.1]
batch_size=[16, 32, 64]
hidden_size=[128, 256, 512]""",
                            placeholder="param_name=[value1, value2, value3]"
                        )
                        
                        optimize_btn = gr.Button("开始优化搜索", variant="primary")
                        
                        gr.Markdown("""
                        ### 使用说明
                        1. 每行定义一个参数
                        2. 格式：参数名=[值1,值2,值3]
                        3. 支持数字和字符串值
                        4. 系统会自动搜索最佳组合
                        """)
                    
                    with gr.Column():
                        optimize_output = gr.Textbox(
                            label="优化结果",
                            lines=20
                        )
                
                optimize_btn.click(
                    fn=hyperparameter_optimization_demo,
                    inputs=[param_input],
                    outputs=[optimize_output]
                )
            
            # Tab 5: 项目信息
            with gr.TabItem("ℹ️ 项目信息"):
                gr.Markdown("""
                ## 🎓 关于本项目
                
                这是一个完整的大语言模型学习项目，包含从基础神经网络到GPT的完整实现路径。
                
                ### 🏗️ 项目特色
                - 📚 **5个渐进阶段**: MLP → RNN/LSTM → Attention → Transformer → GPT
                - 🔧 **完整工具链**: 分词器、预处理、训练、评估、优化
                - 📊 **实验管理**: MLflow集成、超参数搜索、结果对比
                - 🌐 **Web界面**: Gradio/Streamlit交互式演示
                - 📖 **详细文档**: 理论解释、代码注释、使用指南
                
                ### 🚀 快速开始
                1. **环境配置**: `python scripts/setup_environment.py --full`
                2. **数据下载**: `python scripts/download_datasets.py --all`
                3. **模型训练**: `python training/stage4_transformer/train_transformer.py`
                4. **Web演示**: `python web_interface/gradio_demo.py`
                
                ### 📁 项目结构
                ```
                my_llm/
                ├── models/           # 模型实现
                ├── training/         # 训练脚本
                ├── evaluation/       # 评估工具
                ├── utils/           # 工具模块
                ├── web_interface/   # Web界面
                ├── data/            # 数据集
                └── experiments/     # 实验结果
                ```
                
                ### 🔗 相关资源
                - 📖 [ROADMAP.md](../ROADMAP.md) - 完整开发路线图
                - 📋 [TODO_IMPROVEMENTS.md](../TODO_IMPROVEMENTS.md) - 改进计划
                - 🗂️ [docs/](../docs/) - 理论文档
                
                ### 🤝 参与贡献
                欢迎提交Issues和Pull Requests！
                """)
        
        gr.Markdown("""
        ---
        💡 **提示**: 这是一个演示界面，展示了项目的核心功能。完整功能请参考项目文档。
        """)
    
    return demo


def main():
    """主函数"""
    print("🚀 启动Gradio演示界面...")
    
    if not GRADIO_AVAILABLE:
        print("❌ Gradio未安装，请运行: pip install gradio")
        return
    
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    try:
        demo.launch(
            server_name="0.0.0.0",  # 允许外部访问
            server_port=7860,       # 默认端口
            share=False,            # 不创建公共链接
            debug=True,             # 调试模式
            show_error=True,        # 显示错误
            quiet=False             # 不静默
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 尝试使用不同端口或检查防火墙设置")


if __name__ == "__main__":
    main()