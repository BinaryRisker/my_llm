#!/usr/bin/env python3
"""
LLM项目快速演示脚本

展示项目的核心功能，包括：
- 配置系统使用
- 模型评估
- GLUE基准测试
- API服务
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from evaluation.evaluation_metrics import EvaluationMetrics
from evaluation.glue_benchmark import GLUEBenchmark


def print_header(title):
    """打印格式化的标题"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")


def print_section(section):
    """打印段落标题"""
    print(f"\n📋 {section}")
    print("-" * 40)


def demo_config_system():
    """演示配置系统"""
    print_header("配置系统演示")
    
    # 加载开发环境配置
    config = get_config(env='development')
    
    print_section("项目基础信息")
    project_info = config.get('project', {})
    print(f"项目名称: {project_info.get('name', '未知')}")
    print(f"版本: {project_info.get('version', '未知')}")
    print(f"描述: {project_info.get('description', '未知')}")
    
    print_section("模型配置")
    bert_config = config.get('model', {}).get('bert', {})
    print(f"BERT隐藏层维度: {bert_config.get('hidden_size', '未配置')}")
    print(f"BERT层数: {bert_config.get('num_layers', '未配置')}")
    print(f"注意力头数: {bert_config.get('num_attention_heads', '未配置')}")
    
    print_section("训练配置")
    training_config = config.get('training', {})
    print(f"批次大小: {training_config.get('batch_size', '未配置')}")
    print(f"学习率: {training_config.get('learning_rate', '未配置')}")
    print(f"设备: {training_config.get('device', '未配置')}")


def demo_evaluation_metrics():
    """演示评估指标"""
    print_header("评估指标演示")
    
    metrics = EvaluationMetrics()
    
    print_section("分类任务评估")
    # 模拟分类结果
    predictions = [1, 0, 1, 1, 0, 1, 0, 0]
    labels = [1, 0, 1, 0, 0, 1, 0, 1]
    
    accuracy = metrics.classification_accuracy(predictions, labels)
    f1_score = metrics.classification_f1_score(predictions, labels)
    precision = metrics.classification_precision(predictions, labels)
    recall = metrics.classification_recall(predictions, labels)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    
    print_section("回归任务评估")
    # 模拟回归结果
    pred_scores = [2.1, 3.5, 1.8, 4.2, 2.9]
    true_scores = [2.0, 3.7, 1.9, 4.0, 3.1]
    
    mse = metrics.regression_mse(pred_scores, true_scores)
    rmse = metrics.regression_rmse(pred_scores, true_scores)
    mae = metrics.regression_mae(pred_scores, true_scores)
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    
    print_section("文本生成评估")
    # 模拟文本生成结果
    predictions = ["the cat is on the mat"]
    references = [["the cat is on the mat"]]
    
    bleu = metrics.text_generation_bleu_score(predictions, references)
    print(f"BLEU分数: {bleu:.4f}")
    
    # 困惑度
    loss = 2.5
    perplexity = metrics.perplexity(loss)
    print(f"困惑度 (loss={loss}): {perplexity:.4f}")


def demo_glue_benchmark():
    """演示GLUE基准测试"""
    print_header("GLUE基准测试演示")
    
    glue = GLUEBenchmark()
    
    print_section("GLUE任务信息")
    tasks = glue.get_task_names()
    print(f"支持的GLUE任务数量: {len(tasks)}")
    print("任务列表:")
    for task in tasks[:5]:  # 显示前5个任务
        task_type = glue.get_task_type(task)
        task_metrics = glue.get_task_metrics(task)
        print(f"  - {task}: {task_type} 任务, 指标: {list(task_metrics.keys())}")
    print(f"  ... 还有 {len(tasks)-5} 个任务")
    
    print_section("模拟GLUE评估结果")
    # 模拟各任务的评估结果
    simulated_results = {
        'CoLA': {'matthews_corrcoef': 0.52},
        'SST-2': {'accuracy': 0.91},
        'MRPC': {'accuracy': 0.83, 'f1_score': 0.87},
        'STS-B': {'pearson_corrcoef': 0.85, 'spearman_corrcoef': 0.84},
        'QQP': {'accuracy': 0.89, 'f1_score': 0.86},
        'MNLI': {'accuracy': 0.84},
        'QNLI': {'accuracy': 0.90},
        'RTE': {'accuracy': 0.68},
        'WNLI': {'accuracy': 0.56}
    }
    
    for task, results in simulated_results.items():
        metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in results.items()])
        print(f"  {task}: {metrics_str}")
    
    # 计算总GLUE分数
    glue_score = glue.compute_glue_score(simulated_results)
    print(f"\n🎯 GLUE总分: {glue_score:.2f}/100")


def demo_sample_data():
    """演示样本数据"""
    print_header("样本数据演示")
    
    print_section("文本分类样本")
    classification_samples = [
        ("I love this movie!", "POSITIVE"),
        ("This is terrible", "NEGATIVE"), 
        ("Great product, highly recommend", "POSITIVE"),
        ("Waste of money", "NEGATIVE"),
        ("Average quality, nothing special", "NEUTRAL")
    ]
    
    for text, label in classification_samples:
        print(f"  '{text}' -> {label}")
    
    print_section("命名实体识别样本")
    ner_samples = [
        ("Apple Inc. is located in Cupertino.", "B-ORG I-ORG O O O B-LOC"),
        ("John works at Microsoft", "B-PER O O B-ORG"),
        ("Visit New York City next week", "O B-LOC I-LOC I-LOC O O")
    ]
    
    for text, labels in ner_samples:
        print(f"  '{text}'")
        print(f"    -> {labels}")
    
    print_section("问答样本")
    qa_samples = [
        {
            "context": "The Eiffel Tower is located in Paris, France.",
            "question": "Where is the Eiffel Tower?",
            "answer": "Paris, France"
        },
        {
            "context": "Machine learning is a subset of artificial intelligence.",
            "question": "What is machine learning?", 
            "answer": "a subset of artificial intelligence"
        }
    ]
    
    for sample in qa_samples:
        print(f"  问题: {sample['question']}")
        print(f"  上下文: {sample['context']}")
        print(f"  答案: {sample['answer']}")
        print()


def demo_api_usage():
    """演示API使用方法"""
    print_header("API使用演示")
    
    print_section("启动API服务")
    print("启动命令:")
    print("  python api/main.py")
    print("\nAPI文档地址:")
    print("  http://localhost:8000/docs")
    print("\nAPI健康检查:")
    print("  GET http://localhost:8000/health")
    
    print_section("主要API端点")
    endpoints = [
        ("GET /", "API根端点，返回欢迎信息"),
        ("GET /models", "列出可用的模型"),
        ("GET /glue/tasks", "获取GLUE任务列表"),
        ("POST /evaluate/classification", "评估分类任务"),
        ("POST /evaluate/regression", "评估回归任务"),
        ("POST /predict/text-classification", "文本分类预测"),
        ("POST /predict/text-generation", "文本生成预测"),
        ("GET /stats", "获取API统计信息")
    ]
    
    for endpoint, description in endpoints:
        print(f"  {endpoint:35} - {description}")
    
    print_section("使用示例")
    print("Python客户端示例:")
    print("""
import requests

# 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 文本分类预测
data = {"text": "This is a great product!"}
response = requests.post("http://localhost:8000/predict/text-classification", json=data)
print(response.json())
    """)


def main():
    """主演示函数"""
    print("🎉 欢迎使用LLM从零实现项目！")
    print("这是一个完整的大语言模型开发和训练平台的演示")
    
    try:
        # 配置系统演示
        demo_config_system()
        
        # 评估指标演示
        demo_evaluation_metrics()
        
        # GLUE基准测试演示  
        demo_glue_benchmark()
        
        # 样本数据演示
        demo_sample_data()
        
        # API使用演示
        demo_api_usage()
        
        print_header("演示完成")
        print("🎯 项目特点:")
        print("  ✅ 6个主要阶段全部完成")
        print("  ✅ 24个核心模块")
        print("  ✅ 完整的评估体系")
        print("  ✅ GLUE基准测试支持")
        print("  ✅ 分布式训练支持")
        print("  ✅ Web界面和API接口")
        print("  ✅ 工业级代码质量")
        
        print("\n🚀 下一步:")
        print("  1. 运行 'python test_bert.py' 测试BERT模型")
        print("  2. 启动 'python web_interface/gradio_demo.py' 查看Web界面")
        print("  3. 启动 'python api/main.py' 使用API服务")
        print("  4. 查看 README.md 了解更多信息")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查项目依赖是否正确安装")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)