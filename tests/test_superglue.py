"""
SuperGLUE评估测试脚本

测试SuperGLUE评估框架的基本功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.superglue import SuperGLUEEvaluator, SuperGLUEMetrics, list_all_tasks
from evaluation.superglue.tasks import get_task, convert_examples_to_text2text, extract_predictions
from evaluation.superglue.data_loader import SuperGLUEDataLoader


def test_task_conversion():
    """测试任务数据转换"""
    print("测试任务数据转换...")
    
    # 测试每个任务的转换
    tasks_to_test = ['boolq', 'cb', 'copa', 'rte', 'wic']
    
    for task_name in tasks_to_test:
        print(f"\n测试 {task_name} 任务:")
        
        # 获取任务处理器
        task = get_task(task_name)
        print(f"  任务类型: {task.task_type}")
        
        # 生成模拟数据
        data_loader = SuperGLUEDataLoader()
        mock_data = data_loader._generate_mock_data(task_name, 'validation')
        print(f"  生成了 {len(mock_data)} 个模拟样本")
        
        if mock_data:
            # 转换第一个样本
            example = mock_data[0]
            converted = task.convert_to_text2text(example)
            
            print(f"  输入文本: {converted['input_text'][:100]}...")
            print(f"  目标文本: {converted['target_text']}")
            
            # 测试答案提取
            prediction = task.extract_answer(converted['target_text'])
            print(f"  提取的答案: {prediction}")


def test_metrics_computation():
    """测试指标计算"""
    print("\n测试指标计算...")
    
    # 测试准确率计算
    predictions = [0, 1, 1, 0, 1]
    references = [0, 1, 0, 0, 1]
    
    accuracy = SuperGLUEMetrics.compute_task_metrics('boolq', predictions, references)
    print(f"BoolQ准确率: {accuracy}")
    
    # 测试F1分数计算  
    cb_predictions = [0, 1, 2, 1, 0]
    cb_references = [0, 1, 2, 2, 0]
    
    f1_macro = SuperGLUEMetrics.compute_task_metrics('cb', cb_predictions, cb_references)
    print(f"CB F1分数: {f1_macro}")
    
    # 测试字符串匹配
    record_predictions = ['answer1', 'answer2', 'answer3']
    record_references = ['answer1', 'different', 'answer3']
    
    em_f1 = SuperGLUEMetrics.compute_task_metrics('record', record_predictions, record_references)
    print(f"ReCoRD EM/F1: {em_f1}")


def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    
    data_loader = SuperGLUEDataLoader()
    
    # 测试加载模拟数据
    for task_name in ['boolq', 'cb', 'copa'][:2]:  # 只测试前2个任务
        print(f"\n加载 {task_name} 数据:")
        
        data = data_loader.load_task_data(
            task_name=task_name,
            split='validation',
            data_source='mock',
            max_samples=5
        )
        
        print(f"  加载了 {len(data)} 个样本")
        
        # 验证数据
        is_valid, errors = data_loader.validate_data(task_name, data)
        print(f"  数据验证: {'通过' if is_valid else '失败'}")
        if errors:
            print(f"  错误: {errors}")


def test_evaluator():
    """测试评估器"""
    print("\n测试SuperGLUE评估器...")
    
    # 创建评估器（不使用真实模型）
    evaluator = SuperGLUEEvaluator(
        model=None,  # 使用模拟预测
        tokenizer=None,
        device='cpu',
        max_length=512,
        batch_size=4
    )
    
    print(f"支持的任务: {evaluator.list_supported_tasks()}")
    
    # 测试单个任务评估
    print("\n测试单个任务评估:")
    result = evaluator.evaluate_task('boolq', split='validation', save_predictions=False)
    print(f"BoolQ评估结果: {result}")
    
    # 测试多任务评估（只测试几个任务以节省时间）
    test_tasks = ['boolq', 'cb', 'copa']
    print(f"\n测试多任务评估: {test_tasks}")
    
    summary = evaluator.evaluate_all(
        tasks=test_tasks,
        split='validation', 
        save_predictions=False
    )
    
    print(f"\n总体评估结果:")
    print(f"  总体分数: {summary['overall_score']:.4f}")
    print(f"  完成任务数: {summary['num_tasks_completed']}")
    print(f"  失败任务数: {summary['num_tasks_failed']}")


def test_text2text_conversion():
    """测试Text-to-Text转换"""
    print("\n测试批量Text-to-Text转换...")
    
    # 生成一些测试数据
    test_data = [
        {
            'question': 'Is Python a programming language?',
            'passage': 'Python is a high-level programming language known for its simplicity.',
            'label': 1
        },
        {
            'question': 'Is the sky green?', 
            'passage': 'The sky appears blue during the day due to light scattering.',
            'label': 0
        }
    ]
    
    # 转换为Text-to-Text格式
    converted = convert_examples_to_text2text('boolq', test_data)
    
    print(f"转换了 {len(converted)} 个样本:")
    for i, item in enumerate(converted):
        print(f"  样本 {i}:")
        print(f"    输入: {item['input_text'][:80]}...")
        print(f"    目标: {item['target_text']}")
    
    # 测试答案提取
    generated_texts = ['True', 'False', 'True']
    predictions = extract_predictions('boolq', generated_texts)
    
    print(f"\n答案提取测试:")
    for i, (text, pred) in enumerate(zip(generated_texts, predictions)):
        print(f"  生成文本: '{text}' -> 预测: {pred}")


def main():
    """主测试函数"""
    print("开始SuperGLUE评估框架测试")
    print("=" * 60)
    
    try:
        test_task_conversion()
        test_metrics_computation()
        test_data_loader()
        test_text2text_conversion()
        test_evaluator()
        
        print("\n" + "=" * 60)
        print("🎉 所有SuperGLUE测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()