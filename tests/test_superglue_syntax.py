"""
SuperGLUE模块语法检查

检查SuperGLUE模块的语法和基本结构，不依赖外部库
"""

import sys
import os
import ast
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_file_syntax(file_path):
    """检查文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content, filename=file_path)
        return True, None
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"


def test_superglue_syntax():
    """测试SuperGLUE模块语法"""
    print("检查SuperGLUE模块语法...")
    
    files_to_check = [
        "evaluation/superglue/__init__.py",
        "evaluation/superglue/metrics.py",
        "evaluation/superglue/tasks.py",
        "evaluation/superglue/evaluator.py", 
        "evaluation/superglue/data_loader.py",
        "evaluation/superglue/processor.py"
    ]
    
    all_passed = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            success, error = check_file_syntax(file_path)
            if success:
                print(f"✓ {file_path}")
            else:
                print(f"❌ {file_path}: {error}")
                all_passed = False
        else:
            print(f"⚠️  {file_path}: 文件不存在")
            all_passed = False
    
    return all_passed


def test_class_definitions():
    """检查关键类定义"""
    print("\n检查类定义...")
    
    classes_to_check = {
        "evaluation/superglue/metrics.py": ["SuperGLUEMetrics"],
        "evaluation/superglue/tasks.py": [
            "SuperGLUETask", "BoolQTask", "CBTask", "COPATask",
            "MultiRCTask", "ReCoRDTask", "RTETask", "WiCTask", "WSCTask", "AXbTask"
        ],
        "evaluation/superglue/evaluator.py": ["SuperGLUEEvaluator"],
        "evaluation/superglue/data_loader.py": ["SuperGLUEDataLoader"],
        "evaluation/superglue/processor.py": ["SuperGLUEProcessor"]
    }
    
    all_passed = True
    
    for file_path, expected_classes in classes_to_check.items():
        if not os.path.exists(file_path):
            print(f"⚠️  {file_path}: 文件不存在")
            all_passed = False
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            found_classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    found_classes.append(node.name)
            
            missing_classes = set(expected_classes) - set(found_classes)
            if missing_classes:
                print(f"❌ {file_path}: 缺少类 {missing_classes}")
                all_passed = False
            else:
                print(f"✓ {file_path}: 所有期望的类都存在")
                
        except Exception as e:
            print(f"❌ {file_path}: 解析失败 {e}")
            all_passed = False
    
    return all_passed


def test_function_definitions():
    """检查关键函数定义"""
    print("\n检查函数定义...")
    
    functions_to_check = {
        "evaluation/superglue/metrics.py": [
            "compute_accuracy", "compute_f1_macro", "compute_matthews_correlation", 
            "compute_em_f1", "normalize_answer"
        ],
        "evaluation/superglue/tasks.py": [
            "get_task", "convert_examples_to_text2text", "extract_predictions"
        ]
    }
    
    all_passed = True
    
    for file_path, expected_functions in functions_to_check.items():
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            found_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    found_functions.append(node.name)
            
            missing_functions = set(expected_functions) - set(found_functions)
            if missing_functions:
                print(f"❌ {file_path}: 缺少函数 {missing_functions}")
                all_passed = False
            else:
                print(f"✓ {file_path}: 所有期望的函数都存在")
                
        except Exception as e:
            print(f"❌ {file_path}: 解析失败 {e}")
            all_passed = False
    
    return all_passed


def test_imports_structure():
    """检查导入结构"""
    print("\n检查导入结构...")
    
    try:
        # 检查__init__.py的导入
        init_file = "evaluation/superglue/__init__.py"
        if os.path.exists(init_file):
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            expected_imports = [
                'SUPERGLUE_TASKS', 'get_task_info', 'list_all_tasks'
            ]
            
            for import_item in expected_imports:
                if import_item in content:
                    print(f"✓ 找到导入: {import_item}")
                else:
                    print(f"⚠️  未找到导入: {import_item}")
        
        print("✓ 导入结构检查完成")
        return True
        
    except Exception as e:
        print(f"❌ 导入结构检查失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始SuperGLUE模块语法和结构检查")
    print("=" * 60)
    
    tests = [
        ("文件语法检查", test_superglue_syntax),
        ("类定义检查", test_class_definitions),
        ("函数定义检查", test_function_definitions),
        ("导入结构检查", test_imports_structure)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if not success:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} 失败: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有SuperGLUE语法和结构检查通过！")
        print("\n📝 注意: 这只是语法检查，功能测试需要安装完整依赖")
        print("运行 'pip install -r requirements.txt' 来安装依赖")
    else:
        print("❌ 部分检查失败，请修复上述问题")
    
    return all_passed


if __name__ == "__main__":
    main()