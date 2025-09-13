"""
训练优化模块语法测试

测试混合精度训练和内存优化模块的语法和结构
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


def test_mixed_precision_syntax():
    """测试混合精度模块语法"""
    print("检查混合精度训练模块语法...")
    
    files_to_check = [
        "training/mixed_precision/__init__.py",
        "training/mixed_precision/loss_scaler.py",
        "training/mixed_precision/gradient_clipper.py",
        "training/mixed_precision/precision_utils.py",
        "training/mixed_precision/amp_trainer.py"
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


def test_memory_optimization_syntax():
    """测试内存优化模块语法"""
    print("\n检查内存优化模块语法...")
    
    files_to_check = [
        "training/memory_optimization/__init__.py",
        "training/memory_optimization/gradient_accumulation.py"
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
    
    return all_passed


def test_class_definitions():
    """检查关键类定义"""
    print("\n检查类定义...")
    
    classes_to_check = {
        "training/mixed_precision/loss_scaler.py": [
            "LossScaler", "StaticLossScaler", "DynamicLossScaler"
        ],
        "training/mixed_precision/gradient_clipper.py": [
            "GradientClipper", "AdaptiveGradientClipper"
        ],
        "training/mixed_precision/precision_utils.py": [
            "PrecisionManager", "PrecisionConfig", "AutoPrecisionSelector"
        ],
        "training/mixed_precision/amp_trainer.py": [
            "AMPConfig", "AMPTrainer"
        ],
        "training/memory_optimization/gradient_accumulation.py": [
            "AccumulationConfig", "GradientAccumulator"
        ]
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


def test_configuration_structure():
    """测试配置结构"""
    print("\n检查配置结构...")
    
    try:
        # 检查混合精度配置
        from training.mixed_precision import DEFAULT_AMP_CONFIG, SUPPORTED_PRECISIONS
        print(f"✓ 找到默认AMP配置: {len(DEFAULT_AMP_CONFIG)} 项")
        print(f"✓ 支持的精度类型: {SUPPORTED_PRECISIONS}")
        
        # 检查内存优化配置
        from training.memory_optimization import DEFAULT_ACCUMULATION_CONFIG
        print(f"✓ 找到默认累积配置: {len(DEFAULT_ACCUMULATION_CONFIG)} 项")
        
        return True
        
    except ImportError as e:
        print(f"❌ 配置导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始训练优化模块语法和结构检查")
    print("=" * 60)
    
    tests = [
        ("混合精度模块语法检查", test_mixed_precision_syntax),
        ("内存优化模块语法检查", test_memory_optimization_syntax),
        ("类定义检查", test_class_definitions),
        ("配置结构检查", test_configuration_structure)
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
        print("🎉 所有训练优化模块语法和结构检查通过！")
        print("\n📝 实现的功能:")
        print("  • FP16/BF16混合精度训练")
        print("  • 动态损失缩放")
        print("  • 自适应梯度裁剪")
        print("  • 梯度累积")
        print("  • 内存优化")
        print("\n⚠️  注意: 这只是语法检查，完整功能测试需要安装PyTorch等依赖")
    else:
        print("❌ 部分检查失败，请修复上述问题")
    
    return all_passed


if __name__ == "__main__":
    main()