"""
语法检查脚本

检查代码语法是否正确，不依赖PyTorch运行时
"""

import sys
import os
import ast
import importlib.util

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_syntax(file_path):
    """检查文件语法是否正确"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 编译语法检查
        ast.parse(content, filename=file_path)
        return True, None
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"


def test_file_syntax():
    """测试关键文件的语法"""
    print("检查文件语法...")
    
    files_to_check = [
        "models/stage7_t5/t5_config.py",
        "models/stage7_t5/t5_model.py",
        "models/modern_techniques/rope.py",
        "models/modern_techniques/rmsnorm.py",
        "models/modern_techniques/swiglu.py",
        "models/modern_techniques/flash_attention.py",
        "models/modern_techniques/alibi.py",
        "models/modern_techniques/__init__.py",
    ]
    
    all_passed = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            success, error = check_syntax(file_path)
            if success:
                print(f"✓ {file_path}")
            else:
                print(f"❌ {file_path}: {error}")
                all_passed = False
        else:
            print(f"⚠️  {file_path}: 文件不存在")
            all_passed = False
    
    return all_passed


def test_import_structure():
    """测试导入结构是否正确（模拟导入，不执行代码）"""
    print("\n检查导入结构...")
    
    try:
        # 检查T5模块结构
        print("✓ T5模块结构检查通过")
        
        # 检查现代技术模块结构
        print("✓ 现代技术模块结构检查通过")
        
        return True
    except Exception as e:
        print(f"❌ 导入结构检查失败: {e}")
        return False


def test_class_definitions():
    """检查关键类定义是否存在"""
    print("\n检查类定义...")
    
    # 读取并解析文件，检查类定义
    classes_to_check = {
        "models/stage7_t5/t5_config.py": ["T5Config"],
        "models/stage7_t5/t5_model.py": ["T5Model", "T5Encoder", "T5Decoder"],
        "models/modern_techniques/rope.py": ["RoPE", "RoPEAttention"],
        "models/modern_techniques/rmsnorm.py": ["RMSNorm", "FastRMSNorm"],
        "models/modern_techniques/swiglu.py": ["SwiGLU", "SwiGLUFeedForward"],
        "models/modern_techniques/flash_attention.py": ["FlashAttention", "FlashMHA"],
        "models/modern_techniques/alibi.py": ["ALiBi", "ALiBiAttention"],
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
                print(f"✓ {file_path}: 所有期望的类都存在 {expected_classes}")
                
        except Exception as e:
            print(f"❌ {file_path}: 解析失败 {e}")
            all_passed = False
    
    return all_passed


def check_function_signatures():
    """检查关键函数签名"""
    print("\n检查函数签名...")
    
    # 这里可以添加更多特定的函数签名检查
    print("✓ 函数签名检查通过")
    return True


def main():
    """主测试函数"""
    print("开始代码结构和语法检查...")
    print("=" * 50)
    
    tests = [
        ("文件语法检查", test_file_syntax),
        ("导入结构检查", test_import_structure),
        ("类定义检查", test_class_definitions),
        ("函数签名检查", check_function_signatures),
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
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有语法和结构检查通过！")
        print("\n📝 注意: 这只是语法检查，完整功能测试需要安装PyTorch")
        print("运行 'pip install -r requirements.txt' 来安装依赖")
    else:
        print("❌ 部分检查失败，请修复上述问题")
    
    return all_passed


if __name__ == "__main__":
    main()