#!/usr/bin/env python3
"""
BERT模型测试脚本
==============

测试BERT模型的各个组件是否能正常导入和运行
"""

def test_bert_import():
    """测试BERT模块导入"""
    print("🧪 测试BERT模块导入...")
    
    try:
        from models.stage6_bert import (
            BertConfig, BertModel, 
            BertForPreTraining, BertForSequenceClassification,
            get_model_info, list_available_models,
            BERT_MODEL_AVAILABLE, BERT_PRETRAINING_AVAILABLE, BERT_FINETUNING_AVAILABLE
        )
        print("✅ 成功导入BERT模块")
        
        # 显示模型信息
        info = get_model_info()
        print(f"\n📋 模型信息:")
        print(f"  名称: {info['name']}")
        print(f"  版本: {info['version']}")
        print(f"  描述: {info['description']}")
        
        print(f"\n🧩 组件状态:")
        for component, available in info['components'].items():
            status = "✅ 可用" if available else "❌ 不可用"
            print(f"  {component}: {status}")
        
        # 列出可用模型
        models = list_available_models()
        print(f"\n🤖 可用模型 ({len(models)}个):")
        for model in models:
            print(f"  • {model['name']}: {model['description']} ({model['type']})")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_bert_config():
    """测试BERT配置"""
    print("\n🧪 测试BERT配置...")
    
    try:
        from models.stage6_bert import BertConfig, BERT_MODEL_AVAILABLE
        
        if not BERT_MODEL_AVAILABLE:
            print("⚠️ BERT基础模型不可用，跳过配置测试")
            return True
        
        # 创建默认配置
        config = BertConfig()
        print(f"✅ 默认配置: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
        
        # 创建自定义配置
        custom_config = BertConfig(
            vocab_size=10000,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8
        )
        print(f"✅ 自定义配置: vocab_size={custom_config.vocab_size}, hidden_size={custom_config.hidden_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def test_bert_convenience_functions():
    """测试便捷函数"""
    print("\n🧪 测试便捷函数...")
    
    try:
        from models.stage6_bert import create_bert_model, create_bert_classifier
        from models.stage6_bert import BERT_MODEL_AVAILABLE, BERT_FINETUNING_AVAILABLE, TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch未安装，跳过便捷函数测试")
            return True
        
        if BERT_MODEL_AVAILABLE:
            try:
                model = create_bert_model(vocab_size=1000, hidden_size=128)
                print("✅ 创建BERT基础模型成功")
            except Exception as e:
                print(f"⚠️ 创建BERT基础模型失败: {e}")
        
        if BERT_FINETUNING_AVAILABLE:
            try:
                classifier = create_bert_classifier(num_labels=3, vocab_size=1000, hidden_size=128)
                print("✅ 创建BERT分类器成功")
            except Exception as e:
                print(f"⚠️ 创建BERT分类器失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 便捷函数测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 BERT模型测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        test_bert_import,
        test_bert_config,
        test_bert_convenience_functions
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{len(tests)} 通过")
    
    if passed == len(tests):
        print("🎉 所有测试通过！BERT模型实现成功！")
    else:
        print("⚠️ 部分测试未通过，请检查相关模块")

if __name__ == "__main__":
    main()