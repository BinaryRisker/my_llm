"""
基本测试框架 - 测试各阶段模型的基本功能
===========================================

这个测试文件验证所有阶段的基本功能，包括：
- 模型创建和初始化
- 前向传播
- 基本训练步骤
- 数据加载

运行测试: python -m pytest tests/test_basic.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBasicFunctionality:
    """基本功能测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device('cpu')  # 测试中使用CPU
    
    def test_project_structure(self):
        """测试项目结构完整性"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 检查主要目录存在
        expected_dirs = [
            'docs', 'scripts', 'tests',
            'stage1_mlp', 'stage2_rnn_lstm', 'stage3_attention_seq2seq',
            'stage4_transformer', 'stage5_gpt'
        ]
        
        for dir_name in expected_dirs:
            dir_path = os.path.join(project_root, dir_name)
            assert os.path.exists(dir_path), f"目录不存在: {dir_name}"
        
        # 检查重要文件存在
        expected_files = [
            'README.md', 'requirements.txt', 'config.yaml', '.gitignore'
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(project_root, file_name)
            assert os.path.exists(file_path), f"文件不存在: {file_name}"
    
    def test_config_loading(self):
        """测试配置文件加载"""
        try:
            import yaml
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, 'config.yaml')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查配置文件的基本结构
            assert 'project' in config
            assert 'training' in config
            assert 'stages' in config
            
            # 检查所有阶段的配置
            expected_stages = ['stage1_mlp', 'stage2_rnn_lstm', 'stage3_attention', 
                             'stage4_transformer', 'stage5_gpt']
            
            for stage in expected_stages:
                assert stage in config['stages'], f"配置中缺少阶段: {stage}"
            
        except ImportError:
            pytest.skip("PyYAML not installed")
    
    def test_stage1_mlp_basic(self):
        """测试阶段1 MLP基本功能"""
        try:
            from stage1_mlp.models.mlp import MLP, MLPConfig
            
            # 创建模型
            config = MLPConfig(
                input_size=784,
                hidden_sizes=[128, 64],
                output_size=10,
                activation='relu',
                dropout=0.1
            )
            
            model = MLP(config)
            
            # 测试前向传播
            batch_size = 16
            x = torch.randn(batch_size, config.input_size)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, config.output_size)
            assert not torch.isnan(output).any()
            
        except ImportError as e:
            pytest.skip(f"Stage 1 MLP模块导入失败: {e}")
    
    def test_stage2_rnn_lstm_basic(self):
        """测试阶段2 RNN/LSTM基本功能"""
        try:
            from stage2_rnn_lstm.models.lstm import LSTMLanguageModel
            
            # 创建LSTM语言模型
            vocab_size = 1000
            embed_size = 128
            hidden_size = 256
            num_layers = 2
            
            model = LSTMLanguageModel(
                vocab_size=vocab_size,
                embed_size=embed_size, 
                hidden_size=hidden_size,
                num_layers=num_layers
            )
            
            # 测试前向传播
            batch_size = 8
            seq_len = 20
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids)
            
            assert output.shape == (batch_size, seq_len, vocab_size)
            assert not torch.isnan(output).any()
            
        except ImportError as e:
            pytest.skip(f"Stage 2 RNN/LSTM模块导入失败: {e}")
    
    def test_stage3_attention_basic(self):
        """测试阶段3 注意力机制基本功能"""
        try:
            from stage3_attention_seq2seq.models.attention import BahdanauAttention
            
            # 创建注意力机制
            encoder_dim = 256
            decoder_dim = 256
            attention_dim = 128
            
            attention = BahdanauAttention(
                encoder_dim=encoder_dim,
                decoder_dim=decoder_dim, 
                attention_dim=attention_dim
            )
            
            # 测试注意力计算
            batch_size = 4
            encoder_seq_len = 10
            decoder_seq_len = 8
            
            encoder_outputs = torch.randn(batch_size, encoder_seq_len, encoder_dim)
            decoder_hidden = torch.randn(batch_size, decoder_dim)
            
            with torch.no_grad():
                context, attention_weights = attention(decoder_hidden, encoder_outputs)
            
            assert context.shape == (batch_size, encoder_dim)
            assert attention_weights.shape == (batch_size, encoder_seq_len)
            assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-6)
            
        except ImportError as e:
            pytest.skip(f"Stage 3 Attention模块导入失败: {e}")
    
    def test_stage4_transformer_basic(self):
        """测试阶段4 Transformer基本功能"""
        try:
            from stage4_transformer.models.multi_head_attention import MultiHeadAttention
            
            # 创建多头注意力
            d_model = 512
            num_heads = 8
            
            mha = MultiHeadAttention(d_model, num_heads)
            
            # 测试多头注意力
            batch_size = 2
            seq_len = 16
            
            x = torch.randn(batch_size, seq_len, d_model)
            
            with torch.no_grad():
                output = mha(x, x, x)
            
            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.isnan(output).any()
            
        except ImportError as e:
            pytest.skip(f"Stage 4 Transformer模块导入失败: {e}")
    
    def test_stage5_gpt_basic(self):
        """测试阶段5 GPT基本功能"""
        try:
            from stage5_gpt.models.gpt_mini import GPTMini, GPTConfig
            
            # 创建GPT-Mini模型
            config = GPTConfig.gpt_mini()
            model = GPTMini(config)
            
            # 测试前向传播
            batch_size = 2
            seq_len = 32
            
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                outputs = model(input_ids)
            
            assert 'logits' in outputs
            logits = outputs['logits']
            assert logits.shape == (batch_size, seq_len, config.vocab_size)
            assert not torch.isnan(logits).any()
            
            # 测试生成功能
            prompt = torch.randint(0, config.vocab_size, (1, 10))
            
            with torch.no_grad():
                generated = model.generate(prompt, max_new_tokens=5, do_sample=False)
            
            assert generated.shape[0] == 1
            assert generated.shape[1] == 15  # 10 + 5
            
        except ImportError as e:
            pytest.skip(f"Stage 5 GPT模块导入失败: {e}")
    
    def test_common_utils(self):
        """测试通用工具函数"""
        try:
            from scripts.train_utils import TrainingConfig, UniversalTrainer
            
            # 测试训练配置
            config = TrainingConfig(
                learning_rate=1e-3,
                batch_size=32,
                num_epochs=10
            )
            
            assert config.learning_rate == 1e-3
            assert config.batch_size == 32
            assert config.num_epochs == 10
            
        except ImportError as e:
            pytest.skip(f"通用工具导入失败: {e}")
    
    def test_data_files_exist(self):
        """测试示例数据文件存在"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 检查新添加的示例数据
        data_files = [
            'stage3_attention_seq2seq/data/sample_en.txt',
            'stage3_attention_seq2seq/data/sample_fr.txt', 
            'stage4_transformer/data/sample_translation_pairs.txt',
            'stage2_rnn_lstm/data/sample_corpus.txt'
        ]
        
        for file_path in data_files:
            full_path = os.path.join(project_root, file_path)
            assert os.path.exists(full_path), f"数据文件不存在: {file_path}"
            
            # 检查文件不为空
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                assert len(content) > 0, f"数据文件为空: {file_path}"


class TestDocumentation:
    """文档完整性测试"""
    
    def test_readme_files(self):
        """测试README文件存在"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        readme_files = [
            'README.md',
            'stage1_mlp/README.md',
            'stage2_rnn_lstm/README.md', 
            'stage3_attention_seq2seq/README.md',
            'stage4_transformer/README.md'
        ]
        
        for readme in readme_files:
            readme_path = os.path.join(project_root, readme)
            assert os.path.exists(readme_path), f"README文件不存在: {readme}"
    
    def test_theory_docs(self):
        """测试理论文档存在"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        theory_docs = [
            'docs/stage2_rnn_lstm.md',
            'docs/stage3_attention.md',
            'docs/stage4_transformer.md',
            'docs/stage5_gpt.md',
            'docs/roadmap.md'
        ]
        
        for doc in theory_docs:
            doc_path = os.path.join(project_root, doc)
            assert os.path.exists(doc_path), f"理论文档不存在: {doc}"
            
            # 检查文档有实质内容
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 1000, f"理论文档内容过少: {doc}"


@pytest.mark.integration
class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_gpt_mini(self):
        """端到端测试GPT-Mini流程"""
        try:
            from stage5_gpt.models.gpt_mini import GPTMini, GPTConfig
            from stage5_gpt.utils.helpers import SimpleTokenizer
            
            # 创建简单分词器
            tokenizer = SimpleTokenizer()
            texts = ["Hello world", "GPT is great", "AI is the future"]
            tokenizer.fit(texts)
            
            # 创建模型
            config = GPTConfig.gpt_mini()
            config.vocab_size = tokenizer.vocab_size
            model = GPTMini(config)
            
            # 测试编码解码
            test_text = "Hello AI"
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            
            assert isinstance(encoded, list)
            assert isinstance(decoded, str)
            
            # 测试模型推理
            input_ids = torch.tensor([encoded], dtype=torch.long)
            
            with torch.no_grad():
                outputs = model(input_ids)
                assert 'logits' in outputs
                
                # 测试生成
                generated = model.generate(input_ids[:, :3], max_new_tokens=5)
                assert generated.shape[1] > input_ids.shape[1]
            
        except ImportError as e:
            pytest.skip(f"GPT集成测试导入失败: {e}")


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "--tb=short"])