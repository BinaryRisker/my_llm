"""
T5模型测试脚本

测试T5模型的基本功能，包括：
- 模型初始化
- 前向传播
- 编码器-解码器架构
- 现代技术集成
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stage7_t5.t5_config import T5Config
from models.stage7_t5.t5_model import T5Model
from models.modern_techniques import RoPE, RMSNorm, SwiGLU, FlashAttention, ALiBi

def test_t5_config():
    """测试T5配置"""
    print("测试T5配置...")
    
    # 测试默认配置
    config = T5Config()
    print(f"默认配置: vocab_size={config.vocab_size}, d_model={config.d_model}")
    
    # 测试预定义配置
    small_config = T5Config.get_config('small')
    print(f"Small配置: vocab_size={small_config.vocab_size}, d_model={small_config.d_model}")
    
    base_config = T5Config.get_config('base')
    print(f"Base配置: vocab_size={base_config.vocab_size}, d_model={base_config.d_model}")
    
    print("T5配置测试通过 ✓")


def test_modern_techniques():
    """测试现代技术组件"""
    print("\n测试现代技术组件...")
    
    batch_size, seq_len, embed_dim, num_heads = 2, 32, 512, 8
    head_dim = embed_dim // num_heads
    
    # 测试RoPE
    print("测试RoPE...")
    rope = RoPE(dim=head_dim, max_seq_length=seq_len)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("RoPE测试通过 ✓")
    
    # 测试RMSNorm
    print("测试RMSNorm...")
    rmsnorm = RMSNorm(embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    x_norm = rmsnorm(x)
    assert x_norm.shape == x.shape
    print("RMSNorm测试通过 ✓")
    
    # 测试SwiGLU
    print("测试SwiGLU...")
    swiglu = SwiGLU(embed_dim * 2)  # 输入维度是输出的2倍
    x_swiglu = torch.randn(batch_size, seq_len, embed_dim * 2)
    x_out = swiglu(x_swiglu)
    assert x_out.shape == (batch_size, seq_len, embed_dim)
    print("SwiGLU测试通过 ✓")
    
    # 测试FlashAttention
    print("测试FlashAttention...")
    flash_attn = FlashAttention(embed_dim, num_heads)
    attn_out, _ = flash_attn(x, x, x)
    assert attn_out.shape == x.shape
    print("FlashAttention测试通过 ✓")
    
    # 测试ALiBi
    print("测试ALiBi...")
    alibi = ALiBi(num_heads)
    bias = alibi(seq_len, device=x.device, dtype=x.dtype)
    assert bias.shape == (1, num_heads, seq_len, seq_len)
    print("ALiBi测试通过 ✓")
    
    print("所有现代技术组件测试通过 ✓")


def test_t5_model_basic():
    """测试T5模型基本功能"""
    print("\n测试T5模型基本功能...")
    
    # 使用小配置进行测试
    config = T5Config.get_config('small')
    config.d_model = 256  # 减小模型尺寸用于测试
    config.d_ff = 512
    config.num_layers = 2
    config.num_heads = 4
    config.vocab_size = 1000
    
    model = T5Model(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试输入
    batch_size = 2
    src_len = 16
    tgt_len = 12
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, src_len))
    decoder_input_ids = torch.randint(0, config.vocab_size, (batch_size, tgt_len))
    
    print(f"输入形状: {input_ids.shape}")
    print(f"解码器输入形状: {decoder_input_ids.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids
        )
    
    print(f"输出形状: {outputs.logits.shape}")
    expected_shape = (batch_size, tgt_len, config.vocab_size)
    assert outputs.logits.shape == expected_shape, f"期望形状 {expected_shape}, 得到 {outputs.logits.shape}"
    
    print("T5模型基本功能测试通过 ✓")


def test_t5_encoder_decoder():
    """测试T5编码器和解码器分离功能"""
    print("\n测试T5编码器-解码器分离功能...")
    
    config = T5Config.get_config('small')
    config.d_model = 128
    config.d_ff = 256
    config.num_layers = 2
    config.num_heads = 4
    config.vocab_size = 500
    
    model = T5Model(config)
    
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, src_len))
    decoder_input_ids = torch.randint(0, config.vocab_size, (batch_size, tgt_len))
    
    with torch.no_grad():
        # 测试只编码器
        encoder_outputs = model.encoder(input_ids)
        print(f"编码器输出形状: {encoder_outputs.last_hidden_state.shape}")
        expected_enc_shape = (batch_size, src_len, config.d_model)
        assert encoder_outputs.last_hidden_state.shape == expected_enc_shape
        
        # 测试解码器（使用编码器输出）
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state
        )
        print(f"解码器输出形状: {decoder_outputs.last_hidden_state.shape}")
        expected_dec_shape = (batch_size, tgt_len, config.d_model)
        assert decoder_outputs.last_hidden_state.shape == expected_dec_shape
        
        # 测试语言模型头
        lm_logits = model.lm_head(decoder_outputs.last_hidden_state)
        print(f"语言模型输出形状: {lm_logits.shape}")
        expected_lm_shape = (batch_size, tgt_len, config.vocab_size)
        assert lm_logits.shape == expected_lm_shape
    
    print("T5编码器-解码器分离功能测试通过 ✓")


def test_t5_generation():
    """测试T5生成功能"""
    print("\n测试T5生成功能...")
    
    config = T5Config.get_config('small')
    config.d_model = 128
    config.d_ff = 256
    config.num_layers = 1
    config.num_heads = 4
    config.vocab_size = 100
    
    model = T5Model(config)
    model.eval()
    
    batch_size = 1
    src_len = 8
    max_length = 10
    
    input_ids = torch.randint(1, config.vocab_size - 1, (batch_size, src_len))
    
    with torch.no_grad():
        # 编码输入
        encoder_outputs = model.encoder(input_ids)
        
        # 贪婪解码生成
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long)  # 开始标记
        generated_ids = [decoder_input_ids]
        
        for _ in range(max_length - 1):
            current_input = torch.cat(generated_ids, dim=1)
            
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=current_input
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids.append(next_token_id)
            
            # 简单的停止条件
            if next_token_id.item() == 0:  # 假设0是结束标记
                break
        
        generated_sequence = torch.cat(generated_ids, dim=1)
        print(f"生成的序列形状: {generated_sequence.shape}")
        print(f"生成的序列: {generated_sequence.tolist()}")
    
    print("T5生成功能测试通过 ✓")


def main():
    """主测试函数"""
    print("开始T5模型测试...")
    print("=" * 50)
    
    try:
        test_t5_config()
        test_modern_techniques()
        test_t5_model_basic()
        test_t5_encoder_decoder()
        test_t5_generation()
        
        print("\n" + "=" * 50)
        print("🎉 所有T5模型测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()