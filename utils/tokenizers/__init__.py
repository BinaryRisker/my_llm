"""
分词器模块
=========

包含各种分词器的实现和包装器：
- BPETokenizer: 从零实现的BPE分词器
- SentencePieceWrapper: SentencePiece包装器
- 统一的分词器接口

使用方法:
    from utils.tokenizers import BPETokenizer, SentencePieceWrapper
"""

from .bpe_tokenizer import BPETokenizer
try:
    from .sentencepiece_wrapper import SentencePieceWrapper
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    
    class SentencePieceWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("SentencePiece未安装。请运行: pip install sentencepiece")

__all__ = ['BPETokenizer', 'SentencePieceWrapper', 'SENTENCEPIECE_AVAILABLE']