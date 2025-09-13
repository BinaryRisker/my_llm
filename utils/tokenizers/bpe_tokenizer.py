"""
BPE (Byte Pair Encoding) 分词器实现
==================================

从零实现BPE分词器，支持：
- 基于语料库训练BPE模型
- 编码和解码文本
- 词汇表管理
- 特殊token处理

使用方法:
    from utils.tokenizers.bpe_tokenizer import BPETokenizer
    
    tokenizer = BPETokenizer()
    tokenizer.train(['Hello world', 'Hello BPE'], vocab_size=1000)
    tokens = tokenizer.encode('Hello world!')
    text = tokenizer.decode(tokens)
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BPETokenizer:
    """BPE分词器实现"""
    
    def __init__(self, 
                 vocab_size: int = 10000,
                 min_frequency: int = 2,
                 special_tokens: Optional[List[str]] = None,
                 lowercase: bool = False,
                 dropout: float = 0.0):
        """
        初始化BPE分词器
        
        Args:
            vocab_size: 词汇表大小
            min_frequency: 最小词频阈值
            special_tokens: 特殊token列表
            lowercase: 是否转小写
            dropout: BPE dropout率
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self.dropout = dropout
        
        # 特殊token
        self.special_tokens = special_tokens or [
            '<pad>', '<unk>', '<bos>', '<eos>'
        ]
        
        # 初始化词汇表和编码表
        self.word_to_id = {}
        self.id_to_word = {}
        self.bpe_ranks = {}
        self.word_freqs = {}
        
        # 编译正则表达式
        self.word_pattern = re.compile(r'\b\w+\b|[^\w\s]')
        
        # 是否已训练
        self.trained = False
    
    def _get_word_tokens(self, word: str) -> List[str]:
        """将单词分解为字符列表，最后一个字符加上</w>标记"""
        if not word:
            return []
        chars = list(word)
        chars[-1] += '</w>'
        return chars
    
    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """获取相邻字符对"""
        pairs = set()
        prev_char = word_tokens[0]
        for char in word_tokens[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """统计字符对频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_symbols(self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """合并字符对"""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab:
            new_word = p.sub(''.join(pair), ' '.join(word))
            new_vocab[tuple(new_word.split())] = vocab[word]
        return new_vocab
    
    def train(self, 
              texts: List[str], 
              progress_callback: Optional[callable] = None) -> None:
        """
        训练BPE模型
        
        Args:
            texts: 训练文本列表
            progress_callback: 进度回调函数
        """
        logger.info(f"开始训练BPE模型，目标词汇表大小: {self.vocab_size}")
        
        # 1. 预处理文本和统计词频
        word_freqs = Counter()
        
        for text in texts:
            if self.lowercase:
                text = text.lower()
            
            words = self.word_pattern.findall(text)
            for word in words:
                word_freqs[word] += 1
        
        # 过滤低频词
        word_freqs = {word: freq for word, freq in word_freqs.items() 
                     if freq >= self.min_frequency}
        
        logger.info(f"统计到 {len(word_freqs)} 个不重复单词")
        
        # 2. 初始化词汇表
        vocab = {}
        for word, freq in word_freqs.items():
            word_tokens = self._get_word_tokens(word)
            vocab[tuple(word_tokens)] = freq
        
        # 添加特殊token
        for special_token in self.special_tokens:
            self.word_to_id[special_token] = len(self.word_to_id)
            self.id_to_word[len(self.id_to_word)] = special_token
        
        # 3. 进行BPE训练
        num_merges = self.vocab_size - len(self.special_tokens) - len(set(''.join(vocab.keys())))
        
        for i in range(num_merges):
            # 统计字符对频率
            pairs = self._get_stats(vocab)
            
            if not pairs:
                break
                
            # 选择频率最高的字符对
            best_pair = max(pairs, key=pairs.get)
            
            # 记录合并操作
            self.bpe_ranks[best_pair] = i
            
            # 执行合并
            vocab = self._merge_symbols(best_pair, vocab)
            
            # 进度回调
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, num_merges)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"完成 {i + 1}/{num_merges} 次合并")
        
        # 4. 构建最终词汇表
        for word_tokens in vocab.keys():
            for token in word_tokens:
                if token not in self.word_to_id:
                    self.word_to_id[token] = len(self.word_to_id)
                    self.id_to_word[len(self.id_to_word)] = token
        
        self.word_freqs = word_freqs
        self.trained = True
        
        logger.info(f"BPE训练完成，词汇表大小: {len(self.word_to_id)}")
    
    def _bpe_encode(self, word: str) -> List[str]:
        """对单个单词进行BPE编码"""
        if not word:
            return []
            
        # 添加词结尾标记
        word_tokens = self._get_word_tokens(word)
        
        if len(word_tokens) == 1:
            return word_tokens
        
        # 迭代合并
        pairs = self._get_pairs(word_tokens)
        
        if not pairs:
            return word_tokens
        
        while True:
            # 找到rank最小的pair（最早被合并的）
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
                
            # 合并bigram
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word_tokens):
                try:
                    j = word_tokens.index(first, i)
                    new_word.extend(word_tokens[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word_tokens[i:])
                    break
                
                if i < len(word_tokens) - 1 and word_tokens[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            
            word_tokens = new_word
            
            if len(word_tokens) == 1:
                break
            
            pairs = self._get_pairs(word_tokens)
        
        return word_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        编码文本为token ID列表
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊token
            
        Returns:
            token ID列表
        """
        if not self.trained:
            raise ValueError("BPE模型未训练，请先调用train()方法")
        
        if self.lowercase:
            text = text.lower()
        
        # 提取单词
        words = self.word_pattern.findall(text)
        
        # 编码每个单词
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.word_to_id.get('<bos>', 0))
        
        for word in words:
            # BPE编码
            bpe_tokens = self._bpe_encode(word)
            
            # 转换为ID
            for token in bpe_tokens:
                token_id = self.word_to_id.get(token, self.word_to_id.get('<unk>', 1))
                token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.word_to_id.get('<eos>', 3))
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        解码token ID列表为文本
        
        Args:
            token_ids: token ID列表
            skip_special_tokens: 是否跳过特殊token
            
        Returns:
            解码后的文本
        """
        if not self.trained:
            raise ValueError("BPE模型未训练，请先调用train()方法")
        
        tokens = []
        for token_id in token_ids:
            token = self.id_to_word.get(token_id, '<unk>')
            
            if skip_special_tokens and token in self.special_tokens:
                continue
                
            tokens.append(token)
        
        # 重构文本
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        
        return text.strip()
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """批量编码文本"""
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def decode_batch(self, token_ids_list: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """批量解码token ID"""
        return [self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_list]
    
    def get_vocab(self) -> Dict[str, int]:
        """获取词汇表"""
        return self.word_to_id.copy()
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.word_to_id)
    
    def save(self, save_path: Union[str, Path]) -> None:
        """保存BPE模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'lowercase': self.lowercase,
            'dropout': self.dropout,
            'trained': self.trained
        }
        
        with open(save_path / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 保存词汇表
        with open(save_path / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self.word_to_id, f, indent=2, ensure_ascii=False)
        
        # 保存BPE规则
        with open(save_path / 'merges.txt', 'w', encoding='utf-8') as f:
            # 按rank排序
            sorted_merges = sorted(self.bpe_ranks.items(), key=lambda x: x[1])
            for (first, second), rank in sorted_merges:
                f.write(f"{first} {second}\n")
        
        # 保存词频
        with open(save_path / 'word_freqs.json', 'w', encoding='utf-8') as f:
            json.dump(self.word_freqs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"BPE模型已保存到: {save_path}")
    
    @classmethod
    def load(cls, load_path: Union[str, Path]) -> 'BPETokenizer':
        """加载BPE模型"""
        load_path = Path(load_path)
        
        # 加载配置
        with open(load_path / 'config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建实例
        tokenizer = cls(**{k: v for k, v in config.items() if k != 'trained'})
        
        # 加载词汇表
        with open(load_path / 'vocab.json', 'r', encoding='utf-8') as f:
            tokenizer.word_to_id = json.load(f)
        
        tokenizer.id_to_word = {v: k for k, v in tokenizer.word_to_id.items()}
        
        # 加载BPE规则
        tokenizer.bpe_ranks = {}
        with open(load_path / 'merges.txt', 'r', encoding='utf-8') as f:
            for rank, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 2:
                    tokenizer.bpe_ranks[(parts[0], parts[1])] = rank
        
        # 加载词频
        with open(load_path / 'word_freqs.json', 'r', encoding='utf-8') as f:
            tokenizer.word_freqs = json.load(f)
        
        tokenizer.trained = config['trained']
        
        logger.info(f"BPE模型已从 {load_path} 加载")
        return tokenizer


def demo():
    """演示BPE分词器的使用"""
    print("🚀 BPE分词器演示")
    print("=" * 50)
    
    # 创建分词器
    tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
    
    # 训练语料
    texts = [
        "Hello world! This is a test.",
        "Hello BPE tokenizer.",
        "Byte pair encoding is powerful.",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing with BPE.",
        "Machine learning and artificial intelligence."
    ]
    
    print("📚 训练语料:")
    for text in texts:
        print(f"  - {text}")
    
    # 训练模型
    print("\n🔧 训练BPE模型...")
    tokenizer.train(texts)
    
    print(f"✅ 训练完成，词汇表大小: {tokenizer.get_vocab_size()}")
    
    # 测试编码和解码
    test_text = "Hello BPE! This is amazing."
    print(f"\n📝 测试文本: {test_text}")
    
    # 编码
    token_ids = tokenizer.encode(test_text)
    print(f"🔢 编码结果: {token_ids}")
    
    # 解码
    decoded_text = tokenizer.decode(token_ids)
    print(f"📄 解码结果: {decoded_text}")
    
    # 显示词汇表
    print(f"\n📖 词汇表 (前20个):")
    vocab = tokenizer.get_vocab()
    for i, (word, id_) in enumerate(list(vocab.items())[:20]):
        print(f"  {id_:3d}: '{word}'")
    
    # 保存模型
    save_path = "temp_bpe_model"
    tokenizer.save(save_path)
    print(f"\n💾 模型已保存到: {save_path}")
    
    # 测试加载
    loaded_tokenizer = BPETokenizer.load(save_path)
    test_ids = loaded_tokenizer.encode("Test loading")
    print(f"🔄 加载测试: {loaded_tokenizer.decode(test_ids)}")


if __name__ == "__main__":
    demo()