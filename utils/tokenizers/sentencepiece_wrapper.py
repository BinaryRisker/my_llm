"""
SentencePiece分词器包装器
========================

对SentencePiece进行封装，提供统一的接口和额外功能：
- 统一的训练和使用接口
- 支持多种模型类型 (BPE, Unigram, Word, Char)
- 批量处理功能
- 模型管理和持久化

使用方法:
    from utils.tokenizers.sentencepiece_wrapper import SentencePieceWrapper
    
    tokenizer = SentencePieceWrapper()
    tokenizer.train(['Hello world', 'Hello SentencePiece'], vocab_size=1000)
    tokens = tokenizer.encode('Hello world!')
    text = tokenizer.decode(tokens)
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    spm = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentencePieceWrapper:
    """SentencePiece分词器包装器"""
    
    def __init__(self,
                 vocab_size: int = 10000,
                 model_type: str = 'bpe',
                 character_coverage: float = 0.9995,
                 normalization_rule_name: str = 'nmt_nfkc_cf',
                 split_by_unicode_script: bool = True,
                 split_by_number: bool = True,
                 split_by_whitespace: bool = True,
                 treat_whitespace_as_suffix: bool = False,
                 allow_whitespace_only_pieces: bool = False,
                 split_digits: bool = False,
                 control_symbols: Optional[List[str]] = None,
                 user_defined_symbols: Optional[List[str]] = None,
                 byte_fallback: bool = False,
                 vocabulary_output_piece_score: bool = True,
                 train_extremely_large_corpus: bool = False,
                 enable_sampling: bool = False,
                 nbest_size: int = -1,
                 alpha: float = 0.1):
        """
        初始化SentencePiece包装器
        
        Args:
            vocab_size: 词汇表大小
            model_type: 模型类型 ('bpe', 'unigram', 'word', 'char')
            character_coverage: 字符覆盖率
            normalization_rule_name: 标准化规则
            split_by_unicode_script: 按Unicode脚本分割
            split_by_number: 按数字分割
            split_by_whitespace: 按空白字符分割
            treat_whitespace_as_suffix: 将空白字符视为后缀
            allow_whitespace_only_pieces: 允许仅空白字符的片段
            split_digits: 分割数字
            control_symbols: 控制符号
            user_defined_symbols: 用户定义符号
            byte_fallback: 字节回退
            vocabulary_output_piece_score: 输出片段分数
            train_extremely_large_corpus: 训练极大语料库
            enable_sampling: 启用采样
            nbest_size: N-best大小
            alpha: 采样参数
        """
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("SentencePiece未安装。请运行: pip install sentencepiece")
        
        self.vocab_size = vocab_size
        self.model_type = model_type.lower()
        
        # 验证模型类型
        if self.model_type not in ['bpe', 'unigram', 'word', 'char']:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练参数
        self.training_args = {
            'vocab_size': vocab_size,
            'model_type': self.model_type,
            'character_coverage': character_coverage,
            'normalization_rule_name': normalization_rule_name,
            'split_by_unicode_script': split_by_unicode_script,
            'split_by_number': split_by_number,
            'split_by_whitespace': split_by_whitespace,
            'treat_whitespace_as_suffix': treat_whitespace_as_suffix,
            'allow_whitespace_only_pieces': allow_whitespace_only_pieces,
            'split_digits': split_digits,
            'byte_fallback': byte_fallback,
            'vocabulary_output_piece_score': vocabulary_output_piece_score,
            'train_extremely_large_corpus': train_extremely_large_corpus
        }
        
        # 控制符号和用户定义符号
        if control_symbols:
            self.training_args['control_symbols'] = ','.join(control_symbols)
        
        if user_defined_symbols:
            self.training_args['user_defined_symbols'] = ','.join(user_defined_symbols)
        
        # 采样参数
        self.enable_sampling = enable_sampling
        self.nbest_size = nbest_size
        self.alpha = alpha
        
        # SentencePiece处理器
        self.sp = None
        self.trained = False
        self.model_path = None
    
    def train(self, 
              texts: List[str],
              model_prefix: str = 'sentencepiece_model',
              temp_dir: Optional[str] = None) -> None:
        """
        训练SentencePiece模型
        
        Args:
            texts: 训练文本列表
            model_prefix: 模型文件前缀
            temp_dir: 临时目录
        """
        logger.info(f"开始训练SentencePiece模型，词汇表大小: {self.vocab_size}")
        
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        else:
            os.makedirs(temp_dir, exist_ok=True)
        
        # 创建训练数据文件
        train_file = os.path.join(temp_dir, 'train.txt')
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text.strip() + '\n')
        
        # 设置训练参数
        train_args = self.training_args.copy()
        train_args['input'] = train_file
        train_args['model_prefix'] = os.path.join(temp_dir, model_prefix)
        
        # 构建训练命令
        cmd_args = []
        for key, value in train_args.items():
            cmd_args.append(f'--{key}={value}')
        
        logger.info("训练SentencePiece模型...")
        
        # 训练模型
        spm.SentencePieceTrainer.Train(' '.join(cmd_args))
        
        # 加载训练好的模型
        self.model_path = f"{train_args['model_prefix']}.model"
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        
        self.trained = True
        
        logger.info(f"SentencePiece训练完成，实际词汇表大小: {self.sp.get_piece_size()}")
        
        # 清理临时文件
        try:
            os.remove(train_file)
        except OSError:
            pass
    
    def train_from_file(self, 
                       input_file: str,
                       model_prefix: str = 'sentencepiece_model',
                       output_dir: str = '.') -> None:
        """
        从文件训练SentencePiece模型
        
        Args:
            input_file: 输入文件路径
            model_prefix: 模型文件前缀
            output_dir: 输出目录
        """
        logger.info(f"从文件训练SentencePiece模型: {input_file}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置训练参数
        train_args = self.training_args.copy()
        train_args['input'] = input_file
        train_args['model_prefix'] = os.path.join(output_dir, model_prefix)
        
        # 构建训练命令
        cmd_args = []
        for key, value in train_args.items():
            cmd_args.append(f'--{key}={value}')
        
        # 训练模型
        spm.SentencePieceTrainer.Train(' '.join(cmd_args))
        
        # 加载训练好的模型
        self.model_path = f"{train_args['model_prefix']}.model"
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        
        self.trained = True
        
        logger.info(f"SentencePiece训练完成，实际词汇表大小: {self.sp.get_piece_size()}")
    
    def encode(self, 
               text: str, 
               out_type: str = 'int',
               add_bos: bool = False,
               add_eos: bool = False,
               enable_sampling: bool = None,
               nbest_size: int = None,
               alpha: float = None) -> Union[List[int], List[str]]:
        """
        编码文本
        
        Args:
            text: 输入文本
            out_type: 输出类型 ('int' 或 'str')
            add_bos: 添加开始标记
            add_eos: 添加结束标记
            enable_sampling: 启用采样
            nbest_size: N-best大小
            alpha: 采样参数
            
        Returns:
            编码结果 (token IDs 或 token strings)
        """
        if not self.trained:
            raise ValueError("SentencePiece模型未训练，请先调用train()方法")
        
        # 设置采样参数
        if enable_sampling is None:
            enable_sampling = self.enable_sampling
        if nbest_size is None:
            nbest_size = self.nbest_size
        if alpha is None:
            alpha = self.alpha
        
        # 编码文本
        if out_type == 'int':
            tokens = self.sp.encode(text, 
                                   out_type=int,
                                   add_bos=add_bos,
                                   add_eos=add_eos,
                                   enable_sampling=enable_sampling,
                                   nbest_size=nbest_size,
                                   alpha=alpha)
        elif out_type == 'str':
            tokens = self.sp.encode(text, 
                                   out_type=str,
                                   add_bos=add_bos,
                                   add_eos=add_eos,
                                   enable_sampling=enable_sampling,
                                   nbest_size=nbest_size,
                                   alpha=alpha)
        else:
            raise ValueError(f"不支持的输出类型: {out_type}")
        
        return tokens
    
    def decode(self, tokens: Union[List[int], List[str]]) -> str:
        """
        解码token序列
        
        Args:
            tokens: token ID列表或token字符串列表
            
        Returns:
            解码后的文本
        """
        if not self.trained:
            raise ValueError("SentencePiece模型未训练，请先调用train()方法")
        
        return self.sp.decode(tokens)
    
    def encode_batch(self, 
                    texts: List[str],
                    out_type: str = 'int',
                    add_bos: bool = False,
                    add_eos: bool = False) -> List[Union[List[int], List[str]]]:\n        \"\"\"批量编码文本\"\"\"\n        return [self.encode(text, out_type, add_bos, add_eos) for text in texts]\n    \n    def decode_batch(self, token_lists: List[Union[List[int], List[str]]]) -> List[str]:\n        \"\"\"批量解码token列表\"\"\"\n        return [self.decode(tokens) for tokens in token_lists]\n    \n    def get_vocab_size(self) -> int:\n        \"\"\"获取词汇表大小\"\"\"\n        if not self.trained:\n            return self.vocab_size\n        return self.sp.get_piece_size()\n    \n    def get_vocab(self) -> Dict[str, int]:\n        \"\"\"获取词汇表字典\"\"\"\n        if not self.trained:\n            raise ValueError(\"SentencePiece模型未训练\")\n        \n        vocab = {}\n        for i in range(self.sp.get_piece_size()):\n            piece = self.sp.id_to_piece(i)\n            vocab[piece] = i\n        \n        return vocab\n    \n    def piece_to_id(self, piece: str) -> int:\n        \"\"\"将piece转换为ID\"\"\"\n        if not self.trained:\n            raise ValueError(\"SentencePiece模型未训练\")\n        return self.sp.piece_to_id(piece)\n    \n    def id_to_piece(self, id_: int) -> str:\n        \"\"\"将ID转换为piece\"\"\"\n        if not self.trained:\n            raise ValueError(\"SentencePiece模型未训练\")\n        return self.sp.id_to_piece(id_)\n    \n    def is_unknown(self, id_: int) -> bool:\n        \"\"\"检查ID是否为未知token\"\"\"\n        if not self.trained:\n            raise ValueError(\"SentencePiece模型未训练\")\n        return self.sp.is_unknown(id_)\n    \n    def is_control(self, id_: int) -> bool:\n        \"\"\"检查ID是否为控制token\"\"\"\n        if not self.trained:\n            raise ValueError(\"SentencePiece模型未训练\")\n        return self.sp.is_control(id_)\n    \n    def get_score(self, id_: int) -> float:\n        \"\"\"获取token的分数\"\"\"\n        if not self.trained:\n            raise ValueError(\"SentencePiece模型未训练\")\n        return self.sp.get_score(id_)\n    \n    def save(self, save_dir: Union[str, Path]) -> None:\n        \"\"\"保存模型\"\"\"\n        if not self.trained:\n            raise ValueError(\"SentencePiece模型未训练\")\n        \n        save_dir = Path(save_dir)\n        save_dir.mkdir(parents=True, exist_ok=True)\n        \n        # 保存模型文件\n        import shutil\n        model_file = save_dir / 'sentencepiece.model'\n        vocab_file = save_dir / 'sentencepiece.vocab'\n        \n        shutil.copy2(self.model_path, model_file)\n        if os.path.exists(self.model_path.replace('.model', '.vocab')):\n            shutil.copy2(self.model_path.replace('.model', '.vocab'), vocab_file)\n        \n        # 保存配置\n        config = {\n            'vocab_size': self.vocab_size,\n            'model_type': self.model_type,\n            'training_args': self.training_args,\n            'enable_sampling': self.enable_sampling,\n            'nbest_size': self.nbest_size,\n            'alpha': self.alpha,\n            'trained': self.trained\n        }\n        \n        with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:\n            json.dump(config, f, indent=2, ensure_ascii=False)\n        \n        logger.info(f\"SentencePiece模型已保存到: {save_dir}\")\n    \n    @classmethod\n    def load(cls, load_dir: Union[str, Path]) -> 'SentencePieceWrapper':\n        \"\"\"加载模型\"\"\"\n        load_dir = Path(load_dir)\n        \n        # 加载配置\n        config_file = load_dir / 'config.json'\n        with open(config_file, 'r', encoding='utf-8') as f:\n            config = json.load(f)\n        \n        # 创建实例\n        instance = cls(\n            vocab_size=config['vocab_size'],\n            model_type=config['model_type'],\n            enable_sampling=config.get('enable_sampling', False),\n            nbest_size=config.get('nbest_size', -1),\n            alpha=config.get('alpha', 0.1)\n        )\n        \n        # 加载SentencePiece模型\n        model_file = load_dir / 'sentencepiece.model'\n        instance.sp = spm.SentencePieceProcessor(model_file=str(model_file))\n        instance.model_path = str(model_file)\n        instance.trained = config['trained']\n        \n        logger.info(f\"SentencePiece模型已从 {load_dir} 加载\")\n        return instance\n    \n    def get_model_info(self) -> Dict[str, Any]:\n        \"\"\"获取模型信息\"\"\"\n        if not self.trained:\n            return {\n                'trained': False,\n                'vocab_size': self.vocab_size,\n                'model_type': self.model_type\n            }\n        \n        return {\n            'trained': True,\n            'vocab_size': self.sp.get_piece_size(),\n            'model_type': self.model_type,\n            'bos_id': self.sp.bos_id(),\n            'eos_id': self.sp.eos_id(),\n            'unk_id': self.sp.unk_id(),\n            'pad_id': self.sp.pad_id(),\n            'model_path': self.model_path\n        }\n\n\ndef demo():\n    \"\"\"演示SentencePiece包装器的使用\"\"\"\n    if not SENTENCEPIECE_AVAILABLE:\n        print(\"❌ SentencePiece未安装，无法运行演示\")\n        print(\"请运行: pip install sentencepiece\")\n        return\n    \n    print(\"🚀 SentencePiece包装器演示\")\n    print(\"=\" * 50)\n    \n    # 创建分词器\n    tokenizer = SentencePieceWrapper(\n        vocab_size=100, \n        model_type='bpe',\n        character_coverage=0.995\n    )\n    \n    # 训练语料\n    texts = [\n        \"Hello world! This is a test.\",\n        \"Hello SentencePiece tokenizer.\",\n        \"Byte pair encoding with SentencePiece is powerful.\",\n        \"The quick brown fox jumps over the lazy dog.\",\n        \"Natural language processing with advanced tokenization.\",\n        \"Machine learning and artificial intelligence research.\",\n        \"Deep learning models for text generation.\",\n        \"Transformer architecture revolutionized NLP.\"\n    ]\n    \n    print(\"📚 训练语料:\")\n    for i, text in enumerate(texts[:3]):\n        print(f\"  {i+1}. {text}\")\n    print(f\"  ... 共 {len(texts)} 条\")\n    \n    # 训练模型\n    print(\"\\n🔧 训练SentencePiece模型...\")\n    try:\n        tokenizer.train(texts)\n        print(f\"✅ 训练完成，实际词汇表大小: {tokenizer.get_vocab_size()}\")\n        \n        # 显示模型信息\n        model_info = tokenizer.get_model_info()\n        print(f\"\\n📊 模型信息:\")\n        for key, value in model_info.items():\n            print(f\"  {key}: {value}\")\n        \n        # 测试编码和解码\n        test_text = \"Hello SentencePiece! This is amazing.\"\n        print(f\"\\n📝 测试文本: {test_text}\")\n        \n        # 编码为ID\n        token_ids = tokenizer.encode(test_text, out_type='int')\n        print(f\"🔢 编码结果 (IDs): {token_ids}\")\n        \n        # 编码为字符串\n        token_strs = tokenizer.encode(test_text, out_type='str')\n        print(f\"📝 编码结果 (tokens): {token_strs}\")\n        \n        # 解码\n        decoded_text = tokenizer.decode(token_ids)\n        print(f\"📄 解码结果: {decoded_text}\")\n        \n        # 显示部分词汇表\n        print(f\"\\n📖 词汇表示例 (前20个):\")\n        for i in range(min(20, tokenizer.get_vocab_size())):\n            piece = tokenizer.id_to_piece(i)\n            score = tokenizer.get_score(i)\n            print(f\"  {i:3d}: '{piece}' (score: {score:.4f})\")\n        \n        # 批量处理\n        batch_texts = [\"Hello world\", \"SentencePiece rocks\", \"AI is great\"]\n        batch_encoded = tokenizer.encode_batch(batch_texts)\n        batch_decoded = tokenizer.decode_batch(batch_encoded)\n        \n        print(f\"\\n🔄 批量处理测试:\")\n        for i, (orig, encoded, decoded) in enumerate(zip(batch_texts, batch_encoded, batch_decoded)):\n            print(f\"  {i+1}. '{orig}' -> {encoded} -> '{decoded}'\")\n        \n        # 保存和加载测试\n        save_path = \"temp_spm_model\"\n        tokenizer.save(save_path)\n        print(f\"\\n💾 模型已保存到: {save_path}\")\n        \n        # 测试加载\n        loaded_tokenizer = SentencePieceWrapper.load(save_path)\n        test_result = loaded_tokenizer.decode(loaded_tokenizer.encode(\"Test loading\"))\n        print(f\"🔄 加载测试: {test_result}\")\n        \n    except Exception as e:\n        print(f\"❌ 演示过程中出现错误: {e}\")\n\n\nif __name__ == \"__main__\":\n    demo()