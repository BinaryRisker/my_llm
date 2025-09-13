"""
数据集下载脚本
================

下载各种用于语言模型训练的标准数据集：
- WMT翻译数据集（英法、英德等）
- OpenWebText预训练数据
- WikiText语言建模数据
- IMDB情感分析数据
- 其他常用NLP数据集

使用方法:
    python scripts/download_datasets.py --dataset wmt_en_fr --output_dir ./data
    python scripts/download_datasets.py --all --output_dir ./data
"""

import os
import sys
import argparse
import requests
import tarfile
import zipfile
import gzip
import json
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        # 数据集配置
        self.datasets = {
            'wmt_en_fr': {
                'name': 'WMT English-French Translation',
                'url': 'http://www.statmt.org/wmt14/training-parallel-europarl-v7.tgz',
                'description': 'WMT 2014 English-French parallel corpus',
                'size_mb': 500,
                'files': ['europarl-v7.fr-en.en', 'europarl-v7.fr-en.fr'],
                'md5': 'c1ba6e207f1e3b15bf9b7b70f24db672'
            },
            
            'wmt_en_de': {
                'name': 'WMT English-German Translation',
                'url': 'http://www.statmt.org/wmt14/training-parallel-europarl-v7.tgz',
                'description': 'WMT 2014 English-German parallel corpus',
                'size_mb': 500,
                'files': ['europarl-v7.de-en.en', 'europarl-v7.de-en.de'],
                'md5': 'c1ba6e207f1e3b15bf9b7b70f24db672'
            },
            
            'openwebtext': {
                'name': 'OpenWebText',
                'url': 'https://skylion007.github.io/OpenWebTextCorpus/',
                'description': 'Open source recreation of WebText corpus',
                'size_mb': 40000,  # ~40GB
                'note': 'Large dataset - requires significant disk space',
                'manual': True  # 需要手动下载
            },
            
            'wikitext_103': {
                'name': 'WikiText-103',
                'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
                'description': 'WikiText-103 language modeling dataset',
                'size_mb': 500,
                'files': ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens'],
                'md5': '0ca3512bd7a238be4a63ce7b434f8935'
            },
            
            'wikitext_2': {
                'name': 'WikiText-2',
                'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
                'description': 'WikiText-2 language modeling dataset (smaller)',
                'size_mb': 10,
                'files': ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens'],
                'md5': '3c914d17d80b1459be871a5039ac23e0'
            },
            
            'imdb_reviews': {
                'name': 'IMDB Movie Reviews',
                'url': 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                'description': 'IMDB sentiment classification dataset',
                'size_mb': 80,
                'files': ['train/pos/', 'train/neg/', 'test/pos/', 'test/neg/'],
                'md5': '7c2ac02c03563afcf9b574c7e56c153a'
            },
            
            'penn_treebank': {
                'name': 'Penn Treebank',
                'url': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/penn/',
                'description': 'Penn Treebank language modeling dataset',
                'size_mb': 5,
                'files': ['train.txt', 'valid.txt', 'test.txt'],
                'base_urls': [
                    'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/penn/train.txt',
                    'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/penn/valid.txt', 
                    'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/penn/test.txt'
                ]
            },
            
            'multi30k': {
                'name': 'Multi30K Translation',
                'url': 'https://github.com/multi30k/dataset/raw/master/data/task1/tok/',
                'description': 'Multi30K multimodal translation dataset',
                'size_mb': 10,
                'files': ['train.en', 'train.de', 'val.en', 'val.de', 'test_2016_flickr.en', 'test_2016_flickr.de'],
                'base_urls': [
                    'https://github.com/multi30k/dataset/raw/master/data/task1/tok/train.en',
                    'https://github.com/multi30k/dataset/raw/master/data/task1/tok/train.de',
                    'https://github.com/multi30k/dataset/raw/master/data/task1/tok/val.en',
                    'https://github.com/multi30k/dataset/raw/master/data/task1/tok/val.de',
                    'https://github.com/multi30k/dataset/raw/master/data/task1/tok/test_2016_flickr.en',
                    'https://github.com/multi30k/dataset/raw/master/data/task1/tok/test_2016_flickr.de'
                ]
            }
        }
    
    def download_file(self, url: str, output_path: Path, expected_md5: str = None) -> bool:
        """下载单个文件"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=f"下载 {output_path.name}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            # 验证MD5
            if expected_md5:
                actual_md5 = self.compute_md5(output_path)
                if actual_md5 != expected_md5:
                    logger.warning(f"MD5校验失败: 期望 {expected_md5}, 实际 {actual_md5}")
                    return False
                else:
                    logger.info(f"MD5校验通过: {actual_md5}")
            
            return True
            
        except Exception as e:
            logger.error(f"下载失败 {url}: {e}")
            return False
    
    def compute_md5(self, file_path: Path) -> str:
        """计算文件MD5"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """解压缩文件"""
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_file:
                    zip_file.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.tgz'] or '.tar.' in archive_path.name:
                with tarfile.open(archive_path, 'r:*') as tar_file:
                    tar_file.extractall(extract_to)
            elif archive_path.suffix == '.gz' and not '.tar.' in archive_path.name:
                with gzip.open(archive_path, 'rb') as gz_file:
                    with open(extract_to / archive_path.stem, 'wb') as out_file:
                        out_file.write(gz_file.read())
            else:
                logger.warning(f"不支持的压缩格式: {archive_path}")
                return False
            
            logger.info(f"解压完成: {archive_path} -> {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"解压失败 {archive_path}: {e}")
            return False
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """下载指定数据集"""
        if dataset_name not in self.datasets:
            logger.error(f"未知数据集: {dataset_name}")
            return False
        
        config = self.datasets[dataset_name]
        dataset_dir = self.base_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"开始下载数据集: {config['name']}")
        logger.info(f"描述: {config['description']}")
        logger.info(f"大小: {config['size_mb']} MB")
        
        # 检查是否需要手动下载
        if config.get('manual'):
            logger.warning(f"数据集 {dataset_name} 需要手动下载")
            logger.info(f"请访问: {config['url']}")
            return False
        
        # 检查是否已存在
        if not force and self.is_dataset_downloaded(dataset_name):
            logger.info(f"数据集 {dataset_name} 已存在，跳过下载")
            return True
        
        # 下载数据集
        success = False
        
        if 'base_urls' in config:
            # 多个文件分别下载
            success = True
            for i, url in enumerate(config['base_urls']):
                filename = config['files'][i]
                output_path = dataset_dir / filename
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                if not self.download_file(url, output_path):
                    success = False
                    break
        else:
            # 单个压缩文件下载
            url = config['url']
            filename = Path(url).name
            archive_path = dataset_dir / filename
            
            # 下载
            expected_md5 = config.get('md5')
            if self.download_file(url, archive_path, expected_md5):
                # 解压
                if self.extract_archive(archive_path, dataset_dir):
                    # 删除压缩文件（可选）
                    # archive_path.unlink()
                    success = True
        
        if success:
            # 创建数据集信息文件
            info_file = dataset_dir / 'dataset_info.json'
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'name': config['name'],
                    'description': config['description'],
                    'downloaded_at': str(Path(__file__).stat().st_mtime),
                    'files': config.get('files', []),
                    'size_mb': config['size_mb']
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"数据集 {dataset_name} 下载完成!")
        else:
            logger.error(f"数据集 {dataset_name} 下载失败!")
        
        return success
    
    def is_dataset_downloaded(self, dataset_name: str) -> bool:
        """检查数据集是否已下载"""
        dataset_dir = self.base_dir / dataset_name
        if not dataset_dir.exists():
            return False
        
        info_file = dataset_dir / 'dataset_info.json'
        if not info_file.exists():
            return False
        
        # 检查关键文件是否存在
        config = self.datasets[dataset_name]
        if 'files' in config:
            for filename in config['files']:
                file_path = dataset_dir / filename
                # 检查文件或目录
                if not (file_path.exists() or any(dataset_dir.rglob(filename))):
                    return False
        
        return True
    
    def list_available_datasets(self):
        """列出可用的数据集"""
        print("\n📚 可用数据集:")
        print("=" * 80)
        
        for name, config in self.datasets.items():
            status = "✅ 已下载" if self.is_dataset_downloaded(name) else "📥 未下载"
            manual = " 🔗 需手动下载" if config.get('manual') else ""
            
            print(f"\n🔹 {name}")
            print(f"   名称: {config['name']}")
            print(f"   描述: {config['description']}")
            print(f"   大小: {config['size_mb']} MB")
            print(f"   状态: {status}{manual}")
            
            if config.get('note'):
                print(f"   注意: {config['note']}")
    
    def download_all(self, force: bool = False, skip_manual: bool = True):
        """下载所有数据集"""
        logger.info("开始下载所有数据集...")
        
        success_count = 0
        total_count = 0
        
        for dataset_name, config in self.datasets.items():
            if skip_manual and config.get('manual'):
                logger.info(f"跳过需要手动下载的数据集: {dataset_name}")
                continue
            
            total_count += 1
            if self.download_dataset(dataset_name, force):
                success_count += 1
        
        logger.info(f"下载完成: {success_count}/{total_count} 个数据集下载成功")
    
    def create_sample_configs(self):
        """为各阶段创建示例数据集配置"""
        configs = {
            'stage2_rnn_lstm': {
                'datasets': ['penn_treebank', 'wikitext_2'],
                'task': 'language_modeling',
                'description': 'RNN/LSTM语言建模数据'
            },
            'stage3_attention': {
                'datasets': ['multi30k', 'wmt_en_fr'],
                'task': 'translation', 
                'description': '注意力机制翻译数据'
            },
            'stage4_transformer': {
                'datasets': ['wmt_en_fr', 'wmt_en_de'],
                'task': 'translation',
                'description': 'Transformer翻译数据'
            },
            'stage5_gpt': {
                'datasets': ['wikitext_103', 'openwebtext'],
                'task': 'language_modeling',
                'description': 'GPT预训练数据'
            }
        }
        
        configs_dir = self.base_dir / 'configs'
        configs_dir.mkdir(exist_ok=True)
        
        for stage, config in configs.items():
            config_file = configs_dir / f'{stage}_datasets.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集配置文件已创建在: {configs_dir}")


def main():
    parser = argparse.ArgumentParser(description='下载语言模型训练数据集')
    
    parser.add_argument('--dataset', type=str, 
                       help='要下载的数据集名称')
    parser.add_argument('--all', action='store_true',
                       help='下载所有数据集')
    parser.add_argument('--list', action='store_true', 
                       help='列出所有可用数据集')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='数据下载目录 (默认: ./data)')
    parser.add_argument('--force', action='store_true',
                       help='强制重新下载已存在的数据集')
    parser.add_argument('--skip_manual', action='store_true', default=True,
                       help='跳过需要手动下载的数据集')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output_dir)
    
    if args.list:
        downloader.list_available_datasets()
        return
    
    if args.all:
        downloader.download_all(args.force, args.skip_manual)
        downloader.create_sample_configs()
        return
    
    if args.dataset:
        success = downloader.download_dataset(args.dataset, args.force)
        if success:
            print(f"\n✅ 数据集 {args.dataset} 下载成功!")
        else:
            print(f"\n❌ 数据集 {args.dataset} 下载失败!")
        return
    
    # 默认显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()