#!/usr/bin/env python3
"""
环境设置脚本
===========

快速设置项目环境，包括：
- 创建必要的目录结构
- 检查依赖包安装情况
- 下载必要的预训练模型和数据
- 配置开发环境

使用方法:
    python scripts/setup_environment.py --full
    python scripts/setup_environment.py --check-only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import urllib.request
import platform
import pkg_resources
from typing import List, Dict, Tuple

# ANSI颜色代码
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_colored(text: str, color: str = Colors.WHITE):
    """打印彩色文本"""
    print(f"{color}{text}{Colors.END}")

def print_header(text: str):
    """打印标题"""
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f" {text}", Colors.BOLD + Colors.CYAN)
    print_colored(f"{'='*60}", Colors.CYAN)

def print_success(text: str):
    """打印成功信息"""
    print_colored(f"✅ {text}", Colors.GREEN)

def print_warning(text: str):
    """打印警告信息"""
    print_colored(f"⚠️  {text}", Colors.YELLOW)

def print_error(text: str):
    """打印错误信息"""
    print_colored(f"❌ {text}", Colors.RED)

def print_info(text: str):
    """打印信息"""
    print_colored(f"ℹ️  {text}", Colors.BLUE)

class EnvironmentSetup:
    """环境设置类"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.required_dirs = [
            'data',
            'data/configs', 
            'experiments',
            'experiments/stage4_transformer',
            'experiments/stage4_transformer/models',
            'experiments/stage4_transformer/logs',
            'experiments/stage4_transformer/results',
            'experiments/stage5_gpt',
            'experiments/stage5_gpt/models',
            'experiments/stage5_gpt/logs',
            'models',
            'training',
            'evaluation',
            'utils',
            'docs',
            'docs/theory',
            'docs/tutorials',
            'docs/api'
        ]
        
        self.core_requirements = [
            'torch',
            'numpy',
            'pandas',
            'matplotlib',
            'tqdm',
            'pyyaml',
            'requests'
        ]
        
        self.optional_requirements = [
            'transformers',
            'tokenizers', 
            'sacrebleu',
            'tensorboard',
            'wandb',
            'rich'
        ]
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print_header("检查Python版本")
        
        version = sys.version_info
        required_version = (3, 8)
        
        if version[:2] >= required_version:
            print_success(f"Python版本: {version.major}.{version.minor}.{version.micro} ✓")
            return True
        else:
            print_error(f"需要Python {required_version[0]}.{required_version[1]}+, 当前版本: {version.major}.{version.minor}.{version.micro}")
            return False
    
    def check_system_info(self):
        """显示系统信息"""
        print_header("系统信息")
        
        print_info(f"操作系统: {platform.system()} {platform.release()}")
        print_info(f"处理器: {platform.processor()}")
        print_info(f"Python版本: {platform.python_version()}")
        print_info(f"项目根目录: {self.project_root.absolute()}")
        
        # 检查GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print_success(f"GPU可用: {gpu_count} 个GPU, 主GPU: {gpu_name}")
            else:
                print_warning("未检测到GPU，将使用CPU训练（速度较慢）")
        except ImportError:
            print_info("PyTorch未安装，跳过GPU检查")
    
    def create_directories(self):
        """创建必要的目录结构"""
        print_header("创建目录结构")
        
        created_count = 0
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                print_success(f"创建目录: {dir_path}")
                created_count += 1
            else:
                print_info(f"目录已存在: {dir_path}")
        
        if created_count > 0:
            print_success(f"成功创建 {created_count} 个目录")
        else:
            print_info("所有必要目录已存在")
    
    def check_requirements(self) -> Tuple[List[str], List[str]]:
        """检查依赖包安装情况"""
        print_header("检查依赖包")
        
        installed_packages = {pkg.project_name.lower(): pkg.version 
                            for pkg in pkg_resources.working_set}
        
        missing_core = []
        missing_optional = []
        
        # 检查核心依赖
        print_info("检查核心依赖包...")
        for package in self.core_requirements:
            if package.lower() in installed_packages:
                version = installed_packages[package.lower()]
                print_success(f"{package}: {version}")
            else:
                print_error(f"{package}: 未安装")
                missing_core.append(package)
        
        # 检查可选依赖
        print_info("\n检查可选依赖包...")
        for package in self.optional_requirements:
            if package.lower() in installed_packages:
                version = installed_packages[package.lower()]
                print_success(f"{package}: {version}")
            else:
                print_warning(f"{package}: 未安装（可选）")
                missing_optional.append(package)
        
        return missing_core, missing_optional
    
    def install_requirements(self, missing_core: List[str], missing_optional: List[str], install_optional: bool = True):
        """安装缺失的依赖包"""
        if not missing_core and (not missing_optional or not install_optional):
            print_success("所有必要的依赖包已安装")
            return
        
        print_header("安装依赖包")
        
        # 安装核心依赖
        if missing_core:
            print_info("安装核心依赖包...")
            try:
                requirements_file = self.project_root / "requirements.txt"
                if requirements_file.exists():
                    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
                    subprocess.run(cmd, check=True)
                    print_success("从requirements.txt安装依赖包完成")
                else:
                    # 单独安装核心包
                    for package in missing_core:
                        cmd = [sys.executable, "-m", "pip", "install", package]
                        subprocess.run(cmd, check=True)
                        print_success(f"安装 {package} 完成")
            except subprocess.CalledProcessError as e:
                print_error(f"安装依赖包失败: {e}")
                return False
        
        return True
    
    def create_config_files(self):
        """创建配置文件"""
        print_header("创建配置文件")
        
        # 创建数据集配置
        configs_dir = self.project_root / "data" / "configs"
        
        # Stage 4 Transformer配置
        stage4_config = {
            "datasets": ["multi30k", "wmt_en_fr"],
            "task": "translation",
            "description": "Transformer翻译数据",
            "default_dataset": "multi30k"
        }
        
        config_file = configs_dir / "stage4_transformer_datasets.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(stage4_config, f, indent=2, ensure_ascii=False)
        print_success(f"创建配置文件: {config_file.name}")
        
        # Stage 5 GPT配置
        stage5_config = {
            "datasets": ["wikitext_103", "openwebtext"],
            "task": "language_modeling", 
            "description": "GPT预训练数据",
            "default_dataset": "wikitext_103"
        }
        
        config_file = configs_dir / "stage5_gpt_datasets.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(stage5_config, f, indent=2, ensure_ascii=False)
        print_success(f"创建配置文件: {config_file.name}")
    
    def check_gpu_setup(self):
        """检查GPU设置"""
        print_header("GPU环境检查")
        
        try:
            import torch
            
            print_info(f"PyTorch版本: {torch.__version__}")
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print_success(f"CUDA可用，检测到 {gpu_count} 个GPU")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print_info(f"GPU {i}: {gpu_name}, 内存: {gpu_memory:.1f} GB")
                
                # 测试GPU计算
                try:
                    x = torch.randn(1000, 1000).cuda()
                    y = torch.matmul(x, x)
                    print_success("GPU计算测试通过")
                except Exception as e:
                    print_warning(f"GPU计算测试失败: {e}")
            else:
                print_warning("CUDA不可用，将使用CPU进行训练")
                print_info("如需GPU加速，请安装CUDA版本的PyTorch")
                
        except ImportError:
            print_error("PyTorch未安装，无法进行GPU检查")
    
    def download_sample_data(self):
        """下载示例数据"""
        print_header("准备示例数据")
        
        # 这里可以下载一些小的示例数据集用于快速测试
        print_info("示例数据下载功能已预留")
        print_info("请使用 python scripts/download_datasets.py 下载完整数据集")
    
    def create_gitignore(self):
        """创建.gitignore文件"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
*.ckpt

# Data
data/*/
!data/configs/
*.csv
*.json
*.txt
*.tsv

# Logs
*.log
logs/
runs/
experiments/*/logs/
experiments/*/models/
experiments/*/results/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.venv
venv/
ENV/

# Temporary
tmp/
temp/
"""
        
        gitignore_path = self.project_root / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(gitignore_content)
            print_success("创建 .gitignore 文件")
        else:
            print_info(".gitignore 文件已存在")
    
    def run_setup(self, full_setup: bool = True, check_only: bool = False):
        """运行完整设置流程"""
        print_colored("""
🚀 大语言模型学习项目 - 环境设置
================================
        """, Colors.BOLD + Colors.CYAN)
        
        # 1. 检查Python版本
        if not self.check_python_version():
            return False
        
        # 2. 显示系统信息
        self.check_system_info()
        
        if check_only:
            # 3. 只检查依赖包
            missing_core, missing_optional = self.check_requirements()
            
            if missing_core:
                print_error(f"缺少核心依赖包: {', '.join(missing_core)}")
                print_info("请运行: python scripts/setup_environment.py --full")
                return False
            
            if missing_optional:
                print_warning(f"缺少可选依赖包: {', '.join(missing_optional)}")
                print_info("建议安装以获得完整功能")
            
            print_success("环境检查完成")
            return True
        
        if full_setup:
            # 3. 创建目录结构
            self.create_directories()
            
            # 4. 检查并安装依赖
            missing_core, missing_optional = self.check_requirements()
            self.install_requirements(missing_core, missing_optional)
            
            # 5. 创建配置文件
            self.create_config_files()
            
            # 6. 检查GPU
            self.check_gpu_setup()
            
            # 7. 创建.gitignore
            self.create_gitignore()
            
            # 8. 准备示例数据
            self.download_sample_data()
        
        print_colored("""
✅ 环境设置完成！

🎯 下一步建议:
1. 下载数据集: python scripts/download_datasets.py --list
2. 训练第一个模型: python training/stage4_transformer/train_transformer.py --help  
3. 查看项目文档: 浏览 docs/ 目录

祝学习愉快！🚀
        """, Colors.GREEN + Colors.BOLD)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='设置项目环境')
    parser.add_argument('--full', action='store_true', 
                       help='执行完整环境设置')
    parser.add_argument('--check-only', action='store_true',
                       help='只检查环境，不进行安装')
    parser.add_argument('--project-root', type=Path,
                       help='项目根目录路径')
    
    args = parser.parse_args()
    
    # 如果没有指定参数，默认执行完整设置
    if not args.full and not args.check_only:
        args.full = True
    
    setup = EnvironmentSetup(args.project_root)
    
    try:
        success = setup.run_setup(
            full_setup=args.full,
            check_only=args.check_only
        )
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print_warning("\n设置被用户中断")
        sys.exit(1)
    except Exception as e:
        print_error(f"设置过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()