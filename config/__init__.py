"""
统一配置管理系统

提供统一的配置加载和管理功能，支持不同环境的配置切换。
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_cache = {}
    
    def load_config(self, env: str = "development") -> Dict[str, Any]:
        """加载指定环境的配置
        
        Args:
            env: 环境名称 (development, testing, production)
            
        Returns:
            配置字典
        """
        if env in self.config_cache:
            return self.config_cache[env]
        
        config_file = self.config_dir / f"{env}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.config_cache[env] = config
        return config
    
    def get_current_env(self) -> str:
        """获取当前环境"""
        return os.environ.get('LLM_ENV', 'development')
    
    def get_config(self, key: Optional[str] = None, env: Optional[str] = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点分割的嵌套键 (如 'model.bert.hidden_size')
            env: 环境名称，默认使用当前环境
            
        Returns:
            配置值
        """
        if env is None:
            env = self.get_current_env()
        
        config = self.load_config(env)
        
        if key is None:
            return config
        
        # 支持嵌套键访问
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"配置键不存在: {key}")
        
        return value


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config(key: Optional[str] = None, env: Optional[str] = None) -> Any:
    """便捷的配置获取函数"""
    return config_manager.get_config(key, env)


def load_config(env: str = "development") -> Dict[str, Any]:
    """便捷的配置加载函数"""
    return config_manager.load_config(env)