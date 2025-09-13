"""
配置系统单元测试
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path

from config import ConfigManager, get_config, load_config


@pytest.mark.unit
class TestConfigManager:
    """测试配置管理器"""
    
    def test_init(self):
        """测试配置管理器初始化"""
        config_manager = ConfigManager("test_config")
        assert config_manager.config_dir == Path("test_config")
        assert config_manager.config_cache == {}
    
    def test_get_current_env(self, monkeypatch):
        """测试获取当前环境"""
        config_manager = ConfigManager()
        
        # 默认环境
        assert config_manager.get_current_env() == "development"
        
        # 设置环境变量
        monkeypatch.setenv("LLM_ENV", "testing")
        assert config_manager.get_current_env() == "testing"
    
    def test_load_config_file_not_found(self, temp_dir):
        """测试加载不存在的配置文件"""
        config_manager = ConfigManager(str(temp_dir))
        
        with pytest.raises(FileNotFoundError):
            config_manager.load_config("nonexistent")
    
    def test_load_config_success(self, temp_dir):
        """测试成功加载配置文件"""
        # 创建测试配置文件
        config_data = {
            'project': {'name': 'Test Project'},
            'environment': {'name': 'testing'}
        }
        
        config_file = temp_dir / "testing.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(str(temp_dir))
        loaded_config = config_manager.load_config("testing")
        
        assert loaded_config == config_data
        assert "testing" in config_manager.config_cache
    
    def test_get_config_full(self, temp_dir):
        """测试获取完整配置"""
        config_data = {'test': 'value'}
        
        config_file = temp_dir / "development.yaml"  
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(str(temp_dir))
        result = config_manager.get_config()
        
        assert result == config_data
    
    def test_get_config_nested_key(self, temp_dir):
        """测试获取嵌套键的配置值"""
        config_data = {
            'model': {
                'bert': {
                    'hidden_size': 768
                }
            }
        }
        
        config_file = temp_dir / "development.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(str(temp_dir))
        
        # 测试多级嵌套
        assert config_manager.get_config('model.bert.hidden_size') == 768
        assert config_manager.get_config('model.bert') == {'hidden_size': 768}
        assert config_manager.get_config('model') == {'bert': {'hidden_size': 768}}
    
    def test_get_config_key_not_found(self, temp_dir):
        """测试获取不存在的配置键"""
        config_data = {'existing_key': 'value'}
        
        config_file = temp_dir / "development.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(str(temp_dir))
        
        with pytest.raises(KeyError):
            config_manager.get_config('nonexistent_key')
    
    def test_config_caching(self, temp_dir):
        """测试配置缓存功能"""
        config_data = {'test': 'cached_value'}
        
        config_file = temp_dir / "caching_test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(str(temp_dir))
        
        # 第一次加载
        config1 = config_manager.load_config("caching_test")
        assert "caching_test" in config_manager.config_cache
        
        # 修改文件
        new_data = {'test': 'modified_value'}
        with open(config_file, 'w') as f:
            yaml.dump(new_data, f)
        
        # 第二次加载应该返回缓存的值
        config2 = config_manager.load_config("caching_test")
        assert config1 == config2 == config_data


@pytest.mark.unit
class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_get_config_function(self, monkeypatch):
        """测试get_config便捷函数"""
        # 模拟配置管理器
        mock_config = {'test': 'value'}
        
        class MockConfigManager:
            def get_config(self, key=None, env=None):
                if key == 'test':
                    return mock_config[key]
                return mock_config
        
        # 替换全局配置管理器
        import config
        original_manager = config.config_manager
        config.config_manager = MockConfigManager()
        
        try:
            assert get_config('test') == 'value'
            assert get_config() == mock_config
        finally:
            config.config_manager = original_manager
    
    def test_load_config_function(self, monkeypatch):
        """测试load_config便捷函数"""
        mock_config = {'environment': 'test'}
        
        class MockConfigManager:
            def load_config(self, env="development"):
                return mock_config
        
        import config
        original_manager = config.config_manager
        config.config_manager = MockConfigManager()
        
        try:
            assert load_config() == mock_config
            assert load_config("production") == mock_config
        finally:
            config.config_manager = original_manager


@pytest.mark.unit  
def test_yaml_parsing():
    """测试YAML解析功能"""
    yaml_content = """
    project:
      name: Test Project
      version: 1.0.0
    
    training:
      batch_size: 32
      learning_rate: 0.001
      
    paths:
      - /path/one
      - /path/two
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_file = f.name
    
    try:
        with open(temp_file, 'r') as f:
            data = yaml.safe_load(f)
        
        assert data['project']['name'] == 'Test Project'
        assert data['training']['batch_size'] == 32
        assert len(data['paths']) == 2
    finally:
        os.unlink(temp_file)


@pytest.mark.unit
def test_environment_integration(monkeypatch, temp_dir):
    """测试环境变量集成"""
    # 创建测试配置
    config_data = {
        'environment': {'name': 'test_env'},
        'debug': True
    }
    
    config_file = temp_dir / "test_env.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    # 设置环境变量
    monkeypatch.setenv("LLM_ENV", "test_env")
    
    config_manager = ConfigManager(str(temp_dir))
    
    # 应该自动使用环境变量指定的配置
    current_config = config_manager.get_config()
    assert current_config['environment']['name'] == 'test_env'