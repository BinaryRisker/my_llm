"""
LLM项目测试套件

提供完整的单元测试和集成测试，确保代码质量和功能正确性。

测试结构:
- unit/: 单元测试
- integration/: 集成测试  
- fixtures/: 测试数据和夹具
- conftest.py: 共享配置和夹具
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 测试配置
TEST_CONFIG = {
    'test_data_dir': PROJECT_ROOT / 'test_data',
    'test_outputs_dir': PROJECT_ROOT / 'test_outputs',
    'fixtures_dir': Path(__file__).parent / 'fixtures',
    'temp_dir': PROJECT_ROOT / 'temp_test',
}

# 创建测试目录
for dir_path in TEST_CONFIG.values():
    if isinstance(dir_path, Path):
        dir_path.mkdir(exist_ok=True)