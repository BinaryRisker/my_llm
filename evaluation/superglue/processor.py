"""
SuperGLUE数据处理器

提供数据预处理、后处理和格式转换功能
"""

from typing import Dict, List, Any, Optional, Tuple
from .data_loader import SuperGLUEDataLoader


class SuperGLUEProcessor:
    """SuperGLUE数据处理器"""
    
    def __init__(self, data_loader: Optional[SuperGLUEDataLoader] = None):
        self.data_loader = data_loader or SuperGLUEDataLoader()
    
    def process_task_data(self, 
                         task_name: str, 
                         split: str = 'validation',
                         data_source: str = 'mock',
                         max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """处理任务数据的完整流程"""
        
        # 1. 加载数据
        raw_data = self.data_loader.load_task_data(
            task_name=task_name,
            split=split, 
            data_source=data_source,
            max_samples=max_samples
        )
        
        # 2. 预处理数据
        processed_data = self.data_loader.preprocess_data(task_name, raw_data)
        
        # 3. 验证数据
        is_valid, errors = self.data_loader.validate_data(task_name, processed_data)
        if not is_valid:
            print(f"数据验证失败: {errors}")
        
        return processed_data