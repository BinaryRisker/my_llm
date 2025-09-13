"""
网格搜索超参数优化
================

实现网格搜索算法进行超参数优化：
- 穷举所有超参数组合
- 支持多种评估指标
- 并行执行和结果缓存
- 详细的搜索报告

使用方法:
    from utils.hyperparameter_optimization.grid_search import GridSearchOptimizer
    
    optimizer = GridSearchOptimizer()
    best_params = optimizer.search(param_grid, objective_function)
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from itertools import product
import concurrent.futures
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    best_params: Dict[str, Any]
    best_score: float
    best_std: float
    cv_results: List[Dict[str, Any]]
    search_time: float
    total_trials: int
    successful_trials: int
    failed_trials: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class TrialResult:
    """单次试验结果"""
    params: Dict[str, Any]
    score: float
    std: float
    execution_time: float
    status: str  # 'success', 'failed', 'timeout'
    error_message: Optional[str] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class GridSearchOptimizer:
    """网格搜索优化器"""
    
    def __init__(self,
                 scoring: str = 'accuracy',
                 cv: int = 3,
                 n_jobs: int = 1,
                 verbose: bool = True,
                 cache_results: bool = True,
                 timeout: Optional[float] = None):
        """
        初始化网格搜索优化器
        
        Args:
            scoring: 评分方式
            cv: 交叉验证折数
            n_jobs: 并行任务数
            verbose: 是否显示详细信息
            cache_results: 是否缓存结果
            timeout: 单次试验超时时间
        """
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cache_results = cache_results
        self.timeout = timeout
        
        # 结果缓存
        self.results_cache = {}
        self.trial_history = []
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        if not param_grid:
            return [{}]
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = []
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _params_to_key(self, params: Dict[str, Any]) -> str:
        """将参数字典转换为缓存键"""
        sorted_items = sorted(params.items())
        return json.dumps(sorted_items, sort_keys=True, default=str)
    
    def _evaluate_single_trial(self, 
                              params: Dict[str, Any], 
                              objective_function: Callable,
                              trial_id: int) -> TrialResult:
        """评估单次试验"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._params_to_key(params)
        if self.cache_results and cache_key in self.results_cache:
            cached_result = self.results_cache[cache_key]
            if self.verbose:
                logger.info(f"试验 {trial_id}: 使用缓存结果 - {cached_result.score:.4f}")
            return cached_result
        
        try:
            if self.verbose:
                logger.info(f"试验 {trial_id}: 参数 {params}")
            
            # 执行目标函数
            if self.timeout:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(objective_function, params)
                    result = future.result(timeout=self.timeout)
            else:
                result = objective_function(params)
            
            execution_time = time.time() - start_time
            
            # 处理返回结果
            if isinstance(result, (int, float)):
                score = float(result)
                std = 0.0
                additional_metrics = None
            elif isinstance(result, dict):
                score = float(result.get('score', result.get('accuracy', 0.0)))
                std = float(result.get('std', 0.0))
                additional_metrics = {k: v for k, v in result.items() 
                                    if k not in ['score', 'accuracy', 'std']}
            elif isinstance(result, (list, tuple)) and len(result) >= 2:
                score = float(result[0])
                std = float(result[1])
                additional_metrics = result[2] if len(result) > 2 else None
            else:
                raise ValueError(f"不支持的返回结果格式: {type(result)}")
            
            trial_result = TrialResult(
                params=params,
                score=score,
                std=std,
                execution_time=execution_time,
                status='success',
                additional_metrics=additional_metrics
            )
            
            # 缓存结果
            if self.cache_results:
                self.results_cache[cache_key] = trial_result
            
            if self.verbose:
                logger.info(f"试验 {trial_id}: 得分 {score:.4f} (±{std:.4f}), 用时 {execution_time:.2f}s")
            
            return trial_result
            
        except concurrent.futures.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"试验超时 ({self.timeout}s)"
            logger.warning(f"试验 {trial_id}: {error_msg}")
            
            return TrialResult(
                params=params,
                score=float('-inf'),
                std=0.0,
                execution_time=execution_time,
                status='timeout',
                error_message=error_msg
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"试验失败: {str(e)}"
            logger.error(f"试验 {trial_id}: {error_msg}")
            
            return TrialResult(
                params=params,
                score=float('-inf'),
                std=0.0,
                execution_time=execution_time,
                status='failed',
                error_message=error_msg
            )
    
    def search(self, 
               param_grid: Dict[str, List[Any]],
               objective_function: Callable,
               maximize: bool = True) -> OptimizationResult:
        """
        执行网格搜索
        
        Args:
            param_grid: 参数网格
            objective_function: 目标函数
            maximize: 是否最大化目标函数
            
        Returns:
            优化结果
        """
        logger.info("开始网格搜索优化...")
        start_time = time.time()
        
        # 生成所有参数组合
        param_combinations = self._generate_param_combinations(param_grid)
        total_trials = len(param_combinations)
        
        logger.info(f"总共需要评估 {total_trials} 个参数组合")
        
        # 执行搜索
        trial_results = []
        
        if self.n_jobs == 1:
            # 串行执行
            for i, params in enumerate(param_combinations):
                result = self._evaluate_single_trial(params, objective_function, i + 1)
                trial_results.append(result)
                self.trial_history.append(result)
        else:
            # 并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(self._evaluate_single_trial, params, objective_function, i + 1): i
                    for i, params in enumerate(param_combinations)
                }
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    trial_results.append(result)
                    self.trial_history.append(result)
        
        # 分析结果
        successful_results = [r for r in trial_results if r.status == 'success']
        failed_trials = len([r for r in trial_results if r.status == 'failed'])
        timeout_trials = len([r for r in trial_results if r.status == 'timeout'])
        
        if not successful_results:
            raise ValueError("所有试验都失败了，无法找到最佳参数")
        
        # 找到最佳结果
        if maximize:
            best_result = max(successful_results, key=lambda x: x.score)
        else:
            best_result = min(successful_results, key=lambda x: x.score)
        
        # 准备CV结果
        cv_results = []
        for result in trial_results:
            cv_result = {
                'params': result.params,
                'mean_test_score': result.score,
                'std_test_score': result.std,
                'rank_test_score': 0,  # 稍后填充
                'execution_time': result.execution_time,
                'status': result.status
            }
            if result.additional_metrics:
                cv_result.update(result.additional_metrics)
            cv_results.append(cv_result)
        
        # 计算排名
        successful_cv_results = [r for r in cv_results if r['status'] == 'success']
        if maximize:
            successful_cv_results.sort(key=lambda x: x['mean_test_score'], reverse=True)
        else:
            successful_cv_results.sort(key=lambda x: x['mean_test_score'])
        
        # 填充排名
        for i, result in enumerate(successful_cv_results):
            result['rank_test_score'] = i + 1
        
        search_time = time.time() - start_time
        
        # 创建优化结果
        optimization_result = OptimizationResult(
            best_params=best_result.params,
            best_score=best_result.score,
            best_std=best_result.std,
            cv_results=cv_results,
            search_time=search_time,
            total_trials=total_trials,
            successful_trials=len(successful_results),
            failed_trials=failed_trials + timeout_trials
        )
        
        logger.info(f"网格搜索完成!")
        logger.info(f"最佳参数: {best_result.params}")
        logger.info(f"最佳得分: {best_result.score:.4f} (±{best_result.std:.4f})")
        logger.info(f"总用时: {search_time:.2f}s")
        logger.info(f"成功/失败/总数: {len(successful_results)}/{failed_trials + timeout_trials}/{total_trials}")
        
        return optimization_result
    
    def get_top_results(self, n: int = 5, maximize: bool = True) -> List[TrialResult]:
        """获取前N个最佳结果"""
        successful_results = [r for r in self.trial_history if r.status == 'success']
        
        if maximize:
            sorted_results = sorted(successful_results, key=lambda x: x.score, reverse=True)
        else:
            sorted_results = sorted(successful_results, key=lambda x: x.score)
        
        return sorted_results[:n]
    
    def save_results(self, filepath: Union[str, Path], 
                    optimization_result: OptimizationResult) -> None:
        """保存搜索结果"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(optimization_result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"搜索结果已保存到: {filepath}")
    
    def load_results(self, filepath: Union[str, Path]) -> OptimizationResult:
        """加载搜索结果"""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return OptimizationResult(**data)
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.results_cache.clear()
        self.trial_history.clear()
        logger.info("缓存已清空")
    
    def get_search_summary(self) -> Dict[str, Any]:
        """获取搜索摘要统计"""
        if not self.trial_history:
            return {"message": "尚未执行搜索"}
        
        successful_results = [r for r in self.trial_history if r.status == 'success']
        failed_results = [r for r in self.trial_history if r.status != 'success']
        
        if successful_results:
            scores = [r.score for r in successful_results]
            execution_times = [r.execution_time for r in successful_results]
            
            summary = {
                'total_trials': len(self.trial_history),
                'successful_trials': len(successful_results),
                'failed_trials': len(failed_results),
                'success_rate': len(successful_results) / len(self.trial_history),
                'best_score': max(scores),
                'worst_score': min(scores),
                'mean_score': sum(scores) / len(scores),
                'score_std': (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
                'total_time': sum(execution_times),
                'mean_time_per_trial': sum(execution_times) / len(execution_times),
                'max_time_per_trial': max(execution_times),
                'min_time_per_trial': min(execution_times)
            }
        else:
            summary = {
                'total_trials': len(self.trial_history),
                'successful_trials': 0,
                'failed_trials': len(failed_results),
                'success_rate': 0.0,
                'message': '所有试验都失败了'
            }
        
        return summary


def demo():
    """演示网格搜索的使用"""
    print("🚀 网格搜索超参数优化演示")
    print("=" * 50)
    
    # 定义一个模拟的目标函数
    def objective_function(params):
        """模拟的目标函数"""
        import random
        import time
        
        # 模拟计算时间
        time.sleep(random.uniform(0.1, 0.5))
        
        # 模拟一些参数的影响
        lr = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 32)
        hidden_size = params.get('hidden_size', 128)
        
        # 人工设计的函数，有一个最优点
        score = (
            0.9 - abs(lr - 0.001) * 100 +  # 最佳lr是0.001
            0.1 - abs(batch_size - 64) * 0.001 +  # 最佳batch_size是64
            0.1 - abs(hidden_size - 256) * 0.0001  # 最佳hidden_size是256
        )
        
        # 添加一些随机噪声
        noise = random.uniform(-0.05, 0.05)
        score += noise
        
        # 模拟交叉验证的标准差
        std = random.uniform(0.01, 0.03)
        
        # 返回额外指标
        additional_metrics = {
            'train_accuracy': score + random.uniform(-0.02, 0.02),
            'val_loss': 1.0 - score + random.uniform(-0.1, 0.1)
        }
        
        return {
            'score': max(0, score),  # 确保分数非负
            'std': std,
            'train_accuracy': additional_metrics['train_accuracy'],
            'val_loss': additional_metrics['val_loss']
        }
    
    # 定义搜索空间
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64, 128],
        'hidden_size': [64, 128, 256, 512]
    }
    
    print("📋 搜索空间:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"  总组合数: {total_combinations}")
    
    # 创建网格搜索优化器
    optimizer = GridSearchOptimizer(
        scoring='accuracy',
        cv=3,
        n_jobs=2,  # 使用2个并行任务
        verbose=True,
        cache_results=True,
        timeout=10.0  # 10秒超时
    )
    
    print(f"\n🔍 开始网格搜索...")
    
    # 执行搜索
    result = optimizer.search(
        param_grid=param_grid,
        objective_function=objective_function,
        maximize=True
    )
    
    print(f"\n📊 搜索结果:")
    print(f"  最佳参数: {result.best_params}")
    print(f"  最佳得分: {result.best_score:.4f} (±{result.best_std:.4f})")
    print(f"  搜索时间: {result.search_time:.2f}秒")
    print(f"  成功试验: {result.successful_trials}/{result.total_trials}")
    
    # 显示前5个最佳结果
    print(f"\n🏆 前5个最佳结果:")
    top_results = optimizer.get_top_results(n=5, maximize=True)
    for i, trial in enumerate(top_results):
        print(f"  {i+1}. 得分: {trial.score:.4f}, 参数: {trial.params}")
    
    # 搜索摘要
    print(f"\n📈 搜索摘要:")
    summary = optimizer.get_search_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 保存结果
    save_path = "temp_grid_search_results.json"
    optimizer.save_results(save_path, result)
    print(f"\n💾 结果已保存到: {save_path}")
    
    # 测试加载结果
    loaded_result = optimizer.load_results(save_path)
    print(f"🔄 加载测试 - 最佳得分: {loaded_result.best_score:.4f}")
    
    print(f"\n🎉 演示完成!")


if __name__ == "__main__":
    demo()