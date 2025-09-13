"""
LLM项目API服务 - FastAPI实现

提供RESTful API接口，用于模型推理和管理。
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import logging
from pathlib import Path

from config import get_config
from evaluation.evaluation_metrics import EvaluationMetrics
from evaluation.glue_benchmark import GLUEBenchmark

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="LLM从零实现 API",
    description="完整的大语言模型开发与训练平台 - API接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
config = get_config()
if config.get('api', {}).get('cors', {}).get('enabled', True):
    origins = config.get('api', {}).get('cors', {}).get('origins', ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 全局变量
models_cache = {}
metrics_instance = None
glue_instance = None


# 数据模型定义
class TextInput(BaseModel):
    text: str
    max_length: Optional[int] = 512


class ClassificationInput(BaseModel):
    texts: List[str]
    labels: Optional[List[int]] = None


class RegressionInput(BaseModel):
    texts: List[str]
    scores: Optional[List[float]] = None


class EvaluationResult(BaseModel):
    metrics: Dict[str, float]
    task_type: str
    num_samples: int


class ModelInfo(BaseModel):
    name: str
    type: str
    parameters: int
    status: str


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


# 依赖函数
def get_evaluation_metrics():
    """获取评估指标实例"""
    global metrics_instance
    if metrics_instance is None:
        metrics_instance = EvaluationMetrics()
    return metrics_instance


def get_glue_benchmark():
    """获取GLUE基准实例"""
    global glue_instance
    if glue_instance is None:
        glue_instance = GLUEBenchmark()
    return glue_instance


# API端点定义

@app.get("/", response_model=APIResponse)
async def root():
    """API根端点"""
    return APIResponse(
        success=True,
        message="欢迎使用LLM从零实现API！",
        data={
            "version": "1.0.0",
            "docs": "/docs",
            "status": "运行中"
        }
    )


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "健康",
        "timestamp": "2025-01-15",
        "version": "1.0.0",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """列出可用的模型"""
    # 这里模拟返回可用模型列表
    models = [
        ModelInfo(
            name="BERT-base",
            type="classification", 
            parameters=110_000_000,
            status="可用"
        ),
        ModelInfo(
            name="GPT-small",
            type="generation",
            parameters=125_000_000, 
            status="可用"
        ),
        ModelInfo(
            name="Transformer-base",
            type="translation",
            parameters=65_000_000,
            status="可用"
        )
    ]
    return models


@app.post("/evaluate/classification", response_model=EvaluationResult)
async def evaluate_classification(
    input_data: ClassificationInput,
    metrics: EvaluationMetrics = Depends(get_evaluation_metrics)
):
    """评估分类任务"""
    try:
        if input_data.labels is None:
            raise HTTPException(
                status_code=400,
                detail="分类评估需要提供标签"
            )
        
        if len(input_data.texts) != len(input_data.labels):
            raise HTTPException(
                status_code=400,
                detail="文本和标签数量不匹配"
            )
        
        # 模拟预测结果 (实际应该调用真实模型)
        predictions = [0 if len(text.split()) > 5 else 1 for text in input_data.texts]
        
        # 计算评估指标
        results = {
            'predictions': predictions,
            'labels': input_data.labels
        }
        
        computed_metrics = metrics.compute_metrics(results, task_type='classification')
        
        return EvaluationResult(
            metrics=computed_metrics,
            task_type="classification",
            num_samples=len(input_data.texts)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@app.post("/evaluate/regression", response_model=EvaluationResult)
async def evaluate_regression(
    input_data: RegressionInput,
    metrics: EvaluationMetrics = Depends(get_evaluation_metrics)
):
    """评估回归任务"""
    try:
        if input_data.scores is None:
            raise HTTPException(
                status_code=400,
                detail="回归评估需要提供分数"
            )
            
        if len(input_data.texts) != len(input_data.scores):
            raise HTTPException(
                status_code=400,
                detail="文本和分数数量不匹配"
            )
        
        # 模拟预测结果
        predictions = [len(text.split()) / 10.0 for text in input_data.texts]
        
        # 计算评估指标
        results = {
            'predictions': predictions,
            'labels': input_data.scores
        }
        
        computed_metrics = metrics.compute_metrics(results, task_type='regression')
        
        return EvaluationResult(
            metrics=computed_metrics,
            task_type="regression",
            num_samples=len(input_data.texts)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@app.get("/glue/tasks")
async def get_glue_tasks(glue: GLUEBenchmark = Depends(get_glue_benchmark)):
    """获取GLUE任务列表"""
    try:
        tasks = glue.get_task_names()
        task_info = []
        
        for task in tasks:
            task_info.append({
                "name": task,
                "type": glue.get_task_type(task),
                "metrics": glue.get_task_metrics(task),
                "description": f"GLUE {task} 任务"
            })
        
        return APIResponse(
            success=True,
            message="GLUE任务列表获取成功",
            data=task_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@app.post("/predict/text-classification")
async def predict_text_classification(input_data: TextInput):
    """文本分类预测"""
    try:
        # 这里应该加载真实的模型进行预测
        # 现在返回模拟结果
        text_length = len(input_data.text.split())
        
        if text_length > 10:
            prediction = {"label": "POSITIVE", "confidence": 0.85}
        else:
            prediction = {"label": "NEGATIVE", "confidence": 0.72}
        
        return APIResponse(
            success=True,
            message="预测完成",
            data={
                "text": input_data.text,
                "prediction": prediction,
                "text_length": text_length
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict/text-generation")
async def predict_text_generation(input_data: TextInput):
    """文本生成预测"""
    try:
        # 模拟文本生成
        generated_text = input_data.text + " This is a generated continuation of the input text."
        
        return APIResponse(
            success=True,
            message="生成完成",
            data={
                "input_text": input_data.text,
                "generated_text": generated_text,
                "generation_length": len(generated_text.split()) - len(input_data.text.split())
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.get("/stats")
async def get_stats():
    """获取API统计信息"""
    return APIResponse(
        success=True,
        message="统计信息获取成功",
        data={
            "total_models": 3,
            "available_tasks": ["classification", "regression", "generation", "glue"],
            "supported_languages": ["中文", "English"],
            "max_sequence_length": 512,
            "api_version": "1.0.0"
        }
    )


# 异常处理器
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "API端点不存在",
            "data": {"path": str(request.url.path)}
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "内部服务器错误",
            "data": {"error": str(exc)}
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # 从配置中获取服务器设置
    host = config.get('api', {}).get('host', '127.0.0.1')
    port = config.get('api', {}).get('port', 8000)
    debug = config.get('api', {}).get('debug', True)
    
    logger.info(f"启动API服务器: http://{host}:{port}")
    logger.info(f"API文档: http://{host}:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        debug=debug,
        reload=debug
    )