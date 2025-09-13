"""
SuperGLUE任务实现

实现所有9个SuperGLUE任务的数据处理和文本转换功能：
- 将各种NLP任务统一转换为Text-to-Text格式
- 支持T5模型的训练和评估
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import json
import re


class SuperGLUETask(ABC):
    """SuperGLUE任务基类"""
    
    def __init__(self, task_name: str, task_type: str):
        self.task_name = task_name
        self.task_type = task_type  # 'classification' or 'generation'
    
    @abstractmethod
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        """将样本转换为Text-to-Text格式"""
        pass
    
    @abstractmethod
    def extract_answer(self, generated_text: str) -> Any:
        """从生成的文本中提取答案"""
        pass
    
    def get_template(self) -> str:
        """获取任务模板"""
        return getattr(self, 'template', '')


class BoolQTask(SuperGLUETask):
    """BoolQ: Boolean Questions"""
    
    def __init__(self):
        super().__init__('boolq', 'classification')
        self.template = "Answer the following question with True or False.\nQuestion: {question}\nPassage: {passage}\nAnswer:"
    
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        input_text = self.template.format(
            question=example['question'],
            passage=example['passage']
        )
        target_text = "True" if example.get('label', 0) == 1 else "False"
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def extract_answer(self, generated_text: str) -> int:
        """提取布尔答案"""
        text = generated_text.strip().lower()
        if 'true' in text:
            return 1
        elif 'false' in text:
            return 0
        else:
            # 默认返回False
            return 0


class CBTask(SuperGLUETask):
    """CB: CommitmentBank"""
    
    def __init__(self):
        super().__init__('cb', 'classification')
        self.template = "Given the premise, determine if the hypothesis is entailed, contradicted, or neutral.\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
        self.label_map = {0: "entailment", 1: "contradiction", 2: "neutral"}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        input_text = self.template.format(
            premise=example['premise'],
            hypothesis=example['hypothesis']
        )
        target_text = self.label_map[example.get('label', 2)]
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def extract_answer(self, generated_text: str) -> int:
        """提取蕴含关系答案"""
        text = generated_text.strip().lower()
        for label_text, label_id in self.reverse_label_map.items():
            if label_text in text:
                return label_id
        return 2  # 默认返回neutral


class COPATask(SuperGLUETask):
    """COPA: Choice of Plausible Alternatives"""
    
    def __init__(self):
        super().__init__('copa', 'classification')
        self.template = "Choose the more plausible alternative.\nPremise: {premise}\nQuestion: What was the {question}?\nChoice 1: {choice1}\nChoice 2: {choice2}\nAnswer:"
    
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        question_word = "cause" if example['question'] == "cause" else "effect"
        
        input_text = self.template.format(
            premise=example['premise'],
            question=question_word,
            choice1=example['choice1'],
            choice2=example['choice2']
        )
        target_text = f"Choice {example.get('label', 0) + 1}"
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def extract_answer(self, generated_text: str) -> int:
        """提取选择答案"""
        text = generated_text.strip().lower()
        if 'choice 1' in text or '1' in text:
            return 0
        elif 'choice 2' in text or '2' in text:
            return 1
        else:
            return 0  # 默认选择1


class MultiRCTask(SuperGLUETask):
    """MultiRC: Multi-Sentence Reading Comprehension"""
    
    def __init__(self):
        super().__init__('multirc', 'classification')
        self.template = "Answer the question based on the passage. Answer with True or False.\nPassage: {passage}\nQuestion: {question}\nAnswer option: {answer}\nIs this answer correct?"
    
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        input_text = self.template.format(
            passage=example['passage']['text'],
            question=example['question'],
            answer=example['answer']
        )
        target_text = "True" if example.get('label', 0) == 1 else "False"
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def extract_answer(self, generated_text: str) -> int:
        """提取布尔答案"""
        text = generated_text.strip().lower()
        if 'true' in text:
            return 1
        elif 'false' in text:
            return 0
        else:
            return 0


class ReCoRDTask(SuperGLUETask):
    """ReCoRD: Reading Comprehension with Commonsense Reasoning Dataset"""
    
    def __init__(self):
        super().__init__('record', 'generation')
        self.template = "Fill in the blank with the correct entity from the passage.\nPassage: {passage}\nQuery: {query}\nAnswer:"
    
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        # 处理passage中的@highlight
        passage = example['passage']['text'].replace('@highlight\n', '')
        
        input_text = self.template.format(
            passage=passage,
            query=example['query']
        )
        
        # 目标答案通常是第一个正确答案
        answers = example.get('answers', [])
        target_text = answers[0] if answers else ""
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def extract_answer(self, generated_text: str) -> str:
        """提取生成的答案"""
        return generated_text.strip()


class RTETask(SuperGLUETask):
    """RTE: Recognizing Textual Entailment"""
    
    def __init__(self):
        super().__init__('rte', 'classification')
        self.template = "Does the premise entail the hypothesis? Answer with entailment or not_entailment.\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
        self.label_map = {0: "entailment", 1: "not_entailment"}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        input_text = self.template.format(
            premise=example['premise'],
            hypothesis=example['hypothesis']
        )
        target_text = self.label_map[example.get('label', 1)]
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def extract_answer(self, generated_text: str) -> int:
        """提取蕴含答案"""
        text = generated_text.strip().lower()
        if 'entailment' in text and 'not_entailment' not in text:
            return 0
        else:
            return 1


class WiCTask(SuperGLUETask):
    """WiC: Words in Context"""
    
    def __init__(self):
        super().__init__('wic', 'classification')
        self.template = "Determine if the word '{word}' has the same meaning in both sentences.\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nAnswer with True or False:"
    
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        input_text = self.template.format(
            word=example['word'],
            sentence1=example['sentence1'],
            sentence2=example['sentence2']
        )
        target_text = "True" if example.get('label', 0) == 1 else "False"
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def extract_answer(self, generated_text: str) -> int:
        """提取布尔答案"""
        text = generated_text.strip().lower()
        if 'true' in text:
            return 1
        elif 'false' in text:
            return 0
        else:
            return 0


class WSCTask(SuperGLUETask):
    """WSC: Winograd Schema Challenge"""
    
    def __init__(self):
        super().__init__('wsc', 'classification')
        self.template = "In the sentence '{text}', does the pronoun '{span2_text}' refer to '{span1_text}'? Answer with True or False:"
    
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        input_text = self.template.format(
            text=example['text'],
            span1_text=example['span1_text'],
            span2_text=example['span2_text']
        )
        target_text = "True" if example.get('label', 0) == 1 else "False"
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def extract_answer(self, generated_text: str) -> int:
        """提取布尔答案"""
        text = generated_text.strip().lower()
        if 'true' in text:
            return 1
        elif 'false' in text:
            return 0
        else:
            return 0


class AXbTask(SuperGLUETask):
    """AX-b: Broad Coverage Diagnostic"""
    
    def __init__(self):
        super().__init__('axb', 'classification')
        self.template = "Given the premise, determine if the hypothesis is entailed, contradicted, or neutral.\nPremise: {sentence1}\nHypothesis: {sentence2}\nAnswer:"
        self.label_map = {0: "entailment", 1: "not_entailment", 2: "neutral"}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def convert_to_text2text(self, example: Dict[str, Any]) -> Dict[str, str]:
        input_text = self.template.format(
            sentence1=example['sentence1'],
            sentence2=example['sentence2']
        )
        target_text = self.label_map[example.get('label', 2)]
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def extract_answer(self, generated_text: str) -> int:
        """提取蕴含关系答案"""
        text = generated_text.strip().lower()
        for label_text, label_id in self.reverse_label_map.items():
            if label_text in text:
                return label_id
        return 2  # 默认返回neutral


# 任务工厂
def get_task(task_name: str) -> SuperGLUETask:
    """获取指定的任务处理器"""
    task_map = {
        'boolq': BoolQTask,
        'cb': CBTask,
        'copa': COPATask,
        'multirc': MultiRCTask,
        'record': ReCoRDTask,
        'rte': RTETask,
        'wic': WiCTask,
        'wsc': WSCTask,
        'axb': AXbTask
    }
    
    task_name = task_name.lower()
    if task_name not in task_map:
        raise ValueError(f"未知的任务: {task_name}")
    
    return task_map[task_name]()


def convert_examples_to_text2text(task_name: str, examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """批量转换样本为Text-to-Text格式"""
    task = get_task(task_name)
    converted_examples = []
    
    for example in examples:
        try:
            converted = task.convert_to_text2text(example)
            converted_examples.append(converted)
        except Exception as e:
            print(f"转换样本失败 {task_name}: {e}")
            continue
    
    return converted_examples


def extract_predictions(task_name: str, generated_texts: List[str]) -> List[Any]:
    """批量提取预测结果"""
    task = get_task(task_name)
    predictions = []
    
    for text in generated_texts:
        try:
            prediction = task.extract_answer(text)
            predictions.append(prediction)
        except Exception as e:
            print(f"提取预测失败 {task_name}: {e}")
            # 根据任务类型提供默认值
            if task.task_type == 'classification':
                predictions.append(0)
            else:
                predictions.append("")
    
    return predictions