"""baseline 使用的答案生成组件。"""

# 对外暴露生成器和统一结果结构。
from src.generators.llm_generator import GenerationResult, LLMGenerator

__all__ = ["GenerationResult", "LLMGenerator"]
