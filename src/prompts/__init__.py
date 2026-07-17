# prompt 包入口：导出固定模板版本和构造函数。
from src.prompts.fixed_prompt import PROMPT_VERSION, build_prompt

__all__ = ["PROMPT_VERSION", "build_prompt"]
