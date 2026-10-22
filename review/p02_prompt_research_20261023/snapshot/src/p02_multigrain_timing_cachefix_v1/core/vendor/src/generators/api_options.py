"""Pinned API choices for new queries; existing experiment configs stay explicit."""
from copy import deepcopy
API_OPTIONS = {1: {'provider': 'openai', 'model': 'gpt-4.1-mini-2025-04-14', 'temperature': 0.0, 'max_output_tokens': 64, 'timeout_seconds': 60.0, 'max_retries': 0}, 2: {'provider': 'dashscope', 'model': 'qwen-plus-2025-12-01', 'base_url': 'https://ws-memqyuz7m8e9oxs1.cn-beijing.maas.aliyuncs.com/compatible-mode/v1', 'temperature': 0.0, 'max_output_tokens': 64, 'timeout_seconds': 60.0, 'max_retries': 0, 'enable_thinking': False}}
DASHSCOPE_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

def api_key_env(provider: str) -> str:
    return 'DASHSCOPE_API_KEY' if provider == 'dashscope' else 'OPENAI_API_KEY'

def get_api_option(option: int) -> dict:
    if isinstance(option, bool) or not isinstance(option, int) or option not in API_OPTIONS:
        raise ValueError('generation.api_option must be 1 (OpenAI) or 2 (Qwen Plus)')
    return deepcopy(API_OPTIONS[option])
