import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

KEY_MAP = {
    "llm": "DASHSCOPE_API_KEY_LLM",
    "vlm": "DASHSCOPE_API_KEY_VLM",
    "embedding": "DASHSCOPE_API_KEY_EMBEDDING",
    "rerank": "DASHSCOPE_API_KEY_RERANK",
    "summary": "DASHSCOPE_API_KEY_SUMMARY",
    "qa": "DASHSCOPE_API_KEY_QA",
}

MODEL_MAP = {
    "llm": "qwen3-vl-plus",
    "vlm": "qwen3-vl-plus",
    "embedding": "text-embedding-v4",
    "rerank": "qwen3-vl-plus",
    "summary": "qwen3-vl-plus",
    "qa": "qwen3-vl-plus",
}


def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_base_url():
    return os.getenv("DASHSCOPE_BASE_URL", BASE_URL)


def get_api_key(purpose: str):
    env_name = KEY_MAP.get(purpose)
    key = os.getenv(env_name, "") if env_name else ""
    if not key:
        key = os.getenv("DASHSCOPE_API_KEY", "")
    return key


def get_model_name(purpose: str):
    env_name = f"DASHSCOPE_MODEL_{purpose.upper()}"
    return os.getenv(env_name, MODEL_MAP.get(purpose, "qwen3-vl-plus"))


def get_embedding_dimension():
    return int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSION", "1024"))


def get_embedding_max_batch_size():
    return int(os.getenv("DASHSCOPE_EMBEDDING_MAX_BATCH_SIZE", "10"))


def get_openai_client(purpose: str):
    from openai import OpenAI

    return OpenAI(
        api_key=get_api_key(purpose),
        base_url=get_base_url(),
    )


def set_model_cache_env():
    cache_root = PROJECT_ROOT / "model_cache"
    hf_cache = cache_root / "hf"
    torch_cache = cache_root / "torch"
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("TORCH_HOME", str(torch_cache))
    os.environ.setdefault("DOCLING_CACHE_DIR", str(cache_root / "docling"))


load_env()
