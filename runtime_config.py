import json
import os
from typing import Dict

from paths import PROJECT_ROOT, SYSTEM_CONFIG_PATH


DEFAULT_CONFIG: Dict[str, str] = {
    # 第三方 Claude 网关：可填 https://vip.aipro.love 或 https://vip.aipro.love/v1（见 anthropic_messages_endpoint）
    "OPENAI_BASE_URL": "https://vip.aipro.love/v1",
    "CHAT_MODEL": "claude-sonnet-4-6",
    "REASONING_MODEL": "claude-sonnet-4-6",
    "CHEAP_MODEL": "",
    "AUTOALPHA_DEFAULT_ROUNDS": "0",
    "AUTOALPHA_DEFAULT_IDEAS": "4",
    "AUTOALPHA_DEFAULT_DAYS": "0",
    "AUTOALPHA_DEFAULT_TARGET_VALID": "10",
    "AUTOALPHA_PROMPT_CONTEXT_LIMIT": "6",
    "AUTOALPHA_QUOTA_DISPLAY_FX": "7.3",
    "AUTOALPHA_MIN_FULL_EVAL_DAYS": "700",
    "AUTOALPHA_SCREEN_DAYS": "160",
    "AUTOALPHA_SCREEN_MIN_IC": "0.12",
    "AUTOALPHA_SCREEN_MIN_IR": "0.6",
    "AUTOALPHA_SCREEN_MAX_TVR": "420",
    "AUTOALPHA_RESEARCH_MIN_IC": "0.3",
    "AUTOALPHA_RESEARCH_MIN_IR": "1.2",
    "AUTOALPHA_IDEA_PAUSE_MS": "250",
    "AUTOALPHA_ROUND_PAUSE_SEC": "1",
    "AUTOALPHA_ROLLING_TARGET_VALID": "100",
    "AUTOALPHA_ROLLING_TRAIN_DAYS": "126",
    "AUTOALPHA_ROLLING_TEST_DAYS": "126",
    "AUTOALPHA_ROLLING_STEP_DAYS": "126",
}


def anthropic_messages_endpoint(api_base: str) -> str:
    """
    构建 Anthropic /messages 完整 URL。
    - Base 为 https://host → https://host/v1/messages
    - Base 为 https://host/v1 → https://host/v1/messages（避免 /v1/v1）
    """
    b = (api_base or "").strip().rstrip("/")
    if not b:
        return b
    if b.endswith("/v1"):
        return f"{b}/messages"
    return f"{b}/v1/messages"


def openai_chat_completions_url(api_base: str) -> str:
    """
    OpenAI 兼容 chat.completions（Bearer sk-...），与第三方站说明一致。
    - https://vip.aipro.love/v1 → https://vip.aipro.love/v1/chat/completions
    """
    b = (api_base or "").strip().rstrip("/")
    if not b:
        return b
    if b.endswith("/v1"):
        return f"{b}/chat/completions"
    return f"{b}/v1/chat/completions"

SECRET_KEYS = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY"}

_DOTENV_KEYS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_API_KEY",
    "OPENAI_BASE_URL",
    "CHAT_MODEL",
    "REASONING_MODEL",
    "CHEAP_MODEL",
    "AUTOALPHA_DEFAULT_ROUNDS",
    "AUTOALPHA_DEFAULT_IDEAS",
    "AUTOALPHA_DEFAULT_DAYS",
    "AUTOALPHA_DEFAULT_TARGET_VALID",
    "AUTOALPHA_PROMPT_CONTEXT_LIMIT",
    "AUTOALPHA_QUOTA_DISPLAY_FX",
    "AUTOALPHA_MIN_FULL_EVAL_DAYS",
    "AUTOALPHA_SCREEN_DAYS",
    "AUTOALPHA_SCREEN_MIN_IC",
    "AUTOALPHA_SCREEN_MIN_IR",
    "AUTOALPHA_SCREEN_MAX_TVR",
    "AUTOALPHA_RESEARCH_MIN_IC",
    "AUTOALPHA_RESEARCH_MIN_IR",
    "AUTOALPHA_IDEA_PAUSE_MS",
    "AUTOALPHA_ROUND_PAUSE_SEC",
    "AUTOALPHA_ROLLING_TARGET_VALID",
    "AUTOALPHA_ROLLING_TRAIN_DAYS",
    "AUTOALPHA_ROLLING_TEST_DAYS",
    "AUTOALPHA_ROLLING_STEP_DAYS",
}


def _merge_project_dotenv(config: Dict[str, str]) -> None:
    """Merge project-root .env so CLI (e.g. research_loop.py) sees the same keys as the Flask UI."""
    path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except Exception:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        key = key.strip()
        if key not in _DOTENV_KEYS:
            continue
        val = val.strip().strip('"').strip("'")
        if val:
            config[key] = val


def _ensure_parent() -> None:
    os.makedirs(os.path.dirname(SYSTEM_CONFIG_PATH), exist_ok=True)


def load_runtime_config() -> Dict[str, str]:
    _ensure_parent()
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(SYSTEM_CONFIG_PATH):
        try:
            with open(SYSTEM_CONFIG_PATH, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                config.update({str(k): "" if v is None else str(v) for k, v in stored.items()})
        except Exception:
            pass

    _merge_project_dotenv(config)

    for key in DEFAULT_CONFIG:
        env_val = os.environ.get(key)
        if env_val:
            config[key] = env_val

    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or config.get("OPENAI_API_KEY", "")
    )
    if api_key:
        config["OPENAI_API_KEY"] = api_key

    return config


def save_runtime_config(updates: Dict[str, str]) -> Dict[str, str]:
    current = load_runtime_config()
    for key, value in updates.items():
        if value is None:
            continue
        current[str(key)] = str(value)

    _ensure_parent()
    with open(SYSTEM_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(current, handle, indent=2, ensure_ascii=False)
    return current


def masked_runtime_config() -> Dict[str, str]:
    config = load_runtime_config()
    masked = dict(config)
    for key in SECRET_KEYS:
        value = masked.get(key)
        if not value:
            continue
        if len(value) <= 12:
            masked[key] = "***"
        else:
            masked[key] = f"{value[:8]}...{value[-4:]}"
    return masked


def get_llm_config() -> Dict[str, str]:
    config = load_runtime_config()
    return {
        "api_key": config.get("OPENAI_API_KEY", ""),
        "api_base": config.get("OPENAI_BASE_URL", DEFAULT_CONFIG["OPENAI_BASE_URL"]).strip().rstrip("/"),
        "model": config.get("CHAT_MODEL", DEFAULT_CONFIG["CHAT_MODEL"]),
    }


def get_llm_routing() -> Dict[str, str]:
    config = load_runtime_config()
    chat_model = config.get("CHAT_MODEL", DEFAULT_CONFIG["CHAT_MODEL"])
    return {
        "api_key": config.get("OPENAI_API_KEY", ""),
        "api_base": config.get("OPENAI_BASE_URL", DEFAULT_CONFIG["OPENAI_BASE_URL"]).strip().rstrip("/"),
        "chat_model": chat_model,
        "reasoning_model": config.get("REASONING_MODEL", chat_model) or chat_model,
        "cheap_model": config.get("CHEAP_MODEL", "") or chat_model,
    }
