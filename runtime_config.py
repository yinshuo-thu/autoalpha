import json
import os
import platform
import subprocess
from typing import Dict

from paths import PROJECT_ROOT, SYSTEM_CONFIG_PATH


DEFAULT_CONFIG: Dict[str, str] = {
    # 第三方 Claude 网关：可填 https://vip.aipro.love 或 https://vip.aipro.love/v1（见 anthropic_messages_endpoint）
    "OPENAI_BASE_URL": "https://vip.aipro.love/v1",
    "CHAT_MODEL": "claude-sonnet-4-6",
    "REASONING_MODEL": "claude-haiku-4-5-20251001",
    "CHEAP_MODEL": "claude-haiku-4-5-20251001",
    "EMBEDDING_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "EMBEDDING_API_KEY": "",
    "EMBEDDING_MODEL": "text-embedding-v4",
    "AUTOALPHA_DEFAULT_ROUNDS": "10",
    "AUTOALPHA_DEFAULT_IDEAS": "4",
    "AUTOALPHA_DEFAULT_DAYS": "0",
    "AUTOALPHA_DEFAULT_TARGET_VALID": "0",
    "AUTOALPHA_PROMPT_CONTEXT_LIMIT": "6",
    "AUTOALPHA_PROMPT_VERSION": "v2-diversity-20260423",
    "AUTOALPHA_INSPIRATION_SOURCES": "paper,llm,future,manual",
    "AUTOALPHA_EXPLORATION_RATIO": "0.35",
    "AUTOALPHA_EXPLORATION_SOURCES": "paper,llm,future",
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
    "AUTOALPHA_ROLLING_TARGET_VALID": "0",
    "AUTOALPHA_ROLLING_TRAIN_DAYS": "126",
    "AUTOALPHA_ROLLING_TEST_DAYS": "126",
    "AUTOALPHA_ROLLING_STEP_DAYS": "126",
    "AUTOALPHA_LLM_CONNECT_TIMEOUT": "15",
    "AUTOALPHA_LLM_CHEAP_READ_TIMEOUT": "55",
    "AUTOALPHA_LLM_CHAT_READ_TIMEOUT": "55",
    "AUTOALPHA_LLM_REASONING_READ_TIMEOUT": "90",
    "AUTOALPHA_LLM_REQUEST_RETRIES": "2",
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


def openai_embeddings_url(api_base: str) -> str:
    """
    OpenAI 兼容 embeddings endpoint。
    - https://vip.aipro.love/v1 → https://vip.aipro.love/v1/embeddings
    """
    b = (api_base or "").strip().rstrip("/")
    if not b:
        return b
    if b.endswith("/v1"):
        return f"{b}/embeddings"
    return f"{b}/v1/embeddings"

SECRET_KEYS = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY"}
_KEYCHAIN_SERVICE_PREFIX = "scientech.runtime_config"

_DOTENV_KEYS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_API_KEY",
    "OPENAI_BASE_URL",
    "CHAT_MODEL",
    "REASONING_MODEL",
    "CHEAP_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "AUTOALPHA_DEFAULT_ROUNDS",
    "AUTOALPHA_DEFAULT_IDEAS",
    "AUTOALPHA_DEFAULT_DAYS",
    "AUTOALPHA_DEFAULT_TARGET_VALID",
    "AUTOALPHA_PROMPT_CONTEXT_LIMIT",
    "AUTOALPHA_PROMPT_VERSION",
    "AUTOALPHA_INSPIRATION_SOURCES",
    "AUTOALPHA_EXPLORATION_RATIO",
    "AUTOALPHA_EXPLORATION_SOURCES",
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
    "AUTOALPHA_LLM_CONNECT_TIMEOUT",
    "AUTOALPHA_LLM_CHEAP_READ_TIMEOUT",
    "AUTOALPHA_LLM_CHAT_READ_TIMEOUT",
    "AUTOALPHA_LLM_REASONING_READ_TIMEOUT",
    "AUTOALPHA_LLM_REQUEST_RETRIES",
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


def _read_stored_runtime_config() -> Dict[str, str]:
    if not os.path.exists(SYSTEM_CONFIG_PATH):
        return {}
    try:
        with open(SYSTEM_CONFIG_PATH, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        if isinstance(stored, dict):
            return {str(k): "" if v is None else str(v) for k, v in stored.items()}
    except Exception:
        pass
    return {}


def _write_stored_runtime_config(stored: Dict[str, str]) -> None:
    _ensure_parent()
    with open(SYSTEM_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(stored, handle, indent=2, ensure_ascii=False)


def _keychain_available() -> bool:
    return platform.system() == "Darwin" and os.path.exists("/usr/bin/security")


def _keychain_service_name(key: str) -> str:
    return f"{_KEYCHAIN_SERVICE_PREFIX}.{key}"


def _keychain_get_secret(key: str) -> str:
    if not _keychain_available():
        return ""
    try:
        proc = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-a", key, "-s", _keychain_service_name(key), "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _keychain_set_secret(key: str, value: str) -> bool:
    if not _keychain_available():
        return False
    secret = str(value or "")
    try:
        proc = subprocess.run(
            ["/usr/bin/security", "add-generic-password", "-a", key, "-s", _keychain_service_name(key), "-U", "-w"],
            input=(secret + "\n") * 2,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _keychain_delete_secret(key: str) -> None:
    if not _keychain_available():
        return
    try:
        subprocess.run(
            ["/usr/bin/security", "delete-generic-password", "-a", key, "-s", _keychain_service_name(key)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        pass


def _migrate_legacy_secret_storage(stored: Dict[str, str]) -> Dict[str, str]:
    if not stored:
        return stored
    sanitized = dict(stored)
    changed = False
    for key in SECRET_KEYS:
        if key not in sanitized:
            continue
        legacy = sanitized.get(key, "").strip()
        if _keychain_available():
            existing = _keychain_get_secret(key)
            if not legacy or existing or _keychain_set_secret(key, legacy):
                sanitized.pop(key, None)
                changed = True
    if changed:
        _write_stored_runtime_config(sanitized)
    return sanitized


def load_runtime_config() -> Dict[str, str]:
    _ensure_parent()
    config = dict(DEFAULT_CONFIG)

    _merge_project_dotenv(config)

    stored = _migrate_legacy_secret_storage(_read_stored_runtime_config())
    if stored:
        # Runtime saves from the UI should override repo-level .env defaults.
        config.update(stored)

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
    elif _keychain_available():
        stored_openai = _keychain_get_secret("OPENAI_API_KEY")
        if stored_openai:
            config["OPENAI_API_KEY"] = stored_openai

    for secret_key in SECRET_KEYS:
        if config.get(secret_key):
            continue
        secret_val = _keychain_get_secret(secret_key)
        if secret_val:
            config[secret_key] = secret_val

    return config


def save_runtime_config(updates: Dict[str, str]) -> Dict[str, str]:
    current = load_runtime_config()
    stored = _migrate_legacy_secret_storage(_read_stored_runtime_config())
    for key, value in updates.items():
        if value is None:
            continue
        key = str(key)
        value = str(value)
        current[key] = value
        if key in SECRET_KEYS:
            if value.strip():
                if _keychain_available():
                    _keychain_set_secret(key, value)
                else:
                    stored[key] = value
            else:
                current[key] = ""
                if _keychain_available():
                    _keychain_delete_secret(key)
                    stored.pop(key, None)
                else:
                    stored[key] = ""
            continue
        stored[key] = value

    if _keychain_available():
        for key in SECRET_KEYS:
            stored.pop(key, None)
    _write_stored_runtime_config(stored)
    return current


def masked_runtime_config() -> Dict[str, str]:
    config = load_runtime_config()
    masked = dict(config)
    for key in SECRET_KEYS:
        value = masked.get(key)
        if not value:
            continue
        masked[key] = "***"
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


def get_embedding_routing() -> Dict[str, str]:
    config = load_runtime_config()
    api_key = (config.get("EMBEDDING_API_KEY", "") or config.get("OPENAI_API_KEY", "")).strip()
    api_base = (config.get("EMBEDDING_BASE_URL", "") or config.get("OPENAI_BASE_URL", DEFAULT_CONFIG["OPENAI_BASE_URL"])).strip().rstrip("/")
    model = config.get("EMBEDDING_MODEL", DEFAULT_CONFIG["EMBEDDING_MODEL"]).strip() or DEFAULT_CONFIG["EMBEDDING_MODEL"]
    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
    }
