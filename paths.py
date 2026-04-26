import os
import platform


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _default_storage_root():
    env_root = os.environ.get("SCIENTECH_STORAGE_ROOT")
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))

    project_parent = os.path.dirname(PROJECT_ROOT)
    sibling_data = os.path.join(project_parent, "data")
    if platform.system() == "Windows":
        return r"e:\data"
    return sibling_data


def _detect_data_root():
    env_root = os.environ.get("SCIENTECH_DATA_ROOT")
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))

    candidates = [
        _default_storage_root(),
        PROJECT_ROOT,
        os.path.expanduser("~/autoalpha_v2"),
        r"e:\autoalpha_v2",
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "eq_data_stage1")):
            return candidate
    return _default_storage_root()


STORAGE_ROOT = _default_storage_root()
DATA_ROOT = _detect_data_root()
OUTPUTS_ROOT = os.path.join(DATA_ROOT, "outputs")
RESEARCH_ARTIFACTS_ROOT = os.path.join(DATA_ROOT, "research_runs")
CACHE_ROOT = os.path.join(DATA_ROOT, "cache")
RESEARCH_LOG_PATH = os.path.join(OUTPUTS_ROOT, "research.log")
LEADERBOARD_PATH = os.path.join(OUTPUTS_ROOT, "leaderboard.json")
AUTORESEARCH_LOG_PATH = os.path.join(OUTPUTS_ROOT, "autoresearch_log.json")
SUBMIT_ROOT = os.path.join(PROJECT_ROOT, "submit")
SUBMISSIONS_ROOT = SUBMIT_ROOT
SYSTEM_CONFIG_PATH = os.path.join(OUTPUTS_ROOT, "system_config.json")
# LLM 因子挖掘：追加式 JSONL，与 autoresearch 分离便于审计
LLM_MINING_LOG_DIR = os.path.join(OUTPUTS_ROOT, "llm_mining")
LLM_MINING_JSONL = os.path.join(LLM_MINING_LOG_DIR, "mining_log.jsonl")
LLM_EXPERIENCE_JSONL = os.path.join(LLM_MINING_LOG_DIR, "factor_experience.jsonl")
LLM_EXPERIENCE_DOC_PATH = os.path.join(PROJECT_ROOT, "summaries", "llm_factor_experience.md")
