"""
autoalpha_v2/llm_client.py

LLM client for AutoAlpha factor generation.

Generation strategy (paper-inspired upgrades):

  AlphaLogics 2026 — Hypothesis-first two-stage generation:
    Stage 1 (cheap model): articulate a market logic / mechanism hypothesis.
    Stage 2 (reasoning model): translate the hypothesis into a DSL formula.
    This separates "what mechanism am I exploiting?" from "how do I encode it?",
    producing more coherent, less pattern-matched factors.

  Hubble 2026 — Family-aware negative RAG:
    Structural fingerprints of exhausted formula families are injected into the
    prompt so the LLM avoids re-generating operator-skeleton near-duplicates even
    when field names differ.

  FactorMiner 2026 — Experience-informed guidance:
    Productive operator pairs (high win-rate from KB skill_memory) are surfaced
    as positive hints; crowded token patterns as negative hints.
"""
from __future__ import annotations

import json
import time
import warnings
from typing import Any, Dict, Iterable, List

import requests

from autoalpha_v2.error_utils import AutoAlphaRuntimeError, as_runtime_error
from autoalpha_v2.inspiration_db import compose_inspiration_context
from autoalpha_v2.knowledge_base import get_generation_guidance
from runtime_config import get_llm_routing, openai_chat_completions_url, load_runtime_config

warnings.filterwarnings("ignore")

TIMEOUT = 60

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — research goal and DSL specification
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior quantitative alpha researcher for intraday A-share equities.
Generate ONE original 15-minute alpha factor in JSON only.

# EXACT DSL (these names only)
Fields   : open_trade_px, high_trade_px, low_trade_px, close_trade_px, trade_count, volume, dvolume, vwap
Time-series: lag(x,d), delta(x,d), ts_pct_change(x,d), ts_mean(x,d), ts_ema(x,d),
             ts_std(x,d), ts_sum(x,d), ts_max(x,d), ts_min(x,d), ts_median(x,d),
             ts_quantile(x,d,q), ts_zscore(x,d), ts_rank(x,d), ts_minmax_norm(x,d),
             ts_decay_linear(x,d), ts_corr(x,y,d), ts_cov(x,y,d),
             ts_skew(x,d), ts_kurt(x,d), ts_argmax(x,d), ts_argmin(x,d)
Cross-section: cs_rank(x), cs_zscore(x), cs_demean(x), cs_scale(x),
               cs_winsorize(x,p), cs_quantile(x,q), cs_neutralize(x,y)
Math       : safe_div(a,b), signed_power(a,b), abs(x), sign(x), neg(x),
             log(x), signed_log(x), sqrt(x), clip(x,a,b), min_of(x,y), max_of(x,y),
             sigmoid(x), tanh(x), mean_of(x1,x2,...), weighted_sum(w1,x1,w2,x2,...),
             combine_rank(x1,x2,...)
Condition  : ifelse(cond,a,b), gt(x,y), ge(x,y), lt(x,y), le(x,y), eq(x,y),
             and_op(a,b), or_op(a,b), not_op(a)
Infix      : +, -, *, /

# HARD CONSTRAINTS
- NEVER use: resp, trading_restriction, lead(), future_, or any forward information.
- Every lookback d must be a positive integer literal.
- lag(x, 0) is forbidden.
- safe_div(a, b) must always have a SERIES denominator b; never pass a scalar literal as b.
- If you need to divide by a scalar constant, use infix / instead of safe_div.
- Keep the final formula compact: ideally <= 4 functional layers and <= 2 multiplicative blocks.
- Prefer smooth operators over hard conditions unless the condition encodes a clear regime filter.

# RESEARCH GOAL
- Prefer signals that can realistically pass competition gates on full-history 2022-2024 evaluation.
- Build signals with a market mechanism in mind: price-vs-vwap mean reversion, range-location
  persistence, failed breakout, intraday continuation with volume confirmation, volatility
  compression / release, or short-horizon reversal after exhaustion.
- Prefer a structure like signal core + stabilizer + cross-sectional normalization.
- Keep turnover controllable: use ts_decay_linear or ts_mean as outer smoothers when needed.
- New useful patterns: robust baselines via ts_median/ts_quantile, soft clipping via tanh/sigmoid,
  liquidity-neutral residuals via cs_neutralize, and multi-leg blends via mean_of/combine_rank.
- Aim for diversity versus prior factors. Do not paraphrase existing formulas.
- Favor full coverage, low concentration, and stable cross-sectional behavior.

# COMPETITION GATES
IC > 0.6, IR > 2.5, Turnover < 330 (local target), full 2022-2024 date coverage.

# OUTPUT
Return ONLY raw JSON:
{
  "thought_process": "2-4 sentence market mechanism and why it may work",
  "formula": "<DSL string>",
  "postprocess": "rank_clip",
  "lookback_days": 20
}
"""

# Stage-1 system prompt for hypothesis generation (AlphaLogics-inspired)
_HYPOTHESIS_SYSTEM = """\
You are a quantitative research strategist for intraday A-share equities.
Your job is to articulate a specific, testable market mechanism hypothesis.
Do NOT write formulas. Focus on the "why" — the economic or behavioral logic.

Output ONLY raw JSON:
{
  "hypothesis": "2-3 sentences: what phenomenon, what causes it, why it persists",
  "key_fields": ["up to 3 DSL fields most relevant"],
  "time_horizon": <positive integer, number of 15-min bars>,
  "archetype": "<one of: mean_reversion | momentum | volatility | volume_signal | range_location | exhaustion>"
}
"""

ARCHETYPES = [
    "price-vs-vwap intraday mean reversion with smoothing",
    "close-location / bar-shape persistence with low-turnover stabilization",
    "short-term continuation confirmed by volume or trade_count expansion",
    "volatility-compression then release using range and participation changes",
    "exhaustion reversal after large move and weak follow-through",
    "relative trend-strength versus recent baseline with cross-sectional normalization",
]


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_error_message(resp: requests.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        return (resp.text or "").strip()
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or error.get("code") or payload)
        return str(error or payload.get("message") or payload)
    return str(payload)


def _extract_content(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content or message.get("text") or "").strip()


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def _candidate_urls(api_base: str) -> List[str]:
    primary = openai_chat_completions_url(api_base)
    urls = [primary]
    if "vip.aipro.love" in primary:
        urls.append("https://free.aipro.love/v1/chat/completions")
    return list(dict.fromkeys(urls))


def _pick_model(tier: str) -> tuple[str, Dict[str, str]]:
    routing = get_llm_routing()
    if tier == "cheap":
        return routing.get("cheap_model") or routing.get("chat_model"), routing
    if tier == "chat":
        return routing.get("chat_model"), routing
    return routing.get("reasoning_model") or routing.get("chat_model"), routing


def _request_text(messages: list[dict[str, str]], *, max_tokens: int, tier: str) -> str:
    model, routing = _pick_model(tier)
    api_key = routing.get("api_key", "")
    api_base = routing.get("api_base", "")
    if not api_key:
        raise AutoAlphaRuntimeError(
            "当前没有可用的 API Key，无法调用 LLM。",
            raw_message="Missing OPENAI_API_KEY / ANTHROPIC_API_KEY / LLM_API_KEY.",
            suggestion="在系统设置中补充 API Key 后重试。",
            error_code="missing_api_key",
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.2,
    }
    last_err: Exception | None = None
    for _ in range(3):
        for url in _candidate_urls(api_base):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT, verify=False)
                if resp.status_code >= 400:
                    raise as_runtime_error(_extract_error_message(resp), status_code=resp.status_code)
                data = resp.json()
                if isinstance(data, dict) and data.get("error"):
                    raise as_runtime_error(data.get("error"))
                content = _extract_content(data)
                if content:
                    return _strip_fences(content)
                last_err = AutoAlphaRuntimeError(
                    "LLM 网关返回了空响应，没有生成可用内容。",
                    raw_message=f"url={url} status={resp.status_code} body={resp.text[:400]}",
                    suggestion="稍后重试；如果持续出现，优先检查额度、模型状态和网关健康度。",
                    error_code="empty_response",
                )
            except requests.RequestException as exc:
                last_err = as_runtime_error(exc)
            except AutoAlphaRuntimeError as exc:
                last_err = exc
            except Exception as exc:
                last_err = as_runtime_error(exc)
        time.sleep(2)

    if last_err is not None:
        raise last_err
    raise AutoAlphaRuntimeError(
        "LLM 网关返回了空响应，没有生成可用内容。",
        raw_message="Empty content after retries.",
        suggestion="稍后重试；如果持续出现，优先检查额度、模型状态和网关健康度。",
        error_code="empty_response",
    )


def call_llm(messages: list[dict[str, str]], max_tokens: int = 512, tier: str = "reasoning") -> dict:
    text = _request_text(messages, max_tokens=max_tokens, tier=tier)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AutoAlphaRuntimeError(
            "LLM 返回了无法解析的 JSON，输出格式不符合预期。",
            raw_message=f"{exc}. Raw: {text[:400]}",
            suggestion="重试该轮请求，或收紧提示词中的 JSON 输出约束。",
            error_code="bad_json",
        )


def _call_cheap_model(system: str, user: str, max_tokens: int = 800) -> str:
    """Thin wrapper: call cheap-tier model with a system + user message pair."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _request_text(messages, max_tokens=max_tokens, tier="cheap")


def summarize_inspiration_text(text: str, source_hint: str = "") -> str:
    content = (text or "").strip()
    if not content:
        return ""
    clipped = content[:3000]
    prompt = (
        "请把下面这段研究灵感提炼成 2 句话，突出潜在市场机制、可转化的价量结构、"
        "以及它能启发什么类型的 Alpha；不要复述废话，控制在 120 个中文字符内。"
    )
    if source_hint:
        prompt += f"\n来源: {source_hint}"
    prompt += f"\n\n内容:\n{clipped}"
    try:
        text = _request_text([{"role": "user", "content": prompt}], max_tokens=180, tier="cheap")
        return text.strip()[:180]
    except Exception:
        fallback = clipped.replace("\n", " ").strip()
        return fallback[:180]


# ─────────────────────────────────────────────────────────────────────────────
# Context formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_contrast_examples(title: str, items: list[dict[str, Any]] | None) -> str:
    rows = list(items or [])
    if not rows:
        return ""
    lines = [title]
    for item in rows[:4]:
        lines.append(
            "  run_id={run_id} | motif={motif} | IC={ic:.3f} | IR={ir:.2f} | score={score:.1f} | formula={formula}".format(
                run_id=item.get("run_id", ""),
                motif=item.get("motif", "generic"),
                ic=float(item.get("IC", 0) or 0),
                ir=float(item.get("IR", 0) or 0),
                score=float(item.get("Score", 0) or 0),
                formula=item.get("formula", "")[:220],
            )
        )
    return "\n".join(lines)


def _format_parent_lines(parents: Iterable[Dict[str, Any]] | None) -> str:
    lines: List[str] = []
    for parent in list(parents or [])[:5]:
        lines.append(
            "  formula={formula} | IC={ic} | tvr={tvr} | score={score} | thought={thought}".format(
                formula=parent.get("formula", ""),
                ic=parent.get("IC", "?"),
                tvr=parent.get("tvr", parent.get("Turnover", "?")),
                score=parent.get("score", parent.get("Score", "?")),
                thought=(parent.get("thought_process") or "")[:120],
            )
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Hypothesis generation (AlphaLogics-inspired)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_hypothesis(
    archetype: str,
    parents: list[dict[str, Any]] | None,
    inspirations: str,
    guidance: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Stage 1 (cheap model): articulate a market logic hypothesis before writing
    the formula.  Returns a dict with keys: hypothesis, key_fields,
    time_horizon, archetype — or None if the call fails.
    """
    sections: list[str] = [f"Target archetype: {archetype}"]

    parent_lines = _format_parent_lines(parents)
    if parent_lines:
        sections.append(
            "Prior factors in this research session (for contrast, NOT to copy):\n"
            f"{parent_lines}"
        )

    if inspirations:
        sections.append(
            "Research inspirations (distill the mechanism, do not quote literally):\n"
            f"{inspirations}"
        )

    exhausted = guidance.get("exhausted_families") or []
    if exhausted:
        family_strs = [
            f"  [{rec['attempts']} attempts, 0 wins] e.g. {rec.get('example','')[:100]}"
            for rec in exhausted[:4]
        ]
        sections.append(
            "Structurally exhausted search regions — these operator skeletons always failed, "
            "avoid their structure:\n" + "\n".join(family_strs)
        )

    sections.append(
        "Describe a NEW intraday A-share market mechanism. "
        "Be specific: name the phenomenon, the cause, and the expected time-horizon. "
        "Suggest 1-3 DSL fields that best capture it. Output JSON only."
    )

    messages = [
        {"role": "system", "content": _HYPOTHESIS_SYSTEM},
        {"role": "user",   "content": "\n\n".join(sections)},
    ]
    try:
        result = call_llm(messages, max_tokens=280, tier="cheap")
        if isinstance(result, dict) and result.get("hypothesis"):
            return result
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 / main: Formula generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_idea(
    parents: list[dict[str, Any]] | None = None,
    inspirations: str | None = None,
    idea_index: int = 0,
    total_ideas: int = 1,
) -> dict:
    """
    Two-stage factor generation:
      1. Cheap model produces a market mechanism hypothesis (AlphaLogics).
      2. Reasoning model translates hypothesis into a DSL formula, guided by:
         - Productive operator pairs from skill_memory (FactorMiner).
         - Exhausted formula families to avoid (Hubble negative RAG).
         - Classic parent + inspiration context.
    """
    cfg = load_runtime_config()
    context_limit = int(cfg.get("AUTOALPHA_PROMPT_CONTEXT_LIMIT", "6") or 6)
    inspiration_text = inspirations or compose_inspiration_context(limit=max(1, context_limit))
    guidance = get_generation_guidance()

    archetype = ARCHETYPES[idea_index % len(ARCHETYPES)]

    # ── Stage 1: Hypothesis (AlphaLogics) ────────────────────────────────────
    hypothesis_obj = _generate_hypothesis(archetype, parents, inspiration_text, guidance)

    # ── Stage 2: Formula ─────────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    sections: List[str] = []

    # Inject Stage-1 hypothesis as grounding context
    if hypothesis_obj:
        h_text = hypothesis_obj.get("hypothesis", "")
        h_fields = ", ".join(hypothesis_obj.get("key_fields") or [])
        h_horizon = hypothesis_obj.get("time_horizon", "")
        sections.append(
            "Market mechanism hypothesis to implement (AlphaLogics stage-1 output):\n"
            f"  Mechanism: {h_text}\n"
            f"  Key fields: {h_fields}\n"
            f"  Time horizon: {h_horizon} bars\n"
            "Translate this mechanism into a compact, competition-ready DSL formula."
        )
    else:
        sections.append(f"Target archetype: {archetype}")

    parent_lines = _format_parent_lines(parents)
    if parent_lines:
        sections.append(
            "Prior tested factors (use as contrast set, not templates to copy):\n"
            f"{parent_lines}"
        )

    if inspiration_text:
        sections.append(
            "Fresh inspirations (convert mechanisms into factor structures, do not quote literally):\n"
            f"{inspiration_text}"
        )

    # Hubble: family-aware negative RAG — exhausted structural skeletons
    exhausted_families = guidance.get("exhausted_families") or []
    if exhausted_families:
        family_strs = [
            f"  [{rec['attempts']} attempts, 0 wins] operator skeleton: {rec.get('example','')[:100]}"
            for rec in exhausted_families[:5]
        ]
        sections.append(
            "Exhausted structural families — different field names will NOT rescue these "
            "operator skeletons, which have never passed despite multiple attempts:\n"
            + "\n".join(family_strs)
        )

    # FactorMiner: productive operator pairs from experience memory
    productive_pairs = guidance.get("productive_operator_pairs") or []
    if productive_pairs:
        sections.append(
            "Operator combinations with proven win-rate in this research session "
            "(consider using these building blocks):\n  " + "\n  ".join(productive_pairs[:5])
        )

    # Token-level crowding from recent failures
    crowded_tokens = guidance.get("crowded_tokens") or []
    if crowded_tokens:
        sections.append(
            "Overused tokens in recent failed factors (avoid as the primary motif unless "
            "you materially transform the structure):\n"
            f"  {', '.join(crowded_tokens[:8])}"
        )

    failed_examples = _format_contrast_examples(
        "Recent weak or non-passing examples to avoid cloning:",
        guidance.get("recent_failed_examples"),
    )
    if failed_examples:
        sections.append(failed_examples)

    strong_examples = _format_contrast_examples(
        "Strong examples that passed or stayed useful as references:",
        guidance.get("strong_examples"),
    )
    if strong_examples:
        sections.append(strong_examples)

    sections.append(
        f"This is idea {idea_index + 1} of {total_ideas}. "
        "Generate ONE novel factor with a competition-oriented market rationale. "
        "Prefer a strong but compact design: regime detector + signal core + stabilizer. "
        "Bias toward formulas smooth enough to pass IR/turnover gates on full-history evaluation. "
        "Maximize structural diversity versus recent failures. "
        "Return JSON only."
    )

    messages.append({"role": "user", "content": "\n\n".join(sections)})
    return call_llm(messages, max_tokens=700, tier="reasoning")
