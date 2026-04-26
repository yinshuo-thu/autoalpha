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

import hashlib
import json
import re
import time
import urllib3
import warnings
from typing import Any, Dict, Iterable, List

import requests

from autoalpha_v2.error_utils import AutoAlphaRuntimeError, as_runtime_error
from autoalpha_v2.inspiration_db import compose_inspiration_context_with_sources, normalize_source_type
from autoalpha_v2.idea_cache import get_default_cache
from autoalpha_v2.knowledge_base import (
    get_generation_guidance,
    compose_passing_factors_rag,
    compose_failure_pattern_summary,
    find_relevant_experience,
    formula_structural_fingerprint,
)
from runtime_config import get_llm_routing, openai_chat_completions_url, load_runtime_config

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_READ_TIMEOUT = 45
DEFAULT_CONNECT_TIMEOUT = 8
PROMPT_TOTAL_CHAR_BUDGET = 7600
PROMPT_STRICT_CHAR_BUDGET = 5200
PROMPT_EMERGENCY_CHAR_BUDGET = 3400
_SECTION_COMPACT_CACHE: Dict[str, str] = {}
_SECTION_TARGET_CHARS = {
    "passing_rag": 1100,
    "failure_summary": 320,
    "hypothesis_context": 420,
    "parent_context": 520,
    "inspiration_text": 520,
    "exhausted_families": 260,
    "productive_pairs": 180,
    "crowded_tokens": 120,
    "failed_examples": 320,
    "generation_experience": 420,
    "relevant_experience": 180,
    "mode_rules": 260,
    "novelty_rules": 260,
    "fresh_blood": 180,
}
_LLM_COMPACTABLE_SECTIONS = {
    "passing_rag",
    "parent_context",
    "inspiration_text",
    "generation_experience",
    "failed_examples",
}


def _runtime_int(cfg: dict[str, str], key: str, default: int) -> int:
    try:
        return int(str(cfg.get(key, default) or default).strip())
    except (TypeError, ValueError):
        return default

SYSTEM_PROMPT_COMPACT = """\
You are a senior quantitative alpha researcher for 15-minute A-share equities.
Return ONE factor as raw JSON only.

Allowed fields:
open_trade_px, high_trade_px, low_trade_px, close_trade_px, trade_count, volume, dvolume, vwap

Allowed operators:
lag, delta, ts_pct_change, ts_mean, ts_ema, ts_std, ts_sum, ts_max, ts_min, ts_median,
ts_quantile, ts_zscore, ts_rank, ts_minmax_norm, ts_decay_linear, ts_corr, ts_cov,
ts_skew, ts_kurt, ts_argmax, ts_argmin, cs_rank, cs_zscore, cs_demean, cs_scale,
cs_winsorize, cs_quantile, cs_neutralize, safe_div, signed_power, abs, sign, neg,
log, signed_log, sqrt, clip, min_of, max_of, sigmoid, tanh, mean_of, weighted_sum,
combine_rank, ifelse, gt, ge, lt, le, eq, and_op, or_op, not_op, +, -, *, /

Hard constraints:
- No future info, no resp, no trading_restriction, no lead, no future_ fields.
- Positive integer lookbacks only; lag(x,0) forbidden.
- safe_div denominator must be a series, not a scalar literal.
- Keep formula compact, usually <= 4 functional layers and <= 2 multiplicative blocks.
- Outer smoother must be >= 10 bars, e.g. ts_mean(x,10), ts_ema(x,10), ts_decay_linear(x,15).

Research goal:
- Favor stable, low-turnover, full-coverage factors that can pass IC>0.6, IR>2.5, TVR<330.
- Good motifs: VWAP reversion, range-location persistence, failed breakout, volume-confirmed continuation,
  volatility compression/release, exhaustion reversal.
- Prefer smooth signal core + stabilizer + cross-sectional normalization.

Output JSON keys:
thought_process, formula, postprocess, lookback_days
"""


def _transport_profile(tier: str) -> tuple[int, int, int]:
    """
    Return retries, connect timeout, read timeout for a request tier.

    Relay (vip.aipro.love) has a hard ~60s server-side timeout; Haiku model
    typically responds in 30-50s with a full prompt.  Timeouts here are the
    client-side socket limits — set generously above the relay cutoff so we
    always see the relay error rather than a premature client abort.
    """
    cfg = load_runtime_config()
    connect_timeout = _runtime_int(cfg, "AUTOALPHA_LLM_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT)
    retries = max(1, _runtime_int(cfg, "AUTOALPHA_LLM_REQUEST_RETRIES", 2))
    if tier == "cheap":
        return retries, max(10, connect_timeout), max(55, _runtime_int(cfg, "AUTOALPHA_LLM_CHEAP_READ_TIMEOUT", 55))
    if tier == "chat":
        return retries, connect_timeout, max(55, _runtime_int(cfg, "AUTOALPHA_LLM_CHAT_READ_TIMEOUT", 55))
    return retries, connect_timeout, max(55, _runtime_int(cfg, "AUTOALPHA_LLM_REASONING_READ_TIMEOUT", 90))


def _trim_block(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _local_compact_block(name: str, text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text

    if name in {"passing_rag", "parent_context", "failed_examples"}:
        lines = []
        for raw in text.splitlines():
            line = " ".join(raw.split()).strip()
            if not line:
                continue
            if "formula=" in line:
                prefix, _, suffix = line.partition("formula=")
                line = prefix + "formula=" + _trim_block(suffix, 90)
            lines.append(_trim_block(line, 170))
        return _trim_block("\n".join(lines), limit)

    units = [part.strip() for part in re.split(r"(?:\n+|(?<=[。.!?])\s+)", text) if part.strip()]
    compact = " ".join(" ".join(unit.split()) for unit in units)
    return _trim_block(compact, limit)


def _compact_block_with_llm(name: str, text: str, limit: int, *, allow_llm: bool = True) -> str:
    text = (text or "").strip()
    if not text or len(text) <= limit:
        return text

    cache_key = hashlib.sha1(f"{name}|{limit}|{text}".encode("utf-8")).hexdigest()
    cached = _SECTION_COMPACT_CACHE.get(cache_key)
    if cached:
        print(f"[compact] section={name} mode=cache before={len(text)} after={len(cached)}")
        return cached

    fallback = _local_compact_block(name, text, limit)
    if (not allow_llm) or name not in _LLM_COMPACTABLE_SECTIONS:
        print(f"[compact] section={name} mode=local before={len(text)} after={len(fallback)}")
        _SECTION_COMPACT_CACHE[cache_key] = fallback
        return fallback

    prompt = [
        {
            "role": "system",
            "content": (
                "Compress research context without dropping concrete facts. "
                "Preserve identifiers, metrics, formulas, and explicit do/don't guidance. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Section: {name}\n"
                f"Target chars: <= {limit}\n"
                "Rewrite the text densely for reuse inside another LLM prompt. "
                "Keep all concrete facts, but remove repetition and prose padding.\n"
                f"Text:\n{_trim_block(text, 5000)}\n\n"
                'Return JSON: {"compressed": "..."}'
            ),
        },
    ]
    try:
        result = call_llm(prompt, max_tokens=min(220, max(120, limit // 3)), tier="cheap")
        compressed = str(result.get("compressed") or "").strip()
        if compressed:
            fallback = _local_compact_block(name, compressed, limit)
            print(f"[compact] section={name} mode=llm before={len(text)} after={len(fallback)}")
    except Exception:
        print(f"[compact] section={name} mode=local-after-llm-fail before={len(text)} after={len(fallback)}")

    _SECTION_COMPACT_CACHE[cache_key] = fallback
    return fallback


def _auto_compact_sections(
    sections: list[tuple[str, str]],
    *,
    system_prompt: str,
    total_budget: int = PROMPT_TOTAL_CHAR_BUDGET,
) -> list[str]:
    rendered = [(name, (text or "").strip()) for name, text in sections if (text or "").strip()]
    total_chars = len(system_prompt) + sum(len(text) + 2 for _, text in rendered)
    if total_chars <= total_budget:
        return [text for _, text in rendered]

    # First pass: cheap, deterministic local compaction on the biggest blocks.
    order = sorted(
        range(len(rendered)),
        key=lambda idx: (
            -len(rendered[idx][1]),
            idx,
        ),
    )
    for idx in order:
        if total_chars <= total_budget:
            break
        name, text = rendered[idx]
        target = min(len(text), _SECTION_TARGET_CHARS.get(name, max(180, len(text) // 2)))
        if len(text) <= target:
            continue
        compacted = _compact_block_with_llm(name, text, target, allow_llm=False)
        delta = len(text) - len(compacted)
        if delta > 0:
            rendered[idx] = (name, compacted)
            total_chars -= delta

    # Second pass: only in normal mode, allow a tiny number of LLM-assisted compactions.
    llm_compactions_used = 0
    allow_llm_compaction = total_budget >= PROMPT_TOTAL_CHAR_BUDGET
    if total_chars > total_budget and allow_llm_compaction:
        order = sorted(range(len(rendered)), key=lambda idx: len(rendered[idx][1]), reverse=True)
        for idx in order:
            if total_chars <= total_budget or llm_compactions_used >= 1:
                break
            name, text = rendered[idx]
            if name not in _LLM_COMPACTABLE_SECTIONS:
                continue
            target = min(len(text), _SECTION_TARGET_CHARS.get(name, max(180, len(text) // 2)))
            compacted = _compact_block_with_llm(name, text, target, allow_llm=True)
            delta = len(text) - len(compacted)
            if delta > 0:
                rendered[idx] = (name, compacted)
                total_chars -= delta
                llm_compactions_used += 1

    if total_chars > total_budget:
        overflow = total_chars - total_budget
        order = sorted(range(len(rendered)), key=lambda idx: len(rendered[idx][1]), reverse=True)
        for idx in order:
            if overflow <= 0:
                break
            name, text = rendered[idx]
            cut = min(max(80, overflow + 40), max(0, len(text) - 120))
            if cut <= 0:
                continue
            trimmed = _local_compact_block(name, text, len(text) - cut)
            overflow -= max(0, len(text) - len(trimmed))
            rendered[idx] = (name, trimmed)

    return [text for _, text in rendered]

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
- TURNOVER IS THE #1 KILLER: outer smoother window MUST be >= 10 bars.
  ts_decay_linear(x, 2) or ts_decay_linear(x, 3) produces TVR > 700 — FATAL, will not pass.
  The outermost layer MUST be ts_mean(x, 10), ts_decay_linear(x, 15), or ts_ema(x, 10) or longer.
  Never use delta(close,1), ts_pct_change(close,1), or sign(delta(x,1)) directly as the core
  signal without >= 10-bar outer smoothing. Short-window signals MUST be pre-smoothed before rank.
- New useful patterns: robust baselines via ts_median/ts_quantile, soft clipping via tanh/sigmoid,
  liquidity-neutral residuals via cs_neutralize, and multi-leg blends via mean_of/combine_rank.
- Aim for diversity versus prior factors. Do not paraphrase existing formulas.
- Novelty is mandatory across at least 2 axes: mechanism, baseline, operator chain, or field pairing.
- Favor full coverage, low concentration, and stable cross-sectional behavior.

# COMPETITION GATES
IC > 0.6, IR > 2.5, Turnover < 330 (local target), full 2022-2024 date coverage.
Turnover is the dominant failure — tested formulas average TVR > 700. Fix: use window >= 10.

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
    ("mean_reversion", "price-vs-vwap intraday mean reversion with smoothing"),
    ("range_location", "close-location / bar-shape persistence with low-turnover stabilization"),
    ("momentum", "short-term continuation confirmed by volume or trade_count expansion"),
    ("volatility", "volatility-compression then release using range and participation changes"),
    ("exhaustion", "exhaustion reversal after large move and weak follow-through"),
    ("volume_signal", "relative trend-strength versus recent baseline with cross-sectional normalization"),
]

EXPLORATION_ARCHETYPES = [
    ("mean_reversion", "new paper-derived microstructure anomaly unlike current parents"),
    ("volume_signal", "LLM brainstormed orthogonal field interaction with low formula correlation"),
    ("momentum", "factor-prompt market mechanism translated conservatively to equities"),
    ("volatility", "rare regime filter using range, liquidity, and participation divergence"),
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


def _oai_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Connection": "close",
        "User-Agent": "AutoAlpha/2.0",
    }


def _collect_openai_stream_lines(lines: Iterable[bytes | str]) -> str:
    parts: List[str] = []
    for line in lines:
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            break
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, list):
                for item in text:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                    elif item:
                        parts.append(str(item))
            elif text:
                parts.append(str(text))
    return "".join(parts).strip()


def _stream_text(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    *,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    body = {**payload, "stream": True}
    session = requests.Session()
    session.verify = False
    session.trust_env = False
    try:
        with session.post(
            url,
            headers=headers,
            json=body,
            stream=True,
            timeout=(connect_timeout, read_timeout),
        ) as resp:
            if resp.status_code >= 400:
                raise as_runtime_error(_extract_error_message(resp), status_code=resp.status_code)
            content = _collect_openai_stream_lines(resp.iter_lines())
            if content:
                return _strip_fences(content)
            raw = (resp.text or "")[:400]
            raise AutoAlphaRuntimeError(
                "LLM 网关返回了空响应，没有生成可用内容。",
                raw_message=f"url={url} status={resp.status_code} stream_body={raw}",
                suggestion="稍后重试；如果持续出现，优先检查模型状态、网关兼容性和额度。",
                error_code="empty_response",
            )
    finally:
        session.close()


def _nonstream_text(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    *,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    session = requests.Session()
    session.verify = False
    session.trust_env = False
    try:
        resp = session.post(url, headers=headers, json=payload, timeout=(connect_timeout, read_timeout))
        if resp.status_code >= 400:
            raise as_runtime_error(_extract_error_message(resp), status_code=resp.status_code)
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise as_runtime_error(data.get("error"))
        content = _extract_content(data)
        if content:
            return _strip_fences(content)
        raise AutoAlphaRuntimeError(
            "LLM 网关返回了空响应，没有生成可用内容。",
            raw_message=f"url={url} status={resp.status_code} body={resp.text[:400]}",
            suggestion="稍后重试；如果持续出现，优先检查模型状态、网关兼容性和额度。",
            error_code="empty_response",
        )
    finally:
        session.close()


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
    retries, connect_timeout, read_timeout = _transport_profile(tier)
    if not api_key:
        raise AutoAlphaRuntimeError(
            "当前没有可用的 API Key，无法调用 LLM。",
            raw_message="Missing OPENAI_API_KEY / ANTHROPIC_API_KEY / LLM_API_KEY.",
            suggestion="在系统设置中补充 API Key 后重试。",
            error_code="missing_api_key",
        )

    headers = _oai_headers(api_key)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    last_err: Exception | None = None
    fallback_codes = {"empty_response", "timeout", "network_error"}
    for attempt in range(retries):
        for url in _candidate_urls(api_base):
            try:
                return _stream_text(
                    url,
                    headers,
                    payload,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                )
            except AutoAlphaRuntimeError as exc:
                print(f"[llm] tier={tier} url={url} mode=stream error={exc.error_code}: {exc.friendly_message}")
                last_err = exc
                if getattr(exc, "error_code", "") not in fallback_codes:
                    continue
                try:
                    return _nonstream_text(
                        url,
                        headers,
                        payload,
                        connect_timeout=connect_timeout,
                        read_timeout=read_timeout,
                    )
                except Exception as fallback_exc:
                    fallback_err = as_runtime_error(fallback_exc)
                    print(
                        f"[llm] tier={tier} url={url} mode=nonstream "
                        f"error={fallback_err.error_code}: {fallback_err.friendly_message}"
                    )
                    last_err = fallback_err
            except requests.RequestException as exc:
                last_err = as_runtime_error(exc)
                print(
                    f"[llm] tier={tier} url={url} mode=stream "
                    f"error={last_err.error_code}: {last_err.friendly_message}"
                )
                if getattr(last_err, "error_code", "") in fallback_codes:
                    try:
                        return _nonstream_text(
                            url,
                            headers,
                            payload,
                            connect_timeout=connect_timeout,
                            read_timeout=read_timeout,
                        )
                    except Exception as fallback_exc:
                        fallback_err = as_runtime_error(fallback_exc)
                        print(
                            f"[llm] tier={tier} url={url} mode=nonstream "
                            f"error={fallback_err.error_code}: {fallback_err.friendly_message}"
                        )
                        last_err = fallback_err
            except Exception as exc:
                last_err = as_runtime_error(exc)
                print(f"[llm] tier={tier} url={url} mode=stream error={last_err.error_code}: {last_err.friendly_message}")
        if attempt < retries - 1:
            backoff = 3.0 * (attempt + 1)
            print(f"[llm] retry attempt={attempt + 1}/{retries} sleeping {backoff:.0f}s before next try")
            time.sleep(backoff)

    if last_err is not None:
        raise last_err
    raise AutoAlphaRuntimeError(
        "LLM 网关返回了空响应，没有生成可用内容。",
        raw_message="Empty content after retries.",
        suggestion="稍后重试；如果持续出现，优先检查额度、模型状态和网关健康度。",
        error_code="empty_response",
    )


def _should_retry_with_compact_prompt(exc: Exception) -> bool:
    code = getattr(as_runtime_error(exc), "error_code", "")
    return code in {"timeout", "network_error", "empty_response", "bad_json"}


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


def summarize_factor_tldr(description: str, formula: str = "", source_hint: str = "") -> str:
    """
    Convert the existing English factor description into a concise, easy-to-read Chinese TL;DR.
    Keep it short and explanatory rather than decorative.
    """
    content = (description or "").strip()
    if not content:
        return ""

    clipped_description = content[:2400]
    clipped_formula = (formula or "").strip()[:400]
    prompt = (
        "请把下面这个因子说明压缩成 1-2 句中文 TL;DR，目标是让非作者也能快速理解。"
        "要求：\n"
        "1. 只做精简转述，不要发散，不要编造新机制。\n"
        "2. 优先说明它抓的市场现象、核心信号，以及大致在什么情况下有效。\n"
        "3. 语言尽量白话、易懂，避免堆术语。\n"
        "4. 控制在 70 个中文字符内，直接输出正文，不要加标题、引号或项目符号。"
    )
    if source_hint:
        prompt += f"\n补充上下文: {source_hint[:200]}"
    if clipped_formula:
        prompt += f"\n公式: {clipped_formula}"
    prompt += f"\n\n原始说明:\n{clipped_description}"
    try:
        text = _request_text([{"role": "user", "content": prompt}], max_tokens=120, tier="cheap")
        return text.strip().replace("\n", " ")[:100]
    except Exception:
        fallback = clipped_description.split("\n", 1)[0].strip()
        fallback = re.sub(r"\s+", " ", fallback)
        return fallback[:100]


def summarize_generation_experience(payload: dict[str, Any], previous_context: str = "") -> str:
    """Ask the LLM to write a generation-level research note for memory reuse."""
    compact = json.dumps(payload, ensure_ascii=False, indent=2)[:9000]
    prompt = (
        "请作为量化研究负责人，总结下面这个 AutoAlpha Generation 的完整实验经验。"
        "目标是形成后续 LLM 生成因子的上文，而不是普通日报。\n\n"
        "必须覆盖：\n"
        "1. 本代总体表现：测试数、通过数、best score、主要失败模式。\n"
        "2. 可复用的正向经验：哪些市场机制、字段组合、平滑/归一化方式值得继续。\n"
        "3. 明确的负向经验：例如 TVR 过高、IR 不稳、IC 方向差、重复结构、DSL 语法问题分别如何避免。\n"
        "4. 下一代探索启示：给出 5-8 条具体可执行的生成约束或方向。\n"
        "5. 写给下一代 Prompt 的短指令：控制在 120 中文字内。\n\n"
        "输出 Markdown，使用清晰小标题。不要编造不存在的通过因子；如果没有通过因子，要直说。"
    )
    if previous_context:
        prompt += f"\n\n最近几代经验摘要，可用于判断趋势：\n{previous_context[:2400]}"
    prompt += f"\n\n本代结构化实验数据：\n{compact}"
    try:
        text = _request_text([{"role": "user", "content": prompt}], max_tokens=1200, tier="reasoning")
        return text.strip()
    except Exception as exc:
        failures = payload.get("failure_counts", {})
        failure_text = ", ".join(f"{k}={v}" for k, v in failures.items()) or "none"
        return (
            f"# Generation {payload.get('generation')} Experience\n\n"
            f"- Tested: {payload.get('total', 0)}\n"
            f"- Passing: {payload.get('passing', 0)}\n"
            f"- Best Score: {float(payload.get('best_score', 0) or 0):.2f}\n"
            f"- Failure Pattern: {failure_text}\n\n"
            "## 后续启示\n"
            "- 优先降低换手：增加 `ts_decay_linear` / `ts_mean` 平滑，避免直接追逐 1-3 bar 的成交量尖峰。\n"
            "- 避免重复近期失败的 operator skeleton，尝试更稳健的 median/quantile baseline。\n"
            "- 继续保持公式紧凑，先解决 IC/IR/TVR 的门槛，再扩展复杂结构。\n\n"
            f"_LLM summary fallback because: {exc}_"
        )


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
    for parent in list(parents or [])[:3]:
        lines.append(
            "  formula={formula} | IC={ic} | tvr={tvr} | score={score} | thought={thought}".format(
                formula=_trim_block(parent.get("formula", ""), 140),
                ic=parent.get("IC", "?"),
                tvr=parent.get("tvr", parent.get("Turnover", "?")),
                score=parent.get("score", parent.get("Score", "?")),
                thought=_trim_block(parent.get("thought_process", ""), 70),
            )
        )
    return "\n".join(lines)


def _format_novelty_rules(parents: Iterable[Dict[str, Any]] | None) -> str:
    rows = list(parents or [])[:3]
    if not rows:
        return ""
    lines = [
        "Novelty guardrails versus current parents:",
        "  - Do not reuse the same outer smoother + normalization combo as a parent.",
        "  - Do not keep the same primary field pair and only change lookback constants.",
        "  - Do not keep the same structural fingerprint with renamed fields.",
    ]
    for item in rows:
        formula = str(item.get("formula", "") or "")
        if not formula:
            continue
        lines.append(
            f"  parent {item.get('run_id', '')}: fingerprint={formula_structural_fingerprint(formula)[:120]}"
        )
    return "\n".join(lines)


def _archetype_weight(
    archetype_key: str,
    stats: Dict[str, Dict[str, int]],
) -> float:
    bucket = stats.get(archetype_key, {})
    total = sum(int(v or 0) for v in bucket.values())
    if total <= 0:
        return 1.0
    severe = int(bucket.get("syntax_error", 0) or 0) + int(bucket.get("compute_error", 0) or 0)
    screened = int(bucket.get("screened_out", 0) or 0)
    passing = int(bucket.get("passing", 0) or 0)
    weight = 1.0
    weight *= max(0.2, 1.0 - 0.8 * (severe / total))
    weight *= max(0.4, 1.0 - 0.35 * (screened / total))
    if passing > 0:
        weight *= 1.0 + min(0.35, 0.12 * passing)
    return max(0.15, min(weight, 1.5))


def _pick_archetype(
    options: List[tuple[str, str]],
    idea_index: int,
    stats: Dict[str, Dict[str, int]],
) -> tuple[str, str]:
    ordered = sorted(
        enumerate(options),
        key=lambda item: (
            _archetype_weight(item[1][0], stats),
            -(sum(int(v or 0) for v in stats.get(item[1][0], {}).values())),
            -(item[0] == (idea_index % max(1, len(options)))),
        ),
        reverse=True,
    )
    slot = idea_index % max(1, len(ordered))
    return ordered[slot][1]


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
        result = call_llm(messages, max_tokens=220, tier="cheap")
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
    prompt_version = str(cfg.get("AUTOALPHA_PROMPT_VERSION", "v2-diversity-20260424b") or "v2-diversity-20260424b")
    explore_ratio = float(cfg.get("AUTOALPHA_EXPLORATION_RATIO", "0.35") or 0.35)
    explore_every = max(3, round(1 / max(0.05, min(explore_ratio, 0.8))))
    fresh_blood_every = max(2, _runtime_int(cfg, "AUTOALPHA_FRESH_BLOOD_EVERY", 3))
    source_cycle = [
        normalize_source_type(source.strip())
        for source in str(cfg.get("AUTOALPHA_INSPIRATION_SOURCES", "paper,llm,manual")).split(",")
        if source.strip()
    ]
    source_cycle = [source for source in source_cycle if source in {"paper", "llm", "manual"}]
    if not source_cycle:
        source_cycle = ["paper", "llm", "manual"]
    exploration_sources = [
        normalize_source_type(source.strip())
        for source in str(cfg.get("AUTOALPHA_EXPLORATION_SOURCES", "paper,llm,manual")).split(",")
        if source.strip()
    ]
    exploration_sources = [source for source in exploration_sources if source in {"paper", "llm", "manual"}] or ["paper", "llm", "manual"]
    is_exploration = (idea_index + 1) % explore_every == 0
    is_fresh_blood_slot = (idea_index + 1) % fresh_blood_every == 0
    target_source = (
        exploration_sources[idea_index % len(exploration_sources)]
        if is_exploration
        else source_cycle[idea_index % len(source_cycle)]
    )
    if is_fresh_blood_slot and "manual" in exploration_sources:
        target_source = "manual"
    if inspirations:
        inspiration_text = inspirations
        inspiration_rows: list[dict[str, Any]] = []
    else:
        inspiration_text, inspiration_rows = compose_inspiration_context_with_sources(
            limit=max(4, context_limit),
            preferred_source=target_source,
            prefer_unused=is_fresh_blood_slot or is_exploration,
        )
    guidance = get_generation_guidance()
    archetype_stats = get_default_cache().recent_archetype_outcomes(limit=80)
    source_types = [
        normalize_source_type(row.get("source_type") or row.get("kind") or "manual")
        for row in inspiration_rows
    ]
    source_ids = [
        int(row.get("id"))
        for row in inspiration_rows
        if row.get("id") is not None
    ]
    primary_source = (
        target_source if target_source in source_types
        else (source_types[idea_index % len(source_types)] if source_types else ("custom" if inspirations else "none"))
    )

    archetype_key, archetype = _pick_archetype(
        EXPLORATION_ARCHETYPES if is_exploration else ARCHETYPES,
        idea_index=idea_index,
        stats=archetype_stats,
    )
    relevant_experience = find_relevant_experience(
        archetype=archetype,
        failure_mode=str(guidance.get("dominant_failure_mode", "")),
    )

    # ── Stage 1: Hypothesis (AlphaLogics) ────────────────────────────────────
    hypothesis_obj = _generate_hypothesis(archetype, parents, inspiration_text, guidance)
    rag_query = archetype
    if hypothesis_obj:
        rag_query = " | ".join(
            part for part in [
                str(hypothesis_obj.get("hypothesis", "") or "").strip(),
                f"fields: {', '.join(hypothesis_obj.get('key_fields') or [])}".strip(),
                f"archetype: {hypothesis_obj.get('archetype', archetype_key)}",
                f"horizon: {hypothesis_obj.get('time_horizon', '')}",
            ]
            if part and part.strip()
        ) or archetype

    # ── Stage 2: Formula ─────────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    sections: List[tuple[str, str]] = []

    # RAG: inject all verified passing factors first — highest-signal context
    passing_rag = compose_passing_factors_rag(
        query_text=rag_query,
        max_factors=3 if is_exploration else 4,
        semantic_k=2 if is_exploration else 3,
        anchor_count=1,
        include_formulas=not is_exploration,
        include_template=False,
    )
    if passing_rag:
        sections.append(("passing_rag", passing_rag))

    # Failure pattern summary — explicit remediation guidance
    failure_summary = compose_failure_pattern_summary()
    if failure_summary:
        sections.append(("failure_summary", failure_summary))

    # Inject Stage-1 hypothesis as grounding context
    if hypothesis_obj:
        h_text = hypothesis_obj.get("hypothesis", "")
        h_fields = ", ".join(hypothesis_obj.get("key_fields") or [])
        h_horizon = hypothesis_obj.get("time_horizon", "")
        sections.append(
            (
            "hypothesis_context",
            "Market mechanism hypothesis to implement (AlphaLogics stage-1 output):\n"
            f"  Mechanism: {h_text}\n"
            f"  Key fields: {h_fields}\n"
            f"  Time horizon: {h_horizon} bars\n"
            "Translate this mechanism into a compact, competition-ready DSL formula."
            )
        )
    else:
        sections.append(("hypothesis_context", f"Target archetype: {archetype}"))

    parent_lines = _format_parent_lines(parents)
    novelty_rules = _format_novelty_rules(parents)
    if parent_lines:
        if is_exploration:
            sections.append(
                (
                "parent_context",
                "Current strong parent factors (treat as an exclusion/contrast set; do NOT iterate their skeleton):\n"
                f"{parent_lines}"
                )
            )
        else:
            sections.append(
                (
                "parent_context",
                "Prior tested factors (use as contrast set, not templates to copy):\n"
                f"{parent_lines}"
                )
            )
    if novelty_rules:
        sections.append(("novelty_rules", novelty_rules))

    if inspiration_text:
        sections.append(
            (
            "inspiration_text",
            f"Fresh inspirations. Primary source target for this idea: {primary_source}. "
            "Convert the source mechanism into a factor structure; do not quote literally:\n"
            f"{inspiration_text}"
            )
        )
    if is_fresh_blood_slot:
        sections.append(
            (
            "fresh_blood",
            "Fresh-blood slot is active for this idea. Pull at least one mechanism from a low-usage or newly added inspiration, "
            "then combine it with turnover-safe implementation. Do not merely restate recent parent motifs."
            )
        )

    # Hubble: family-aware negative RAG — exhausted structural skeletons
    exhausted_families = guidance.get("exhausted_families") or []
    if exhausted_families:
        family_strs = [
            f"  [{rec['attempts']} attempts, 0 wins] operator skeleton: {rec.get('example','')[:100]}"
            for rec in exhausted_families[:5]
        ]
        sections.append(
            (
            "exhausted_families",
            "Exhausted structural families — different field names will NOT rescue these "
            "operator skeletons, which have never passed despite multiple attempts:\n"
            + "\n".join(family_strs)
            )
        )

    # FactorMiner: productive operator pairs from experience memory
    productive_pairs = guidance.get("productive_operator_pairs") or []
    if productive_pairs:
        sections.append(
            (
            "productive_pairs",
            "Operator combinations with proven win-rate in this research session "
            "(consider using these building blocks):\n  " + "\n  ".join(productive_pairs[:5])
            )
        )

    # Token-level crowding from recent failures
    crowded_tokens = guidance.get("crowded_tokens") or []
    if crowded_tokens:
        sections.append(
            (
            "crowded_tokens",
            "Overused tokens in recent failed factors (avoid as the primary motif unless "
            "you materially transform the structure):\n"
            f"  {', '.join(crowded_tokens[:8])}"
            )
        )

    failed_examples = _format_contrast_examples(
        "Recent weak or non-passing examples to avoid cloning:",
        guidance.get("recent_failed_examples"),
    )
    if failed_examples:
        sections.append(("failed_examples", failed_examples))

    tvr_alert = guidance.get("tvr_alert") or ""
    if tvr_alert:
        sections.append(("tvr_alert", f"⚠️ {tvr_alert}"))

    generation_experience = guidance.get("generation_experience_context") or ""
    if generation_experience:
        sections.append(
            (
            "generation_experience",
            "Generation-level research experience from previous rounds. Treat this as hard-earned lab memory; "
            "especially obey repeated failure lessons such as high turnover, unstable IR, duplicate structures, "
            "and DSL syntax pitfalls:\n"
            f"{generation_experience}"
            )
        )
    if relevant_experience:
        sections.append(
            (
            "relevant_experience",
            "Older but relevant generation lesson matched to the current archetype/failure mode:\n"
            f"{relevant_experience}"
            )
        )

    mode_rules = (
        "1. This slot is EXPLORATION: start from fresh Paper/LLM/Manual inspiration and target low correlation to parents; do not reuse the parent operator skeleton.\n"
        "2. You may change both mechanism and skeleton, but keep turnover-safe smoothing and compact DSL.\n"
        if is_exploration
        else
        "1. Follow the proven structural template from the RAG above when it fits: "
        "neg(outer_smoother(cs_zscore(ts_mean(price_core * sigmoid_volume_gate, 3-4)))).\n"
        "2. Vary the MECHANISM (what price signal, what confirmation), NOT only constants.\n"
    )
    sections.append(
        (
        "mode_rules",
        f"This is idea {idea_index + 1} of {total_ideas}. "
        "Generate ONE novel factor. STRICT RULES:\n"
        f"{mode_rules}"
        "3. Outer smoother window MUST be >= 10 (ts_decay_linear >= 15, ts_ema >= 10).\n"
        "4. Use sigmoid/tanh softening on all sub-signals before multiplying.\n"
        "5. Do NOT copy any formula from the RAG verbatim — new price signal or new baseline.\n"
        "6. Keep at least two novelty changes versus the closest parent: e.g. different baseline + different signal core, or different field pair + different outer smoother.\n"
        "Return JSON only."
        )
    )

    raw_section_lengths = {name: len((text or "").strip()) for name, text in sections}
    raw_sections_total = sum(len((text or "").strip()) + 2 for _, text in sections)

    def _build_formula_messages(
        *,
        system_prompt: str,
        budget: int,
        log_prefix: str,
        section_names: set[str] | None = None,
    ) -> list[dict[str, str]]:
        prompt_sections = [
            (name, text)
            for name, text in sections
            if section_names is None or name in section_names
        ]
        rendered_sections = _auto_compact_sections(
            prompt_sections,
            system_prompt=system_prompt,
            total_budget=budget,
        )
        compacted_section_lengths = {
            name: len(text)
            for (name, _), text in zip(prompt_sections, rendered_sections)
        }
        raw_total = len(system_prompt) + sum(len((text or "").strip()) + 2 for _, text in prompt_sections)
        compact_total = len(system_prompt) + sum(len(text) + 2 for text in rendered_sections)
        print(
            f"{log_prefix} formula_context chars "
            f"before={raw_total} after={compact_total} "
            f"budget={budget} sections="
            + ", ".join(
                f"{name}:{raw_section_lengths.get(name, 0)}->{compacted_section_lengths.get(name, 0)}"
                for name, _ in prompt_sections
            )
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(rendered_sections)},
        ]

    messages = _build_formula_messages(
        system_prompt=SYSTEM_PROMPT,
        budget=PROMPT_TOTAL_CHAR_BUDGET,
        log_prefix="[prompt]",
    )
    try:
        idea = call_llm(messages, max_tokens=420, tier="reasoning")
    except Exception as exc:
        if not _should_retry_with_compact_prompt(exc):
            raise
        strict_messages = _build_formula_messages(
            system_prompt=SYSTEM_PROMPT_COMPACT,
            budget=PROMPT_STRICT_CHAR_BUDGET,
            log_prefix="[prompt-retry]",
        )
        try:
            idea = call_llm(strict_messages, max_tokens=280, tier="reasoning")
        except Exception as retry_exc:
            if not _should_retry_with_compact_prompt(retry_exc):
                raise
            emergency_messages = _build_formula_messages(
                system_prompt=SYSTEM_PROMPT_COMPACT,
                budget=PROMPT_EMERGENCY_CHAR_BUDGET,
                log_prefix="[prompt-emergency]",
                section_names={
                    "hypothesis_context",
                    "passing_rag",
                    "inspiration_text",
                    "failure_summary",
                    "generation_experience",
                    "mode_rules",
                },
            )
            idea = call_llm(emergency_messages, max_tokens=220, tier="reasoning")
    idea["inspiration_source_type"] = primary_source
    idea["inspiration_source_types"] = sorted(set(source_types))
    idea["inspiration_ids"] = source_ids
    idea["archetype"] = archetype_key
    idea["archetype_label"] = archetype
    idea["generation_mode"] = "exploration" if is_exploration else "iteration"
    idea["target_source"] = target_source
    idea["prompt_version"] = prompt_version
    return idea
