import os
import sys
import json
import time
import random
import yaml
import traceback
import httpx
import requests
import urllib3
from datetime import datetime
from paths import AUTORESEARCH_LOG_PATH, OUTPUTS_ROOT
from runtime_config import anthropic_messages_endpoint, get_llm_config, openai_chat_completions_url

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = """You are a Quantitative Alpha Researcher at a top-tier hedge fund. Your task is to generate high-capacity, original Python-based factor formulas using our proprietary DSL.

# CORE DSL COMPONENTS
Available inputs: open_trade_px, high_trade_px, low_trade_px, close_trade_px, trade_count, volume, dvolume, vwap
Available Time-Series operators: lag(x, d), delta(x, d), ts_mean(x, d), ts_std(x, d), ts_sum(x, d), ts_max(x, d), ts_min(x, d), ts_rank(x, d), ts_zscore(x, d), ts_decay_linear(x, d)
Available Cross-Sectional operators: cs_rank(x), cs_zscore(x), cs_demean(x)
Available Math operators: safe_div(a, b), signed_power(a, b), abs(x), sign(x), neg(x), log(x)

# RESEARCH GUIDELINES
1. **Structural Complexity**: Aim for an AST depth between 2 and 4. Avoid overly shallow (1 level) or overly deep (>4 levels) formulas to balance signal strength and robustness.
2. **Economic Intuition**: Your 'thought_process' must describe a clear market hypothesis (e.g., liquidity provision, information asymmetry, institutional flow, or price-volume divergence).
3. **Feature Interaction**: Professional alphas often combine price movement (delta) with volume or volatility signals. Pure price momentum is usually crowded.
4. **DSL Compliance**: Only use the provided operators. Ensure all numeric parameters (d) are logical (e.g., 5 for short-term, 20 for monthly, 60 for quarterly).
5. **Gates to Pass**: Your target factor must aim for: IC > 0.006 (0.6 displayed), IR > 2.5, Turnover < 400.

# OUTPUT FORMAT
Output MUST be valid JSON matching this schema exactly:
{
  "thought_process": "Detailed economic intuition and how this idea relates to or differs from prior factors.",
  "formula": "The mathematical formula string (DSL only)",
  "postprocess": "stable_low_turnover" or "aggressive_high_ic",
  "lookback_days": 20
}
DO NOT output markdown formatting like `json`. Output ONLY the raw JSON object.
"""

_BOOT_LLM_CONFIG = get_llm_config()
API_BASE = _BOOT_LLM_CONFIG["api_base"]
MODEL_NAME = _BOOT_LLM_CONFIG["model"]
# Backup base URL for automatic failover
BACKUP_API_BASES = [
    # Disabled by default: free gateway is noisier/slower in this environment.
]
REQUEST_TIMEOUT = 60
CONNECT_TIMEOUT = 10
TRANSPORT_RETRIES = 2
LLM_MAX_TOKENS = 320
LLM_COOLDOWN_SECONDS = 90

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_LLM_COOLDOWN_UNTIL = 0.0

# ===== LOCAL OFFLINE FORMULA BANK (used when API is unreachable) =====
OFFLINE_FORMULAS = [
    {"thought": "Short-term mean reversion: if price deviates from its 10-bar mean, it will revert.",
     "formula": "cs_rank(-ts_zscore(close_trade_px, 10))", "post": "stable_low_turnover", "lb": 10},
    {"thought": "Volume-weighted momentum: higher volume confirms direction, rank cross-sectionally.",
     "formula": "cs_rank(ts_decay_linear(delta(close_trade_px, 5) * volume, 10))", "post": "stable_low_turnover", "lb": 15},
    {"thought": "VWAP deviation reversal: securities far below VWAP tend to bounce.",
     "formula": "cs_rank(ts_decay_linear(-(close_trade_px / vwap - 1), 10))", "post": "stable_low_turnover", "lb": 10},
    {"thought": "Multi-horizon mean reversion combining 5 and 20 bar lookbacks.",
     "formula": "cs_rank(-ts_zscore(close_trade_px, 5)) + cs_rank(-ts_zscore(close_trade_px, 20))", "post": "stable_low_turnover", "lb": 20},
    {"thought": "Intraday range compression signals breakout direction via volume.",
     "formula": "cs_rank(safe_div(ts_std(close_trade_px, 10), ts_mean(close_trade_px, 10)) * sign(delta(volume, 5)))", "post": "stable_low_turnover", "lb": 10},
    {"thought": "Trade count anomaly: unusual trade count relative to recent history signals informed trading.",
     "formula": "cs_rank(ts_zscore(trade_count, 15))", "post": "stable_low_turnover", "lb": 15},
    {"thought": "Price-volume divergence: rising price with falling volume signals weakness.",
     "formula": "cs_rank(-delta(close_trade_px, 5) * ts_rank(-volume, 10))", "post": "stable_low_turnover", "lb": 10},
    {"thought": "Smooth momentum using decay-weighted delta, penalizing erratic changes.",
     "formula": "cs_rank(ts_decay_linear(delta(vwap, 3), 15))", "post": "stable_low_turnover", "lb": 15},
    {"thought": "High-low range contraction predicts continuation, weighted by recent volume trend.",
     "formula": "cs_rank(safe_div(high_trade_px - low_trade_px, ts_mean(high_trade_px - low_trade_px, 10)) * sign(delta(dvolume, 5)))", "post": "aggressive_high_ic", "lb": 10},
    {"thought": "Cross-sectional ranking of smoothed close-to-open gap captures overnight drift.",
     "formula": "cs_rank(ts_decay_linear(close_trade_px - open_trade_px, 10))", "post": "stable_low_turnover", "lb": 10},
]

def log_agent_thought(generation_id, status, thought, metrics=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file = AUTORESEARCH_LOG_PATH
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except: pass
        
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "generation": generation_id,
        "status": status,
        "thought": thought,
        "metrics": metrics,
    }
    logs.insert(0, entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    print(f"[AutoAgent] {status} -> {thought[:120]}...")

def _last_user_excerpt(history, limit: int = 900) -> str:
    for msg in reversed(history):
        if msg.get("role") == "user":
            c = msg.get("content", "") or ""
            return c[:limit]
    return ""


def _oai_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Connection": "close",
        "User-Agent": "AutoAlpha/1.0",
    }


def _anthropic_headers(api_key: str) -> dict:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Connection": "close",
        "User-Agent": "AutoAlpha/1.0",
    }


def _collect_openai_stream_lines(lines) -> str:
    parts: list = []
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
        for ch in obj.get("choices") or []:
            delta = ch.get("delta") or {}
            t = delta.get("content")
            if t:
                parts.append(t)
    return "".join(parts).strip()


def _openai_stream_collect_text_requests(url: str, headers: dict, payload: dict) -> str:
    """为规避第三方网关偶发断连，每次流式请求都使用新连接。"""
    body = {**payload, "stream": True}
    session = requests.Session()
    session.verify = False
    try:
        with session.post(
            url,
            json=body,
            headers=headers,
            stream=True,
            timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
        ) as response:
            response.raise_for_status()
            return _collect_openai_stream_lines(response.iter_lines())
    finally:
        session.close()


def _openai_stream_collect_text_httpx(url: str, headers: dict, payload: dict) -> str:
    body = {**payload, "stream": True}
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
    with httpx.Client(verify=False, timeout=timeout, follow_redirects=True) as client:
        with client.stream("POST", url, json=body, headers=headers) as response:
            response.raise_for_status()
            return _collect_openai_stream_lines(response.iter_lines())


def _openai_non_stream_collect_text_requests(url: str, headers: dict, payload: dict) -> str:
    session = requests.Session()
    session.verify = False
    try:
        response = session.post(
            url,
            json=payload,
            headers=headers,
            timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
        )
        response.raise_for_status()
        return _extract_openai_message_text(response.json())
    finally:
        session.close()


def _anthropic_collect_text_requests(url: str, headers: dict, payload: dict) -> str:
    session = requests.Session()
    session.verify = False
    try:
        response = session.post(
            url,
            json=payload,
            headers=headers,
            timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
        )
        response.raise_for_status()
        return _extract_anthropic_message_text(response.json())
    finally:
        session.close()


def _extract_openai_message_text(data: dict) -> str:
    try:
        ch0 = (data.get("choices") or [{}])[0]
        msg = ch0.get("message") or {}
        c = msg.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts).strip()
    except Exception:
        pass
    return ""


def _extract_anthropic_message_text(data: dict) -> str:
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            t = (block.get("text") or "").strip()
            if t:
                return t
    return ""


def _retry_transport(name: str, fn, errors: list, retries: int = TRANSPORT_RETRIES) -> str:
    for attempt in range(1, retries + 1):
        try:
            text = fn()
            if text:
                return text
            errors.append(f"{name} attempt {attempt}: empty response")
        except Exception as exc:
            errors.append(f"{name} attempt {attempt}: {type(exc).__name__}: {exc}")
        if attempt < retries:
            time.sleep(min(0.8 * attempt, 1.6))
    return ""


def _summarize_transport_errors(errors: list, limit: int = 6) -> str:
    if not errors:
        return "no transport details"
    if len(errors) <= limit:
        return " | ".join(errors)
    visible = " | ".join(errors[:limit])
    return f"{visible} | ... ({len(errors)} transport issues total)"


def _parse_json_object_from_text(text: str) -> dict:
    """Parse first JSON object from model output (handles extra prose / fences)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty LLM text")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found in LLM output")


def _strip_markdown_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    t = t.strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def _complete_chat_to_text(api_key: str, api_base: str, model_name: str, history: list) -> tuple:
    """
    优先 OpenAI 兼容 chat.completions（Bearer），其次 Anthropic Messages（x-api-key）。
    自动尝试 backup base URL。全部使用 proxy-bypass session。
    返回 (raw_text, transport) 其中 transport 为 'openai'|'anthropic'
    """
    system_content = ""
    messages = []
    for msg in history:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            messages.append({"role": msg["role"], "content": msg["content"]})

    oa_messages = []
    if system_content:
        oa_messages.append({"role": "system", "content": system_content})
    oa_messages.extend(messages)

    headers_oai = _oai_headers(api_key)
    stream_payload = {
        "model": model_name,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": oa_messages,
    }

    # Try primary + backup base URLs
    bases_to_try = [api_base] + [b for b in BACKUP_API_BASES if b != api_base]

    transport_errors = []

    # Anthropic-style messages (no separate system in messages list)
    anthropic_messages = list(messages)
    anthropic_payload = {
        "model": model_name,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": anthropic_messages,
    }
    if system_content:
        anthropic_payload["system"] = system_content

    for base in bases_to_try:
        url_oai = openai_chat_completions_url(base)
        text = _retry_transport(
            f"openai_stream_requests@{base}",
            lambda url=url_oai: _openai_stream_collect_text_requests(url, headers_oai, stream_payload),
            transport_errors,
        )
        if text:
            return text, "openai_stream"

        text = _retry_transport(
            f"openai_stream_httpx@{base}",
            lambda url=url_oai: _openai_stream_collect_text_httpx(url, headers_oai, stream_payload),
            transport_errors,
            retries=1,
        )
        if text:
            return text, "openai_stream_httpx"

        # 部分网关（如 DashScope 兼容接口）流式返回不稳定，改非流式
        text = _retry_transport(
            f"openai_non_stream@{base}",
            lambda url=url_oai: _openai_non_stream_collect_text_requests(
                url, headers_oai, stream_payload
            ),
            transport_errors,
        )
        if text:
            return text, "openai_non_stream"

        url_anth = anthropic_messages_endpoint(base)
        headers_ant = _anthropic_headers(api_key)
        text = _retry_transport(
            f"anthropic_messages@{base}",
            lambda url=url_anth: _anthropic_collect_text_requests(url, headers_ant, anthropic_payload),
            transport_errors,
        )
        if text:
            return text, "anthropic_messages"

    raise RuntimeError(
        "LLM 返回空文本（OpenAI 流式/非流式与 Anthropic 回退均失败）；"
        f"请检查网关、模型名或套餐权限。详情: {_summarize_transport_errors(transport_errors)}"
    )


def query_llm(history, mining_source: str = "query_llm", extra=None):
    """
    优先 OpenAI 兼容 chat.completions（流式→非流式），其次 Anthropic /messages。
    传输失败回退离线库并触发冷却；JSON 解析失败仅回退离线库，不触发冷却。
    """
    from core.llm_mining_log import append_llm_mining_record

    extra = dict(extra or {})
    user_excerpt = _last_user_excerpt(history)

    global _LLM_COOLDOWN_UNTIL

    llm_config = get_llm_config()
    api_key = llm_config["api_key"]
    api_base = llm_config["api_base"]
    model_name = llm_config["model"]

    def _offline_fallback(err: Exception, event: str):
        err_s = f"{type(err).__name__}: {err}"
        pick = random.choice(OFFLINE_FORMULAS)
        out = {
            "thought_process": f"[OFFLINE MODE] {pick['thought']}",
            "formula": pick["formula"],
            "postprocess": pick["post"],
            "lookback_days": pick["lb"],
        }
        append_llm_mining_record(
            {
                "event": event,
                "mining_source": mining_source,
                "error": err_s,
                "user_prompt_excerpt": user_excerpt,
                "parsed": out,
                **extra,
            }
        )
        return out

    if not api_key:
        print("[LLM Fallback] using local candidate bank (No LLM API key configured)")
        return _offline_fallback(RuntimeError("No LLM API key configured"), "llm_offline_fallback")

    now_ts = time.time()
    if _LLM_COOLDOWN_UNTIL > now_ts:
        wait_s = int(_LLM_COOLDOWN_UNTIL - now_ts)
        print(f"[LLM Fallback] using local candidate bank (cooldown {wait_s}s)")
        return _offline_fallback(
            RuntimeError(f"LLM cooldown active ({wait_s}s remaining)"), "llm_offline_fallback"
        )

    try:
        content, transport = _complete_chat_to_text(api_key, api_base, model_name, history)
        _LLM_COOLDOWN_UNTIL = 0.0
    except Exception as e:
        if "No LLM API key configured" not in str(e) and "LLM cooldown active" not in str(e):
            _LLM_COOLDOWN_UNTIL = max(_LLM_COOLDOWN_UNTIL, time.time() + LLM_COOLDOWN_SECONDS)
        print(f"[LLM Fallback] using local candidate bank ({type(e).__name__}: {e})")
        return _offline_fallback(e, "llm_offline_fallback")

    stripped = _strip_markdown_fences(content)
    try:
        try:
            parsed = json.loads(stripped.strip())
        except json.JSONDecodeError:
            parsed = _parse_json_object_from_text(stripped)
    except Exception as e:
        print(f"[LLM Fallback] JSON parse failed, using offline bank ({type(e).__name__}: {e})")
        return _offline_fallback(e, "llm_parse_fallback")

    append_llm_mining_record(
        {
            "event": "llm_ok",
            "mining_source": mining_source,
            "model": model_name,
            "api_base": api_base,
            "transport": transport,
            "user_prompt_excerpt": user_excerpt,
            "parsed": parsed,
            "raw_text_excerpt": (content or "")[:2000],
            **extra,
        }
    )
    _tp = " ".join(str(parsed.get("thought_process") or "").split())[:600]
    _fm = str(parsed.get("formula") or "")[:220]
    print(
        f"[LLM_OK] {mining_source} transport={transport} model={model_name} | formula={_fm} | thought={_tp}",
        flush=True,
    )
    return parsed


def test_llm_connection():
    """Connectivity smoke test used by server and local scripts."""
    llm_config = get_llm_config()
    api_key = llm_config["api_key"]
    api_base = llm_config["api_base"]
    model_name = llm_config["model"]
    if not api_key:
        return {
            "ok": False,
            "reason": "No API key configured",
            "api_base": api_base,
            "model": model_name,
        }

    history = [
        {"role": "system", "content": "You are a helpful assistant. Reply very briefly."},
        {"role": "user", "content": "Reply with exactly: AUTOALPHA API OK"},
    ]

    try:
        text, transport = _complete_chat_to_text(api_key, api_base, model_name, history)
        return {
            "ok": bool(text),
            "reply": text,
            "transport": transport,
            "model": model_name,
            "api_base": api_base,
            "usage": {},
        }
    except Exception as exc:
        return {
            "ok": False,
            "reason": str(exc),
            "model": model_name,
            "api_base": api_base,
        }

def run_auto_loop(max_iters=5):
    import pandas as pd
    from core.genalpha import GenAlpha
    from core.datahub import get_trading_days, load_pv_days, load_universe, load_resp_days, load_restriction_days
    
    print("\n[AutoAgent] Booting up and preloading 2022-01-01 to 2024-12-31...")
    start_date, end_date = "2022-01-01", "2024-12-31"
    all_days = get_trading_days(start=None, end=end_date)
    target_start_idx = all_days.index(start_date) if start_date in all_days else 0
    eval_slice = all_days[target_start_idx:]
    
    shared_ram = {
        'pv': load_pv_days(all_days),  
        'univ': load_universe(all_days),
        'resp': load_resp_days(eval_slice),
        'rest': load_restriction_days(eval_slice)
    }
    
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Generate the baseline alpha 001. Keep it simple, e.g., using moving average crossover or price reversal."}
    ]
    
    for i in range(1, max_iters + 1):
        run_id = f"auto_alpha_{i:03d}"
        log_agent_thought(i, "THINKING", "Querying LLM API (with offline fallback)...")
        
        try:
            res_json = query_llm(conversation, mining_source="run_auto_loop")
            thought = res_json['thought_process']
            formula = res_json['formula']
            post = res_json.get('postprocess', 'stable_low_turnover')
            lb = res_json.get('lookback_days', 20)
            
            log_agent_thought(i, "WRITING_CODE", f"Formula: {formula}\nThought: {thought}")
            
            yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "configs", f"{run_id}.yaml")
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, 'w') as f:
                yaml.dump({
                    'formula': formula,
                    'preprocess': {'lookback_days': lb},
                    'postprocess': post,
                    'export_submission': True
                }, f)
                
            # Execute evaluation
            log_agent_thought(i, "EVALUATING", f"Running 3-year backtest engine on '{formula}'...")
            eval_res = GenAlpha.run(
                formula=formula,
                start=start_date, end=end_date,
                preprocess={'lookback_days': lb}, postprocess=post,
                export_submission=True, run_id=run_id, preloaded_data=shared_ram
            )
            
            overall = eval_res.get('metrics', {}).get('overall', {})
            score = overall.get('Score', 0)
            ic = overall.get('IC', 0)
            tvr = overall.get('Turnover', 0)
            ir = overall.get('IR', 0)
            passed = overall.get('PassGates', False)
            
            metrics_dict = {
                'IC': round(ic,4), 'IR': round(ir,4), 'Turnover': round(tvr,4), 
                'Score': round(score,2), 'Passed': passed,
                'Formula': formula, 'PostProcess': post
            }
            
            status_emoji = "✅ PASS" if passed else "❌ FAIL"
            log_agent_thought(i, "RESULT", 
                f"{status_emoji} | Score: {score:.2f} | IC: {ic:.4f} | IR: {ir:.4f} | Tvr: {tvr:.2f}\nFormula: {formula}", 
                metrics_dict)
            
            # Append to auto_leaderboard.csv
            summary_path = os.path.join(OUTPUTS_ROOT, "research_runs", "auto_leaderboard.csv")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            df_new = pd.DataFrame([metrics_dict], index=[run_id])
            if os.path.exists(summary_path):
                df_old = pd.read_csv(summary_path, index_col=0)
                df_all = pd.concat([df_old, df_new])
                df_all = df_all[~df_all.index.duplicated(keep='last')]
                df_all = df_all.sort_values(by='Score', ascending=False)
            else:
                df_all = df_new
            df_all.to_csv(summary_path)
            
            # Build next iteration prompt
            conversation.append({"role": "assistant", "content": json.dumps(res_json)})
            if passed:
                msg = f"Excellent! The factor passed with Score={score:.2f}, IC={ic:.4f}, Turnover={tvr:.2f}. Now, generate a completely orthogonal, distinct factor using different features to diversify our pipeline."
            else:
                msg = f"Factor Failed. Score={score:.2f}, IC={ic:.4f}, IR={ir:.4f}, Turnover={tvr:.2f}. "
                if ic < 0.6: msg += "IC is too low. Try a stronger momentum or volume indicator. "
                if tvr >= 400: msg += "Turnover is too high! Use ts_decay_linear or ts_mean to smooth the signal. "
                if ir <= 2.5: msg += "IR is low, it means the signal is not consistent over time. Adjust rolling periods."
                msg += " Please mutate the formula to fix these bottlenecks."
                
            conversation.append({"role": "user", "content": msg})
            
        except Exception as e:
            trace_str = traceback.format_exc()
            log_agent_thought(i, "ERROR", f"Engine Error: {str(e)}\n{trace_str}")
            conversation.append({"role": "user", "content": f"Your last JSON caused an error: {str(e)}. Fix the syntax and return strictly raw JSON."})
            
        import gc
        gc.collect()

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    run_auto_loop()
