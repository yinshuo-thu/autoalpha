"""
autoalpha_v1/error_utils.py

Shared helpers for turning raw runtime exceptions into short, human-readable
messages for logs, API responses, and Feishu notifications.
"""

from __future__ import annotations

from typing import Any, Tuple


class AutoAlphaRuntimeError(RuntimeError):
    """Structured runtime error that keeps both friendly and raw context."""

    def __init__(
        self,
        friendly_message: str,
        *,
        raw_message: str = "",
        suggestion: str = "",
        error_code: str = "unknown",
    ) -> None:
        super().__init__(friendly_message)
        self.friendly_message = friendly_message
        self.raw_message = raw_message
        self.suggestion = suggestion
        self.error_code = error_code


def stringify_error(error: Any) -> str:
    """Normalize an exception or arbitrary payload into a single-line string."""
    if error is None:
        return ""
    if isinstance(error, AutoAlphaRuntimeError):
        if error.raw_message:
            return " ".join(str(error.raw_message).split())
        return " ".join(str(error.friendly_message).split())
    if isinstance(error, BaseException):
        return " ".join(str(error).split())
    return " ".join(str(error).split())


def humanize_error(error: Any, status_code: int | None = None) -> Tuple[str, str, str, str]:
    """
    Convert a raw error into:
      friendly summary, suggestion, error code, raw detail
    """
    if isinstance(error, AutoAlphaRuntimeError):
        return (
            error.friendly_message,
            error.suggestion,
            error.error_code,
            stringify_error(error),
        )

    raw = stringify_error(error)
    lowered = raw.lower()

    if status_code == 401 or any(
        token in lowered
        for token in ("invalid_api_key", "unauthorized", "incorrect api key", "authentication")
    ):
        return (
            "LLM API 认证失败，当前 Key 不可用或已经失效。",
            "检查 API Key、网关地址和模型权限配置。",
            "auth_error",
            raw,
        )

    if status_code == 429 or "rate limit" in lowered:
        return (
            "LLM API 触发限流，短时间内请求过多。",
            "稍后重试，或降低并发与调用频率。",
            "rate_limited",
            raw,
        )

    if any(
        token in lowered
        for token in (
            "insufficient_quota",
            "quota",
            "billing hard limit",
            "credit balance",
            "额度",
            "余额不足",
            "欠费",
            "充值",
        )
    ):
        return (
            "LLM API 额度已用尽或余额不足，当前无法继续生成新因子。",
            "充值当前账号、切换到可用 API Key，或改用备用网关。",
            "quota_exhausted",
            raw,
        )

    if any(token in lowered for token in ("timed out", "timeout", "read timeout", "connect timeout")):
        return (
            "请求 LLM 网关超时，可能是上游模型响应过慢或网关不稳定。",
            "稍后重试，必要时切换到备用网关。",
            "timeout",
            raw,
        )

    if any(
        token in lowered
        for token in (
            "ssl",
            "connection aborted",
            "connection reset",
            "max retries exceeded",
            "temporarily unavailable",
            "name or service not known",
            "failed to establish a new connection",
            "network is unreachable",
        )
    ):
        return (
            "网络或网关连接异常，当前无法访问 LLM / 计费服务。",
            "检查网络连通性、域名可用性，必要时切换协议或备用地址。",
            "network_error",
            raw,
        )

    if any(token in lowered for token in ("empty content", "empty response", "no content", "无内容")):
        return (
            "LLM 网关返回了空响应，没有生成可用的因子内容。",
            "稍后重试；如果持续出现，优先检查额度、模型状态和网关健康度。",
            "empty_response",
            raw,
        )

    if any(token in lowered for token in ("bad json", "jsondecodeerror", "expecting value", "无法解析")):
        return (
            "LLM 返回了无法解析的 JSON，输出格式不符合预期。",
            "重试该轮请求，或收紧提示词中的 JSON 输出约束。",
            "bad_json",
            raw,
        )

    if "no module named" in lowered:
        return (
            "运行环境缺少必要依赖，导致计算流程无法继续。",
            "安装缺失依赖后重新运行挖掘。",
            "missing_dependency",
            raw,
        )

    if "syntax error" in lowered:
        return (
            "生成出的公式语法不合法，无法进入计算阶段。",
            "检查 DSL 语法约束，或让模型重新生成公式。",
            "syntax_error",
            raw,
        )

    if "runtime error evaluating formula" in lowered:
        return (
            "公式在实际数据计算阶段报错，当前因子无法评估。",
            "检查公式中算子、字段和维度是否匹配。",
            "formula_runtime_error",
            raw,
        )

    if "research report not found" in lowered:
        return (
            "该因子的研究报告还没有生成完成。",
            "等当前评估结束后再刷新页面查看。",
            "report_not_ready",
            raw,
        )

    return (
        raw or "发生了未知错误。",
        "查看日志中的原始报错，必要时重试当前步骤。",
        "unknown",
        raw,
    )


def as_runtime_error(error: Any, status_code: int | None = None) -> AutoAlphaRuntimeError:
    """Wrap a raw exception into AutoAlphaRuntimeError."""
    friendly, suggestion, error_code, raw = humanize_error(error, status_code=status_code)
    return AutoAlphaRuntimeError(
        friendly,
        raw_message=raw,
        suggestion=suggestion,
        error_code=error_code,
    )
