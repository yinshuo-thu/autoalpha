import os

import requests


class FeishuNotifier:
    def __init__(self, webhook_url=None):
        """
        webhook_url: 飞书群机器人的 Webhook 地址
        """
        self.webhook_url = webhook_url or os.environ.get("FEISHU_WEBHOOK_URL", "")

    @staticmethod
    def _clip(text, limit=800):
        if text is None:
            return ""
        value = str(text).strip()
        if len(value) <= limit:
            return value
        return value[: limit - 1] + "…"

    def _post_payload(self, payload):
        if not self.webhook_url:
            print("[Feishu] Warning: No Webhook URL provided. Skipping notification.")
            return False
        try:
            res = requests.post(self.webhook_url, json=payload, timeout=5)
            data = res.json()
            if data.get("code") != 0:
                print(f"[Feishu] Webhook sending failed (Code {data.get('code')}): {data.get('msg')}")
                return False
            print("[Feishu] Webhook notification sent successfully.")
            return True
        except Exception as e:
            print(f"[Feishu] Network error when sending webhook: {e}")
            return False

    def send_factor_notification_from_metadata(self, metadata: dict):
        """
        Build the card strictly from submit metadata JSON (single source of truth).
        Expected keys: display_name, factor_name, Score, PassGates, IC, rank_ic, IR, Turnover,
        formula, description, hypothesis, sequence_index, timestamp.
        """
        display = metadata.get("display_name") or metadata.get("factor_name") or "unknown"
        count_text = ""

        score = float(metadata.get("Score", 0.0))
        ic = float(metadata.get("IC", 0.0))
        ir = float(metadata.get("IR", 0.0))
        turnover = float(metadata.get("Turnover", 0.0))
        pass_gates = bool(metadata.get("PassGates", False))
        color = "green" if pass_gates else "red"
        formula = metadata.get("formula") or "N/A"
        hyp = metadata.get("hypothesis") or metadata.get("description") or ""
        tldr = self._clip(metadata.get("tldr") or metadata.get("summary") or "", 90)

        card_content = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"✨ AutoAlpha 新成效因子产出{count_text}: {display}",
                },
                "template": color,
            },
            "elements": [
                {
                    "tag": "div",
                    "fields": [
                        {
                            "is_short": True,
                            "text": {"tag": "lark_md", "content": f"**Score**\n{score:.4f}"},
                        },
                        {
                            "is_short": True,
                            "text": {"tag": "lark_md", "content": f"**IC / IR / TVR**\n{ic:.4f} / {ir:.3f} / {turnover:.2f}"},
                        },
                        {
                            "is_short": True,
                            "text": {"tag": "lark_md", "content": f"**TL;DR**\n{tldr or '-'}"},
                        },
                    ],
                },
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**公式**\n{formula}"}},
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**AI 假设 / 描述**\n{hyp}"},
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": f"Mining Time: {metadata.get('timestamp', '')}",
                        }
                    ],
                },
            ],
        }

        payload = {"msg_type": "interactive", "card": card_content}
        return self._post_payload(payload)

    def send_error_notification(
        self,
        title,
        summary,
        *,
        stage="AutoAlpha",
        error_code="unknown",
        suggestion="",
        raw_detail="",
        run_id="",
        formula="",
        timestamp="",
    ):
        """Send a human-readable error card for pipeline/loop failures."""
        display_summary = self._clip(summary, 320)
        display_suggestion = self._clip(suggestion, 240)
        display_raw = self._clip(raw_detail, 700)
        display_formula = self._clip(formula, 320)

        fields = [
            {
                "is_short": True,
                "text": {"tag": "lark_md", "content": f"**阶段**\n{stage}"},
            },
            {
                "is_short": True,
                "text": {"tag": "lark_md", "content": f"**错误类型**\n{error_code}"},
            },
        ]

        if run_id:
            fields.append(
                {
                    "is_short": True,
                    "text": {"tag": "lark_md", "content": f"**Run ID**\n{self._clip(run_id, 80)}"},
                }
            )
        if timestamp:
            fields.append(
                {
                    "is_short": True,
                    "text": {"tag": "lark_md", "content": f"**时间**\n{self._clip(timestamp, 80)}"},
                }
            )

        elements = [
            {
                "tag": "div",
                "fields": fields,
            },
            {"tag": "hr"},
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**可理解报错**\n{display_summary}"},
            },
        ]

        if display_suggestion:
            elements.append(
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**建议处理**\n{display_suggestion}"},
                }
            )
        if display_formula:
            elements.append(
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**相关公式**\n`{display_formula}`"},
                }
            )
        if display_raw:
            elements.append(
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**原始错误**\n{display_raw}"},
                }
            )

        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"🚨 {self._clip(title, 80)}",
                    },
                    "template": "red",
                },
                "elements": elements,
            },
        }
        return self._post_payload(payload)

    def send_factor_notification(self, factor_name, description, metrics, formula, timestamp, valid_count=None):
        """Legacy wrapper — normalizes to metadata shape."""
        meta = {
            "display_name": factor_name,
            "factor_name": factor_name,
            "Score": float(metrics.get("Score", 0.0)),
            "PassGates": bool(metrics.get("PassGates", False)),
            "IC": float(metrics.get("IC", 0.0)),
            "rank_ic": float(metrics.get("rank_ic", 0.0)),
            "IR": float(metrics.get("IR", 0.0)),
            "Turnover": float(metrics.get("Turnover", 0.0)),
            "tldr": metrics.get("tldr") or metrics.get("summary") or "",
            "formula": formula,
            "hypothesis": description,
            "timestamp": timestamp,
            "sequence_index": valid_count,
        }
        return self.send_factor_notification_from_metadata(meta)


# 单例提供给整个系统调用
default_notifier = FeishuNotifier()
