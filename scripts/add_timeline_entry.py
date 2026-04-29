from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


TIMELINE_PATH = Path(__file__).resolve().parents[1] / "frontend" / "src" / "data" / "devTimeline.ts"


def _quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a dev timeline entry.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--bullet", action="append", default=[])
    parser.add_argument("--timestamp", default=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    args = parser.parse_args()

    text = TIMELINE_PATH.read_text(encoding="utf-8")
    marker = "export const devTimeline: DevTimelineEntry[] = ["
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit(f"Cannot find timeline array in {TIMELINE_PATH}")
    insert_at = idx + len(marker)
    tags = args.tag or ["Update"]
    bullets = args.bullet or ["Updated project behavior; validation pending."]
    entry = (
        "\n  {\n"
        f"    timestamp: {_quote(args.timestamp)},\n"
        f"    title: {_quote(args.title)},\n"
        f"    summary: {_quote(args.summary)},\n"
        f"    tags: [{', '.join(_quote(t) for t in tags)}],\n"
        "    bullets: [\n"
        + "".join(f"      {_quote(b)},\n" for b in bullets)
        + "    ],\n"
        "  },"
    )
    TIMELINE_PATH.write_text(text[:insert_at] + entry + text[insert_at:], encoding="utf-8")


if __name__ == "__main__":
    main()
