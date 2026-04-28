"""Make the flat autoalpha_v3 package importable when commands run in repo root."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _path in (_PROJECT_PARENT, _PROJECT_ROOT):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)
