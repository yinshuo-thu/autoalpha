"""Flat-package shim for running commands from the project root.

The v3 project is intentionally flattened so the project root is also the
`autoalpha_v3` package body.  When Python is launched from this directory,
`import autoalpha_v3.pipeline` would normally look for a nested directory.  This
module marks the root directory as the package search path instead.
"""

from __future__ import annotations

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent)]
