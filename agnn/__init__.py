"""AGNN - Adaptive Goal Negotiation Network"""

from __future__ import annotations

import sys


def _enable_safe_stdio() -> None:
    """
    Windows consoles often expose cp1252/cp437 stdout encodings. AGNN prints
    box-drawing characters in several modules, so fail closed by replacing
    unsupported characters instead of crashing the whole run.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            continue


_enable_safe_stdio()
