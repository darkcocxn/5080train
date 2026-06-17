from __future__ import annotations

from pathlib import Path


BRIDGE_DIR = Path(__file__).resolve().parent
V8_SOURCE_PATH = BRIDGE_DIR.parent / "2dcnnv8" / "2dcnnv8.py"

globals()["__file__"] = str(V8_SOURCE_PATH)
globals()["__package__"] = None

source = V8_SOURCE_PATH.read_text(encoding="utf-8")
code = compile(source, str(V8_SOURCE_PATH), "exec")
exec(code, globals())
