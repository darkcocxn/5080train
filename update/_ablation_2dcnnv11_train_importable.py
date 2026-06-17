# -*- coding: utf-8 -*-
"""Importable wrapper for 2dcnnv11.py during ablation DataLoader workers.

Windows DataLoader workers need Dataset/transform classes to live in an
importable module. The source script directory starts with a digit, so the
ablation runner executes the original file in this module namespace.
"""

from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "2dcnnv11" / "2dcnnv11.py"
exec(compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec"), globals(), globals())

