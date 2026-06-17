# -*- coding: utf-8 -*-
"""Importable wrapper for 2dcnnv11test.py during ablation DataLoader workers."""

from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "2dcnnv11" / "2dcnnv11test.py"
exec(compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec"), globals(), globals())

