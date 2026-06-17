# -*- coding: utf-8 -*-
"""Rebuild the publication figure manifest and generated figures.

This wrapper uses the same non-training production path as
``make_paper_tables.py`` so figure/table paths stay synchronized in
``figure_table_manifest.json``.
"""

from __future__ import annotations

from complete_remaining_todos import main


if __name__ == "__main__":
    main()
