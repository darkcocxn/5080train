# -*- coding: utf-8 -*-
"""Rebuild the publication table manifest and generated tables.

This lightweight wrapper delegates to ``complete_remaining_todos.py`` because
the remaining TODO14 table outputs depend on the final TODO12-TODO16 package.
It is kept as a stable paper-production entry point.
"""

from __future__ import annotations

from complete_remaining_todos import main


if __name__ == "__main__":
    main()
