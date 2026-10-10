"""Backlot — the living board.

A read-only, disk-derived board for OpenDrawer. A small local web server
watches ``projects/`` and renders each coloring project's stages, cartoon
candidates, lines, zones, final page, QA report, decisions and activity —
live, plus the pages of a coloring book.

Design contract:
- Observation, not reporting: all state derives from files the pipeline
  already writes. Agents never update the UI.
- Never block, never break: malformed or missing state degrades gracefully.
- The agent's only duty: ``python -m backlot open <project>`` at pipeline init.
"""

__version__ = "0.2.0"

DEFAULT_PORT = 4750
