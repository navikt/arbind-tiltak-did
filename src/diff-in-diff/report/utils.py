"""Shared utilities for report generation.

Provides common helpers used by both :mod:`report` (standard DiD) and
:mod:`report_triple_diff` (triple-diff) to avoid code duplication.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# ── Nav brand colours ─────────────────────────────────────────────────────────
# Primary palette (report.py / standard DiD)
BLUE = "#003366"
RED = "#C8102E"
LIGHT_BLUE = "#66A3C8"
LIGHT_RED = "#f4a582"

# Extended palette (report_triple_diff.py)
BLUE_DARK = "#254B6D"
RED_DARK = "#B90000"
LIGHT_BLUE_BRIGHT = "#66CBEC"
ORANGE = "#FF9100"
GREEN = "#858E00"
DIRT = "#C1A753"


# ── Figure helpers ────────────────────────────────────────────────────────────


def save_fig(fig: plt.Figure, path: Path) -> None:
    """Save *fig* to *path*, creating parent directories as needed.

    PNG files are saved at 96 dpi; SVG files use vector output.
    Closes the figure after saving.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if path.suffix == ".png":
        kwargs["dpi"] = 96
    fig.savefig(path, **kwargs)
    plt.close(fig)


def rel(path: Path, base: Path) -> str:
    """Return *path* as a POSIX-style relative path string from *base*.

    Uses :func:`os.path.relpath` so the result is always relative even when
    *path* is not a descendant of *base*.  Forward slashes are used regardless
    of platform so the output is safe to embed in Quarto/Markdown files.
    """
    return Path(os.path.relpath(path, base)).as_posix()


# ── Statistical helpers ───────────────────────────────────────────────────────


def sig_stars(p: float) -> str:
    """Return conventional significance stars for p-value *p*."""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ── Config helpers ────────────────────────────────────────────────────────────


def get_tiltak_label(cfg: dict[str, Any]) -> str:
    """Return a human-readable label for the tiltak data source.

    Reads ``data.tiltak_label`` from *cfg* if present; otherwise derives a
    label from the filename of ``data.tiltak_file``.
    """
    explicit = cfg.get("data", {}).get("tiltak_label")
    if explicit:
        return str(explicit)
    path = cfg.get("data", {}).get("tiltak_file", "")
    stem = Path(path).stem.lower()
    if "lønnstilskudd" in stem or "lonnstilskudd" in stem:
        return "midlertidig lønnstilskudd"
    if "alle-tiltak" in stem or "alle_tiltak" in stem:
        return "alle arbeidsmarkedstiltak"
    return stem
