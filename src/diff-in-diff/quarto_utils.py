"""Quarto project configuration helpers.

Updates ``_quarto.yml`` chapter listings after new reports are written.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _update_quarto_chapters(quarto_dir: Path, variation: str) -> None:
    """Scan quarto/<variation>/*/ for report QMDs and update _quarto.yml chapters."""
    quarto_yml = quarto_dir / "_quarto.yml"
    if not quarto_yml.exists():
        logger.warning(
            "_quarto.yml not found at %s, skipping chapter update", quarto_yml
        )
        return

    variation_dir = quarto_dir / variation
    report_qmds = sorted(
        p.relative_to(quarto_dir).as_posix()
        for p in variation_dir.glob("*/report_*.qmd")
    )

    with open(quarto_yml, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    # PyYAML parses `lang: no` / `lang: nb` as boolean False. Restore the
    # Norwegian Bokmål language tag so Quarto doesn't receive `lang: false`.
    try:
        if cfg_yaml["format"]["html"].get("lang") is False:
            cfg_yaml["format"]["html"]["lang"] = "nb"
    except KeyError, TypeError:
        pass

    part_path = f"{variation}/intro.qmd"
    chapters: list[Any] = cfg_yaml["book"]["chapters"]

    part_entry = next(
        (ch for ch in chapters if isinstance(ch, dict) and ch.get("part") == part_path),
        None,
    )
    if part_entry is None:
        part_entry = {"part": part_path, "chapters": []}
        chapters.append(part_entry)

    part_entry["chapters"] = report_qmds

    with open(quarto_yml, "w", encoding="utf-8") as f:
        yaml.dump(
            cfg_yaml, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )

    logger.info(
        "Updated _quarto.yml: %d chapters for '%s'", len(report_qmds), variation
    )


def _update_quarto_triple_diff_chapters(quarto_dir: Path, config_slug: str) -> None:
    """Register a triple-diff multi-chapter folder in _quarto.yml.

    The folder ``quarto/<config_slug>/`` is added as a top-level ``part:`` entry
    with its numbered sub-chapter QMD files.
    """
    quarto_yml = quarto_dir / "_quarto.yml"
    if not quarto_yml.exists():
        logger.warning(
            "_quarto.yml not found at %s, skipping chapter update", quarto_yml
        )
        return

    slug_dir = quarto_dir / config_slug
    sub_qmds = sorted(
        p.relative_to(quarto_dir).as_posix()
        for p in slug_dir.glob("*.qmd")
        if p.name != "intro.qmd"
    )

    with open(quarto_yml, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    try:
        if cfg_yaml["format"]["html"].get("lang") is False:
            cfg_yaml["format"]["html"]["lang"] = "nb"
    except KeyError, TypeError:
        pass

    intro_path = f"{config_slug}/intro.qmd"
    chapters: list[Any] = cfg_yaml["book"]["chapters"]

    part_entry = next(
        (
            ch
            for ch in chapters
            if isinstance(ch, dict) and ch.get("part") == intro_path
        ),
        None,
    )
    if part_entry is None:
        part_entry = {"part": intro_path, "chapters": []}
        chapters.append(part_entry)

    part_entry["chapters"] = sub_qmds

    with open(quarto_yml, "w", encoding="utf-8") as f:
        yaml.dump(
            cfg_yaml, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )

    logger.info(
        "Updated _quarto.yml: %d chapters for triple-diff '%s'",
        len(sub_qmds),
        config_slug,
    )
