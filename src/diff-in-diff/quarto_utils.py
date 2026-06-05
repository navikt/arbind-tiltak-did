"""Quarto project configuration helpers.

Updates ``_quarto.yml`` chapter listings after new reports are written.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _fix_lang(cfg_yaml: dict[str, Any]) -> None:
    """PyYAML parses ``lang: no`` / ``lang: nb`` as boolean False.

    Restore the Norwegian Bokmål language tag so Quarto doesn't receive
    ``lang: false``.
    """
    try:
        if cfg_yaml["format"]["html"].get("lang") is False:
            cfg_yaml["format"]["html"]["lang"] = "nb"
    except KeyError, TypeError:
        pass


def _update_quarto_chapters(quarto_dir: Path, variation: str) -> None:
    """Scan ``quarto/<variation>/*/`` for report QMDs and update ``_quarto.yml``.

    The variation path may be nested (e.g., ``did/midl-lonnstilskudd``).  The
    corresponding part entry in ``_quarto.yml`` uses ``<variation>/intro.qmd``
    as its title page and lists all ``report_*.qmd`` files one level below.
    """
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

    _fix_lang(cfg_yaml)

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


def _update_quarto_triple_diff_chapters(quarto_dir: Path, variation: str) -> None:
    """Register a triple-diff multi-chapter section in ``_quarto.yml``.

    The folder ``quarto/<variation>/`` is added as a ``part:`` entry with its
    numbered sub-chapter QMD files (all ``*.qmd`` except ``intro.qmd``).

    The ``variation`` field from the config (e.g.
    ``triple-diff/midl-lonnstilskudd/regioner``) doubles as the quarto output
    path, so the intro is at ``quarto/<variation>/intro.qmd``.
    """
    quarto_yml = quarto_dir / "_quarto.yml"
    if not quarto_yml.exists():
        logger.warning(
            "_quarto.yml not found at %s, skipping chapter update", quarto_yml
        )
        return

    section_dir = quarto_dir / variation
    sub_qmds = sorted(
        p.relative_to(quarto_dir).as_posix()
        for p in section_dir.glob("*.qmd")
        if p.name != "intro.qmd"
    )

    with open(quarto_yml, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    _fix_lang(cfg_yaml)

    intro_path = f"{variation}/intro.qmd"
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
        variation,
    )
