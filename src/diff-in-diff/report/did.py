"""Markdown report generator for the DID analysis.

Generates a Norwegian-language analytical write-up in ``report.qmd`` in a report folder, covering all employment indicators.  The report is
structured for a non-technical audience familiar with Nav's operations. This is a first draft with minimal text, but with figures and tables included. The report includes:
- Descriptive statistics of the indicators and treatment variable.
- Graphs showing trends over time for treated and control regions.
- Regression results from the DiD analysis, presented in a reader-friendly format.
- Plots of the coefficients for the fixed effects.

All figure helpers and section builders live in :mod:`report_sections`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import pandas as pd
from report.sections import (
    _get_tiltak_label,
    _section_descriptive,
    _section_event_study,
    _section_frontmatter,
    _section_intro,
    _section_leave_one_out,
    _section_placebo,
    _section_regression,
    _section_summary_table,
    _section_tiltak,
)

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0


def generate_report(
    all_results: dict[str, dict[str, Any] | None],
    cfg: dict[str, Any],
    output_path: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Write a Quarto markdown report to *output_path*.

    Parameters
    ----------
    all_results:
        Dict mapping indicator name → result dict (or ``None`` if skipped).
    cfg:
        The loaded analysis config dict.
    output_path:
        Destination ``.qmd`` file path.
    figures_dir:
        Directory where figures are saved (referenced relatively in the QMD).
    tables_dir:
        Directory where CSV tables are saved.
    """
    treatment_start = str(cfg["analysis"]["treatment_start"])
    treatment_type = str(cfg["analysis"].get("treatment_type", "continuous"))
    control_regions: list[str] | None = cfg["analysis"].get("control_regions", None)
    report_title = str(
        cfg["analysis"].get("title", "Nav DID-analyse — effekt av tiltaksnedgang")
    )
    report_dir = output_path.parent

    coef_interp = (
        "Koeffisienten for behandlingsvariabelen angir estimert gjennomsnittlig behandlingseffekt (ATT) "
        "på indikatoren, i prosentpoeng. "
        "Et positivt fortegn betyr at behandlede regioner hadde en høyere verdi på indikatoren "
        "i post-perioden sammenlignet med kontrafaktum; et negativt fortegn betyr lavere verdi. "
        "Størrelsen angir den absolutte endringen i prosentpoeng. "
        if treatment_type == "discrete"
        else "Koeffisienten for behandlingsvariabelen angir estimert effekt av å gå fra null til full "
        "tiltaksnedgang (behandlingsintensitet = 1) på indikatoren, i prosentpoeng. "
        "Et positivt fortegn betyr at regioner med større tiltaksnedgang hadde høyere indikatorverdier "
        "i post-perioden sammenlignet med kontrafaktum; et negativt fortegn betyr lavere verdier. "
        "Størrelsen angir den absolutte endringen i prosentpoeng ved full tiltaksnedgang. "
    )

    ind_labels: dict[str, str] = {
        ind["name"]: ind.get("description", ind["name"])
        for ind in cfg["data"]["indikatorer"]
    }

    lines: list[str] = []
    lines += _section_frontmatter(report_title)
    lines += _section_summary_table(cfg)
    tiltak_label = _get_tiltak_label(cfg)
    lines += _section_intro(
        treatment_start, treatment_type, control_regions, tiltak_label
    )

    first_res = next((r for r in all_results.values() if r is not None), None)
    if first_res is not None:
        lines += _section_tiltak(
            panel=first_res["panel"],
            treatment_start=treatment_start,
            figures_dir=figures_dir,
            report_dir=report_dir,
            tiltak_label=tiltak_label,
        )

    for ind_name, res in all_results.items():
        if res is None:
            continue

        base_ind_name = str(res.get("indicator_name", ind_name))
        label = ind_labels.get(base_ind_name, base_ind_name)
        panel: pd.DataFrame = res["panel"]
        baseline = res["baseline"]
        preferred = res["preferred"]

        lines += [f"## {label} (`{ind_name}`)", ""]

        lines += _section_descriptive(
            panel=panel,
            ind_name=ind_name,
            label=label,
            treatment_start=treatment_start,
            treatment_type=treatment_type,
            control_regions=control_regions,
            figures_dir=figures_dir,
            report_dir=report_dir,
        )

        lines += _section_regression(
            baseline=baseline,
            preferred=preferred,
            bootstrap_baseline=res.get("bootstrap_baseline"),
            bootstrap_preferred=res.get("bootstrap_preferred"),
            coef_interp=coef_interp,
            baseline_mean=res.get("baseline_mean"),
            mde=res.get("mde"),
            ind_name=ind_name,
            label=label,
            figures_dir=figures_dir,
            report_dir=report_dir,
        )

        event_study = res.get("event_study")
        event_study_baseline = res.get("event_study_baseline")
        if event_study is not None and event_study_baseline is not None:
            lines += _section_event_study(
                event_study_baseline=event_study_baseline,
                event_study_preferred=event_study,
                ind_name=ind_name,
                label=label,
                treatment_type=treatment_type,
                figures_dir=figures_dir,
                report_dir=report_dir,
            )

        placebo = res.get("placebo")
        if placebo is not None:
            lines += _section_placebo(
                placebo_baseline=res.get("placebo_baseline"),
                placebo_preferred=placebo,
                baseline=baseline,
                preferred=preferred,
                ind_name=ind_name,
                label=label,
                figures_dir=figures_dir,
                report_dir=report_dir,
            )

        leave_one_out = res.get("leave_one_out")
        leave_one_out_baseline = res.get("leave_one_out_baseline")
        if (
            leave_one_out is not None
            and leave_one_out_baseline is not None
            and not leave_one_out.rows.empty
        ):
            lines += _section_leave_one_out(
                loo_baseline=leave_one_out_baseline,
                loo_preferred=leave_one_out,
                ind_name=ind_name,
                label=label,
                figures_dir=figures_dir,
                report_dir=report_dir,
            )

    lines += [
        "---",
        "",
        "*Rapporten er automatisk generert av analysepipelinen.*",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
