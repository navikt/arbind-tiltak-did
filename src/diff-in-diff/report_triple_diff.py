"""Multi-chapter report generator for the triple-diff analysis.

Generates four separate ``.qmd`` files per config, plus an intro, for
the triple-diff deep analysis.  Each file is self-contained with its own
figures.

Chapters:
    1. ``intro.qmd``               — Part intro (brief description)
    2. ``01-deskriptiv.qmd``       — Descriptive statistics & background
    3. ``02-regresjonsresultater.qmd`` — Regression results & coefficients
    4. ``03-bootstrap.qmd``        — Bootstrap & inference
    5. ``04-antagelser.qmd``       — Assumption tests
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0

if TYPE_CHECKING:
    from cluster_bootstrap import BootstrapResult
    from event_study import EventStudyResult
    from regression import LeaveOneOutResult

logger = logging.getLogger(__name__)

_BLUE = "#003366"
_RED = "#C8102E"
_LIGHT_BLUE = "#66A3C8"
_LIGHT_RED = "#f4a582"
_GREEN = "#2E8B57"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if path.suffix == ".png":
        kwargs["dpi"] = 96
    fig.savefig(path, **kwargs)
    plt.close(fig)


def _rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _sig_stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ── Figures ──────────────────────────────────────────────────────────────────


def _plot_group_trends(
    panel: pd.DataFrame,
    indicator_name: str,
    figures_dir: Path,
) -> Path:
    """Plot indicator trends for treated vs control group over time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for group_label, color in [("treated", _RED), ("control", _BLUE)]:
        sub = panel[panel["group"] == group_label]
        monthly = sub.groupby("aarmnd")["indikator"].mean().sort_index()
        ax.plot(
            monthly.index,
            monthly.values,
            color=color,
            linewidth=1.5,
            label="Veiledning" if group_label == "treated" else "Gode muligheter",
        )

    treatment_start = panel.loc[panel["relative_month"] == 0, "aarmnd"]
    if not treatment_start.empty:
        ax.axvline(treatment_start.iloc[0], color="grey", linestyle="--", alpha=0.7)

    ax.set_xlabel("År-måned")
    ax.set_ylabel("Indikator")
    ax.set_title(f"{indicator_name} — gjennomsnitt per gruppe over tid")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()

    path = figures_dir / f"group_trends_{indicator_name}.svg"
    _save_fig(fig, path)
    return path


def _plot_treatment_intensity(
    panel: pd.DataFrame,
    indicator_name: str,
    figures_dir: Path,
) -> Path:
    """Plot treatment intensity (tiltaksnedgang) per region over time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    regions = sorted(panel["region"].unique())
    for region in regions:
        sub = panel[(panel["region"] == region) & (panel["group"] == "treated")]
        if sub.empty:
            sub = panel[panel["region"] == region]
        monthly = sub.drop_duplicates(subset=["aarmnd"]).sort_values("aarmnd")
        ax.plot(
            monthly["aarmnd"],
            monthly["tiltaksnedgang"],
            linewidth=1,
            alpha=0.7,
            label=region,
        )

    ax.set_xlabel("År-måned")
    ax.set_ylabel("Tiltaksnedgang")
    ax.set_title(f"Behandlingsintensitet per region — {indicator_name}")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()

    path = figures_dir / f"treatment_intensity_{indicator_name}.svg"
    _save_fig(fig, path)
    return path


def _plot_event_study(
    event_study: EventStudyResult,
    indicator_name: str,
    label: str,
    figures_dir: Path,
) -> Path:
    """Plot event study coefficients with CI."""
    fig, ax = plt.subplots(figsize=(10, 5))

    taus = [c.tau for c in event_study.coefs]
    coefs = [c.coefficient for c in event_study.coefs]
    ci_lo = [c.ci_lower for c in event_study.coefs]
    ci_hi = [c.ci_upper for c in event_study.coefs]

    ax.fill_between(taus, ci_lo, ci_hi, alpha=0.2, color=_BLUE)
    ax.plot(taus, coefs, color=_BLUE, marker="o", markersize=3, linewidth=1.2)
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.8)
    ax.axvline(-0.5, color=_RED, linestyle="--", alpha=0.6)

    ax.set_xlabel("Relativ måned (τ)")
    ax.set_ylabel("Koeffisient")
    ax.set_title(f"Hendelsesstudie — {indicator_name} ({label})")
    ax.grid(axis="y", alpha=0.3)

    path = figures_dir / f"event_study_{indicator_name}_{label}.svg"
    _save_fig(fig, path)
    return path


def _plot_bootstrap_dist(
    bootstrap: BootstrapResult,
    indicator_name: str,
    label: str,
    figures_dir: Path,
) -> Path:
    """Plot bootstrap t-statistic distribution."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(bootstrap.bootstrap_t_stats, bins=80, density=True, alpha=0.6, color=_BLUE)
    ax.axvline(
        bootstrap.observed_t_stat,
        color=_RED,
        linewidth=2,
        label=f"Observert t = {bootstrap.observed_t_stat:.2f}",
    )
    ax.axvline(
        -abs(bootstrap.observed_t_stat),
        color=_RED,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )
    ax.axvline(
        abs(bootstrap.observed_t_stat),
        color=_RED,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel("Bootstrap t-statistikk")
    ax.set_ylabel("Tetthet")
    ax.set_title(f"Wild cluster bootstrap — {indicator_name} ({label})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = figures_dir / f"bootstrap_{indicator_name}_{label}.svg"
    _save_fig(fig, path)
    return path


def _plot_leave_one_out(
    loo: LeaveOneOutResult,
    indicator_name: str,
    label: str,
    figures_dir: Path,
) -> Path:
    """Plot leave-one-out coefficients."""
    fig, ax = plt.subplots(figsize=(10, 5))

    rows = loo.rows.sort_values("coefficient")
    y_pos = range(len(rows))
    ax.errorbar(
        rows["coefficient"],
        y_pos,
        xerr=[
            rows["coefficient"] - rows["ci_lower"],
            rows["ci_upper"] - rows["coefficient"],
        ],
        fmt="o",
        color=_BLUE,
        markersize=4,
        capsize=3,
    )
    ax.axvline(
        loo.full_coefficient,
        color=_RED,
        linestyle="-",
        linewidth=1.5,
        label="Fullmodell",
    )
    ax.axvspan(loo.full_ci_lower, loo.full_ci_upper, alpha=0.1, color=_RED)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(rows["dropped_region"].tolist(), fontsize=8)
    ax.set_xlabel("Koeffisient")
    ax.set_title(f"Leave-one-out — {indicator_name} ({label})")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    path = figures_dir / f"loo_{indicator_name}_{label}.svg"
    _save_fig(fig, path)
    return path


# ── Chapter generators ───────────────────────────────────────────────────────


def _generate_intro(
    cfg: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate the part intro QMD."""
    title = cfg["analysis"].get("title", "Trippel-diff-analyse")
    analysis_level = cfg["analysis"].get("analysis_level", "region")
    treated = cfg["analysis"].get("treated_group", "veiledning")
    control = cfg["analysis"].get("control_group", "gode muligheter")

    lines = [
        "---",
        f'title: "{title}"',
        "---",
        "",
        f"Trippel-diff-analyse med behandlet gruppe **{treated}** og kontrollgruppe **{control}**.",
        f"Analysenivå: **{analysis_level}**.",
        "",
        "Analysen er delt inn i fire kapitler:",
        "",
        "1. Deskriptiv statistikk og bakgrunn",
        "2. Regresjonsresultater og koeffisienter",
        "3. Bootstrap og inferens",
        "4. Test av antagelser",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_descriptive(
    all_results: dict[str, dict[str, Any] | None],
    cfg: dict[str, Any],
    output_path: Path,
    figures_dir: Path,
) -> None:
    """Generate chapter 1: descriptive statistics."""
    title = cfg["analysis"].get("title", "Trippel-diff")
    treated = cfg["analysis"].get("treated_group", "veiledning")
    control = cfg["analysis"].get("control_group", "gode muligheter")
    analysis_level = cfg["analysis"].get("analysis_level", "region")

    lines = [
        "---",
        f'title: "Deskriptiv statistikk — {title}"',
        "---",
        "",
        "## Oversikt",
        "",
        f"Behandlet gruppe: **{treated}**  ",
        f"Kontrollgruppe: **{control}**  ",
        f"Analysenivå: **{analysis_level}**",
        "",
    ]

    for name, res in all_results.items():
        if res is None:
            continue

        panel = res["panel"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Antall observasjoner: **{len(panel)}**",
                f"- Antall regioner: **{panel['region'].nunique()}**",
            ]
        )
        if "entity" in panel.columns:
            lines.append(f"- Antall enheter: **{panel['entity'].nunique()}**")
        lines.extend(
            [
                f"- Antall måneder: **{panel['aarmnd'].nunique()}**",
                f"- Behandlet gruppe (n): **{int((panel['treated'] == 1.0).sum())}**",
                f"- Kontrollgruppe (n): **{int((panel['treated'] == 0.0).sum())}**",
                f"- Gjennomsnittlig indikator (pre, behandlet): **{res.get('baseline_mean_treated', 0):.4f}**",
                f"- Gjennomsnittlig indikator (pre, kontroll): **{res.get('baseline_mean_control', 0):.4f}**",
                "",
            ]
        )

        # Group trend figure
        fig_path = _plot_group_trends(panel, name, figures_dir)
        lines.extend(
            [
                f"### Trender over tid — {name}",
                "",
                f"![Gruppesammenlikning {name}]({_rel(fig_path, output_path.parent)})",
                "",
            ]
        )

        # Treatment intensity figure
        fig_path = _plot_treatment_intensity(panel, name, figures_dir)
        lines.extend(
            [
                f"### Behandlingsintensitet — {name}",
                "",
                f"![Behandlingsintensitet {name}]({_rel(fig_path, output_path.parent)})",
                "",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_regression(
    all_results: dict[str, dict[str, Any] | None],
    cfg: dict[str, Any],
    output_path: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Generate chapter 2: regression results."""
    title = cfg["analysis"].get("title", "Trippel-diff")

    lines = [
        "---",
        f'title: "Regresjonsresultater — {title}"',
        "---",
        "",
        "## Hovedresultater",
        "",
        "Trippel-diff-koeffisienten ($\\beta$) måler den differensielle effekten ",
        "av tiltaksnedgangen på den behandlede gruppen sammenlignet med kontrollgruppen.",
        "",
    ]

    # Summary table
    rows = []
    for name, res in all_results.items():
        if res is None:
            continue
        for model_label, result, boot_key in [
            ("Basis", res["baseline"], "bootstrap_baseline"),
            ("Sesongjustert", res["preferred"], "bootstrap_preferred"),
        ]:
            boot = res.get(boot_key)
            bp = boot.bootstrap_p_value if boot else None
            stars = _sig_stars(bp) if bp is not None else ""
            rows.append(
                f"| {name} | {model_label} | {result.coefficient:.4f}{stars} "
                f"| {result.std_error:.4f} | {result.t_stat:.3f} "
                f"| {result.p_value:.4f} "
                f"| {bp:.4f} | [{result.ci_lower:.4f}, {result.ci_upper:.4f}] "
                f"| {result.n_obs} | {result.n_clusters} |"
            )

    if rows:
        lines.extend(
            [
                "| Indikator | Modell | Koeffisient | SE | t | p (asymp.) | p (bootstrap) | 95% KI | N | Clustere |",
                "|:--|:--|--:|--:|--:|--:|--:|:--|--:|--:|",
                *rows,
                "",
                "::: {.callout-note}",
                "Signifikansnivå basert på bootstrap p-verdi: \\*\\*\\* p < 0,01; \\*\\* p < 0,05; \\* p < 0,10.",
                ":::",
                "",
            ]
        )

    for name, res in all_results.items():
        if res is None:
            continue
        lines.extend(
            [
                f"## Detaljert: {name}",
                "",
                f"- MDE (minimum detekterbar effekt): **{res.get('mde', 0):.4f}** prosentpoeng",
                f"- Gjennomsnittlig indikator i pre-perioden: **{res.get('baseline_mean', 0):.4f}**",
                "",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_bootstrap(
    all_results: dict[str, dict[str, Any] | None],
    cfg: dict[str, Any],
    output_path: Path,
    figures_dir: Path,
) -> None:
    """Generate chapter 3: bootstrap inference."""
    title = cfg["analysis"].get("title", "Trippel-diff")

    lines = [
        "---",
        f'title: "Bootstrap og inferens — {title}"',
        "---",
        "",
        "## Wild cluster bootstrap",
        "",
        "Bootstrap-p-verdien er det primære inferensresultatet. Alle standardfeil ",
        "er clustret på regionsnivå. Webb (6-punkt) vekter brukes for å håndtere ",
        "det lave antallet clustere (G = 12).",
        "",
    ]

    for name, res in all_results.items():
        if res is None:
            continue
        for label, boot_key in [
            ("basis", "bootstrap_baseline"),
            ("sesongjustert", "bootstrap_preferred"),
        ]:
            boot = res.get(boot_key)
            if boot is None:
                continue
            lines.extend(
                [
                    f"### {name} — {label}",
                    "",
                    f"- Observert koeffisient: **{boot.observed_coefficient:.4f}**",
                    f"- Observert SE: **{boot.observed_se:.4f}**",
                    f"- Observert t-statistikk: **{boot.observed_t_stat:.3f}**",
                    f"- Bootstrap p-verdi: **{boot.bootstrap_p_value:.4f}**",
                    f"- Antall replikasjoner: **{boot.n_boot}**",
                    "",
                ]
            )

            fig_path = _plot_bootstrap_dist(boot, name, label, figures_dir)
            lines.extend(
                [
                    f"![Bootstrap-fordeling {name} ({label})]({_rel(fig_path, output_path.parent)})",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_assumptions(
    all_results: dict[str, dict[str, Any] | None],
    cfg: dict[str, Any],
    output_path: Path,
    figures_dir: Path,
) -> None:
    """Generate chapter 4: assumption tests."""
    title = cfg["analysis"].get("title", "Trippel-diff")

    lines = [
        "---",
        f'title: "Test av antagelser — {title}"',
        "---",
        "",
        "## Hendelsesstudie",
        "",
        "Trippel-diff-hendelsesstudien interagerer regionens behandlingsintensitet ",
        "med gruppeindikator og tidsperiode. Koeffisientene i pre-perioden bør være ",
        "nær null under parallell-trendantagelsen.",
        "",
    ]

    for name, res in all_results.items():
        if res is None:
            continue
        for label, es_key in [
            ("sesongjustert", "event_study"),
            ("basis", "event_study_baseline"),
        ]:
            event_study = res.get(es_key)
            if event_study is None:
                continue
            lines.extend(
                [
                    f"### {name} — {label}",
                    "",
                    f"- Pre-trend F-test: F({event_study.pretrend_df_num}, {event_study.pretrend_df_denom}) = {event_study.pretrend_f_stat:.3f}, p = {event_study.pretrend_p_value:.4f}",
                    "",
                ]
            )
            fig_path = _plot_event_study(event_study, name, label, figures_dir)
            lines.extend(
                [
                    f"![Hendelsesstudie {name} ({label})]({_rel(fig_path, output_path.parent)})",
                    "",
                ]
            )

    # Placebo tests
    lines.extend(
        [
            "## Placebotest",
            "",
            "Placebo-analysen bruker en falsk behandlingsdato 12 måneder før den virkelige, ",
            "og estimerer trippel-diff-modellen på pre-perioden. En nær-null koeffisient ",
            "tyder på at parallell-trendantagelsen holder.",
            "",
        ]
    )
    for name, res in all_results.items():
        if res is None:
            continue
        for label, p_key in [
            ("sesongjustert", "placebo"),
            ("basis", "placebo_baseline"),
        ]:
            placebo = res.get(p_key)
            if placebo is None:
                lines.append(f"- **{name}** ({label}): Ikke nok pre-periodedata.\n")
                continue
            stars = _sig_stars(placebo.p_value)
            lines.extend(
                [
                    f"### {name} — {label}",
                    "",
                    f"- Koeffisient: **{placebo.coefficient:.4f}**{stars}",
                    f"- SE: {placebo.std_error:.4f}",
                    f"- p-verdi: {placebo.p_value:.4f}",
                    "",
                ]
            )

    # Leave-one-out
    lines.extend(
        [
            "## Leave-one-out",
            "",
            "Modellen re-estimeres med én region utelatt om gangen. ",
            "Stabile koeffisienter viser at ingen enkeltregion driver resultatene.",
            "",
        ]
    )
    for name, res in all_results.items():
        if res is None:
            continue
        for label, loo_key in [
            ("sesongjustert", "leave_one_out"),
            ("basis", "leave_one_out_baseline"),
        ]:
            loo = res.get(loo_key)
            if loo is None:
                continue
            fig_path = _plot_leave_one_out(loo, name, label, figures_dir)
            lines.extend(
                [
                    f"### {name} — {label}",
                    "",
                    f"![Leave-one-out {name} ({label})]({_rel(fig_path, output_path.parent)})",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main entry point ─────────────────────────────────────────────────────────


def generate_triple_diff_report(
    all_results: dict[str, dict[str, Any] | None],
    cfg: dict[str, Any],
    report_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Generate all four triple-diff report chapters plus intro.

    Parameters
    ----------
    all_results:
        Dict mapping indicator name to a :class:`TripleDiffResult` dict (or None).
    cfg:
        The loaded YAML config.
    report_dir:
        Directory to write the QMD files into.
    figures_dir:
        Directory for figure output.
    tables_dir:
        Directory for table output.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _generate_intro(cfg, report_dir / "intro.qmd")
    _generate_descriptive(
        all_results, cfg, report_dir / "01-deskriptiv.qmd", figures_dir
    )
    _generate_regression(
        all_results,
        cfg,
        report_dir / "02-regresjonsresultater.qmd",
        figures_dir,
        tables_dir,
    )
    _generate_bootstrap(all_results, cfg, report_dir / "03-bootstrap.qmd", figures_dir)
    _generate_assumptions(
        all_results, cfg, report_dir / "04-antagelser.qmd", figures_dir
    )

    logger.info("Generated %d triple-diff report chapters in %s", 5, report_dir)
