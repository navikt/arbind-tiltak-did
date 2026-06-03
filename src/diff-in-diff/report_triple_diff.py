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
from report_utils import (
    BLUE_DARK as _BLUE,
)
from report_utils import (
    RED_DARK as _RED,
)
from report_utils import (
    get_tiltak_label as _get_tiltak_label,
)
from report_utils import (
    rel as _rel,
)
from report_utils import (
    save_fig as _save_fig,
)
from report_utils import (
    sig_stars as _sig_stars,
)

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0

if TYPE_CHECKING:
    from cluster_bootstrap import BootstrapResult
    from event_study import EventStudyResult
    from regression import LeaveOneOutResult

logger = logging.getLogger(__name__)


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
        "## Bakgrunn og metode",
        "",
        "Denne analysen estimerer om nedgangen i arbeidsmarkedstiltak har hatt en",
        f"*differensiell* effekt på innsatsgruppen **{treated}** sammenlignet med",
        f"**{control}**. Logikken er at dersom tiltaksnedgangen rammer de to gruppene",
        "likt, vil en standard diff-in-diff fange opp felles sjokk — trippel-diff",
        "isolerer i stedet den *ekstra* effekten for den behandlede gruppen.",
        "",
        f"Analysen opererer på **{analysis_level}**-nivå. Behandlingsvariabelen",
        "`tiltaksnedgang` er den samme som i den ordinære DiD-analysen: andelen av",
        f"regionens pre-periode-topp i *{_get_tiltak_label(cfg)}* som er avviklet.",
        "",
        "Modellen inkluderer region-faste effekter, tidspunkt-faste effekter og en",
        f"gruppe × tid-interaksjon for å skille {treated}-trenden fra den nasjonale",
        "trenden. Den foretrukne spesifikasjonen legger i tillegg til region ×",
        "kalendermåned-faste effekter for å absorbere regionspesifikke sesongmønstre.",
        "",
        "> **Signifikansnivå:** \\* p < 0,10 &nbsp; \\*\\* p < 0,05 &nbsp; \\*\\*\\* p < 0,01  ",
        "> Standardfeil er clustret på regionnivå (CR1 småutvalgskorrigering).  ",
        "> Primær p-verdi basert på wild cluster bootstrap med Webb-vekter (B = 4 999).",
        "",
        "Analysen er delt inn i fire kapitler:",
        "",
        "1. **Deskriptiv statistikk og bakgrunn** — dataoversikt, trender og behandlingsintensitet",
        "2. **Regresjonsresultater og koeffisienter** — trippel-diff-estimatene",
        "3. **Bootstrap og inferens** — wild cluster bootstrap-fordelinger",
        "4. **Test av antagelser** — hendelsesstudie, placebo og leave-one-out",
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
        "Dette kapittelet gir en oversikt over datamaterialet og viser trender for de",
        "to gruppene over tid. Et viktig visuelt kriterium for trippel-diff-strategien",
        f"er at **{treated}** og **{control}** fulgte parallelle trender i",
        "pre-perioden — avvik her vil svekke identifikasjonen.",
        "",
    ]

    for name, res in all_results.items():
        if res is None:
            continue

        panel = res["panel"]
        n_treated = int((panel["treated"] == 1.0).sum())
        n_control = int((panel["treated"] == 0.0).sum())
        mean_treated = res.get("baseline_mean_treated", 0)
        mean_control = res.get("baseline_mean_control", 0)
        mean_diff = mean_treated - mean_control

        lines.extend(
            [
                f"## {name}",
                "",
                "### Dataoversikt",
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
                f"- Behandlet gruppe ({treated}, n): **{n_treated}**",
                f"- Kontrollgruppe ({control}, n): **{n_control}**",
                "",
                "**Gjennomsnittlig indikatorverdi i pre-perioden:**",
                "",
                "| Gruppe | Gjennomsnitt |",
                "|:--|--:|",
                f"| {treated} (behandlet) | {mean_treated:.4f} |",
                f"| {control} (kontroll) | {mean_control:.4f} |",
                f"| Differanse | {mean_diff:.4f} |",
                "",
                f"I pre-perioden lå {treated} i gjennomsnitt **{abs(mean_diff):.4f}** poeng",
                f"{'høyere' if mean_diff > 0 else 'lavere'} enn {control} på indikatoren `{name}`.",
                "Denne nivåforskjellen absorberes av gruppe-faste effekter i regresjonen.",
                "",
            ]
        )

        # Group trend figure
        fig_path = _plot_group_trends(panel, name, figures_dir)
        lines.extend(
            [
                f"### Trender over tid — {name}",
                "",
                "Figuren viser gjennomsnittlig indikatorverdi per måned for de to gruppene.",
                "Den stiplede linjen markerer behandlingsstart. Parallelle trender i",
                "pre-perioden er en forutsetning for identifikasjonen.",
                "",
                f"![Gruppesammenlikning {name}]({_rel(fig_path, output_path.parent)})",
                "",
            ]
        )

        # Treatment intensity figure
        fig_path = _plot_treatment_intensity(panel, name, figures_dir)
        lines.extend(
            [
                f"### Behandlingsintensitet per region — {name}",
                "",
                "Figuren viser tiltaksnedgangen per region over tid. Regionene skiller seg",
                "fra hverandre i intensitet, noe som er kilden til identifikasjonen i den",
                "kontinuerlige trippel-diff-modellen.",
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
    treated = cfg["analysis"].get("treated_group", "veiledning")
    control = cfg["analysis"].get("control_group", "gode muligheter")
    tiltak_label = _get_tiltak_label(cfg)

    lines = [
        "---",
        f'title: "Regresjonsresultater — {title}"',
        "---",
        "",
        "## Hovedresultater",
        "",
        "Trippel-diff-koeffisienten ($\\beta_{DDD}$) måler den *differensielle* effekten",
        f"av nedgangen i *{tiltak_label}* på **{treated}** sammenlignet med **{control}**.",
        "Koeffisienten angir estimert effekt av å gå fra null til full tiltaksnedgang",
        "(behandlingsintensitet = 1) på indikatoren, i prosentpoeng, *utover* den effekten",
        f"som også observeres for {control}.",
        "",
        "To modellspesifikasjoner estimeres:",
        "",
        "- **Basis:** Region FE + år-måned FE + gruppe × tid",
        "- **Sesongjustert (foretrukket):** Basis + region × kalendermåned FE",
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
                "Signifikansnivå basert på bootstrap p-verdi: \\*\\*\\* p < 0,01; \\*\\* p < 0,05; \\* p < 0,10.  ",
                "Standardfeil er clustret på regionnivå (CR1 småutvalgskorrigering).  ",
                "Med et lavt antall clustere er bootstrap-p-verdien det primære inferensgrunnlaget.",
                ":::",
                "",
            ]
        )

    for name, res in all_results.items():
        if res is None:
            continue

        baseline = res["baseline"]
        preferred = res["preferred"]
        boot_pref = res.get("bootstrap_preferred")
        boot_base = res.get("bootstrap_baseline")
        baseline_mean = res.get("baseline_mean", 0)
        mde = res.get("mde", 0)

        pref_p = boot_pref.bootstrap_p_value if boot_pref else preferred.p_value
        base_p = boot_base.bootstrap_p_value if boot_base else baseline.p_value
        pref_stars = _sig_stars(pref_p)
        base_stars = _sig_stars(base_p)

        pref_dir = "positiv" if preferred.coefficient > 0 else "negativ"
        pref_sig = (
            f"statistisk signifikant på {10 if pref_p < 0.10 else 5 if pref_p < 0.05 else 1}%-nivå ({pref_stars})"
            if pref_stars
            else "ikke statistisk signifikant"
        )

        rel_change = (
            (preferred.coefficient / baseline_mean * 100)
            if baseline_mean != 0
            else float("nan")
        )

        lines.extend(
            [
                f"## Detaljert: {name}",
                "",
                "Den foretrukne (sesongjusterte) modellen gir en koeffisient på",
                f"**{preferred.coefficient:.4f}** (SE = {preferred.std_error:.4f},",
                f"t = {preferred.t_stat:.3f}, bootstrap p = {pref_p:.3f}{pref_stars}).",
                f"Estimatet er **{pref_sig}** og har **{pref_dir}** fortegn.",
            ]
        )

        if baseline_mean != 0 and not (rel_change != rel_change):  # not NaN
            lines.append(
                f"Relativt til det gjennomsnittlige pre-periode-nivået ({baseline_mean:.4f})"
                f" tilsvarer dette en relativ endring på **{rel_change:.1f} %**."
            )

        lines.extend(
            [
                "",
                f"Basis-modellen gir koeffisient {baseline.coefficient:.4f}"
                f" (bootstrap p = {base_p:.3f}{base_stars}).",
                f"At de to modellene {'er i rimelig overenstemmelse' if abs(preferred.coefficient - baseline.coefficient) < 0.05 else 'avviker noe fra hverandre'}"
                f" gir et inntrykk av om sesongmønsteret spiller en rolle for resultatet.",
                "",
                f"**Minimum detekterbar effekt (80 % styrke, α = 0,05):** ±{mde:.4f} pp.  ",
                f"**Gjennomsnittlig pre-periode-nivå:** {baseline_mean:.4f}",
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
        "Med et lavt antall regioner (clustere) er asymptotisk clusterinferens upålitelig.",
        "Primær p-verdi er derfor basert på **wild cluster bootstrap** med Webb 6-punkt-vekter",
        "(B = 4 999 replikasjoner). Fordelingen av bootstrap-t-statistikken vises nedenfor",
        "for å gi et intuitivt bilde av usikkerheten.",
        "",
        "Den røde vertikale linjen markerer den observerte t-statistikken. En smal fordeling",
        "konsentrert rundt null med den observerte t-verdien langt ute i halen indikerer et",
        "statistisk sjeldent resultat; en observert t-verdi tett på fordelingens sentrum",
        "tilsvarer et ikke-signifikant funn.",
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

            stars = _sig_stars(boot.bootstrap_p_value)
            sig_text = (
                f"statistisk signifikant ({stars})"
                if stars
                else "ikke statistisk signifikant"
            )

            lines.extend(
                [
                    f"### {name} — {label}",
                    "",
                    f"Observert koeffisient: **{boot.observed_coefficient:.4f}**  ",
                    f"Observert SE: **{boot.observed_se:.4f}**  ",
                    f"Observert t-statistikk: **{boot.observed_t_stat:.3f}**  ",
                    f"Bootstrap p-verdi: **{boot.bootstrap_p_value:.4f}**  ",
                    f"Antall replikasjoner: **{boot.n_boot}**",
                    "",
                    f"Resultatet er **{sig_text}** på bootstrap-grunnlag.",
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
    treated = cfg["analysis"].get("treated_group", "veiledning")
    control = cfg["analysis"].get("control_group", "gode muligheter")

    lines = [
        "---",
        f'title: "Test av antagelser — {title}"',
        "---",
        "",
        "## Hendelsesstudie",
        "",
        "Trippel-diff-hendelsesstudien interagerer regionens behandlingsintensitet",
        f"med gruppeindikator ({treated} vs. {control}) og tidsperiode.",
        "Koeffisientene for τ < 0 (pre-perioden) bør ligge nær null dersom",
        "parallell-trendantagelsen holder — det vil si at tiltaksnedgangen ikke",
        f"predikerer divergerende trender mellom {treated} og {control} *før*",
        "behandlingsstart.",
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

            f_stat = event_study.pretrend_f_stat
            f_p = event_study.pretrend_p_value
            df_num = event_study.pretrend_df_num
            df_denom = event_study.pretrend_df_denom

            if f_p < 0.05:
                pretrend_verdict = (
                    f"**Det er statistisk grunnlag for å forkaste parallelle trender** "
                    f"(F({df_num},{df_denom}) = {f_stat:.3f}, p = {f_p:.4f}). "
                    "Dette er et advarselssignal for identifikasjonsstrategien."
                )
            elif f_p < 0.10:
                pretrend_verdict = (
                    f"Pre-trend-testen er svakt signifikant "
                    f"(F({df_num},{df_denom}) = {f_stat:.3f}, p = {f_p:.4f}), "
                    "noe som bør tolkes med forsiktighet."
                )
            else:
                pretrend_verdict = (
                    f"Det er **ikke** statistisk grunnlag for å forkaste parallelle trender "
                    f"(F({df_num},{df_denom}) = {f_stat:.3f}, p = {f_p:.4f}). "
                    "Dette styrker identifikasjonsstrategien."
                )

            lines.extend(
                [
                    f"### {name} — {label}",
                    "",
                    pretrend_verdict,
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
            "Placebo-analysen re-estimerer trippel-diff-modellen med en *falsk*",
            "behandlingsdato 12 måneder før den virkelige, utelukkende i pre-perioden.",
            "Et estimat nær null og et ikke-signifikant resultat tyder på at parallell-",
            "trendantagelsen holder og at det ikke er pre-eksisterende divergerende",
            f"trender mellom {treated} og {control}.",
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
                lines.append(
                    f"**{name}** ({label}): Ikke nok pre-periodedata til å kjøre placebo-analyse.\n"
                )
                continue
            stars = _sig_stars(placebo.p_value)
            sig_text = (
                f"statistisk signifikant ({stars}) — dette er et advarselssignal"
                if stars
                else "ikke signifikant, noe som styrker identifikasjonsstrategien"
            )
            lines.extend(
                [
                    f"### {name} — {label}",
                    "",
                    f"Placebo-koeffisienten er **{placebo.coefficient:.4f}**"
                    f" (SE = {placebo.std_error:.4f}, p = {placebo.p_value:.4f}).",
                    f"Dette er **{sig_text}**.",
                    "",
                ]
            )

    # Leave-one-out
    lines.extend(
        [
            "## Leave-one-out robusthet",
            "",
            "Modellen re-estimeres med én region utelatt om gangen.",
            "Stabile koeffisienter på tvers av utelatelsene tilsier at ingen enkeltregion",
            "er avgjørende for resultatet. Dersom en utelatelse endrer koeffisienten",
            "vesentlig eller flytter den over signifikansterskelen, bør den regionen",
            "undersøkes nærmere.",
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

            coef_min = loo.rows["coefficient"].min()
            coef_max = loo.rows["coefficient"].max()
            full_coef = loo.full_coefficient
            same_sign = (coef_min > 0) == (coef_max > 0)
            sign_note = (
                "Fortegnet er stabilt på tvers av alle utelatelser."
                if same_sign
                else "**Fortegnet skifter** for minst én utelatelse — tolkes med forsiktighet."
            )

            lines.extend(
                [
                    f"### {name} — {label}",
                    "",
                    f"Fullmodell-koeffisienten er **{full_coef:.4f}**.",
                    f"Koeffisienten varierer mellom **{coef_min:.4f}** og **{coef_max:.4f}**"
                    f" når én region utelates. {sign_note}",
                    "",
                ]
            )
            fig_path = _plot_leave_one_out(loo, name, label, figures_dir)
            lines.extend(
                [
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
