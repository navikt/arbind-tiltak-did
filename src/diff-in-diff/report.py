"""Markdown report generator for the DID analysis.

Generates a Norwegian-language analytical write-up in ``report.qmd`` in a report folder, covering all employment indicators.  The report is
structured for a non-technical audience familiar with Nav's operations. This is a first draft with minimal text, but with figures and tables included. The report includes:
- Descriptive statistics of the indicators and treatment variable.
- Graphs showing trends over time for treated and control regions.
- Regression results from the DiD analysis, presented in a reader-friendly format.
- Plots of the coefficients for the fixed effects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from regression import RegressionResult

# ── Nav brand colours ─────────────────────────────────────────────────────────

_BLUE = "#003366"
_RED = "#C8102E"
_LIGHT_BLUE = "#66A3C8"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _rel(path: Path, base: Path) -> str:
    """Return *path* as a relative path string from *base*."""
    return os.path.relpath(path, base)


def _sig_stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ── Figures ───────────────────────────────────────────────────────────────────


def _plot_trends(
    panel: pd.DataFrame,
    indicator_name: str,
    indicator_label: str,
    treatment_start: str,
    figures_dir: Path,
    controll_regions: list[str] | None = None,
) -> Path:
    """Plot average indicator trend for treated vs control regions.

    For continuous treatment, regions are split at the median mean post-period
    tiltaksnedgang (high vs low).  For discrete treatment, the split is determined
    directly by ``controll_regions``.
    Returns the saved figure path.
    """
    ts_dt = pd.to_datetime(str(treatment_start), format="%Y%m")

    panel = panel.copy()
    if controll_regions is not None:
        controll_set = set(controll_regions)
        panel["gruppe"] = panel["region"].apply(
            lambda r: "Kontroll" if r in controll_set else "Behandlet"
        )
        group_colors = [("Behandlet", _RED), ("Kontroll", _BLUE)]
    else:
        post = panel[panel["post_treatment"]]
        intensity = post.groupby("region")["tiltaksnedgang"].mean()
        median_intensity = intensity.median()
        high_regions = set(intensity[intensity >= median_intensity].index)
        panel["gruppe"] = panel["region"].apply(
            lambda r: "Høy nedgang" if r in high_regions else "Lav nedgang"
        )
        group_colors = [("Høy nedgang", _RED), ("Lav nedgang", _BLUE)]

    trend = panel.groupby(["aarmnd", "gruppe"])["indikator"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for gruppe, color in group_colors:
        sub = trend[trend["gruppe"] == gruppe].sort_values("aarmnd")
        ax.plot(sub["aarmnd"], sub["indikator"], color=color, label=gruppe, linewidth=2)

    ax.axvline(
        ts_dt, color="black", linestyle="--", linewidth=1, label="Behandlingsstart"
    )
    ax.set_ylabel("Gjennomsnitt (%)", fontsize=11)
    ax.set_title(f"Trender over tid — {indicator_label}", fontsize=13)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    out = figures_dir / f"trend_{indicator_name}.png"
    _save_fig(fig, out)
    return out


def _plot_tiltak_trends(
    panel: pd.DataFrame,
    indicator_name: str,
    indicator_label: str,
    treatment_start: str,
    figures_dir: Path,
) -> Path:
    """Spaghetti plot of tiltak count per region over time.

    Each region is a separate line, with a vertical dashed line at treatment
    start.  Returns the saved figure path.
    """
    ts_dt = pd.to_datetime(str(treatment_start), format="%Y%m")
    tbl = panel.groupby(["aarmnd", "region"])["tiltak"].mean().reset_index()

    regions = sorted(tbl["region"].unique().tolist())
    cmap = plt.cm.get_cmap("tab20", len(regions))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, region in enumerate(regions):
        sub = tbl[tbl["region"] == region].sort_values("aarmnd")
        ax.plot(sub["aarmnd"], sub["tiltak"], color=cmap(i), linewidth=1.4, label=region)

    ax.axvline(ts_dt, color="black", linestyle="--", linewidth=1, label="Behandlingsstart")
    ax.set_ylabel("Antall tiltak", fontsize=11)
    ax.set_title(f"Tiltaksbruk per region — {indicator_label}", fontsize=13)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    out = figures_dir / f"tiltak_trend_{indicator_name}.png"
    _save_fig(fig, out)
    return out


def _plot_placebo(
    placebo: Any,
    preferred: Any,
    indicator_name: str,
    indicator_label: str,
    figures_dir: Path,
) -> Path:
    """Dot plot comparing the placebo coefficient to the main estimate.

    Shows the coefficient ± 95 % CI for the real preferred model (solid) and the
    placebo model (hollow), with a zero reference line.
    Returns the saved figure path.
    """
    fig, ax = plt.subplots(figsize=(7, 3))

    items = [
        ("Placebo (τ=−12)", placebo.coefficient, placebo.ci_lower, placebo.ci_upper, _LIGHT_BLUE, "o"),
        ("Foretrukket modell", preferred.coefficient, preferred.ci_lower, preferred.ci_upper, _RED, "D"),
    ]
    yticks = list(range(len(items)))
    for y, (label, coef, lo, hi, color, marker) in enumerate(items):
        ax.plot([lo, hi], [y, y], color=color, linewidth=2)
        ax.scatter([coef], [y], color=color, marker=marker, s=60, zorder=3)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(yticks)
    ax.set_yticklabels([item[0] for item in items], fontsize=10)
    ax.set_xlabel("Koeffisient (prosentpoeng)", fontsize=10)
    ax.set_title(f"Placebotest — {indicator_label}", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = figures_dir / f"placebo_{indicator_name}.png"
    _save_fig(fig, out)
    return out


def _plot_leave_one_out(
    loo: Any,
    indicator_name: str,
    indicator_label: str,
    figures_dir: Path,
) -> Path:
    """Dot plot of leave-one-out estimates with the full-sample CI as a band.

    Each point is the coefficient when the named region is dropped.  A shaded
    band shows the full-sample 95 % CI.  Returns the saved figure path.
    """
    df = loo.rows.sort_values("coefficient")
    n = len(df)

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.45)))

    # Full-sample CI band
    ax.axvspan(loo.full_ci_lower, loo.full_ci_upper, color=_LIGHT_BLUE, alpha=0.20, label="Full-utvalg 95% KI")
    ax.axvline(loo.full_coefficient, color=_BLUE, linewidth=1.5, linestyle="--", label=f"Full-utvalg: {loo.full_coefficient:.3f}")
    ax.axvline(0, color="black", linewidth=0.8)

    yticks = list(range(n))
    for y, (_, row) in enumerate(df.iterrows()):
        color = _RED if row["p_value"] < 0.05 else _LIGHT_BLUE
        ax.plot([row["ci_lower"], row["ci_upper"]], [y, y], color=color, linewidth=1.5)
        ax.scatter([row["coefficient"]], [y], color=color, s=40, zorder=3)

    ax.set_yticks(yticks)
    ax.set_yticklabels(df["dropped_region"].tolist(), fontsize=8)
    ax.set_xlabel("Koeffisient (prosentpoeng)", fontsize=10)
    ax.set_title(f"Leave-one-out — {indicator_label}", fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = figures_dir / f"loo_{indicator_name}.png"
    _save_fig(fig, out)
    return out


def _plot_fe_coefficients(
    coef_df: pd.DataFrame,
    indicator_name: str,
    indicator_label: str,
    figures_dir: Path,
) -> dict[str, Path]:
    """Generate visualisations for each FE type in the preferred model.

    - Region FE and Time FE → horizontal bar charts coloured by significance.
    - Region × Sesong FE    → heatmap (regions × months) with dropped dummies
                               shown as 0.

    Returns a dict mapping FE type label → saved figure path.
    """
    from matplotlib.patches import Patch

    df = coef_df[
        (coef_df["modell"] != "Baseline")
        & (~coef_df["koeffisient_type"].isin(["Behandling", "Konstantledd"]))
    ].copy()

    paths: dict[str, Path] = {}

    for fe_type, group in df.groupby("koeffisient_type"):
        slug = fe_type.lower().replace(" ", "_").replace("×", "x")
        out = figures_dir / f"fe_{slug}_{indicator_name}.png"

        if fe_type == "Region × Sesong FE":
            _save_fig(
                _plot_region_season_heatmap(group, indicator_label),
                out,
            )
        else:
            group = group.sort_values("koeffisient_navn")
            n = len(group)
            fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(4, n * 0.35)))

            colors = [_RED if p < 0.05 else _LIGHT_BLUE for p in group["p_verdi"]]
            ax.barh(
                group["koeffisient_navn"], group["estimat"], color=colors, edgecolor="none"
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Koeffisient", fontsize=11)
            ax.set_title(f"{fe_type} — {indicator_label}", fontsize=13)
            ax.tick_params(axis="y", labelsize=7 if n > 20 else 9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(
                handles=[
                    Patch(facecolor=_RED, label="p < 0,05"),
                    Patch(facecolor=_LIGHT_BLUE, label="p ≥ 0,05"),
                ],
                frameon=False,
                fontsize=9,
            )
            _save_fig(fig, out)

        paths[fe_type] = out

    return paths


def _plot_region_season_heatmap(
    group: pd.DataFrame,
    indicator_label: str,
) -> plt.Figure:
    """Build a heatmap of Region × calendar-month FE coefficients.

    Coefficient names follow the pattern ``rm_{region_underscored}_m{month}``.
    The reference month (January, month 1) is dropped in the regression; those
    cells are filled with 0 to represent the baseline.

    Parameters
    ----------
    group:
        Rows from the coefficients table for FE type "Region × Sesong FE".
    indicator_label:
        Human-readable indicator name used in the figure title.

    Returns:
    -------
    A :class:`matplotlib.figure.Figure` (not yet saved).
    """
    import re

    MONTH_NAMES = [
        "Jan", "Feb", "Mar", "Apr", "Mai", "Jun",
        "Jul", "Aug", "Sep", "Okt", "Nov", "Des",
    ]

    # Parse rm_{region}_m{month} into (region_display, month_int, coef)
    records = []
    for _, row in group.iterrows():
        m = re.fullmatch(r"rm_(.+)_m(\d+)", row["koeffisient_navn"])
        if m is None:
            continue
        region = m.group(1).replace("_", " ")
        month = int(m.group(2))
        records.append({"region": region, "month": month, "estimat": row["estimat"]})

    if not records:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Ingen data", ha="center", va="center")
        return fig

    pivot = (
        pd.DataFrame(records)
        .pivot(index="month", columns="region", values="estimat")
    )

    # Add any missing months (dropped reference months → 0)
    all_months = list(range(1, 13))
    pivot = pivot.reindex(index=all_months, fill_value=0.0)
    pivot.index = [MONTH_NAMES[m - 1] for m in all_months]

    # Sort regions alphabetically
    pivot = pivot.sort_index(axis=1)

    n_months, n_regions = pivot.shape
    fig_w = max(8, n_regions * 1.1)
    fig_h = max(4, n_months * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = float(np.abs(pivot.values).max()) or 1.0
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    # Axis labels
    ax.set_xticks(range(n_regions))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n_months))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate each cell with its value
    for r in range(n_months):
        for c in range(n_regions):
            val = pivot.values[r, c]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(
                c, r, f"{val:.2f}",
                ha="center", va="center",
                fontsize=6.5, color=text_color,
            )

    ax.set_title(f"Region × Sesong FE — {indicator_label}", fontsize=13)
    ax.set_xlabel("Region", fontsize=10)
    ax.set_ylabel("Kalendermåned", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Koeffisient", fontsize=9)

    # Mark reference cells (value == 0 due to dropped dummy) with a hatching
    for r in range(n_months):
        for c in range(n_regions):
            if pivot.values[r, c] == 0.0:
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        fill=False, hatch="///", edgecolor="grey",
                        linewidth=0, alpha=0.5,
                    )
                )

    fig.tight_layout()
    return fig


def _plot_event_study(
    event_study: Any,
    indicator_name: str,
    indicator_label: str,
    figures_dir: Path,
) -> Path:
    """Plot event-study coefficients with 95% CI bands.

    Pre-period estimates are shown in blue, post-period in red.
    A horizontal line at zero and a vertical dashed line at τ = 0 are included.
    Returns the saved figure path.
    """
    coefs = event_study.coefs
    taus = [c.tau for c in coefs]
    betas = [c.coefficient for c in coefs]
    ci_lo = [c.ci_lower for c in coefs]
    ci_hi = [c.ci_upper for c in coefs]

    colors = [_RED if t >= 0 else _BLUE for t in taus]

    fig, ax = plt.subplots(figsize=(12, 5))
    for tau, beta, lo, hi, color in zip(taus, betas, ci_lo, ci_hi, colors):
        ax.plot([tau, tau], [lo, hi], color=color, linewidth=1.2, alpha=0.6)
    ax.scatter(taus, betas, color=colors, zorder=3, s=18)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(
        0, color="black", linewidth=0.8, linestyle="--", label="Behandlingsstart"
    )
    ax.set_xlabel("Måneder relativt til behandlingsstart (τ)", fontsize=11)
    ax.set_ylabel("Koeffisient (β_τ)", fontsize=11)
    ax.set_title(f"Eventstudie — {indicator_label}", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    p_val = event_study.pretrend_p_value
    f_stat = event_study.pretrend_f_stat
    df_n = event_study.pretrend_df_num
    df_d = event_study.pretrend_df_denom
    ax.text(
        0.02,
        0.97,
        f"Pre-trend F({df_n},{df_d}) = {f_stat:.2f}  p = {p_val:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="dimgray",
    )

    out = figures_dir / f"event_study_{indicator_name}.png"
    _save_fig(fig, out)
    return out


def _plot_bootstrap(
    bootstrap: Any,
    indicator_name: str,
    indicator_label: str,
    model_label: str,
    figures_dir: Path,
) -> Path:
    """Plot the bootstrap t-distribution with the observed t-statistic marked.

    Fills the rejection region (|t| ≥ |t_obs|) in red.
    Returns the saved figure path.
    """
    t_boots = bootstrap.bootstrap_t_stats
    t_obs = bootstrap.observed_t_stat
    p_val = bootstrap.bootstrap_p_value

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(
        t_boots,
        bins=60,
        color=_LIGHT_BLUE,
        edgecolor="none",
        density=True,
        label="Bootstrap t*",
    )

    # Shade the rejection region
    reject_mask = np.abs(t_boots) >= np.abs(t_obs)
    if reject_mask.any():
        ax.hist(
            t_boots[reject_mask],
            bins=60,
            color=_RED,
            edgecolor="none",
            density=True,
            alpha=0.7,
            label=f"|t*| ≥ |t_obs| ({reject_mask.mean():.3f})",
        )

    ax.axvline(
        t_obs, color=_RED, linewidth=1.5, linestyle="--", label=f"t_obs = {t_obs:.2f}"
    )
    ax.axvline(-t_obs, color=_RED, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Bootstrap t-statistikk", fontsize=11)
    ax.set_ylabel("Tetthet", fontsize=11)
    ax.set_title(
        f"Wild cluster bootstrap — {indicator_label} ({model_label})\n"
        f"p-verdi = {p_val:.3f}  (n = {bootstrap.n_boot:,})",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    slug = model_label.lower().replace(" ", "_")
    out = figures_dir / f"bootstrap_{slug}_{indicator_name}.png"
    _save_fig(fig, out)
    return out


# ── Markdown building blocks ──────────────────────────────────────────────────


def _descriptive_stats_md(
    panel: pd.DataFrame, indicator_name: str, treatment_type: str = "continuous"
) -> str:
    """Return a markdown table with descriptive statistics for the panel."""
    stats = panel[["indikator", "tiltaksnedgang", "tiltak"]].describe().T
    treatment_label = (
        "Behandlet (0/1)" if treatment_type == "discrete" else "Tiltaksnedgang (0–1)"
    )
    stats.index = [indicator_name, treatment_label, "Tiltak (antall)"]
    stats = stats[["mean", "std", "min", "max"]].round(3)
    stats.columns = ["Gjennomsnitt", "Std.avvik", "Min", "Maks"]
    return stats.to_markdown() or ""


def _intensity_table_md(
    panel: pd.DataFrame, treatment_type: str = "continuous"
) -> str:
    """Return a markdown table of mean treatment variable per region."""
    post = panel[panel["post_treatment"]]
    tbl = (
        post.groupby("region")["tiltaksnedgang"]
        .mean()
        .reset_index()
        .sort_values("tiltaksnedgang", ascending=False)
        .round(3)
    )
    treatment_col = (
        "Behandlet (0=kontroll, 1=behandlet)"
        if treatment_type == "discrete"
        else "Gj.snitt nedgang (0–1)"
    )
    tbl.columns = ["Region", treatment_col]
    return tbl.to_markdown(index=False) or ""


def _regression_table_md(
    baseline: "RegressionResult",
    preferred: "RegressionResult",
    bootstrap_baseline: Any = None,
    bootstrap_preferred: Any = None,
) -> str:
    """Return a markdown table comparing baseline and preferred model results.

    Significance stars are based on bootstrap p-values when available,
    falling back to asymptotic p-values otherwise.
    """
    rows = []
    for res, boot in [(baseline, bootstrap_baseline), (preferred, bootstrap_preferred)]:
        p_primary = boot.bootstrap_p_value if boot is not None else res.p_value
        sig = _sig_stars(p_primary)
        p_boot_str = f"{boot.bootstrap_p_value:.3f}" if boot is not None else "—"
        rows.append(
            {
                "Modell": res.model_name,
                "Koeffisient": f"{res.coefficient:.4f}{sig}",
                "Std.feil (CR1)": f"{res.std_error:.4f}",
                "t-stat": f"{res.t_stat:.3f}",
                "p (bootstrap)": p_boot_str,
                "p (asymptotisk)": f"{res.p_value:.4f}",
                "95% KI": f"[{res.ci_lower:.4f}, {res.ci_upper:.4f}]",
                "Obs.": res.n_obs,
                "Clustere": res.n_clusters,
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False) or ""


# ── Main entry point ──────────────────────────────────────────────────────────


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
        Dict mapping indicator name → result dict with keys ``"baseline"``,
        ``"preferred"``, ``"event_study"``, and ``"panel"``.
    cfg:
        The loaded analysis config dict.
    output_path:
        Destination ``.qmd`` file path.
    figures_dir:
        Directory where figures are saved (referenced relatively in the QMD).
    tables_dir:
        Directory where CSV tables are saved.
    """
    from regression import extract_all_coefficients

    treatment_start = str(cfg["analysis"]["treatment_start"])
    treatment_type = str(cfg["analysis"].get("treatment_type", "continuous"))
    controll_regions: list[str] | None = cfg["analysis"].get("controll_regions", None)
    report_title = str(
        cfg["analysis"].get("title", "Nav DID-analyse — effekt av tiltaksnedgang")
    )
    report_dir = output_path.parent
    ts_formatted = f"{treatment_start[:4]}-{treatment_start[4:]}"

    ind_labels: dict[str, str] = {
        ind["name"]: ind.get("description", ind["name"])
        for ind in cfg["data"]["indikatorer"]
    }

    lines: list[str] = []

    # ── Quarto YAML frontmatter ───────────────────────────────────────────────
    lines += [
        "---",
        f'title: "{report_title}"',
        "lang: nb",
        "format:",
        "  html:",
        "    toc: true",
        "    toc-depth: 3",
        "    embed-resources: true",
        "  pdf:",
        "    toc: true",
        "execute:",
        "  echo: false",
        "  warning: false",
        "---",
        "",
    ]

    # ── Introduction ─────────────────────────────────────────────────────────
    if treatment_type == "discrete":
        controll_list = (
            ", ".join(controll_regions) if controll_regions else "—"
        )
        method_para = (
            "Analysen bruker en *difference-in-differences*-tilnærming med **diskret behandling**. "
            "Behandlingsvariabelen er binær: 1 for behandlede regioner i post-perioden, 0 ellers. "
            f"Kontrollregioner: {controll_list}. "
            "Alle øvrige regioner klassifiseres som behandlede. "
            "Modellen inkluderer region-faste effekter og tidspunkt-faste effekter "
            "for å kontrollere for tidsinvariante regionforskjeller og felles nasjonale trender."
        )
        coef_interp = (
            "Koeffisienten angir estimert gjennomsnittlig behandlingseffekt (ATT) "
            "på indikatoren, i prosentpoeng. "
        )
    else:
        method_para = (
            "Analysen bruker en *difference-in-differences*-tilnærming med **kontinuerlig behandling**. "
            "Behandlingsvariabelen (`tiltaksnedgang`) måler hvor mye tiltaksnivået i en region har falt "
            "relativt til toppen i pre-perioden, og varierer kontinuerlig mellom 0 (ingen nedgang) "
            "og 1 (full nedgang). Modellen inkluderer region-faste effekter og tidspunkt-faste effekter "
            "for å kontrollere for tidsinvariante regionforskjeller og felles nasjonale trender."
        )
        coef_interp = (
            "Koeffisienten angir estimert effekt av å gå fra null til full tiltaksnedgang "
            "(behandlingsintensitet = 1) på indikatoren, i prosentpoeng. "
        )

    lines += [
        "## Bakgrunn og metode",
        "",
        f"Denne rapporten analyserer om nedgangen i arbeidsmarkedstiltak fra og med {ts_formatted} "
        "har hatt målbar effekt på Nav-indikatorer for overgang til arbeid.",
        "",
        method_para,
        "",
        "To modellspesifikasjoner estimeres:",
        "",
        "- **Basis:** Region FE + år-måned FE",
        "- **Foretrukket:** Basis + region × kalendermåned FE (absorberer regionspesifikke sesongmønstre)",
        "",
        "> **Signifikansnivå:** \\* p < 0,10 &nbsp; \\*\\* p < 0,05 &nbsp; \\*\\*\\* p < 0,01  ",
        "> Standardfeil er clustret på regionnivå (CR1 småutvalgskorrigering).  ",
        "> Med kun G = 12 regioner er asymptotisk clusterinferens upålitelig; "
        "primær p-verdi er basert på wild cluster bootstrap med Webb-vekter (B = 4 999).",
        "",
    ]

    # ── One section per indicator ─────────────────────────────────────────────
    for ind_name, res in all_results.items():
        if res is None:
            continue

        base_ind_name = str(res.get("indicator_name", ind_name))
        prep_variant = str(res.get("prep_variant", "regular"))
        label = ind_labels.get(base_ind_name, base_ind_name)
        if prep_variant == "flattened":
            label = f"{label} — flattenet"
        panel: pd.DataFrame = res["panel"]
        baseline = res["baseline"]
        preferred = res["preferred"]
        bootstrap_baseline = res.get("bootstrap_baseline")
        bootstrap_preferred = res.get("bootstrap_preferred")
        event_study = res.get("event_study")
        placebo = res.get("placebo")
        leave_one_out = res.get("leave_one_out")
        mde: float | None = res.get("mde")
        baseline_mean: float | None = res.get("baseline_mean")

        lines += [f"## {label} (`{ind_name}`)", ""]

        # Tiltak trend (spaghetti)
        tiltak_path = _plot_tiltak_trends(
            panel=panel,
            indicator_name=ind_name,
            indicator_label=label,
            treatment_start=treatment_start,
            figures_dir=figures_dir,
        )
        lines += [
            "### Tiltaksbruk over tid",
            "",
            "Tiltaksbruk per region over tid. Den stiplede linjen markerer behandlingsstart.",
            "",
            f"![]({_rel(tiltak_path, report_dir)}){{fig-align='center' width=95%}}",
            "",
        ]

        # Descriptive statistics
        lines += [
            "### Deskriptiv statistikk",
            "",
            _descriptive_stats_md(panel, ind_name, treatment_type=treatment_type),
            "",
            "**Behandlingsvariabel per region (gjennomsnitt i post-perioden):**",
            "",
            _intensity_table_md(panel, treatment_type=treatment_type),
            "",
        ]

        # Trend plot
        trend_path = _plot_trends(
            panel=panel,
            indicator_name=ind_name,
            indicator_label=label,
            treatment_start=treatment_start,
            figures_dir=figures_dir,
            controll_regions=controll_regions,
        )
        trend_legend = (
            "Regionene er delt inn i behandlet og kontrollgruppe."
            if treatment_type == "discrete"
            else "Regionene er delt i to grupper basert på median behandlingsintensitet "
            "(gjennomsnittlig tiltaksnedgang i post-perioden)."
        )
        lines += [
            "### Trender over tid",
            "",
            trend_legend,
            "",
            f"![]({_rel(trend_path, report_dir)}){{fig-align='center' width=90%}}",
            "",
        ]

        # Regression results
        n_boot = bootstrap_preferred.n_boot if bootstrap_preferred else 0
        coef_val = preferred.coefficient
        baseline_mean_str = f"{baseline_mean:.1f} %" if baseline_mean is not None else "—"
        if baseline_mean and baseline_mean != 0:
            rel_change_str = f"{coef_val / baseline_mean * 100:.1f} %"
        else:
            rel_change_str = "—"
        mde_str = f"±{mde:.2f} pp" if mde is not None else "—"

        lines += [
            "### Regresjonsresultater",
            "",
            coef_interp
            + f"Signifikansstjerner og primær p-verdi basert på wild cluster bootstrap "
            f"(Webb-vekter, G = 12, B = {n_boot:,}).",
            "",
            _regression_table_md(
                baseline,
                preferred,
                bootstrap_baseline=bootstrap_baseline,
                bootstrap_preferred=bootstrap_preferred,
            ),
            "",
            "> **Signifikansnivå (bootstrap):** "
            "\\* p < 0,10 &nbsp; \\*\\* p < 0,05 &nbsp; \\*\\*\\* p < 0,01",
            "",
            f"**Gjennomsnittlig pre-periode-nivå:** {baseline_mean_str} — "
            f"koeffisienten tilsvarer en relativ endring på {rel_change_str}.  ",
            f"**Minimum detekterbar effekt (80 % styrke, α = 0,05):** {mde_str}.",
            "",
        ]

        # Bootstrap distribution plots
        if bootstrap_baseline is not None and bootstrap_preferred is not None:
            boot_paths = [
                _plot_bootstrap(
                    bootstrap_baseline, ind_name, label, "Basis", figures_dir
                ),
                _plot_bootstrap(
                    bootstrap_preferred, ind_name, label, "Sesongjustert", figures_dir
                ),
            ]
            lines += ["### Bootstrap-fordeling", ""]
            for bp in boot_paths:
                lines += [
                    f"![]({_rel(bp, report_dir)}){{fig-align='center' width=90%}}",
                    "",
                ]

        # FE coefficient plots
        coef_df = pd.concat(
            [extract_all_coefficients(baseline), extract_all_coefficients(preferred)],
            ignore_index=True,
        )
        fe_paths = _plot_fe_coefficients(
            coef_df=coef_df,
            indicator_name=ind_name,
            indicator_label=label,
            figures_dir=figures_dir,
        )
        if fe_paths:
            lines += [
                "### Faste effekter",
                "",
                "Stolpediagrammene viser koeffisientene for de faste effektene i den foretrukne modellen. "
                "Røde søyler er signifikante på 5 %-nivå.",
                "",
            ]
            for fe_type, fig_path in fe_paths.items():
                lines += [
                    f"**{fe_type}**",
                    "",
                    f"![]({_rel(fig_path, report_dir)}){{fig-align='center' width=90%}}",
                    "",
                ]

        # Event study
        if event_study is not None:
            es_path = _plot_event_study(
                event_study=event_study,
                indicator_name=ind_name,
                indicator_label=label,
                figures_dir=figures_dir,
            )
            p = event_study.pretrend_p_value
            f = event_study.pretrend_f_stat
            dfn = event_study.pretrend_df_num
            dfd = event_study.pretrend_df_denom
            pretrend_verdict = (
                "Det er **ikke** statistisk grunnlag for å forkaste parallelle trender "
                f"(F({dfn},{dfd}) = {f:.2f}, p = {p:.3f})."
                if p >= 0.10
                else f"**Advarsel:** pre-trend-testen er signifikant "
                f"(F({dfn},{dfd}) = {f:.2f}, p = {p:.3f}), noe som svekker DiD-antakelsen."
            )
            if treatment_type == "discrete":
                es_desc = (
                    "Eventsstudien samhandler periodevise indikatorer med en tidsinvariant "
                    "behandlingindikator per region (0 = kontroll, 1 = behandlet). "
                    "Pre-periode-koeffisientene (τ < 0) bør ligge nær null dersom "
                    "parallelle trender holder."
                )
            else:
                es_desc = (
                    "Eventsstudien samhandler periodevise indikatorer med en tidsinvariant "
                    "intensitetsscore per region (maksimal tiltaksnedgang i post-perioden). "
                    "Pre-periode-koeffisientene (τ < 0) bør ligge nær null dersom "
                    "parallelle trender holder."
                )
            lines += [
                "### Eventstudie og parallell-trend-test",
                "",
                es_desc,
                "",
                pretrend_verdict,
                "",
                f"![]({_rel(es_path, report_dir)}){{fig-align='center' width=95%}}",
                "",
            ]

        # Placebo test
        if placebo is not None:
            placebo_path = _plot_placebo(
                placebo=placebo,
                preferred=preferred,
                indicator_name=ind_name,
                indicator_label=label,
                figures_dir=figures_dir,
            )
            placebo_p_str = f"{placebo.p_value:.3f}"
            placebo_coef_str = f"{placebo.coefficient:.4f}"
            placebo_verdict = (
                f"Placebo-koeffisienten er {placebo_coef_str} (p = {placebo_p_str}). "
                + (
                    "Dette er ikke signifikant, noe som styrker identifikasjonsstrategien."
                    if placebo.p_value >= 0.10
                    else "**Advarsel:** placebo-estimatet er signifikant, noe som kan indikere pre-eksisterende trender."
                )
            )
            lines += [
                "### Placebotest (τ = −12)",
                "",
                "Modellen re-estimeres med en falsk behandlingsstart tolv måneder tidligere, "
                "utelukkende i pre-perioden. Et estimat nær null styrker antakelsen om "
                "at resultatene ikke skyldes pre-eksisterende trender.",
                "",
                placebo_verdict,
                "",
                f"![]({_rel(placebo_path, report_dir)}){{fig-align='center' width=80%}}",
                "",
            ]

        # Leave-one-out
        if leave_one_out is not None and not leave_one_out.rows.empty:
            loo_path = _plot_leave_one_out(
                loo=leave_one_out,
                indicator_name=ind_name,
                indicator_label=label,
                figures_dir=figures_dir,
            )
            coef_range = (
                f"{leave_one_out.rows['coefficient'].min():.4f} "
                f"til {leave_one_out.rows['coefficient'].max():.4f}"
            )
            lines += [
                "### Leave-one-out robusthet",
                "",
                "Den foretrukne modellen re-estimeres tolv ganger, én for hver region "
                "som droppes. Det skyggelagte feltet viser 95 %-konfidensintervallet "
                "for full-utvalgsmodellen.",
                "",
                f"Koeffisienten varierer mellom {coef_range} når én region utelates. "
                "Dette gir et inntrykk av hvor mye enkeltregioner driver resultatet.",
                "",
                f"![]({_rel(loo_path, report_dir)}){{fig-align='center' width=90%}}",
                "",
            ]

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "*Rapporten er automatisk generert av analysepipelinen.*",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
