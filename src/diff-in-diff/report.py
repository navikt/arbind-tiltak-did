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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simplify paths in SVG output — merges nearly-collinear segments, reducing
# file size significantly for line-heavy charts.
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0

if TYPE_CHECKING:
    from regression import RegressionResult

# ── Nav brand colours ─────────────────────────────────────────────────────────

_BLUE = "#003366"
_RED = "#C8102E"
_LIGHT_BLUE = "#66A3C8"
_LIGHT_RED = "#f4a582"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if path.suffix == ".png":
        kwargs["dpi"] = 96
    fig.savefig(path, **kwargs)
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
    control_regions: list[str] | None = None,
) -> Path:
    """Plot average indicator trend for treated vs control regions.

    For continuous treatment, regions are split at the median mean post-period
    tiltaksnedgang (high vs low).  For discrete treatment, the split is determined
    directly by ``control_regions``.
    Returns the saved figure path.
    """
    ts_dt = pd.to_datetime(str(treatment_start), format="%Y%m")

    panel = panel.copy()
    if control_regions is not None:
        controll_set = set(control_regions)
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

    out = figures_dir / f"trend_{indicator_name}.svg"
    _save_fig(fig, out)
    return out


def _plot_tiltak_trends(
    panel: pd.DataFrame,
    treatment_start: str,
    figures_dir: Path,
    filename_slug: str = "shared",
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
        ax.plot(
            sub["aarmnd"], sub["tiltak"], color=cmap(i), linewidth=1.4, label=region
        )

    ax.axvline(
        ts_dt, color="black", linestyle="--", linewidth=1, label="Behandlingsstart"
    )
    ax.set_ylabel("Antall tiltak", fontsize=11)
    ax.set_title("Tiltaksbruk per region", fontsize=13)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    out = figures_dir / f"tiltak_trend_{filename_slug}.svg"
    _save_fig(fig, out)
    return out


def _plot_placebo_combined(
    placebo_baseline: Any,
    placebo_preferred: Any,
    baseline: Any,
    preferred: Any,
    indicator_name: str,
    indicator_label: str,
    figures_dir: Path,
) -> Path:
    """Dot plot comparing placebo and main estimates for both models.

    Shows four rows: Basis (main), Basis (placebo), Sesongjustert (main), Sesongjustert (placebo).
    Returns the saved figure path.
    """
    items = []
    if placebo_baseline is not None:
        items.append(
            (
                "Basis — placebo (τ=−12)",
                placebo_baseline.coefficient,
                placebo_baseline.ci_lower,
                placebo_baseline.ci_upper,
                _LIGHT_BLUE,
                "o",
            )
        )
    items.append(
        (
            "Basis — modell",
            baseline.coefficient,
            baseline.ci_lower,
            baseline.ci_upper,
            _BLUE,
            "D",
        )
    )
    if placebo_preferred is not None:
        items.append(
            (
                "Sesongjustert — placebo (τ=−12)",
                placebo_preferred.coefficient,
                placebo_preferred.ci_lower,
                placebo_preferred.ci_upper,
                _LIGHT_RED,
                "o",
            )
        )
    items.append(
        (
            "Sesongjustert — modell",
            preferred.coefficient,
            preferred.ci_lower,
            preferred.ci_upper,
            _RED,
            "D",
        )
    )

    fig, ax = plt.subplots(figsize=(8, max(3, len(items) * 0.7)))
    yticks = list(range(len(items)))
    for y, (lbl, coef, lo, hi, color, marker) in enumerate(items):
        ax.plot([lo, hi], [y, y], color=color, linewidth=2)
        ax.scatter([coef], [y], color=color, marker=marker, s=60, zorder=3)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(yticks)
    ax.set_yticklabels([item[0] for item in items], fontsize=10)
    ax.set_xlabel("Koeffisient (prosentpoeng)", fontsize=10)
    ax.set_title(f"Placebotest — {indicator_label}", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = figures_dir / f"placebo_{indicator_name}.svg"
    _save_fig(fig, out)
    return out


def _plot_leave_one_out_combined(
    loo_baseline: Any,
    loo_preferred: Any,
    indicator_name: str,
    indicator_label: str,
    figures_dir: Path,
) -> Path:
    """Side-by-side leave-one-out plots for baseline and preferred models.

    Each panel shows per-region LOO estimates with the full-sample CI as a band.
    Returns the saved figure path.
    """

    def _loo_panel(ax: Any, loo: Any, model_label: str) -> None:
        df = loo.rows.sort_values("coefficient")
        n = len(df)
        ax.axvspan(
            loo.full_ci_lower,
            loo.full_ci_upper,
            color=_LIGHT_BLUE,
            alpha=0.20,
            label="Full-utvalg 95% KI",
        )
        ax.axvline(
            loo.full_coefficient,
            color=_BLUE,
            linewidth=1.5,
            linestyle="--",
            label=f"Full: {loo.full_coefficient:.3f}",
        )
        ax.axvline(0, color="black", linewidth=0.8)
        yticks = list(range(n))
        for y, (_, row) in enumerate(df.iterrows()):
            color = _RED if row["p_value"] < 0.05 else _LIGHT_BLUE
            ax.plot(
                [row["ci_lower"], row["ci_upper"]], [y, y], color=color, linewidth=1.5
            )
            ax.scatter([row["coefficient"]], [y], color=color, s=40, zorder=3)
        ax.set_yticks(yticks)
        ax.set_yticklabels(df["dropped_region"].tolist(), fontsize=8)
        ax.set_xlabel("Koeffisient (prosentpoeng)", fontsize=10)
        ax.set_title(model_label, fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

    n = max(len(loo_baseline.rows), len(loo_preferred.rows))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, n * 0.45)), sharey=True)
    _loo_panel(axes[0], loo_baseline, "Basis")
    _loo_panel(axes[1], loo_preferred, "Sesongjustert")
    fig.suptitle(f"Leave-one-out robusthet — {indicator_label}", fontsize=13)
    fig.tight_layout()

    out = figures_dir / f"loo_{indicator_name}.svg"
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
        (coef_df["modell"] == "Sesongjustert")
        & (~coef_df["koeffisient_type"].isin(["Behandling", "Konstantledd"]))
    ].copy()

    paths: dict[str, Path] = {}

    for fe_type, group in df.groupby("koeffisient_type"):
        slug = fe_type.lower().replace(" ", "_").replace("×", "x")
        out = figures_dir / f"fe_{slug}_{indicator_name}.png"

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
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Mai",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Okt",
        "Nov",
        "Des",
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

    pivot = pd.DataFrame(records).pivot(
        index="month", columns="region", values="estimat"
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
                c,
                r,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
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
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="grey",
                        linewidth=0,
                        alpha=0.5,
                    )
                )

    fig.tight_layout()
    return fig


def _plot_event_study_combined(
    event_study_baseline: Any,
    event_study_preferred: Any,
    indicator_name: str,
    indicator_label: str,
    figures_dir: Path,
) -> Path:
    """Plot event-study coefficients for both models on the same axes.

    Baseline (regular panel) in blue, sesongjustert panel in red.
    A horizontal line at zero and a vertical dashed line at τ = 0 are included.
    Returns the saved figure path.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for es, color, model_label in [
        (event_study_baseline, _BLUE, "Basis"),
        (event_study_preferred, _RED, "Sesongjustert"),
    ]:
        coefs = es.coefs
        taus = [c.tau for c in coefs]
        betas = [c.coefficient for c in coefs]
        ci_lo = [c.ci_lower for c in coefs]
        ci_hi = [c.ci_upper for c in coefs]

        for tau, beta, lo, hi in zip(taus, betas, ci_lo, ci_hi):
            ax.plot([tau, tau], [lo, hi], color=color, linewidth=1.2, alpha=0.5)
        ax.scatter(taus, betas, color=color, zorder=3, s=18, label=model_label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(
        0, color="black", linewidth=0.8, linestyle="--", label="Behandlingsstart"
    )
    ax.set_xlabel("Måneder relativt til behandlingsstart (τ)", fontsize=11)
    ax.set_ylabel("Koeffisient (β_τ)", fontsize=11)
    ax.set_title(f"Eventstudie — {indicator_label}", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate pre-trend test for preferred model
    p_val = event_study_preferred.pretrend_p_value
    f_stat = event_study_preferred.pretrend_f_stat
    df_n = event_study_preferred.pretrend_df_num
    df_d = event_study_preferred.pretrend_df_denom
    ax.text(
        0.02,
        0.97,
        f"Pre-trend F({df_n},{df_d}) = {f_stat:.2f}  p = {p_val:.3f}  (sesongjustert modell)",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=_RED,
    )
    p_val_b = event_study_baseline.pretrend_p_value
    f_stat_b = event_study_baseline.pretrend_f_stat
    df_n_b = event_study_baseline.pretrend_df_num
    df_d_b = event_study_baseline.pretrend_df_denom
    ax.text(
        0.02,
        0.89,
        f"Pre-trend F({df_n_b},{df_d_b}) = {f_stat_b:.2f}  p = {p_val_b:.3f}  (basis modell)",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=_BLUE,
    )

    out = figures_dir / f"event_study_{indicator_name}.svg"
    _save_fig(fig, out)
    return out


def _plot_bootstrap_combined(
    bootstrap_baseline: Any,
    bootstrap_preferred: Any,
    indicator_name: str,
    indicator_label: str,
    figures_dir: Path,
) -> Path:
    """Side-by-side bootstrap t-distributions for baseline and preferred models.

    Each panel shows the full bootstrap null distribution with the rejection
    region highlighted and the observed t-statistic marked.
    Returns the saved figure path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

    for ax, boot, model_label in [
        (axes[0], bootstrap_baseline, "Basis"),
        (axes[1], bootstrap_preferred, "Sesongjustert"),
    ]:
        t_boots = boot.bootstrap_t_stats
        t_obs = boot.observed_t_stat
        p_val = boot.bootstrap_p_value

        ax.hist(
            t_boots,
            bins=60,
            color=_LIGHT_BLUE,
            edgecolor="none",
            density=True,
            label="Bootstrap t*",
        )

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
            t_obs,
            color=_RED,
            linewidth=1.5,
            linestyle="--",
            label=f"t_obs = {t_obs:.2f}",
        )
        ax.axvline(-t_obs, color=_RED, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Bootstrap t-statistikk", fontsize=10)
        ax.set_ylabel("Tetthet", fontsize=10)
        ax.set_title(
            f"{model_label} — {indicator_label}\np = {p_val:.3f}  (n = {boot.n_boot:,})",
            fontsize=11,
        )
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Wild cluster bootstrap", fontsize=13)
    fig.tight_layout()
    out = figures_dir / f"bootstrap_{indicator_name}.png"
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


def _intensity_table_md(panel: pd.DataFrame, treatment_type: str = "continuous") -> str:
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


def _infer_subgroup(file_path: str) -> str:
    """Infer the population subgroup from an indicator file path."""
    parts = Path(file_path).parts
    if "nedbrytning" in parts:
        idx = parts.index("nedbrytning")
        if idx + 1 < len(parts):
            subdir = parts[idx + 1]
            return subdir.capitalize()
    return "Alle"


def _infer_outcome_type(file_path: str) -> str:
    """Infer whether the indicator is faktisk, forventet, or ujustert."""
    stem = Path(file_path).stem.lower()
    if stem.startswith("faktisk_"):
        return "Faktisk"
    if stem.startswith("forventet_"):
        return "Forventet"
    return "—"


def _section_summary_table(cfg: dict[str, Any]) -> list[str]:
    """Return a summary table of the analysis configuration."""
    analysis = cfg["analysis"]
    indikatorer = cfg["data"].get("indikatorer", [])

    treatment_type = analysis.get("treatment_type", "continuous")
    treatment_type_label = (
        "Kontinuerlig" if treatment_type == "continuous" else "Diskret"
    )

    ts = str(analysis.get("treatment_start", ""))
    treatment_start_label = f"{ts[:4]}-{ts[4:]}" if len(ts) == 6 else ts

    # Infer subgroup and outcome type from first indicator file (they share the same)
    first_file = indikatorer[0]["file"] if indikatorer else ""
    subgroup = _infer_subgroup(first_file)
    outcome_type = _infer_outcome_type(first_file)

    ind_labels = ", ".join(
        f"{ind['name']} — {ind.get('description', ind['name'])}" for ind in indikatorer
    )

    control_regions = analysis.get("control_regions")
    control_label = ", ".join(control_regions) if control_regions else "—"

    rows = [
        ("Behandlingstype", treatment_type_label),
        ("Behandlingsstart", treatment_start_label),
        ("Variasjon i behandling", "Nav-regioner"),
        ("Undergruppe (populasjon)", subgroup),
        ("Utfallsmål (indikatortype)", outcome_type),
        ("Indikatorer", ind_labels),
    ]
    if treatment_type == "discrete":
        rows.append(("Kontrollregioner", control_label))

    table_lines = [
        "| Egenskap | Verdi |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in rows]

    return [
        "",
        *table_lines,
        "",
    ]


def _section_frontmatter(title: str) -> list[str]:
    """Return Quarto YAML frontmatter lines."""
    return [
        "---",
        f'title: "{title}"',
        "execute:",
        "  echo: false",
        "  warning: false",
        "---",
        "",
    ]


def _section_intro(
    treatment_start: str,
    treatment_type: str,
    control_regions: list[str] | None,
) -> list[str]:
    """Return the method/background introduction section."""
    ts_formatted = f"{treatment_start[:4]}-{treatment_start[4:]}"
    if treatment_type == "discrete":
        control_list = ", ".join(control_regions) if control_regions else "—"
        method_para = (
            "Analysen bruker en *difference-in-differences*-tilnærming med **diskret behandling**. "
            "Behandlingsvariabelen er binær: 1 for behandlede regioner i post-perioden, 0 ellers. "
            f"Kontrollregioner: {control_list}. "
            "Alle øvrige regioner klassifiseres som behandlede. "
            "Modellen inkluderer region-faste effekter og tidspunkt-faste effekter "
            "for å kontrollere for tidsinvariante regionforskjeller og felles nasjonale trender."
        )
    else:
        method_para = (
            "Analysen bruker en *difference-in-differences*-tilnærming med **kontinuerlig behandling**. "
            "Behandlingsvariabelen (`tiltaksnedgang`) måler hvor mye tiltaksnivået i en region har falt "
            "relativt til toppen i pre-perioden, og varierer kontinuerlig mellom 0 (ingen nedgang) "
            "og 1 (full nedgang). Modellen inkluderer region-faste effekter og tidspunkt-faste effekter "
            "for å kontrollere for tidsinvariante regionforskjeller og felles nasjonale trender."
        )
    return [
        "## Bakgrunn og metode",
        "",
        f"Denne rapporten analyserer om nedgangen i arbeidsmarkedstiltak fra og med {ts_formatted} "
        "har hatt målbar effekt på Nav-indikatorer for overgang til arbeid.",
        "",
        method_para,
        "",
        "To modellspesifikasjoner estimeres:",
        "",
        "- **Basis:** Ujustert indikator, region FE + år-måned FE",
        "- **Sesongjustert:** Sesongjustert indikator, region FE + år-måned FE",
        "",
        "> **Signifikansnivå:** \\* p < 0,10 &nbsp; \\*\\* p < 0,05 &nbsp; \\*\\*\\* p < 0,01  ",
        "> Standardfeil er clustret på regionnivå (CR1 småutvalgskorrigering).  ",
        "> Med kun G = 12 regioner er asymptotisk clusterinferens upålitelig; "
        "primær p-verdi er basert på wild cluster bootstrap med Webb-vekter (B = 4 999).",
        "",
    ]


def _section_tiltak(
    panel: pd.DataFrame,
    treatment_start: str,
    figures_dir: Path,
    report_dir: Path,
) -> list[str]:
    """Return the shared tiltaksbruk-over-tid section."""
    tiltak_path = _plot_tiltak_trends(
        panel=panel,
        treatment_start=treatment_start,
        figures_dir=figures_dir,
    )
    return [
        "## Tiltaksbruk over tid",
        "",
        "Tiltaksbruk (midlertidig lønnstilskudd) per region over tid. "
        "Den stiplede linjen markerer behandlingsstart.",
        "",
        f"![]({_rel(tiltak_path, report_dir)}){{fig-align='center' width=95%}}",
        "",
    ]


def _section_descriptive(
    panel: pd.DataFrame,
    ind_name: str,
    label: str,
    treatment_start: str,
    treatment_type: str,
    control_regions: list[str] | None,
    figures_dir: Path,
    report_dir: Path,
) -> list[str]:
    """Return descriptive statistics and trend-plot section for one indicator."""
    trend_path = _plot_trends(
        panel=panel,
        indicator_name=ind_name,
        indicator_label=label,
        treatment_start=treatment_start,
        figures_dir=figures_dir,
        control_regions=control_regions,
    )
    trend_legend = (
        "Regionene er delt inn i behandlet og kontrollgruppe."
        if treatment_type == "discrete"
        else "Regionene er delt i to grupper basert på median behandlingsintensitet "
        "(gjennomsnittlig tiltaksnedgang i post-perioden)."
    )
    return [
        "### Deskriptiv statistikk",
        "",
        _descriptive_stats_md(panel, ind_name, treatment_type=treatment_type),
        "",
        "**Behandlingsvariabel per region (gjennomsnitt i post-perioden):**",
        "",
        _intensity_table_md(panel, treatment_type=treatment_type),
        "",
        "### Trender over tid",
        "",
        trend_legend,
        "",
        f"![]({_rel(trend_path, report_dir)}){{fig-align='center' width=90%}}",
        "",
    ]


def _section_regression(
    baseline: Any,
    preferred: Any,
    bootstrap_baseline: Any,
    bootstrap_preferred: Any,
    coef_interp: str,
    baseline_mean: float | None,
    mde: float | None,
    ind_name: str,
    label: str,
    figures_dir: Path,
    report_dir: Path,
) -> list[str]:
    """Return regression table, bootstrap plot, and FE coefficient sections."""
    from regression import extract_all_coefficients

    n_boot = bootstrap_preferred.n_boot if bootstrap_preferred else 0
    coef_val = preferred.coefficient
    baseline_mean_str = f"{baseline_mean:.1f} %" if baseline_mean is not None else "—"
    rel_change_str = f"{coef_val / baseline_mean * 100:.1f} %" if baseline_mean else "—"
    mde_str = f"±{mde:.2f} pp" if mde is not None else "—"

    lines: list[str] = [
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

    if bootstrap_baseline is not None and bootstrap_preferred is not None:
        boot_path = _plot_bootstrap_combined(
            bootstrap_baseline, bootstrap_preferred, ind_name, label, figures_dir
        )
        lines += [
            "### Bootstrap-fordeling",
            "",
            f"![]({_rel(boot_path, report_dir)}){{fig-align='center' width=95%}}",
            "",
        ]

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
            "Stolpediagrammene viser koeffisientene for de faste effektene i den sesongjusterte modellen. "
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
    return lines


def _section_event_study(
    event_study_baseline: Any,
    event_study_preferred: Any,
    ind_name: str,
    label: str,
    treatment_type: str,
    figures_dir: Path,
    report_dir: Path,
) -> list[str]:
    """Return event-study section with combined figure and pre-trend verdict."""
    es_path = _plot_event_study_combined(
        event_study_baseline=event_study_baseline,
        event_study_preferred=event_study_preferred,
        indicator_name=ind_name,
        indicator_label=label,
        figures_dir=figures_dir,
    )
    p = event_study_preferred.pretrend_p_value
    f = event_study_preferred.pretrend_f_stat
    dfn = event_study_preferred.pretrend_df_num
    dfd = event_study_preferred.pretrend_df_denom
    pretrend_verdict = (
        "Det er **ikke** statistisk grunnlag for å forkaste parallelle trender "
        f"(F({dfn},{dfd}) = {f:.2f}, p = {p:.3f}, sesongjustert modell)."
        if p >= 0.10
        else f"**Advarsel:** pre-trend-testen er signifikant "
        f"(F({dfn},{dfd}) = {f:.2f}, p = {p:.3f}), noe som svekker DiD-antakelsen."
    )
    es_desc = (
        (
            "Eventsstudien samhandler periodevise indikatorer med en tidsinvariant "
            "behandlingindikator per region (0 = kontroll, 1 = behandlet). "
            if treatment_type == "discrete"
            else "Eventsstudien samhandler periodevise indikatorer med en tidsinvariant "
            "intensitetsscore per region (maksimal tiltaksnedgang i post-perioden). "
        )
        + "Pre-periode-koeffisientene (τ < 0) bør ligge nær null dersom parallelle trender holder. Blå = basis, rød = sesongjustert modell."
    )
    return [
        "### Eventstudie og parallell-trend-test",
        "",
        es_desc,
        "",
        pretrend_verdict,
        "",
        f"![]({_rel(es_path, report_dir)}){{fig-align='center' width=95%}}",
        "",
    ]


def _section_placebo(
    placebo_baseline: Any,
    placebo_preferred: Any,
    baseline: Any,
    preferred: Any,
    ind_name: str,
    label: str,
    figures_dir: Path,
    report_dir: Path,
) -> list[str]:
    """Return placebo-test section with combined figure."""
    placebo_path = _plot_placebo_combined(
        placebo_baseline=placebo_baseline,
        placebo_preferred=placebo_preferred,
        baseline=baseline,
        preferred=preferred,
        indicator_name=ind_name,
        indicator_label=label,
        figures_dir=figures_dir,
    )
    p_str = f"{placebo_preferred.p_value:.3f}"
    coef_str = f"{placebo_preferred.coefficient:.4f}"
    verdict = (
        f"Sesongjustert modell — placebo-koeffisienten er {coef_str} (p = {p_str}). "
        + (
            "Dette er ikke signifikant, noe som styrker identifikasjonsstrategien."
            if placebo_preferred.p_value >= 0.10
            else "**Advarsel:** placebo-estimatet er signifikant, noe som kan indikere pre-eksisterende trender."
        )
    )
    return [
        "### Placebotest (τ = −12)",
        "",
        "Begge modeller re-estimeres med en falsk behandlingsstart tolv måneder tidligere, "
        "utelukkende i pre-perioden. Estimater nær null styrker antakelsen om "
        "at resultatene ikke skyldes pre-eksisterende trender.",
        "",
        verdict,
        "",
        f"![]({_rel(placebo_path, report_dir)}){{fig-align='center' width=80%}}",
        "",
    ]


def _section_leave_one_out(
    loo_baseline: Any,
    loo_preferred: Any,
    ind_name: str,
    label: str,
    figures_dir: Path,
    report_dir: Path,
) -> list[str]:
    """Return leave-one-out robustness section with combined figure."""
    loo_path = _plot_leave_one_out_combined(
        loo_baseline=loo_baseline,
        loo_preferred=loo_preferred,
        indicator_name=ind_name,
        indicator_label=label,
        figures_dir=figures_dir,
    )
    coef_range = (
        f"{loo_preferred.rows['coefficient'].min():.4f} "
        f"til {loo_preferred.rows['coefficient'].max():.4f}"
    )
    return [
        "### Leave-one-out robusthet",
        "",
        "Begge modeller re-estimeres tolv ganger, én for hver region som droppes. "
        "Det skyggelagte feltet viser 95 %-konfidensintervallet for full-utvalgsmodellen.",
        "",
        f"Sesongjustert modell: koeffisienten varierer mellom {coef_range} når én region utelates.",
        "",
        f"![]({_rel(loo_path, report_dir)}){{fig-align='center' width=95%}}",
        "",
    ]


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
    lines += _section_intro(treatment_start, treatment_type, control_regions)

    first_res = next((r for r in all_results.values() if r is not None), None)
    if first_res is not None:
        lines += _section_tiltak(
            panel=first_res["panel"],
            treatment_start=treatment_start,
            figures_dir=figures_dir,
            report_dir=report_dir,
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
