"""Exploratory analysis of the treatment variable (tiltaksnedgang).

Compares two definitions of tiltaksnedgang side-by-side:
  peak     – denominator is the highest tiltak count in the pre-period
  last_pre – denominator is tiltak in the last pre-treatment month (t = -1)

Both definitions are derived on the fly from the same raw panel data; no
separate prep step is required.  Analysis parameters (treatment_start,
denominator definitions) are read from ``src/diff-in-diff/analysis-config.yml``.

Outputs (written to ``outputs/exploratory/``):
    figures/tiltaksnedgang_comparison.png  — side-by-side line charts per region
    figures/heatmap_comparison.png         — side-by-side heatmaps
    figures/reference_comparison.png       — side-by-side reference-level bars
    figures/tiltak_over_time.png           — raw headcounts, full period (shared)
    treatment_analysis.md                  — narrative markdown report

Run with:
    uv run src/exploratory/treatment_analysis.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "diff-in-diff" / "analysis-config.yml"
OUT_DIR = PROJECT_ROOT / "outputs" / "exploratory"
FIGURES_DIR = OUT_DIR / "figures"
REPORT_PATH = OUT_DIR / "treatment_analysis.md"

# ── Style ──────────────────────────────────────────────────────────────────────

NAV_BLUE = "#003366"
NAV_RED = "#C8102E"
LIGHT_GREY = "#D9D9D9"
MID_GREY = "#888888"

REGION_PALETTE = [
    "#003366",
    "#C8102E",
    "#E87722",
    "#00A0B0",
    "#6A5ACD",
    "#2E8B57",
    "#8B4513",
    "#D4698B",
    "#708090",
    "#DAA520",
    "#4169E1",
    "#808000",
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _short(region: str) -> str:
    """Strip the 'Nav ' prefix for compact axis labels."""
    return region.replace("Nav ", "")


# ── Config ─────────────────────────────────────────────────────────────────────


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the analysis config from *path*."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


# ── Treatment variable computation ────────────────────────────────────────────


def compute_tiltaksnedgang(df: pd.DataFrame, denominator: str) -> pd.Series:
    """Compute tiltaksnedgang for *df* using the given *denominator* definition.

    Parameters
    ----------
    df:
        Panel with columns ``region``, ``relative_month``, ``tiltak``.
    denominator:
        ``"peak"``     – max tiltak in pre-period per region.
        ``"last_pre"`` – tiltak at relative_month == -1 per region.

    Returns:
    -------
    pd.Series
        tiltaksnedgang, same index as *df*, 0 in pre-period.
    """
    pre_mask = df["relative_month"] < 0
    post_mask = ~pre_mask

    if denominator == "peak":
        ref = df.loc[pre_mask, ["region", "tiltak"]].groupby("region")["tiltak"].max()
    elif denominator == "last_pre":
        ref = df.loc[df["relative_month"] == -1, ["region", "tiltak"]].set_index(
            "region"
        )["tiltak"]
    else:
        raise ValueError(
            f"Unknown denominator '{denominator}'. Use 'peak' or 'last_pre'."
        )

    ref_series = df["region"].map(ref)
    result = pd.Series(0.0, index=df.index)
    result[post_mask] = (
        (ref_series[post_mask] - df.loc[post_mask, "tiltak"]) / ref_series[post_mask]
    ).clip(lower=0.0, upper=1.0)
    return result


def add_all_definitions(
    df: pd.DataFrame, definitions: list[dict[str, Any]]
) -> pd.DataFrame:
    """Add a ``tiltaksnedgang_{id}`` column for each entry in *definitions*.

    Parameters
    ----------
    df:
        Base panel (must have ``region``, ``relative_month``, ``tiltak``).
    definitions:
        List of dicts with keys ``id`` and ``label`` from the config.

    Returns:
    -------
    pd.DataFrame
        *df* with extra columns ``tiltaksnedgang_{id}`` and
        ``ref_tiltak_{id}`` for each definition.
    """
    pre_mask = df["relative_month"] < 0
    for defn in definitions:
        col = f"tiltaksnedgang_{defn['id']}"
        df[col] = compute_tiltaksnedgang(df, defn["id"])
        # Also store the reference level used for each definition
        if defn["id"] == "peak":
            ref = (
                df.loc[pre_mask, ["region", "tiltak"]].groupby("region")["tiltak"].max()
            )
        else:
            ref = df.loc[df["relative_month"] == -1, ["region", "tiltak"]].set_index(
                "region"
            )["tiltak"]
        df[f"ref_tiltak_{defn['id']}"] = df["region"].map(ref)
    return df


# ── Figures ────────────────────────────────────────────────────────────────────


def plot_tiltaksnedgang_comparison(
    df: pd.DataFrame,
    definitions: list[dict[str, Any]],
    treatment_label: str,
    out_dir: Path,
) -> Path:
    """Side-by-side line charts of tiltaksnedgang per region for each definition.

    Parameters
    ----------
    df:
        Panel enriched by ``add_all_definitions``.
    definitions:
        Ordered list of definition dicts (``id``, ``label``).
    treatment_label:
        Human-readable treatment start label for the x-axis.
    out_dir:
        Directory to save the PNG.

    Returns:
    -------
    Path
        Path to the saved figure.
    """
    post = df[df["relative_month"] >= 0].sort_values("relative_month")
    regions = sorted(post["region"].unique())
    n = len(definitions)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, defn in zip(axes, definitions):
        col = f"tiltaksnedgang_{defn['id']}"
        for i, region in enumerate(regions):
            r = post[post["region"] == region]
            ax.plot(
                r["relative_month"],
                r[col],
                color=REGION_PALETTE[i % len(REGION_PALETTE)],
                linewidth=1.8,
                marker="o",
                markersize=4,
                label=_short(region),
            )
        ax.set_title(f"Definisjon: {defn['label']}")
        ax.set_xlabel(f"Måneder etter behandlingsstart  (t = 0 : {treatment_label})")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.axhline(0, color=MID_GREY, linewidth=0.8, linestyle=":")
        ax.set_xlim(left=0)
        ax.set_ylim(-0.02, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[0].set_ylabel("Tiltaksnedgang")
    axes[-1].legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    fig.suptitle(
        "Tiltaksnedgang per region — sammenligning av definisjoner", fontsize=12, y=1.01
    )

    out = out_dir / "tiltaksnedgang_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_tiltak_over_time(
    df: pd.DataFrame, treatment_label: str, out_dir: Path
) -> Path:
    """Line chart of raw tiltak headcounts per region across the full panel.

    Parameters
    ----------
    df:
        Panel with columns ``relative_month`` and ``tiltak``.
    treatment_label:
        Human-readable treatment start label for annotation.
    out_dir:
        Directory to save the PNG.

    Returns:
    -------
    Path
        Path to the saved figure.
    """
    regions = sorted(df["region"].unique())
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, region in enumerate(regions):
        r = df[df["region"] == region].sort_values("relative_month")
        ax.plot(
            r["relative_month"],
            r["tiltak"],
            color=REGION_PALETTE[i % len(REGION_PALETTE)],
            linewidth=1.5,
            alpha=0.85,
            label=_short(region),
        )
    ax.axvline(0, color=NAV_RED, linestyle="--", linewidth=1.2, zorder=3)
    ax.text(
        0.5,
        0.97,
        f"Behandlingsstart ({treatment_label})",
        transform=ax.get_xaxis_transform(),
        color=NAV_RED,
        fontsize=8,
        ha="left",
        va="top",
    )
    ax.set_xlabel("Måneder relativt til behandlingsstart")
    ax.set_ylabel("Antall deltakere")
    ax.set_title("Utvikling i midlertidig lønnstilskudd per region (hele perioden)")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", "\u202f"))
    )
    out = out_dir / "tiltak_over_time.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_heatmap_comparison(
    df: pd.DataFrame,
    definitions: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    """Side-by-side heatmaps of tiltaksnedgang by region × post-treatment month.

    Parameters
    ----------
    df:
        Panel enriched by ``add_all_definitions``.
    definitions:
        Ordered list of definition dicts.
    out_dir:
        Directory to save the PNG.

    Returns:
    -------
    Path
        Path to the saved figure.
    """
    post = df[df["relative_month"] >= 0]
    n = len(definitions)
    n_months = post["relative_month"].nunique()

    fig, axes = plt.subplots(1, n, figsize=(max(6, n_months * 1.3) * n, 5))
    if n == 1:
        axes = [axes]

    for ax, defn in zip(axes, definitions):
        col = f"tiltaksnedgang_{defn['id']}"
        pivot = post.pivot_table(
            index="region", columns="relative_month", values=col, aggfunc="mean"
        )
        pivot.index = [_short(r) for r in pivot.index]

        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n_months))
        ax.set_xticklabels([f"t+{int(c)}" for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Måneder etter behandlingsstart")
        ax.set_title(f"Definisjon: {defn['label']}")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03).ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1)
        )
        for i in range(pivot.shape[0]):
            for j in range(n_months):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val > 0.55 else "black",
                    )

    fig.suptitle(
        "Tiltaksnedgang per region og måned — sammenligning av definisjoner",
        fontsize=12,
        y=1.01,
    )
    out = out_dir / "heatmap_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_reference_comparison(
    df: pd.DataFrame,
    definitions: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    """Side-by-side bar charts showing each definition's reference level per region.

    Parameters
    ----------
    df:
        Panel enriched by ``add_all_definitions``.
    definitions:
        Ordered list of definition dicts.
    out_dir:
        Directory to save the PNG.

    Returns:
    -------
    Path
        Path to the saved figure.
    """
    n = len(definitions)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, defn in zip(axes, definitions):
        ref_col = f"ref_tiltak_{defn['id']}"
        ref = (
            df.groupby("region")[ref_col]
            .first()
            .sort_values(ascending=False)
            .reset_index()
        )
        ref["short"] = ref["region"].map(_short)
        bars = ax.bar(
            ref["short"], ref[ref_col], color=NAV_BLUE, edgecolor="white", linewidth=0.5
        )
        for bar, val in zip(bars, ref[ref_col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(int(val)),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"Definisjon: {defn['label']}")
        ax.set_ylabel("Referansenivå (antall deltakere)")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(0, ref[ref_col].max() * 1.18)

    fig.suptitle(
        "Referansenivå per region — sammenligning av definisjoner", fontsize=12, y=1.01
    )
    out = out_dir / "reference_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ── Report helpers ─────────────────────────────────────────────────────────────


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep, *rows])


def _monthly_summary_table(df: pd.DataFrame, definitions: list[dict[str, Any]]) -> str:
    """Per-month stats table comparing both definitions."""
    post = df[df["relative_month"] >= 0]
    if post.empty:
        return "_Ingen post-behandlingsdata tilgjengelig._"
    rows = []
    for t in sorted(post["relative_month"].unique()):
        sub = post[post["relative_month"] == t]
        row: dict[str, Any] = {"Måned": f"t+{int(t)}"}
        for defn in definitions:
            col = f"tiltaksnedgang_{defn['id']}"
            vals = sub[col]
            row[f"Snitt ({defn['label']})"] = f"{vals.mean():.1%}"
            row[f"Median ({defn['label']})"] = f"{vals.median():.1%}"
        rows.append(row)
    return _md_table(pd.DataFrame(rows))


def _last_month_table(df: pd.DataFrame, definitions: list[dict[str, Any]]) -> str:
    """Region ranking at the last post-treatment month for both definitions."""
    post = df[df["relative_month"] >= 0]
    if post.empty:
        return "_Ingen post-behandlingsdata tilgjengelig._"
    last_t = int(post["relative_month"].max())
    last = (
        post[post["relative_month"] == last_t]
        .sort_values(f"tiltaksnedgang_{definitions[0]['id']}", ascending=False)
        .copy()
    )
    last["Region"] = last["region"].map(_short)
    last["Tiltak (nå)"] = last["tiltak"].astype(int)
    out_cols: dict[str, Any] = {
        "Region": last["Region"],
        "Tiltak (nå)": last["Tiltak (nå)"],
    }
    for defn in definitions:
        col = f"tiltaksnedgang_{defn['id']}"
        out_cols[defn["label"]] = last[col].map(lambda x: f"{x:.1%}")
    return _md_table(pd.DataFrame(out_cols))


# ── Report ─────────────────────────────────────────────────────────────────────


def generate_report(
    panels: dict[str, pd.DataFrame],
    definitions: list[dict[str, Any]],
    treatment_label: str,
) -> None:
    """Write the markdown report to ``REPORT_PATH``."""
    first_df = next(iter(panels.values()))
    post = first_df[first_df["relative_month"] >= 0]
    last_t = int(post["relative_month"].max()) if not post.empty else 0

    def_bullets = "\n".join(
        f"- **{d['label']}** (`{d['id']}`): "
        + (
            "høyeste antall deltakere i pre-perioden."
            if d["id"] == "peak"
            else "antall deltakere måneden før behandlingsstart (t = −1)."
        )
        for d in definitions
    )

    lines: list[str] = [
        "# Eksplorativ analyse: behandlingsvariabelen (tiltaksnedgang)\n",
        "## Bakgrunn\n",
        (
            "**Tiltaksnedgang** måler andelen av et regionens referansenivå i midlertidig "
            f"lønnstilskudd som er avviklet etter behandlingsstart ({treatment_label}). "
            "Denne analysen sammenligner to definisjoner av referansenivået (nevneren):\n\n"
            + def_bullets
            + "\n\n"
            "Formelen er den samme for begge: "
            "tiltaksnedgang = max(0, (ref − tiltak) / ref), klippet til [0, 1]. "
            "I pre-perioden er variabelen 0 per konstruksjon.\n"
        ),
        "## Figurer\n",
        "### Tiltaksnedgang per region — sammenligning av definisjoner\n",
        (
            f"Figurene viser tiltaksnedgang for alle Nav-regioner fra t+0 til t+{last_t}, "
            "med de to definisjonene side om side.\n"
        ),
        "![Sammenligning tiltaksnedgang](figures/tiltaksnedgang_comparison.png)\n",
        "### Råtall: antall deltakere gjennom hele perioden\n",
        (
            "Figuren er felles for begge definisjoner — den viser det absolutte antallet "
            "deltakere per region. Den stiplede røde linjen markerer behandlingsstart.\n"
        ),
        "![Tiltak over tid](figures/tiltak_over_time.png)\n",
        "### Heatmap — sammenligning av definisjoner\n",
        "![Heatmap sammenligning](figures/heatmap_comparison.png)\n",
        "### Referansenivå per definisjon\n",
        (
            "Søylediagrammene viser referansenivået (nevneren) per region for hver definisjon. "
            "Forskjellen mellom definisjonene er størst der tiltak-toppen er langt fra siste pre-måned.\n"
        ),
        "![Referansenivå sammenligning](figures/reference_comparison.png)\n",
        "## Statistikk per post-behandlingsmåned\n",
        _monthly_summary_table(first_df, definitions) + "\n",
        f"\n## Regionstatus ved t+{last_t} — begge definisjoner\n",
        _last_month_table(first_df, definitions) + "\n",
        "\n## Paneler analysert\n",
        *[
            f"- `{name}`: {len(df_):,} observasjoner, {df_['region'].nunique()} regioner\n"
            for name, df_ in panels.items()
        ],
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", REPORT_PATH)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the exploratory treatment analysis pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    treatment_start: str = cfg["analysis"]["treatment_start"]
    definitions: list[dict[str, Any]] = cfg["analysis"].get(
        "denominator_definitions",
        [{"id": "peak", "label": "Topp (pre-periode)"}],
    )
    indikatorer: list[dict[str, Any]] = cfg["data"]["indikatorer"]

    # Human-readable treatment month label (YYYYMM → "mai 2025")
    import locale as _locale

    try:
        _locale.setlocale(_locale.LC_TIME, "nb_NO.UTF-8")
    except _locale.Error:
        pass
    _ts_dt = pd.to_datetime(treatment_start, format="%Y%m")
    treatment_label = _ts_dt.strftime("%B %Y")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panels: dict[str, pd.DataFrame] = {}
    for ind in indikatorer:
        path = PROJECT_ROOT / ind["file"]
        if not path.exists():
            logger.warning("Panel not found: %s — run prep_data.py first", path)
            continue
        # Load the processed panel (already has tiltak, relative_month, region)
        processed = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"panel_regioner_lønnstilskudd_{ind['name']}.csv"
        )
        if processed.exists():
            df = pd.read_csv(processed, parse_dates=["aarmnd"])
        else:
            logger.warning(
                "Processed panel not found: %s — run prep_data.py first", processed
            )
            continue
        df = add_all_definitions(df, definitions)
        panels[ind["name"]] = df
        logger.info(
            "Loaded %s: %d rows, %d regions, %d post-treatment months",
            ind["name"],
            len(df),
            df["region"].nunique(),
            df[df["relative_month"] >= 0]["relative_month"].nunique(),
        )

    if not panels:
        logger.error("No panels loaded — run src/diff-in-diff/prep_data.py first")
        return

    first_df = next(iter(panels.values()))

    plot_tiltaksnedgang_comparison(first_df, definitions, treatment_label, FIGURES_DIR)
    plot_tiltak_over_time(first_df, treatment_label, FIGURES_DIR)
    plot_heatmap_comparison(first_df, definitions, FIGURES_DIR)
    plot_reference_comparison(first_df, definitions, FIGURES_DIR)

    generate_report(panels, definitions, treatment_label)


if __name__ == "__main__":
    main()
