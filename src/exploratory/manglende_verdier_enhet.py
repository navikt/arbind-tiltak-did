"""Exploratory analysis of missing values in enhet-level indicator data.

Covers the two key folders under data/input/indikatorer/enhet/nedbrytning:
  - veiledning_kombinert       (jobb3, atid3)
  - standard                   (jobb3, atid3, faktisk_jobb3, faktisk_atid3,
                                forventet_jobb3, forventet_atid3)

Outputs (written to outputs/exploratory/):
    figures/missing_over_time.png              — line chart: # missing enheter per month
    figures/missing_heatmap_veiledning_kombinert.png  — heatmap enhet × month (Veiledning kombinert)
    figures/missing_heatmap_standard.png       — heatmap enhet × month (Standard)
    figures/missing_per_enhet.png              — bar chart: total missing periods per enhet
    manglende_verdier_enhet.qmd                — narrative QMD report

Run with:
    uv run src/exploratory/manglende_verdier_enhet.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "indikatorer" / "enhet" / "nedbrytning"
OUT_DIR = PROJECT_ROOT / "outputs" / "exploratory"
FIGURES_DIR = OUT_DIR / "figures"
QMD_PATH = OUT_DIR / "manglende_verdier_enhet.qmd"

# ── Style ──────────────────────────────────────────────────────────────────────

NAV_BLUE = "#003366"
NAV_RED = "#C8102E"
LIGHT_GREY = "#D9D9D9"
MID_GREY = "#888888"
NAV_ORANGE = "#E87722"

# ── Helpers ────────────────────────────────────────────────────────────────────


def load_folder(folder: str) -> dict[str, pd.DataFrame]:
    """Load all CSVs from a nedbrytning subfolder into a dict keyed by filename stem."""
    path = DATA_ROOT / folder
    dfs = {}
    for csv_file in sorted(path.glob("*.csv")):
        df = pd.read_csv(csv_file)
        dfs[csv_file.stem] = df
    return dfs


def value_col(df: pd.DataFrame) -> str:
    """Return the name of the value column (the one that's not aarmnd or enhet)."""
    return [c for c in df.columns if c not in ("aarmnd", "enhet")][0]


def missing_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean pivot (enhet × aarmnd) — True where value is missing."""
    col = value_col(df)
    return df.pivot(index="enhet", columns="aarmnd", values=col).isna()


def fmt_aarmnd(val: int) -> str:
    """Format 202305 → 'mai 23'."""
    MONTHS_NO = [
        "jan",
        "feb",
        "mar",
        "apr",
        "mai",
        "jun",
        "jul",
        "aug",
        "sep",
        "okt",
        "nov",
        "des",
    ]
    year = str(val)[2:4]
    month = int(str(val)[4:6])
    return f"{MONTHS_NO[month - 1]} {year}"


# ── Figures ────────────────────────────────────────────────────────────────────


def plot_missing_over_time(
    mat_v: pd.DataFrame,
    mat_g: pd.DataFrame,
    n_enhet_v: int,
    n_enhet_g: int,
) -> Path:
    """Line chart: number of enheter with missing value per month."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    time_v = mat_v.sum(axis=0)
    time_g = mat_g.sum(axis=0)

    xticks = list(time_v.index)
    x = np.arange(len(xticks))
    xlabels = [fmt_aarmnd(v) for v in xticks]

    ax.plot(
        x,
        time_v.values,
        color=NAV_BLUE,
        linewidth=2,
        label=f"Veiledning (n={n_enhet_v})",
    )
    ax.plot(
        x,
        time_g.values,
        color=NAV_RED,
        linewidth=2,
        label=f"Standard (n={n_enhet_g})",
    )

    ax.set_ylabel("Antall enheter med manglende verdi", fontsize=10)
    ax.set_xlabel("")
    ax.set_xticks(x[::3])
    ax.set_xticklabels(xlabels[::3], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Manglende verdier per måned", fontsize=12, pad=10)

    fig.tight_layout()
    out = FIGURES_DIR / "missing_over_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_heatmap(mat: pd.DataFrame, title: str, filename: str) -> Path:
    """Heatmap: enhet × month, coloured by missing (True/False)."""
    # Sort enheter by total missing (descending) so most-problematic appear at top
    row_order = mat.sum(axis=1).sort_values(ascending=False).index
    mat_sorted = mat.loc[row_order]

    n_enhet, n_time = mat_sorted.shape
    fig_h = max(6, n_enhet * 0.18)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    im = ax.imshow(
        mat_sorted.values.astype(float),
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        interpolation="none",
    )

    # X-axis: every 3rd month
    xtick_pos = list(range(0, n_time, 3))
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(
        [fmt_aarmnd(v) for v in mat_sorted.columns[xtick_pos]],
        rotation=45,
        ha="right",
        fontsize=7,
    )

    # Y-axis: enhet names
    ax.set_yticks(range(n_enhet))
    ax.set_yticklabels(mat_sorted.index, fontsize=6.5)

    ax.set_title(title, fontsize=11, pad=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.4, aspect=20, pad=0.01)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Observert", "Mangler"])
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_missing_per_enhet(
    mat_v: pd.DataFrame,
    mat_g: pd.DataFrame,
    n_time: int,
) -> Path:
    """Horizontal bar chart: total missing periods per enhet for both datasets."""
    enhet_v = mat_v.sum(axis=1).rename("Veiledning")
    enhet_g = mat_g.sum(axis=1).rename("Standard")

    # Union of all enheter
    combined = pd.concat([enhet_v, enhet_g], axis=1).fillna(0).astype(int)
    combined["total"] = combined.sum(axis=1)
    combined = combined.sort_values("total", ascending=True)
    # Only show enheter with at least one missing value
    combined = combined[combined["total"] > 0]

    n = len(combined)
    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.22)))

    y = np.arange(n)
    bar_h = 0.38
    ax.barh(
        y + bar_h / 2,
        combined["Veiledning"],
        height=bar_h,
        color=NAV_BLUE,
        label="Veiledning",
    )
    ax.barh(
        y - bar_h / 2,
        combined["Standard"],
        height=bar_h,
        color=NAV_RED,
        label="Standard",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(combined.index, fontsize=7)
    ax.set_xlabel(f"Antall måneder med manglende verdi (av {n_time})", fontsize=9)
    ax.axvline(
        n_time,
        color=MID_GREY,
        linestyle="--",
        linewidth=0.8,
        label=f"Alle {n_time} måneder",
    )
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    ax.set_title(
        "Manglende verdier per enhet (totalt over alle måneder)", fontsize=11, pad=8
    )

    fig.tight_layout()
    out = FIGURES_DIR / "missing_per_enhet.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ── Report helpers ─────────────────────────────────────────────────────────────


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows)


def build_summary_table(
    mat_v: pd.DataFrame,
    mat_g: pd.DataFrame,
) -> str:
    """Markdown table: per-dataset summary stats."""
    rows = []
    for label, mat, n_enhet in [
        ("Veiledning", mat_v, mat_v.shape[0]),
        ("Standard", mat_g, mat_g.shape[0]),
    ]:
        n_time = mat.shape[1]
        total_cells = n_enhet * n_time
        total_missing = int(mat.sum().sum())
        pct = round(total_missing / total_cells * 100, 1)
        enheter_any = int((mat.sum(axis=1) > 0).sum())
        rows.append(
            {
                "Datasett": label,
                "Enheter": n_enhet,
                "Måneder": n_time,
                "Celler totalt": total_cells,
                "Mangler": total_missing,
                "Andel (%)": pct,
                "Enheter med ≥1 mangler": enheter_any,
            }
        )
    df = pd.DataFrame(rows)
    return _md_table(df)


def build_worst_enhet_table(
    mat_v: pd.DataFrame,
    mat_g: pd.DataFrame,
    top_n: int = 15,
) -> str:
    """Markdown table: enheter with most missing values."""
    enhet_v = mat_v.sum(axis=1).rename("Veiledning")
    enhet_g = mat_g.sum(axis=1).rename("Standard")
    combined = pd.concat([enhet_v, enhet_g], axis=1).fillna(0).astype(int)
    combined["Totalt"] = combined.sum(axis=1)
    top = combined.sort_values("Totalt", ascending=False).head(top_n).reset_index()
    top.columns = [
        "Enhet",
        "Veiledning (måneder)",
        "Standard (måneder)",
        "Totalt",
    ]
    return _md_table(top)


def build_enhet_diff_table(
    mat_v: pd.DataFrame,
    mat_g: pd.DataFrame,
) -> str:
    """Markdown table: enheter present in one dataset but not the other."""
    v_only = sorted(set(mat_v.index) - set(mat_g.index))
    g_only = sorted(set(mat_g.index) - set(mat_v.index))
    max_len = max(len(v_only), len(g_only))
    v_only_padded = v_only + [""] * (max_len - len(v_only))
    g_only_padded = g_only + [""] * (max_len - len(g_only))
    df = pd.DataFrame(
        {
            "Kun i Veiledning": v_only_padded,
            "Kun i Standard": g_only_padded,
        }
    )
    return _md_table(df)


# ── QMD output ─────────────────────────────────────────────────────────────────


def write_qmd(
    summary_table: str,
    worst_enhet_table: str,
    enhet_diff_table: str,
    n_veiledning_files: int,
    n_standard_files: int,
) -> None:
    """Write the narrative QMD report, embedding the generated figures and tables."""
    content = f"""\
# Manglende verdier — enhet-data (nedbrytning)

## Bakgrunn

Denne analysen gir en oversikt over manglende verdier i de to viktigste datakildene
for enhet-nivå nedbrytning:

- **Veiledning kombinert** (`data/input/indikatorer/enhet/nedbrytning/veiledning_kombinert/`):
  {n_veiledning_files} filer (`jobb3`, `atid3`). Alle filene har identisk manglende-mønster.
- **Standard** (`data/input/indikatorer/enhet/nedbrytning/standard/`):
  {n_standard_files} filer (`jobb3`, `atid3`, `faktisk_jobb3`, `faktisk_atid3`,
  `forventet_jobb3`, `forventet_atid3`). Alle filene har identisk manglende-mønster.

Innen hvert datasett er manglende celler fullstendig konsistente på tvers av filer —
det vil si at de samme (enhet, måned)-kombinasjonene mangler i alle filene for det
aktuelle datasettet.

## Oppsummering

{summary_table}

## Manglende verdier over tid

Figuren viser antall enheter med manglende verdi per måned for begge datasett.
Manglene i Veiledning kombinert er relativt stabile over tid, mens Standard viser
en tydelig økning fra 2022 og utover.

![Manglende verdier per måned](figures/missing_over_time.png)

## Manglende verdier per enhet (heatmap)

Røde celler betyr manglende verdi, grønne betyr observert verdi.
Enhetene er sortert etter totalt antall manglende måneder (flest øverst).

### Veiledning kombinert

![Heatmap — Veiledning kombinert](figures/missing_heatmap_veiledning_kombinert.png)

### Standard

![Heatmap — Standard](figures/missing_heatmap_standard.png)

## Manglende verdier per enhet (søylediagram)

Diagrammet viser totalt antall måneder med manglende verdi for enheter med minst
én manglende observasjon.

![Manglende per enhet](figures/missing_per_enhet.png)

## Enheter med flest manglende verdier (topp 15)

{worst_enhet_table}

## Enheter som kun finnes i ett av datasettene

{enhet_diff_table}
"""
    QMD_PATH.write_text(content, encoding="utf-8")
    logger.info("Saved %s", QMD_PATH)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Main function to run the analysis and generate outputs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    veiledning_kombinert = load_folder("veiledning_kombinert")
    standard = load_folder("standard")

    # Use jobb3 as representative file for each dataset (all files have same missing pattern)
    mat_v = missing_matrix(veiledning_kombinert["jobb3"])
    mat_g = missing_matrix(standard["jobb3"])

    n_time = mat_v.shape[1]

    # Figures
    plot_missing_over_time(mat_v, mat_g, mat_v.shape[0], mat_g.shape[0])
    plot_heatmap(
        mat_v,
        "Manglende verdier — Veiledning kombinert (enhet × måned)",
        "missing_heatmap_veiledning_kombinert.png",
    )
    plot_heatmap(
        mat_g,
        "Manglende verdier — Standard (enhet × måned)",
        "missing_heatmap_standard.png",
    )
    plot_missing_per_enhet(mat_v, mat_g, n_time)

    # Tables
    summary = build_summary_table(mat_v, mat_g)
    worst = build_worst_enhet_table(mat_v, mat_g)
    diff = build_enhet_diff_table(mat_v, mat_g)

    write_qmd(summary, worst, diff, len(veiledning_kombinert), len(standard))
    logger.info("Done.")


if __name__ == "__main__":
    main()
