"""Data section report generator.

Generates static Quarto chapters for the *Data* part of the book.  Each
chapter contains figures (saved as SVG) and embedded markdown tables.  No
code chunks appear in the resulting ``.qmd`` files — this follows the same
pattern as the analysis report generators in :mod:`report.did` and
:mod:`report.triple_diff`.

Run from ``src/diff-in-diff/``:

    uv run python generate_data_report.py

Outputs written to ``quarto/data/``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from data_io import _load_tiltak_wide_to_long as _load_tiltak_sa
from report.utils import BLUE as _BLUE
from report.utils import LIGHT_BLUE as _LIGHT_BLUE
from report.utils import RED as _RED
from report.utils import rel as _rel
from report.utils import save_fig as _save_fig

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_INPUT = PROJECT_ROOT / "data" / "input"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
QUARTO_DATA = PROJECT_ROOT / "quarto" / "data"
FIGURES_DIR = QUARTO_DATA / "figures"

TREATMENT_START_DATE = pd.Timestamp("2025-06-01")
TREATMENT_START_STR = "202506"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_data_report")

# ── Colour palette ─────────────────────────────────────────────────────────────

_COLORS_REGIONS = [
    "#003366",
    "#C8102E",
    "#66A3C8",
    "#f4a582",
    "#858E00",
    "#254B6D",
    "#FF9100",
    "#C1A753",
    "#66CBEC",
    "#B90000",
    "#4A90D9",
    "#8B4513",
]

_GRUPPE_LABELS: dict[str, str] = {
    "alle": "Alle",
    "standard": "Standard",
    "situasjonsbestemt": "Situasjonsbestemt",
    "spesielt_tilpasset": "Spesielt tilpasset",
    "veiledning_kombinert": "Veiledning kombinert",
}
_GRUPPE_COLORS: dict[str, str] = {
    "alle": _BLUE,
    "standard": _RED,
    "situasjonsbestemt": _LIGHT_BLUE,
    "spesielt_tilpasset": "#858E00",
    "veiledning_kombinert": "#FF9100",
}

# ── Data loading helpers ───────────────────────────────────────────────────────


def _load_tiltak(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a wide-format tiltak CSV.  Returns (df_with_datetime, region_cols)."""
    df = pd.read_csv(path)
    df.columns = [c.strip("'") for c in df.columns]
    df["aarmnd"] = pd.to_datetime(df["aarmnd"])
    df = df.sort_values("aarmnd").reset_index(drop=True)
    regions = [c for c in df.columns if c not in ("aarmnd", "TOTAL")]
    return df, regions


def _load_indicator(path: Path) -> pd.DataFrame:
    """Load a wide-format indicator CSV and melt to long format."""
    df = pd.read_csv(path)
    df["aarmnd"] = df["aarmnd"].astype(str)
    regions = [c for c in df.columns if c != "aarmnd"]
    return df.melt(
        id_vars="aarmnd", value_vars=regions, var_name="region", value_name="verdi"
    )


# ── Figure generators ──────────────────────────────────────────────────────────


def _plot_tiltak_trends(
    df: pd.DataFrame, regions: list[str], title: str, fig_path: Path
) -> None:
    """Line chart: tiltak count per region over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, region in enumerate(regions):
        ax.plot(
            df["aarmnd"],
            df[region],
            linewidth=1.0,
            alpha=0.8,
            color=_COLORS_REGIONS[i % len(_COLORS_REGIONS)],
        )
    ax.axvline(TREATMENT_START_DATE, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Antall deltakere")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    # Small legend for treatment line
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[Line2D([0], [0], color="black", linestyle="--", linewidth=1.2)],
        labels=["Behandlingsstart (2025-06)"],
        loc="upper left",
        fontsize=9,
    )
    plt.tight_layout()
    _save_fig(fig, fig_path)


def _plot_tiltak_sa_trend(tiltak_path: Path, title: str, fig_path: Path) -> None:
    """Plot raw vs. two STL-seasonally-adjusted national totals.

    Shows three series so the reader can compare:
    - Raw total (unajusted)
    - SA full series: STL fit on all observations (current analysis default)
    - SA pre-period only: STL fit on pre-treatment months, seasonal factors
      extrapolated by calendar month to post-treatment months

    All three use the same underlying data loaded via
    :func:`data_io._load_tiltak_wide_to_long` — the same function used by the
    analysis pipeline.
    """
    raw_long = _load_tiltak_sa(tiltak_path, seasonal_adjust=False)
    sa_full_long = _load_tiltak_sa(
        tiltak_path, seasonal_adjust=True, seasonal_adjust_pre_only=False
    )
    sa_pre_long = _load_tiltak_sa(
        tiltak_path, seasonal_adjust=True, seasonal_adjust_pre_only=True
    )

    def _total(long: pd.DataFrame) -> pd.DataFrame:
        return (
            long.groupby("aarmnd")["tiltak"]
            .sum()
            .reset_index()
            .assign(dato=lambda d: pd.to_datetime(d["aarmnd"], format="%Y%m"))
            .sort_values("dato")
        )

    raw_total = _total(raw_long)
    sa_full_total = _total(sa_full_long)
    sa_pre_total = _total(sa_pre_long)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        raw_total["dato"],
        raw_total["tiltak"],
        label="Rå total",
        color=_LIGHT_BLUE,
        linewidth=1.2,
        alpha=0.7,
    )
    ax.plot(
        sa_full_total["dato"],
        sa_full_total["tiltak"],
        label="SK – full serie (analyse)",
        color=_BLUE,
        linewidth=1.5,
    )
    ax.plot(
        sa_pre_total["dato"],
        sa_pre_total["tiltak"],
        label="SK – kun pre-periode",
        color=_RED,
        linewidth=1.5,
        linestyle="--",
    )
    ax.axvline(TREATMENT_START_DATE, color="black", linestyle=":", linewidth=1.2)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Antall deltakere (sum alle regioner)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    from matplotlib.lines import Line2D

    handles = ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1]
    handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=1.2))
    labels.append("Behandlingsstart (2025-06)")
    ax.legend(handles, labels, fontsize=9)
    plt.tight_layout()
    _save_fig(fig, fig_path)


def _plot_sa_regions(tiltak_path: Path, title: str, fig_path: Path) -> None:
    """Small-multiples plot: raw vs. both SA variants for every region.

    Each subplot covers one Nav region.  Shows the same three series as
    :func:`_plot_tiltak_sa_trend` (raw, full-series SA, pre-only SA) so the
    reader can judge whether the seasonal correction is consistent across
    regions.
    """
    import numpy as np

    raw_long = _load_tiltak_sa(tiltak_path, seasonal_adjust=False)
    sa_full_long = _load_tiltak_sa(
        tiltak_path, seasonal_adjust=True, seasonal_adjust_pre_only=False
    )
    sa_pre_long = _load_tiltak_sa(
        tiltak_path, seasonal_adjust=True, seasonal_adjust_pre_only=True
    )

    def _pivot(long: pd.DataFrame) -> pd.DataFrame:
        return (
            long.assign(dato=lambda d: pd.to_datetime(d["aarmnd"], format="%Y%m"))
            .pivot(index="dato", columns="region", values="tiltak")
            .sort_index()
        )

    raw_p = _pivot(raw_long)
    sf_p = _pivot(sa_full_long)
    sp_p = _pivot(sa_pre_long)
    regions = sorted(raw_p.columns)

    n = len(regions)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3), sharex=True)
    axes_flat = axes.flatten()

    for i, region in enumerate(regions):
        ax = axes_flat[i]
        ax.plot(
            raw_p.index,
            raw_p[region],
            color=_LIGHT_BLUE,
            linewidth=1.0,
            alpha=0.7,
            label="Rå",
        )
        ax.plot(
            sf_p.index,
            sf_p[region],
            color=_BLUE,
            linewidth=1.3,
            label="SK – full serie",
        )
        ax.plot(
            sp_p.index,
            sp_p[region],
            color=_RED,
            linewidth=1.3,
            linestyle="--",
            label="SK – pre-periode",
        )
        ax.axvline(TREATMENT_START_DATE, color="black", linestyle=":", linewidth=0.9)
        short = region.replace("Nav ", "")
        ax.set_title(short, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=0.9))
    labels.append("Behandlingsstart")
    fig.legend(
        handles, labels, loc="lower right", fontsize=9, bbox_to_anchor=(0.98, 0.01)
    )

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    _save_fig(fig, fig_path)


def _plot_tiltak_regional_means(
    df: pd.DataFrame, regions: list[str], title: str, fig_path: Path
) -> None:
    """Horizontal bar chart: pre-period regional mean tiltaksbruk."""
    pre = df[df["aarmnd"] < TREATMENT_START_DATE]
    means = pre[regions].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    means.plot(kind="barh", ax=ax, color=_BLUE)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Gjennomsnitt per måned (pre-periode)")
    plt.tight_layout()
    _save_fig(fig, fig_path)


def _plot_tiltak_comparison(
    midl_df: pd.DataFrame,
    midl_regions: list[str],
    alle_df: pd.DataFrame,
    alle_regions: list[str],
    fig_path: Path,
) -> None:
    """Total count over time: alle tiltak vs midlertidig lønnstilskudd."""
    total_alle = alle_df.set_index("aarmnd")[alle_regions].sum(axis=1)
    total_midl = midl_df.set_index("aarmnd")[midl_regions].sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        total_alle.index,
        total_alle.values,
        label="Alle tiltak",
        color=_BLUE,
        linewidth=1.5,
    )
    ax.plot(
        total_midl.index,
        total_midl.values,
        label="Midl. lønnstilskudd",
        color=_RED,
        linestyle="--",
        linewidth=1.5,
    )
    ax.axvline(TREATMENT_START_DATE, color="black", linestyle=":", linewidth=1.2)
    ax.set_title("Totalt tiltaksnivå (alle regioner samlet)", fontsize=12)
    ax.set_ylabel("Antall deltakere")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, fig_path)


def _plot_indicator_trends_by_group(ind_key: str, fig_path: Path) -> None:
    """Average indicator value per month, one line per innsatsgruppe."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for grp, col in _GRUPPE_COLORS.items():
        fp = DATA_INPUT / "indikatorer" / "nedbrytning" / grp / f"{ind_key}.csv"
        if not fp.exists():
            continue
        long = _load_indicator(fp)
        monthly = (
            long.groupby("aarmnd")["verdi"]
            .mean()
            .reset_index()
            .assign(dato=lambda d: pd.to_datetime(d["aarmnd"], format="%Y%m"))
            .sort_values("dato")
        )
        ax.plot(
            monthly["dato"],
            monthly["verdi"],
            label=_GRUPPE_LABELS[grp],
            color=col,
            linewidth=1.3,
        )

    ax.axvline(TREATMENT_START_DATE, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Prosentpoeng (gjennomsnitt over regioner)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_fig(fig, fig_path)


def _plot_indicator_outcome_types(ind_key: str, fig_path: Path) -> None:
    """Compare plain / faktisk / forventet variants for alle innsatsgrupper."""
    variants = [
        ("", "-", _BLUE, "Standard"),
        ("faktisk_", "--", _RED, "Faktisk"),
        ("forventet_", ":", _LIGHT_BLUE, "Forventet"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for prefix, ls, col, lbl in variants:
        fp = (
            DATA_INPUT
            / "indikatorer"
            / "nedbrytning"
            / "alle"
            / f"{prefix}{ind_key}.csv"
        )
        if not fp.exists():
            continue
        long = _load_indicator(fp)
        monthly = (
            long.groupby("aarmnd")["verdi"]
            .mean()
            .reset_index()
            .assign(dato=lambda d: pd.to_datetime(d["aarmnd"], format="%Y%m"))
            .sort_values("dato")
        )
        ax.plot(
            monthly["dato"],
            monthly["verdi"],
            label=lbl,
            color=col,
            linestyle=ls,
            linewidth=1.4,
        )

    ax.axvline(TREATMENT_START_DATE, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Prosentpoeng (gjennomsnitt)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, fig_path)


# ── Table generators ───────────────────────────────────────────────────────────


def _tiltak_stats_table(df: pd.DataFrame, regions: list[str]) -> str:
    """Markdown table: pre/post mean and percentage change per region."""
    long = df.melt(
        id_vars="aarmnd", value_vars=regions, var_name="region", value_name="tiltak"
    )
    long["periode"] = long["aarmnd"].apply(
        lambda d: "Post" if d >= TREATMENT_START_DATE else "Pre"
    )
    tbl = (
        long.groupby(["region", "periode"])["tiltak"]
        .mean()
        .unstack("periode")
        .rename(columns={"Pre": "Gj.snitt pre", "Post": "Gj.snitt post"})
        .sort_values("Gj.snitt pre", ascending=False)
    )
    tbl["Endring (pp)"] = (tbl["Gj.snitt post"] - tbl["Gj.snitt pre"]).round(0)
    tbl["Endring (%)"] = (tbl["Endring (pp)"] / tbl["Gj.snitt pre"] * 100).round(1)
    tbl[["Gj.snitt pre", "Gj.snitt post"]] = tbl[
        ["Gj.snitt pre", "Gj.snitt post"]
    ].round(0)
    return tbl.to_markdown()


def _indicator_stats_table() -> str:
    """Markdown table: pre-period descriptive stats per group × indicator."""
    rows = []
    for grp in _GRUPPE_LABELS:
        for ind_key, ind_label in [("atid3", "atid3"), ("jobb3", "jobb3")]:
            fp = DATA_INPUT / "indikatorer" / "nedbrytning" / grp / f"{ind_key}.csv"
            if not fp.exists():
                continue
            long = _load_indicator(fp)
            pre = long[long["aarmnd"] < TREATMENT_START_STR]["verdi"]
            rows.append(
                {
                    "Innsatsgruppe": _GRUPPE_LABELS[grp],
                    "Indikator": ind_key,
                    "Gj.snitt (pre)": round(float(pre.mean()), 3),
                    "Std.avvik": round(float(pre.std()), 3),
                    "Min": round(float(pre.min()), 3),
                    "Maks": round(float(pre.max()), 3),
                }
            )
    return pd.DataFrame(rows).set_index(["Innsatsgruppe", "Indikator"]).to_markdown()


# ── QMD section builders ───────────────────────────────────────────────────────


def _section_tiltak_chapter(
    tiltak_label: str,
    tiltak_file_label: str,
    description: str,
    regions: list[str],
    df: pd.DataFrame,
    tiltak_path: Path,
    sa_plot: bool,
    figures_dir: Path,
    report_dir: Path,
    fig_prefix: str,
    include_comparison: bool = False,
    midl_df: pd.DataFrame | None = None,
    midl_regions: list[str] | None = None,
) -> list[str]:
    lines: list[str] = []

    date_from = df["aarmnd"].dt.strftime("%Y-%m").iloc[0]
    date_to = df["aarmnd"].dt.strftime("%Y-%m").iloc[-1]
    n_months = len(df)

    lines += [
        "## Om datakilden",
        "",
        description,
        "",
        "| Egenskap | Verdi |",
        "|---|---|",
        "| Kilde | Nav intern statistikk |",
        "| Nivå | Nav-regioner (12 regioner) |",
        "| Frekvens | Månedlig |",
        f"| Tidsperiode | {date_from} – {date_to} |",
        f"| Antall måneder | {n_months} |",
        "",
    ]

    # Trend figure
    fig_trend = figures_dir / f"{fig_prefix}_trend_regioner.svg"
    _plot_tiltak_trends(df, regions, f"{tiltak_label} per region over tid", fig_trend)
    lines += [
        "## Tiltaksbruk over tid",
        "",
        f"Deltakere i *{tiltak_label}* per region. Stiplet linje = behandlingsstart (2025-06).",
        "",
        f"![]({_rel(fig_trend, report_dir)}){{fig-align='center' width=95%}}",
        "",
    ]

    # Seasonally adjusted figure (only for sources with multiplicative seasonality)
    if sa_plot:
        fig_sa = figures_dir / f"{fig_prefix}_sa.svg"
        fig_sa_regions = figures_dir / f"{fig_prefix}_sa_regioner.svg"
        _plot_tiltak_sa_trend(
            tiltak_path,
            f"Sesongkorrigert tiltaksbruk (STL, nasjonal total) – {tiltak_label}",
            fig_sa,
        )
        _plot_sa_regions(
            tiltak_path,
            f"Sesongkorrigering per region – {tiltak_label}",
            fig_sa_regions,
        )
        lines += [
            "## Sesongkorrigert tiltaksbruk",
            "",
            "Sesongkorrigering med STL på log-transformerte verdier "
            "(`period=12`, `seasonal=13`, `robust=True`). "
            "Fordi sesongsvingningen vokser proporsjonalt med nivået "
            "(multiplikativ sesong), passer log-transformasjon bedre enn additiv STL.",
            "",
            "**STL tilpasses den nasjonale totalen** (sum over alle regioner). "
            "Den estimerte multiplikative sesongfaktoren for hver måned "
            "brukes deretter likt på alle regioner — det vil si at hver regions "
            "råverdi divideres med den samme nasjonale faktoren. "
            "Dette innebærer en forutsetning om at sesongmønsteret er proporsjonalt "
            "likt på tvers av regioner, noe regionplottene under kan brukes til å vurdere.",
            "",
            "To varianter vises:",
            "",
            "| Serie | Metode |",
            "|---|---|",
            "| **SK – full serie** (brukes i analysen) | STL tilpasset hele serien; "
            "`robust=True` demper outlier-innflytelse fra post-perioden |",
            "| **SK – kun pre-periode** | STL tilpasset kun pre-perioden; "
            "sesongfaktorer per kalendermåned ekstrapoleres til post-perioden |",
            "",
            "Likheten mellom de to viser i hvilken grad den estimerte sesongen er "
            "stabil over tid.",
            "",
            f"![]({_rel(fig_sa, report_dir)}){{fig-align='center' width=95%}}",
            "",
            "### Sesongkorrigering per region",
            "",
            "Samme tre serier som over, men vist separat per region. "
            "Fordi alle regioner deles med den *samme* nasjonale sesongfaktoren, "
            "reflekterer regionplottene kun regionenes egne nivåer — ikke "
            "regionale avvik i sesongmønster. "
            "Plottene er nyttige for å vurdere om forutsetningen om felles "
            "sesongmønster virker rimelig.",
            "",
            f"![]({_rel(fig_sa_regions, report_dir)}){{fig-align='center' width=100%}}",
            "",
        ]

    # Regional means figure
    fig_reg = figures_dir / f"{fig_prefix}_regional_gjennomsnitt.svg"
    _plot_tiltak_regional_means(
        df,
        regions,
        "Gjennomsnittlig antall deltakere per region (pre-periode)",
        fig_reg,
    )
    lines += [
        "## Regional fordeling (pre-periode)",
        "",
        "Gjennomsnittlig månedlig deltakerantall per region i perioden før behandlingsstart.",
        "",
        f"![]({_rel(fig_reg, report_dir)}){{fig-align='center' width=90%}}",
        "",
    ]

    # Comparison figure (only for alle-tiltak chapter)
    if include_comparison and midl_df is not None and midl_regions is not None:
        fig_cmp = figures_dir / f"{fig_prefix}_sammenligning.svg"
        _plot_tiltak_comparison(midl_df, midl_regions, df, regions, fig_cmp)
        lines += [
            "## Sammenlikning med midlertidig lønnstilskudd",
            "",
            "Totalt tiltaksnivå (sum over alle regioner). Stiplet = behandlingsstart.",
            "",
            f"![]({_rel(fig_cmp, report_dir)}){{fig-align='center' width=95%}}",
            "",
        ]

    # Stats table
    lines += [
        "## Deskriptiv statistikk per region",
        "",
        "Gjennomsnittlig deltakerantall per region i pre- og post-perioden.",
        "",
        _tiltak_stats_table(df, regions),
        "",
    ]

    return lines


def _section_indikatorer(figures_dir: Path, report_dir: Path) -> list[str]:
    lines: list[str] = []

    lines += [
        "## Om indikatorene",
        "",
        "Utfallsmålene i analysene er to indikatorer for overgang til arbeid:",
        "",
        "| Indikator | Beskrivelse |",
        "|---|---|",
        "| **atid3** | Andel brukere som er registrert med arbeidstid de neste tre månedene |",
        "| **jobb3** | Andel brukere som er i jobb etter tre måneder |",
        "",
        "Begge er tilgjengelige i tre varianter:",
        "",
        "| Variant | Beskrivelse |",
        "|---|---|",
        "| Standard (`atid3.csv`) | Ujustert observert verdi |",
        "| Faktisk (`faktisk_atid3.csv`) | Korrigert for forventningseffekter |",
        "| Forventet (`forventet_atid3.csv`) | Modellbasert forventningsverdi |",
        "",
        "Tilgjengelig for innsatsgruppene: **Alle**, **Standard**, **Situasjonsbestemt**,",
        "**Spesielt tilpasset** og **Veiledning kombinert**.",
        "",
        "Data på regionnivå dekker perioden `202306`–`202512`. "
        "Enhetsnivå-data er tilgjengelig fra `202306`.",
        "",
    ]

    for ind_key, ind_label in [
        ("atid3", "Arbeidstid neste tre måneder (atid3)"),
        ("jobb3", "I jobb etter 3 måneder (jobb3)"),
    ]:
        # By-group trend
        fig_grp = figures_dir / f"ind_{ind_key}_per_gruppe.svg"
        _plot_indicator_trends_by_group(ind_key, fig_grp)

        # Outcome-type comparison
        fig_ot = figures_dir / f"ind_{ind_key}_utfallstype.svg"
        _plot_indicator_outcome_types(ind_key, fig_ot)

        lines += [
            f"## {ind_label}",
            "",
            "### Trend per innsatsgruppe",
            "",
            "Gjennomsnitt over regioner. Stiplet linje = behandlingsstart.",
            "",
            f"![]({_rel(fig_grp, report_dir)}){{fig-align='center' width=95%}}",
            "",
            "### Faktisk vs. forventet (alle innsatsgrupper)",
            "",
            f"![]({_rel(fig_ot, report_dir)}){{fig-align='center' width=95%}}",
            "",
        ]

    lines += [
        "## Deskriptiv statistikk (pre-periode)",
        "",
        "Gjennomsnitt og spredning for pre-perioden per innsatsgruppe og indikator.",
        "",
        _indicator_stats_table(),
        "",
    ]

    return lines


# ── QMD writers ────────────────────────────────────────────────────────────────


def _frontmatter(title: str) -> list[str]:
    return ["---", f'title: "{title}"', "---", ""]


def _write_qmd(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Written %s", path)


def generate_data_report() -> None:
    """Generate all three data-section QMD files and their figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tiltak data")
    midl_path = DATA_INPUT / "tiltak" / "midl.-lønnstilskudd.csv"
    alle_path = DATA_INPUT / "tiltak" / "alle-tiltak.csv"
    midl_df, midl_regions = _load_tiltak(midl_path)
    alle_df, alle_regions = _load_tiltak(alle_path)

    # ── Midlertidig lønnstilskudd ────────────────────────────────────────────
    logger.info("Generating midl.-lønnstilskudd chapter")
    midl_lines = _frontmatter("Tiltaksdata: Midlertidig lønnstilskudd")
    midl_lines += _section_tiltak_chapter(
        tiltak_label="midlertidig lønnstilskudd",
        tiltak_file_label="midl.-lønnstilskudd",
        description=(
            "**Midlertidig lønnstilskudd** er et tiltak der Nav subsidierer lønnskostnader "
            "til arbeidsgivere som ansetter personer med nedsatt arbeidsevne eller som har "
            "stått lenge utenfor arbeidsmarkedet. Tiltaket er særlig rettet mot innsatsgruppen "
            "*Veiledning kombinert*, og utgjorde en betydelig del av Nav's tiltaksportefølje "
            "frem til nedgangen i 2025.\n\n"
            "Disse dataene brukes som **behandlingsvariabel** i alle analyser under "
            "*DiD – Midlertidig lønnstilskudd* og *TrippelDiD*-seksjonene."
        ),
        regions=midl_regions,
        df=midl_df,
        tiltak_path=midl_path,
        sa_plot=False,
        figures_dir=FIGURES_DIR,
        report_dir=QUARTO_DATA,
        fig_prefix="midl",
    )
    _write_qmd(QUARTO_DATA / "tiltak-midl-lonnstilskudd.qmd", midl_lines)

    # ── Alle tiltak ──────────────────────────────────────────────────────────
    logger.info("Generating alle-tiltak chapter")
    alle_lines = _frontmatter("Tiltaksdata: Alle tiltak")
    alle_lines += _section_tiltak_chapter(
        tiltak_label="alle tiltak",
        tiltak_file_label="alle-tiltak",
        description=(
            "**Alle tiltak** er summen av samtlige arbeidsmarkedstiltak Nav tilbyr, "
            "inkludert lønnstilskudd, arbeidspraksis, opplæringstiltak og "
            "oppfølgingstiltak. Disse dataene gir et bredere bilde av tiltaksaktiviteten "
            "enn midlertidig lønnstilskudd alene.\n\n"
            "Disse dataene brukes som **behandlingsvariabel** i analysene under "
            "*DiD – Alle tiltak*-seksjonen. Behandlingsvariabelen er beregnet fra "
            "sesongkorrigerte tiltakstall (multiplikativ STL) for å unngå skjevhet "
            "fra sommer-dip-effekter som varierer med regioners nivå."
        ),
        regions=alle_regions,
        df=alle_df,
        tiltak_path=alle_path,
        sa_plot=True,
        figures_dir=FIGURES_DIR,
        report_dir=QUARTO_DATA,
        fig_prefix="alle",
        include_comparison=True,
        midl_df=midl_df,
        midl_regions=midl_regions,
    )
    _write_qmd(QUARTO_DATA / "tiltak-alle-tiltak.qmd", alle_lines)

    # ── Indikatorer ──────────────────────────────────────────────────────────
    logger.info("Generating indikatorer chapter")
    ind_lines = _frontmatter("Indikatordata")
    ind_lines += _section_indikatorer(figures_dir=FIGURES_DIR, report_dir=QUARTO_DATA)
    _write_qmd(QUARTO_DATA / "indikatorer.qmd", ind_lines)

    logger.info("Data report generation complete")


def main() -> int:
    """Generate the data report chapters from command-line invocation."""
    generate_data_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
