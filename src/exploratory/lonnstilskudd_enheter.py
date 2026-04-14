"""Lag en QMD-rapport med tiltak per enhet, gruppert i én figur per region."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "pre"
OUT_DIR = PROJECT_ROOT / "outputs" / "exploratory" / "lonnstilskudd_enheter"
FIG_DIR = OUT_DIR / "figures"
QMD_PATH = OUT_DIR / "lonnstilskudd_enheter.qmd"

TILTAK_FILE = DATA_DIR / "midl-lønnstilskudd-kontor.json"
MAPPING_FILE = DATA_DIR / "enhetsmapping.json"
CRASH_START = pd.Timestamp("2025-06-01")


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    return cleaned or "region"


def _read_items(path: Path) -> list[dict[str, Any]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    try:
        result: list[dict[str, Any]] = payload["results"][0]["items"]
        return result
    except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover
        raise ValueError(f"Ugyldig JSON-format i {path}") from exc


def read_tiltak_data() -> pd.DataFrame:
    """Les tiltakstall per enhet og måned."""
    df = pd.DataFrame(_read_items(TILTAK_FILE))
    df = df.rename(
        columns={
            "aarmnd_dato": "aarmnd",
            "nav_enhet_navn": "enhet",
            "cnt": "tiltak",
        }
    )
    df["aarmnd"] = pd.to_datetime(df["aarmnd"], format="%d.%m.%Y %H:%M:%S")
    df["tiltak"] = pd.to_numeric(df["tiltak"], errors="coerce").fillna(0)
    return df[["aarmnd", "enhet", "tiltak"]].sort_values(["enhet", "aarmnd"]).copy()


def read_mapping_data() -> pd.DataFrame:
    """Les mapping fra enhet til region."""
    mapping = pd.DataFrame(_read_items(MAPPING_FILE))
    mapping = mapping.rename(
        columns={"nav_enhet_navn": "enhet", "nav_region_navn": "region"}
    )
    mapping = mapping.dropna(subset=["enhet", "region"]).drop_duplicates(
        subset=["enhet"]
    )
    return mapping[["enhet", "region"]].copy()


def map_data_to_region(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Koble tiltakstall til region og marker enheter uten mapping."""
    merged = df.merge(mapping, on="enhet", how="left", validate="many_to_one")
    missing = int(merged["region"].isna().sum())
    if missing > 0:
        logger.warning("Fant %s rader uten regionmapping", missing)
    merged["region"] = merged["region"].fillna("Ukjent region")
    return merged


def compute_crash_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn krasj-mål fra siste observasjon før 2025-06 til laveste etter."""
    pre = df[df["aarmnd"] < CRASH_START].sort_values(["enhet", "aarmnd"])
    post = df[df["aarmnd"] >= CRASH_START].sort_values(["enhet", "aarmnd"])

    if pre.empty:
        return pd.DataFrame(
            columns=[
                "enhet",
                "region",
                "ref_mnd",
                "ref_tiltak",
                "min_post",
                "crash_abs",
                "crash_pct",
            ]
        )

    reference = (
        pre.groupby("enhet")
        .tail(1)
        .rename(columns={"aarmnd": "ref_mnd", "tiltak": "ref_tiltak"})
    )
    reference = reference[["enhet", "region", "ref_mnd", "ref_tiltak"]]

    if post.empty:
        out = reference.copy()
        out["min_post"] = pd.NA
        out["crash_abs"] = pd.NA
        out["crash_pct"] = pd.NA
        return out

    post_min = (
        post.groupby("enhet", as_index=False)["tiltak"]
        .min()
        .rename(columns={"tiltak": "min_post"})
    )
    metrics = reference.merge(post_min, on="enhet", how="left")
    metrics["crash_abs"] = (metrics["ref_tiltak"] - metrics["min_post"]).clip(lower=0)
    metrics["crash_pct"] = metrics["crash_abs"] / metrics["ref_tiltak"].where(
        metrics["ref_tiltak"] > 0
    )
    return metrics


def compute_peak_to_latest_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback-mål: nedgang fra historisk topp til siste måned i datagrunnlaget."""
    latest_month = df["aarmnd"].max()
    peak = (
        df.groupby(["enhet", "region"], as_index=False)["tiltak"]
        .max()
        .rename(columns={"tiltak": "peak_tiltak"})
    )
    latest = df[df["aarmnd"] == latest_month][["enhet", "tiltak"]].rename(
        columns={"tiltak": "latest_tiltak"}
    )
    out = peak.merge(latest, on="enhet", how="left")
    out["fallback_abs"] = (out["peak_tiltak"] - out["latest_tiltak"]).clip(lower=0)
    out["fallback_pct"] = out["fallback_abs"] / out["peak_tiltak"].where(
        out["peak_tiltak"] > 0
    )
    return out


def plot_region(df: pd.DataFrame, region: str, output_dir: Path) -> Path:
    """Lag én tidsseriefigur med alle enheter i en region."""
    region_df = df[df["region"] == region].sort_values(["enhet", "aarmnd"])
    enheter = sorted(region_df["enhet"].unique())

    fig, ax = plt.subplots(figsize=(14, max(6, min(12, 4 + 0.18 * len(enheter)))))
    for enhet in enheter:
        subset = region_df[region_df["enhet"] == enhet]
        ax.plot(
            subset["aarmnd"], subset["tiltak"], linewidth=1.1, alpha=0.85, label=enhet
        )

    ax.axvline(CRASH_START, color="#C8102E", linestyle="--", linewidth=1.4)
    ax.set_title(f"Midlertidig lønnstilskudd per enhet – {region}")
    ax.set_xlabel("Måned")
    ax.set_ylabel("Antall deltakere")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)

    filename = f"{_slugify(region)}.png"
    out = output_dir / filename
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_region_normalized(
    df: pd.DataFrame, region: str, output_dir: Path
) -> Path | None:
    """Lag normalisert figur per region med basis i nivå ved 2025-06.

    Kun enheter med tiltak > 0 i 2025-06 tas med. Serien vises som indeks der
    2025-06 = 100, slik at relative fall mellom enheter blir lettere å sammenligne.
    """
    region_df = df[df["region"] == region].sort_values(["enhet", "aarmnd"]).copy()
    base_month = region_df[region_df["aarmnd"] == CRASH_START][
        ["enhet", "tiltak"]
    ].rename(columns={"tiltak": "base_tiltak"})
    base_month = base_month[base_month["base_tiltak"] > 0]
    if base_month.empty:
        return None

    post = region_df[region_df["aarmnd"] >= CRASH_START].merge(
        base_month, on="enhet", how="inner", validate="many_to_one"
    )
    if post.empty:
        return None
    post["indeks_202506"] = (post["tiltak"] / post["base_tiltak"]) * 100

    enheter = sorted(post["enhet"].unique())
    fig, ax = plt.subplots(figsize=(14, max(6, min(12, 4 + 0.16 * len(enheter)))))
    for enhet in enheter:
        subset = post[post["enhet"] == enhet]
        ax.plot(
            subset["aarmnd"],
            subset["indeks_202506"],
            linewidth=1.1,
            alpha=0.9,
            label=enhet,
        )
    ax.axhline(100, color="#888888", linestyle=":", linewidth=1.2)
    ax.axvline(CRASH_START, color="#C8102E", linestyle="--", linewidth=1.4)
    ax.set_title(f"Normalisert lønnstilskudd per enhet – {region} (2025-06 = 100)")
    ax.set_xlabel("Måned")
    ax.set_ylabel("Indeks (2025-06 = 100)")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)

    filename = f"{_slugify(region)}-normalized-202506.png"
    out = output_dir / filename
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def build_region_normalized_comment(df: pd.DataFrame, region: str) -> str:
    """Regionkommentar for normalisert figur (2025-06 = 100)."""
    region_df = df[df["region"] == region].copy()
    base = region_df[region_df["aarmnd"] == CRASH_START][["enhet", "tiltak"]].rename(
        columns={"tiltak": "base_tiltak"}
    )
    base = base[base["base_tiltak"] > 0]
    if base.empty:
        return "Ingen enheter med tiltak > 0 i 2025-06, så normalisert figur er ikke tilgjengelig."

    post = region_df[region_df["aarmnd"] >= CRASH_START].merge(
        base, on="enhet", how="inner", validate="many_to_one"
    )
    if post.empty:
        return (
            "Ingen observasjoner etter 2025-06 for enheter med positivt nivå i 2025-06."
        )

    post["indeks_202506"] = (post["tiltak"] / post["base_tiltak"]) * 100
    post = post.dropna(subset=["enhet"])
    if post.empty:
        return "Ingen gyldige enhetsobservasjoner tilgjengelig for normalisert sammenligning."

    last = post.groupby("enhet", as_index=False).tail(1)
    if last.empty:
        return "Ingen gyldige enhetsobservasjoner tilgjengelig for normalisert sammenligning."
    worst = last.sort_values("indeks_202506", ascending=True).iloc[0]
    best = last.sort_values("indeks_202506", ascending=False).iloc[0]
    spread = float(last["indeks_202506"].max() - last["indeks_202506"].min())

    return (
        "Normalisert sammenligning (2025-06 = 100): "
        f"størst fall har **{worst['enhet']}** (indeks {worst['indeks_202506']:.1f}), "
        f"minst fall/sterkest utvikling har **{best['enhet']}** (indeks {best['indeks_202506']:.1f}), "
        f"og spennet i siste observasjon er **{spread:.1f}** indekspoeng."
    )


def _format_pct(value: Any) -> str:
    if pd.isna(value):
        return "ikke tilgjengelig"
    return f"{value:.1%}"


def build_global_comments(
    df: pd.DataFrame, crash_metrics: pd.DataFrame, fallback_metrics: pd.DataFrame
) -> list[str]:
    """Generer overordnede kommentarpunkter for rapporten."""
    comments: list[str] = []
    latest_month = pd.Timestamp(df["aarmnd"].max()).strftime("%Y-%m")

    has_post_crash = df["aarmnd"].max() >= CRASH_START
    if has_post_crash and crash_metrics["crash_pct"].notna().any():
        valid = crash_metrics.dropna(subset=["crash_pct"])
        by_region = (
            valid.groupby("region")["crash_pct"]
            .agg(region_min="min", region_max="max")
            .assign(spread=lambda x: x["region_max"] - x["region_min"])
            .sort_values("spread", ascending=False)
        )
        spread_row = by_region.iloc[0]
        comments.append(
            "Størst spredning i krasj finnes i "
            f"**{by_region.index[0]}** ({_format_pct(float(spread_row['spread']))} mellom "
            f"største og minste enhetskrasj)."
        )

        crash_worst = valid.sort_values("crash_pct", ascending=False).iloc[0]
        comments.append(
            "Enheten med største krasj fra referansemåned til laveste nivå etter 2025-06 er "
            f"**{crash_worst['enhet']}** i **{crash_worst['region']}** "
            f"({_format_pct(float(crash_worst['crash_pct']))})."
        )
    else:
        comments.append(
            "Ingen observasjoner fra og med **2025-06** i datagrunnlaget. "
            f"Kommentarene under bygger derfor på utviklingen fram til **{latest_month}**."
        )
        valid = fallback_metrics.dropna(subset=["fallback_pct"])
        by_region = (
            valid.groupby("region")["fallback_pct"]
            .agg(region_min="min", region_max="max")
            .assign(spread=lambda x: x["region_max"] - x["region_min"])
            .sort_values("spread", ascending=False)
        )
        spread_row = by_region.iloc[0]
        comments.append(
            "Størst intern spredning (topp-til-siste) finnes i "
            f"**{by_region.index[0]}** ({_format_pct(float(spread_row['spread']))})."
        )
        crash_worst = valid.sort_values("fallback_pct", ascending=False).iloc[0]
        comments.append(
            "Største observerte nedgang (topp-til-siste måned) har "
            f"**{crash_worst['enhet']}** i **{crash_worst['region']}** "
            f"({_format_pct(float(crash_worst['fallback_pct']))})."
        )

    monthly_total = (
        df.groupby("aarmnd", as_index=False)["tiltak"].sum().sort_values("aarmnd")
    )
    peak_row = monthly_total.loc[monthly_total["tiltak"].idxmax()]
    low_row = monthly_total.loc[monthly_total["tiltak"].idxmin()]
    comments.append(
        "På tvers av alle enheter er høyeste totalmåned "
        f"**{pd.Timestamp(peak_row['aarmnd']).strftime('%Y-%m')}** "
        f"({int(peak_row['tiltak'])} deltakere), mens laveste er "
        f"**{pd.Timestamp(low_row['aarmnd']).strftime('%Y-%m')}** "
        f"({int(low_row['tiltak'])} deltakere)."
    )

    return comments


def build_region_comment(
    region: str, crash_metrics: pd.DataFrame, fallback_metrics: pd.DataFrame
) -> str:
    """Lag kort regionkommentar om toppnedgang og spredning."""
    sub = crash_metrics[crash_metrics["region"] == region].dropna(subset=["crash_pct"])
    if not sub.empty:
        worst = sub.sort_values("crash_pct", ascending=False).iloc[0]
        spread = (
            float(sub["crash_pct"].max() - sub["crash_pct"].min())
            if len(sub) > 1
            else 0.0
        )
        return (
            f"Største enhetskrasj i regionen er **{worst['enhet']}** "
            f"({_format_pct(float(worst['crash_pct']))}). "
            f"Intern spredning er **{_format_pct(spread)}**."
        )

    fallback = fallback_metrics[fallback_metrics["region"] == region].dropna(
        subset=["fallback_pct"]
    )
    if fallback.empty:
        return "Ingen tilstrekkelige observasjoner for å beregne nedgangsmål i denne regionen."
    worst = fallback.sort_values("fallback_pct", ascending=False).iloc[0]
    spread = (
        float(fallback["fallback_pct"].max() - fallback["fallback_pct"].min())
        if len(fallback) > 1
        else 0.0
    )
    return (
        f"Ingen data fra 2025-06+, så kommentaren bruker topp-til-siste måned. "
        f"Størst nedgang har **{worst['enhet']}** ({_format_pct(float(worst['fallback_pct']))}), "
        f"med intern spredning **{_format_pct(spread)}**."
    )


def mapping_summary_table(mapping: pd.DataFrame) -> str:
    """Lag markdown-tabell med antall enheter per region."""
    summary = (
        mapping.groupby("region", as_index=False)["enhet"]
        .nunique()
        .rename(columns={"enhet": "Antall enheter"})
        .sort_values(["Antall enheter", "region"], ascending=[False, True])
    )
    header = "| Region | Antall enheter |\n|---|---:|"
    rows = [
        f"| {row['region']} | {int(row['Antall enheter'])} |"
        for _, row in summary.iterrows()
    ]
    return "\n".join([header, *rows])


def write_qmd(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    figure_paths: dict[str, Path],
    normalized_figure_paths: dict[str, Path | None],
    crash_metrics: pd.DataFrame,
    fallback_metrics: pd.DataFrame,
) -> None:
    """Skriv QMD-rapport med figurer og kommentarer."""
    global_comments = build_global_comments(df, crash_metrics, fallback_metrics)
    latest_month = pd.Timestamp(df["aarmnd"].max()).strftime("%Y-%m")
    earliest_month = pd.Timestamp(df["aarmnd"].min()).strftime("%Y-%m")

    unmatched = sorted(
        {
            str(enhet)
            for enhet in df.loc[df["region"] == "Ukjent region", "enhet"]
            if pd.notna(enhet)
        }
    )
    unmatched_text = ", ".join(unmatched) if unmatched else "Ingen"

    lines: list[str] = [
        "---",
        'title: "Lønnstilskudd per enhet og region"',
        "format: html",
        "toc: true",
        "lang: nb",
        "---",
        "",
        "## Om analysen",
        f"- Dataperiode: **{earliest_month}** til **{latest_month}**.",
        "- Tiltakscrash er definert til å starte **2025-06**.",
        f"- Tiltaksdata: `{TILTAK_FILE.relative_to(PROJECT_ROOT)}`.",
        f"- Regionmapping: `{MAPPING_FILE.relative_to(PROJECT_ROOT)}`.",
        "",
        "## Mapping brukt",
        mapping_summary_table(mapping),
        "",
        f"Enheter uten mapping i datasettet: **{unmatched_text}**.",
        "",
        "## Hovedfunn",
        *[f"- {comment}" for comment in global_comments],
        "- I tillegg vises normaliserte regionfigurer der enheter med nivå i **2025-06** settes til indeks **100** for å tydeliggjøre relative fall.",
    ]

    for region in sorted(figure_paths):
        fig_rel = figure_paths[region].relative_to(OUT_DIR).as_posix()
        normalized_fig = normalized_figure_paths.get(region)
        region_enheter = sorted(
            mapping.loc[mapping["region"] == region, "enhet"].tolist()
        )
        lines.extend(
            [
                "",
                f"## {region}",
                f"- Antall enheter i mapping: **{len(region_enheter)}**.",
                f"- {build_region_comment(region, crash_metrics, fallback_metrics)}",
                "",
                f"![Tiltak per enhet i {region}]({fig_rel})",
            ]
        )
        lines.extend(
            [
                "",
                "### Normalisert utvikling (2025-06 = 100)",
                f"- {build_region_normalized_comment(df, region)}",
                "",
            ]
        )
        if normalized_fig is None:
            lines.append(
                "_Ingen normalisert figur for denne regionen (mangler enheter med nivå i 2025-06)._"
            )
        else:
            norm_rel = normalized_fig.relative_to(OUT_DIR).as_posix()
            lines.append(f"![Normalisert tiltak per enhet i {region}]({norm_rel})")

    QMD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Skrev QMD-rapport: %s", QMD_PATH)


def main() -> None:
    """Kjør hele analyseløpet og skriv figurer + QMD."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tiltak = read_tiltak_data()
    mapping = read_mapping_data()
    data = map_data_to_region(tiltak, mapping)

    crash_metrics = compute_crash_metrics(data)
    fallback_metrics = compute_peak_to_latest_metrics(data)

    figure_paths: dict[str, Path] = {}
    normalized_figure_paths: dict[str, Path | None] = {}
    for region in sorted(data["region"].unique()):
        figure_paths[region] = plot_region(data, region, FIG_DIR)
        normalized_figure_paths[region] = plot_region_normalized(data, region, FIG_DIR)
        logger.info("Laget figur for region: %s", region)

    write_qmd(
        data,
        mapping,
        figure_paths,
        normalized_figure_paths,
        crash_metrics,
        fallback_metrics,
    )


if __name__ == "__main__":
    main()
