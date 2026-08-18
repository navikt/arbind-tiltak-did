"""Publication figures for the summary workbook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ("#254B6D", "#66CBEC", "#B90000", "#FF9100", "#C1A753", "#858E00")
GROUPS = (
    "Alle",
    "Alle – diskret",
    "Personer i tiltak",
    "Gode muligheter",
    "Trenger veiledning",
    "Nedsatt arbeidsevne",
    "Veiledning kombinert",
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def _colours(count: int) -> list[object]:
    """Return sufficiently distinct colours for every plotted category."""
    return [plt.colormaps["tab20"](index) for index in range(count)]


def _trend(data: pd.DataFrame, value: str, group: str, title: str, path: Path) -> Path:
    frame = data.copy()
    frame["aarmnd"] = pd.to_datetime(frame["aarmnd"])
    fig, ax = plt.subplots(figsize=(17, 8))
    grouped = list(frame.groupby(group))
    colours = (
        ["#254B6D", "#B90000"]
        if group == "gruppe"
        and {name for name, _ in grouped} == {"Høy nedgang", "Lav nedgang"}
        else _colours(len(grouped))
    )
    for colour, (name, subset) in zip(colours, grouped, strict=True):
        series = subset.groupby("aarmnd")[value].mean()
        ax.plot(series.index, series, label=str(name), color=colour)
    ax.set(title=title, xlabel="Måned", ylabel=value)
    ax.legend(frameon=False)
    return _save(fig, path)


def figure1(
    all_measures: pd.DataFrame, wage_subsidy: pd.DataFrame, output_dir: Path
) -> Path:
    """Write half-year all-measures bars and wage-subsidy line on a right axis."""

    def half_year(data: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(data["aarmnd"])
        return (
            dates.dt.year.astype(str)
            + " H"
            + ((dates.dt.month - 1) // 6 + 1).astype(str)
        )

    all_frame = all_measures.copy()
    wage_frame = wage_subsidy.copy()
    all_frame["halvår"] = half_year(all_frame)
    wage_frame["halvår"] = half_year(wage_frame)
    all_series = all_frame.groupby("halvår", sort=False)["tiltak"].sum()
    wage_series = wage_frame.groupby("halvår", sort=False)["tiltak"].sum()
    periods = all_series.index.union(wage_series.index, sort=False)

    fig, axis = plt.subplots(figsize=(17, 8))
    axis.bar(periods, all_series.reindex(periods), color="#254B6D", label="Alle tiltak")
    axis.set(ylabel="Totalt antall deltakere – alle tiltak", xlabel="Halvår")
    right_axis = axis.twinx()
    right_axis.plot(
        periods,
        wage_series.reindex(periods),
        color="#66CBEC",
        marker="o",
        linewidth=2,
        label="Midlertidig lønnstilskudd",
    )
    right_axis.set_ylabel("Totalt antall deltakere – midlertidig lønnstilskudd")
    axis.set_title("Totalt antall deltakere per halvår")
    handles, labels = axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    axis.legend(handles + right_handles, labels + right_labels, frameon=False)
    return _save(fig, output_dir / "figur1.svg")


def figure2(
    data: pd.DataFrame,
    output_dir: Path,
    treatment_label: str = "Midlertidig lønnstilskudd",
) -> Path:
    """Write treatment participation trends by county."""
    return _trend(
        data,
        "tiltak",
        "region",
        f"{treatment_label} per fylke",
        output_dir / "figur2.svg",
    )


def figure3(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write Atid3 trends for high and low reduction groups."""
    return _trend(
        data, "indikator", "gruppe", "Utvikling i Atid3", output_dir / "figur3.svg"
    )


def figure4(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write Jobb3 trends for high and low reduction groups."""
    return _trend(
        data, "indikator", "gruppe", "Utvikling i Jobb3", output_dir / "figur4.svg"
    )


def figure8(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write heterogeneous preferred-estimate forest plot (TODO Figure 8)."""
    required = {"group", "indicator", "model", "coefficient", "ci_lower", "ci_upper"}
    if missing := required - set(data):
        raise ValueError(f"Figur 8 mangler kolonner: {sorted(missing)}")
    groups = tuple(group for group in GROUPS if group in set(data["group"]))
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharey=True)
    for ax, indicator in zip(axes, ("atid3", "jobb3"), strict=True):
        rows = (
            data[
                (data.group.isin(groups))
                & (data.indicator == indicator)
                & (data.model == "preferred")
            ]
            .set_index("group")
            .reindex(groups)
        )
        if rows[["coefficient", "ci_lower", "ci_upper"]].isna().any().any():
            raise ValueError(f"Figur 8 mangler {indicator}-resultater.")
        y = range(len(groups))
        ax.errorbar(
            rows.coefficient,
            y,
            xerr=[rows.coefficient - rows.ci_lower, rows.ci_upper - rows.coefficient],
            fmt="o",
            color=PALETTE[0],
            ecolor=PALETTE[1],
            capsize=3,
        )
        ax.axvline(0, color="black", linestyle=":")
        ax.set(
            title=indicator.title(),
            xlabel="Estimat (95 % KI)",
            yticks=list(y),
            yticklabels=groups,
        )
    axes[0].invert_yaxis()
    return _save(fig, output_dir / "figur8.svg")


def _faceted_estimates(
    data: pd.DataFrame, title: str, filename: str, output_dir: Path
) -> Path:
    """Write a two-panel forest plot for preferred Atid3 and Jobb3 estimates."""
    required = {"group", "indicator", "coefficient", "ci_lower", "ci_upper"}
    if missing := required - set(data):
        raise ValueError(f"{title} mangler kolonner: {sorted(missing)}")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharey=True)
    for ax, indicator in zip(axes, ("atid3", "jobb3"), strict=True):
        rows = data.loc[data.indicator == indicator].copy()
        if "model" in rows:
            rows = rows.loc[rows.model == "preferred"]
        rows = rows.sort_values("coefficient")
        if rows.empty:
            raise ValueError(f"{title} mangler resultater for {indicator}.")
        y = range(len(rows))
        ax.errorbar(
            rows.coefficient,
            y,
            xerr=[rows.coefficient - rows.ci_lower, rows.ci_upper - rows.coefficient],
            fmt="o",
            color=PALETTE[0],
            ecolor=PALETTE[1],
            capsize=3,
        )
        ax.axvline(0, color="black", linestyle=":")
        ax.set(
            title=indicator.title(),
            xlabel="Estimat (95 % KI)",
            yticks=list(y),
            yticklabels=rows.group,
        )
    fig.suptitle(title)
    return _save(fig, output_dir / filename)


def _diagnostic_axes() -> tuple[plt.Figure, object]:
    """Return eight axes for indicator, treatment, and model combinations."""
    return plt.subplots(2, 4, figsize=(17, 8), sharex=False, sharey=False)


def _diagnostic_title(indicator: str, treatment: str, model: str) -> str:
    """Return a Norwegian facet label for one main-result model."""
    labels = {"discrete": "Diskret", "continuous": "Kontinuerlig"}
    adjustment = "sesongjustert" if model == "preferred" else "ikke sesongjustert"
    return f"{indicator.title()} – {labels[treatment]}, {adjustment}"


def figure5(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write event-study estimates for all eight main-result models."""
    required = {
        "indicator",
        "treatment_type",
        "model",
        "relative_month",
        "coefficient",
        "ci_lower",
        "ci_upper",
    }
    if missing := required - set(data):
        raise ValueError(f"Figur 5 mangler kolonner: {sorted(missing)}")
    fig, axes = _diagnostic_axes()
    for ax, (indicator, treatment, model) in zip(
        axes.flat,
        (
            (indicator, treatment, model)
            for indicator in ("atid3", "jobb3")
            for treatment in ("discrete", "continuous")
            for model in ("baseline", "preferred")
        ),
        strict=True,
    ):
        rows = data.loc[
            (data.indicator == indicator)
            & (data.treatment_type == treatment)
            & (data.model == model)
        ].sort_values("relative_month")
        if rows.empty:
            raise ValueError(f"Figur 5 mangler {indicator}/{treatment}/{model}.")
        ax.errorbar(
            rows.relative_month,
            rows.coefficient,
            yerr=[rows.coefficient - rows.ci_lower, rows.ci_upper - rows.coefficient],
            fmt="o-",
            color=PALETTE[0],
            ecolor=PALETTE[1],
            capsize=2,
        )
        ax.axhline(0, color="black", linestyle=":")
        ax.axvline(-1, color="#858E00", linestyle=":", linewidth=1)
        ax.set(
            title=_diagnostic_title(indicator, treatment, model),
            xlabel="Relativ måned",
            ylabel="Estimat",
        )
    fig.suptitle("Eventstudier")
    return _save(fig, output_dir / "figur5.svg")


def figure6(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write joint parallel-trend test p-values for all eight main models."""
    required = {
        "indicator",
        "treatment_type",
        "model",
        "pretrend_f_stat",
        "pretrend_p_value",
    }
    if missing := required - set(data):
        raise ValueError(f"Figur 6 mangler kolonner: {sorted(missing)}")
    fig, axes = _diagnostic_axes()
    for ax, (indicator, treatment, model) in zip(
        axes.flat,
        (
            (indicator, treatment, model)
            for indicator in ("atid3", "jobb3")
            for treatment in ("discrete", "continuous")
            for model in ("baseline", "preferred")
        ),
        strict=True,
    ):
        rows = data.loc[
            (data.indicator == indicator)
            & (data.treatment_type == treatment)
            & (data.model == model)
        ]
        if len(rows) != 1:
            raise ValueError(f"Figur 6 mangler {indicator}/{treatment}/{model}.")
        row = rows.iloc[0]
        ax.bar(["Felles pretrend-test"], [row.pretrend_p_value], color="#254B6D")
        ax.axhline(0.05, color="#B90000", linestyle=":", label="5 %")
        ax.set(
            title=_diagnostic_title(indicator, treatment, model),
            ylabel="p-verdi",
            ylim=(0, 1),
        )
        ax.text(
            0,
            min(float(row.pretrend_p_value) + 0.05, 0.95),
            f"F={row.pretrend_f_stat:.2f}",
            ha="center",
        )
    fig.suptitle("Tester av parallell trend")
    return _save(fig, output_dir / "figur6.svg")


def figure7(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write leave-one-out estimates for all eight main-result models."""
    required = {
        "indicator",
        "treatment_type",
        "model",
        "dropped_region",
        "coefficient",
        "ci_lower",
        "ci_upper",
        "full_coefficient",
    }
    if missing := required - set(data):
        raise ValueError(f"Figur 7 mangler kolonner: {sorted(missing)}")
    fig, axes = _diagnostic_axes()
    for ax, (indicator, treatment, model) in zip(
        axes.flat,
        (
            (indicator, treatment, model)
            for indicator in ("atid3", "jobb3")
            for treatment in ("discrete", "continuous")
            for model in ("baseline", "preferred")
        ),
        strict=True,
    ):
        rows = data.loc[
            (data.indicator == indicator)
            & (data.treatment_type == treatment)
            & (data.model == model)
        ].sort_values("coefficient")
        if rows.empty:
            raise ValueError(f"Figur 7 mangler {indicator}/{treatment}/{model}.")
        positions = range(len(rows))
        ax.errorbar(
            rows.coefficient,
            positions,
            xerr=[rows.coefficient - rows.ci_lower, rows.ci_upper - rows.coefficient],
            fmt="o",
            color="#254B6D",
            ecolor="#66CBEC",
            capsize=2,
        )
        ax.axvline(rows.full_coefficient.iloc[0], color="#B90000", linestyle=":")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            title=_diagnostic_title(indicator, treatment, model),
            xlabel="Estimat",
            yticks=list(positions),
            yticklabels=rows.dropped_region,
        )
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Robusthetstest: utelat ett fylke")
    return _save(fig, output_dir / "figur7.svg")


def figure9(
    data: pd.DataFrame, output_dir: Path, treatment_label: str = "Alle tiltak"
) -> Path:
    """Write raw and, when available, seasonally adjusted treatment trends."""
    required = {"aarmnd", "tiltak", "model"}
    if missing := required - set(data):
        raise ValueError(f"Figur 9 mangler kolonner: {sorted(missing)}")
    return _trend(
        data,
        "tiltak",
        "model",
        f"{treatment_label}: rådata og sesongkorrigert",
        output_dir / "figur9.svg",
    )


def figure10(
    data: pd.DataFrame, output_dir: Path, treatment_label: str = "Alle tiltak"
) -> Path:
    """Write treatment trends by county for the appendix."""
    return _trend(
        data,
        "tiltak",
        "region",
        f"{treatment_label} per fylke",
        output_dir / "figur10.svg",
    )


def figure11(
    data: pd.DataFrame, output_dir: Path, treatment_label: str = "Alle tiltak"
) -> Path:
    """Write selected treatment subgroup estimates."""
    return _faceted_estimates(
        data, f"Estimater for {treatment_label.lower()}", "figur11.svg", output_dir
    )


def figure12(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write expected Atid3 and Jobb3 trends for high and low reduction groups."""
    required = {"aarmnd", "indicator", "gruppe", "outcome"}
    if missing := required - set(data):
        raise ValueError(f"Figur 12 mangler kolonner: {sorted(missing)}")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True)
    frame = data.loc[data.outcome == "forventet"].copy()
    frame["aarmnd"] = pd.to_datetime(frame["aarmnd"])
    for ax, indicator in zip(axes, ("atid3", "jobb3"), strict=True):
        for colour, (group, rows) in zip(
            PALETTE,
            frame.loc[frame.indicator == indicator].groupby("gruppe"),
            strict=False,
        ):
            series = rows.groupby("aarmnd")["value"].mean()
            ax.plot(series.index, series, label=group, color=colour)
        ax.set(title=indicator.title(), xlabel="Måned", ylabel="Forventet verdi")
        ax.legend(frameon=False)
    return _save(fig, output_dir / "figur12.svg")


def figure13(
    data: pd.DataFrame,
    output_dir: Path,
    treatment_label: str = "Midlertidig lønnstilskudd",
) -> Path:
    """Write expected-outcome extension estimates for one treatment."""
    return _faceted_estimates(
        data,
        f"Estimater for forventede verdier: {treatment_label}",
        "figur13.svg",
        output_dir,
    )


def generate_figures(output_dir: Path, **datasets: pd.DataFrame) -> list[Path]:
    """Generate supplied summary figures by TODO number.

    Keyword names must be ``figure1`` through ``figure13``; unavailable
    analyses are deliberately omitted by the caller rather than fabricated.
    """
    generators = {
        "figure1": figure1,
        "figure2": figure2,
        "figure3": figure3,
        "figure4": figure4,
        "figure5": figure5,
        "figure6": figure6,
        "figure7": figure7,
        "figure8": figure8,
        "figure9": figure9,
        "figure10": figure10,
        "figure11": figure11,
        "figure12": figure12,
        "figure13": figure13,
    }
    unknown = set(datasets) - set(generators)
    if unknown:
        raise ValueError(f"Ukjente figurer: {', '.join(sorted(unknown))}.")
    return [generators[name](data, output_dir) for name, data in datasets.items()]
