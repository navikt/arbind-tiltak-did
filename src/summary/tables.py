"""Publication tables generated from persisted DiD regression summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Font

_INDICATORS = ("atid3", "jobb3")
_TREATMENTS = ("discrete", "continuous")
_MODELS = ("baseline", "preferred")
_LABELS = {"discrete": "Diskret", "continuous": "Kontinuerlig"}
_REQUIRED = {
    "analysis_design",
    "indicator",
    "treatment_type",
    "model",
    "coefficient",
    "std_error",
    "seasonally_adjusted",
    "entity_fixed_effects",
    "time_fixed_effects",
    "r_squared_adjusted",
    "n_obs",
    "n_clusters",
    "period_start",
    "period_end",
}
_ROWS = (
    "Koeffisient",
    "Standardfeil",
    "Sesongjustert",
    "Region faste effekter",
    "Tidspunkt faste effekter",
    "Justert R-kvadrat",
    "Antall observasjoner",
    "Antall klynger",
    "Periode",
)


def _stars(value: float) -> str:
    return (
        "***"
        if value <= 0.01
        else "**"
        if value <= 0.05
        else "*"
        if value <= 0.1
        else ""
    )


def _yes(value: object) -> str:
    return "Ja" if str(value).lower() in {"true", "1", "yes"} else "Nei"


def _result(row: pd.Series) -> list[str]:
    p_value = row.get("p_value_bootstrap")
    if pd.isna(p_value):
        p_value = row.get("p_value_asymp")
    if pd.isna(p_value):
        raise ValueError("Tabellinput mangler både bootstrap- og asymptotisk p-verdi.")
    return [
        f"{row['coefficient']:.3f}{_stars(p_value)}",
        f"({row['std_error']:.3f})",
        _yes(row["seasonally_adjusted"]),
        _yes(row["entity_fixed_effects"]),
        _yes(row["time_fixed_effects"]),
        f"{row['r_squared_adjusted']:.3f}",
        f"{int(row['n_obs']):,}",
        f"{int(row['n_clusters']):,}",
        f"{row['period_start']} til {row['period_end']}",
    ]


def result_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return the requested eight-model presentation table from DiD results."""
    missing = _REQUIRED - set(data)
    if missing:
        raise ValueError(f"Tabellinput mangler kolonner: {', '.join(sorted(missing))}.")
    if {"p_value_bootstrap", "p_value_asymp"}.isdisjoint(data.columns):
        raise ValueError("Tabellinput mangler både p_value_bootstrap og p_value_asymp.")
    keys = ["indicator", "treatment_type", "model"]
    selected = data.loc[
        (data.analysis_design == "did")
        & data.indicator.isin(_INDICATORS)
        & data.treatment_type.isin(_TREATMENTS)
        & data.model.isin(_MODELS)
    ].set_index(keys)
    expected = pd.MultiIndex.from_product(
        [_INDICATORS, _TREATMENTS, _MODELS], names=keys
    )
    if selected.index.has_duplicates:
        raise ValueError("Table input has duplicate model results.")
    missing_keys = expected.difference(selected.index)
    if len(missing_keys):
        raise ValueError(f"Tabellinput mangler modellresultater: {list(missing_keys)}.")
    columns = pd.MultiIndex.from_tuples(
        [
            (
                indicator.title(),
                _LABELS[treatment],
                "Ja" if model == "preferred" else "Nei",
            )
            for indicator in _INDICATORS
            for treatment in _TREATMENTS
            for model in _MODELS
        ],
        names=["Arbeidsindikator", "Behandlingstype", "Sesongjustert"],
    )
    table = pd.DataFrame(index=_ROWS, columns=columns)
    for key in expected:
        table.loc[
            :,
            (key[0].title(), _LABELS[key[1]], "Ja" if key[2] == "preferred" else "Nei"),
        ] = _result(selected.loc[key])
    return table


def table1(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write Table 1 as CSV for backwards-compatible programmatic use."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "table1.csv"
    result_table(data).to_csv(path)
    return path


def write_workbook(
    indicator_data: pd.DataFrame,
    actual_data: pd.DataFrame,
    appendix: pd.DataFrame,
    monthly_treatments: pd.DataFrame,
    output_path: Path,
    treatment_label: str = "Midlertidig lønnstilskudd",
) -> Path:
    """Write titled Tables 1–4 to separate sheets in one Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sheets = (
            (
                "Tabell 1",
                f"Tabell 1: Indikatorutfall - behandling: {treatment_label}",
                result_table(indicator_data),
                True,
            ),
            (
                "Tabell 2",
                f"Tabell 2: Faktiske utfall - behandling: {treatment_label}",
                result_table(actual_data),
                True,
            ),
            (
                "Tabell 3",
                "Tabell 3: Behandlingsintensitet og diskret behandling per region",
                appendix,
                False,
            ),
            (
                "Tabell 4",
                "Tabell 4: Tiltak og behandlingsintensitet per måned og region",
                monthly_treatments,
                False,
            ),
        )
        for sheet_name, title, data, include_index in sheets:
            data.to_excel(
                writer,
                sheet_name=sheet_name,
                index=include_index,
                startrow=2,
            )
            worksheet = writer.sheets[sheet_name]
            worksheet.cell(row=1, column=1, value=title).font = Font(bold=True)
            worksheet.merge_cells(
                start_row=1,
                start_column=1,
                end_row=1,
                end_column=max(1, len(data.columns) + int(include_index)),
            )
    return output_path


def appendix_table(
    wage_subsidy_panel: pd.DataFrame,
    all_measures_panel: pd.DataFrame,
    wage_subsidy_discrete_panel: pd.DataFrame,
    all_measures_discrete_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate Appendix Table 3 from continuous and discrete treatment panels."""
    required = {"region", "post_treatment", "tiltaksnedgang"}
    for name, panel in (
        ("midlertidig lønnstilskudd", wage_subsidy_panel),
        ("alle tiltak", all_measures_panel),
    ):
        if missing := required - set(panel):
            raise ValueError(f"{name} mangler kolonner: {sorted(missing)}")
    for name, panel in (
        ("midlertidig lønnstilskudd (diskret)", wage_subsidy_discrete_panel),
        ("alle tiltak (diskret)", all_measures_discrete_panel),
    ):
        if missing := {"region", "treated"} - set(panel):
            raise ValueError(f"{name} mangler kolonner: {sorted(missing)}")

    def reduction(panel: pd.DataFrame) -> pd.Series:
        return (
            panel.loc[panel.post_treatment].groupby("region")["tiltaksnedgang"].mean()
        )

    def discrete_treatment(panel: pd.DataFrame) -> pd.Series:
        assignments = panel.groupby("region")["treated"].nunique()
        if (assignments != 1).any():
            raise ValueError("Diskret behandlingsvariabel varierer innen en region.")
        return panel.groupby("region")["treated"].first().astype(int)

    wage = reduction(wage_subsidy_panel)
    all_measures = reduction(all_measures_panel)
    wage_discrete = discrete_treatment(wage_subsidy_discrete_panel)
    all_measures_discrete = discrete_treatment(all_measures_discrete_panel)
    regions = (
        wage.index.union(all_measures.index)
        .union(wage_discrete.index)
        .union(all_measures_discrete.index)
    )
    return pd.DataFrame(
        {
            "Region": regions,
            "Gjennomsnittlig nedgang: midlertidig lønnstilskudd": wage.reindex(
                regions
            ).values,
            "Diskret behandlingsvariabel: midlertidig lønnstilskudd": wage_discrete.reindex(
                regions
            ).values,
            "Gjennomsnittlig nedgang: alle tiltak": all_measures.reindex(
                regions
            ).values,
            "Diskret behandlingsvariabel: alle tiltak": all_measures_discrete.reindex(
                regions
            ).values,
        }
    ).sort_values("Region", ignore_index=True)


def monthly_treatment_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Return post-treatment intensity by month, with one column per region."""
    required = {"aarmnd", "region", "post_treatment", "tiltaksnedgang"}
    if missing := required - set(panel):
        raise ValueError(f"Tabell 4 mangler kolonner: {sorted(missing)}")
    values = panel.loc[
        panel.post_treatment, ["aarmnd", "region", "tiltaksnedgang"]
    ].copy()
    variation = values.groupby(["aarmnd", "region"])["tiltaksnedgang"].nunique(
        dropna=False
    )
    if (variation > 1).any():
        raise ValueError("Tiltaksnedgang varierer innen måned og region.")
    return (
        values.drop_duplicates(["aarmnd", "region"])
        .pivot(index="aarmnd", columns="region", values="tiltaksnedgang")
        .rename_axis(index="Måned", columns=None)
        .reset_index()
        .sort_values("Måned", ignore_index=True)
    )


def generate_default_table1(outputs_dir: Path = Path("outputs")) -> Path:
    """Generate Table 1 from standard temporary wage-subsidy output directories."""
    paths: list[Path] = []
    for level in ("enheter", "regioner"):
        paths = list(
            (outputs_dir / "did").glob(
                "did--midl-lonnstilskudd--alle"
                f"--{level}--*/tables/regression_results.csv"
            )
        )
        if paths:
            break
    if not paths:
        raise FileNotFoundError("Fant ingen resultater for Tabell 1.")
    weighted_paths = [path for path in paths if path.parts[-3].endswith("--vektet")]
    paths = weighted_paths or paths
    return table1(
        pd.concat(map(pd.read_csv, paths), ignore_index=True), outputs_dir / "summary"
    )
