"""Presentation tables generated from persisted analysis summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_INDICATORS = ("atid3", "jobb3")
_TREATMENT_TYPES = ("discrete", "continuous")
_MODELS = ("baseline", "preferred")
_TREATMENT_TYPE_LABELS = {"discrete": "Diskret", "continuous": "Kontinuerlig"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TABLE1_CONFIG_PATHS = (
    "did--midl-lonnstilskudd--alle--regioner--diskret--indikator",
    "did--midl-lonnstilskudd--alle--regioner--kontinuerlig--indikator",
)
_REQUIRED_COLUMNS = {
    "analysis_design",
    "indicator",
    "treatment_type",
    "model",
    "coefficient",
    "std_error",
    "p_value_bootstrap",
    "seasonally_adjusted",
    "entity_fixed_effects",
    "time_fixed_effects",
    "r_squared_adjusted",
    "n_obs",
    "n_clusters",
    "period_start",
    "period_end",
}
_ROW_LABELS = (
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

def _table1_data_paths(outputs_dir: Path) -> list[Path]:
    """Return the persisted temporary wage-subsidy results used for Table 1."""
    return [
        outputs_dir / "did" / config_path / "tables" / "regression_results.csv"
        for config_path in _TABLE1_CONFIG_PATHS
    ]


def _significance_stars(p_value: float) -> str:
    """Return conventional significance stars for a bootstrap p-value."""
    if p_value <= 0.01:
        return "***"
    if p_value <= 0.05:
        return "**"
    if p_value <= 0.1:
        return "*"
    return ""


def _yes_no(value: object) -> str:
    """Render a boolean-like summary field as Yes or No."""
    return "Ja" if str(value).lower() in {"true", "1", "yes"} else "Nei"


def _format_result(row: pd.Series) -> list[str]:
    """Format one persisted regression-result row for Table 1."""
    return [
        f"{row['coefficient']:.3f}{_significance_stars(row['p_value_bootstrap'])}",
        f"({row['std_error']:.3f})",
        _yes_no(row["seasonally_adjusted"]),
        _yes_no(row["entity_fixed_effects"]),
        _yes_no(row["time_fixed_effects"]),
        f"{row['r_squared_adjusted']:.3f}",
        f"{int(row['n_obs']):,}",
        f"{int(row['n_clusters']):,}",
        f"{row['period_start']} til {row['period_end']}",
    ]


def table1(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write Table 1 from selected DiD regression summary rows.

    ``data`` must contain baseline and preferred rows for each combination of
    Atid3/Jobb3 and discrete/continuous treatment. The output has eight
    columns: one non-seasonally adjusted and one seasonally adjusted model for
    every indicator/treatment combination. It is normally assembled by reading
    the relevant self-describing ``regression_results.csv`` files. The reported
    adjusted R-squared is Statsmodels' OLS adjusted R-squared, including
    fixed-effect dummies.
    """
    missing_columns = _REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Table 1 input is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    selected = data.loc[
        (data["analysis_design"] == "did")
        & data["indicator"].isin(_INDICATORS)
        & data["treatment_type"].isin(_TREATMENT_TYPES)
        & data["model"].isin(_MODELS)
    ].copy()

    expected_keys = pd.MultiIndex.from_product(
        [_INDICATORS, _TREATMENT_TYPES, _MODELS],
        names=["indicator", "treatment_type", "model"],
    )
    selected = selected.set_index(["indicator", "treatment_type", "model"])
    duplicate_keys = selected.index[selected.index.duplicated()].unique()
    if len(duplicate_keys):
        raise ValueError(
            "Table 1 input has duplicate model rows for: "
            + ", ".join(
                f"{indicator}/{treatment}/{model}"
                for indicator, treatment, model in duplicate_keys
            )
            + ". Select one analysis run per table column."
        )
    missing_keys = expected_keys.difference(selected.index)
    if len(missing_keys):
        raise ValueError(
            "Table 1 input is missing model rows for: "
            + ", ".join(
                f"{indicator}/{treatment}/{model}"
                for indicator, treatment, model in missing_keys
            )
            + "."
        )

    columns = pd.MultiIndex.from_tuples(
        [
            ("Atid3", "Diskret", "Nei"),
            ("Atid3", "Diskret", "Ja"),
            ("Atid3", "Kontinuerlig", "Nei"),
            ("Atid3", "Kontinuerlig", "Ja"),
            ("Jobb3", "Diskret", "Nei"),
            ("Jobb3", "Diskret", "Ja"),
            ("Jobb3", "Kontinuerlig", "Nei"),
            ("Jobb3", "Kontinuerlig", "Ja"),
        ],
        names=["Arbeidsindikator", "Behandlingstype", "Sesongjustert"],
    )
    table = pd.DataFrame(index=_ROW_LABELS, columns=columns)
    for indicator, treatment_type, model in expected_keys:
        table.loc[
            :,
            (
                indicator.title(),
                _TREATMENT_TYPE_LABELS[treatment_type],
                "Ja" if model == "preferred" else "Nei",
            ),
        ] = _format_result(selected.loc[(indicator, treatment_type, model)])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "table1.csv"
    table.to_csv(output_path)
    return output_path


def generate_default_table1(outputs_dir: Path = _PROJECT_ROOT / "outputs") -> Path:
    """Generate Table 1 from the standard midlertidig lønnstilskudd analyses."""
    data_paths = _table1_data_paths(outputs_dir)
    missing_paths = [path for path in data_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing Table 1 regression results:\n"
            + "\n".join(str(path) for path in missing_paths)
        )
    data = pd.concat((pd.read_csv(path) for path in data_paths), ignore_index=True)
    return table1(data, outputs_dir / "summary")


if __name__ == "__main__":
    generate_default_table1()