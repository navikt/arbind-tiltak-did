"""Catalog-driven entry point for the available summary artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from figures import (
    figure1,
    figure2,
    figure3,
    figure4,
    figure5,
    figure6,
    figure7,
    figure8,
    figure9,
    figure10,
    figure11,
    figure12,
    figure13,
)
from tables import appendix_table, write_workbook

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _result_path(outputs_dir: Path, run_id: str) -> Path:
    """Return the persisted regression summary for an explicitly selected run."""
    path = outputs_dir / "did" / run_id / "tables" / "regression_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Mangler analyseoutput: {path}")
    return path


def _results(outputs_dir: Path, run_ids: dict[str, str]) -> pd.DataFrame:
    """Read preferred results and attach publication labels."""
    frames = []
    for label, run_id in run_ids.items():
        frame = pd.read_csv(_result_path(outputs_dir, run_id))
        frame = frame.loc[frame.model == "preferred"].copy()
        frame["group"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _diagnostics(
    outputs_dir: Path, run_ids: tuple[str, ...], filename: str
) -> pd.DataFrame:
    """Load a diagnostic CSV from each main-result run."""
    paths = []
    for run_id in run_ids:
        candidates = (
            run_id,
            run_id.removesuffix("--indikator"),
        )
        for candidate in candidates:
            path = outputs_dir / "did" / candidate / "tables" / filename
            if path.is_file():
                paths.append(path)
                break
        else:
            raise FileNotFoundError(
                f"Mangler diagnostikkfilen {filename} for {run_id}."
            )
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def _panel_path(processed_dir: Path, run_id: str, indicator: str) -> Path:
    """Return the preferred seasonally adjusted panel for a run."""
    parts = run_id.split("--")
    variant = "--".join(parts[4:]).removesuffix("--indikator")
    path = (
        processed_dir
        / "did"
        / parts[1]
        / parts[2]
        / "enheter"
        / variant
        / f"panel_{indicator}_flattened.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Mangler behandlet panel: {path}")
    return path


def _grouped_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Add high/low regional reduction groups used by the trend figures."""
    result = panel.copy()
    post = result.loc[result.post_treatment].groupby("region")["tiltaksnedgang"].mean()
    high = set(post.loc[post >= post.median()].index)
    result["gruppe"] = result.region.map(
        lambda region: "Høy nedgang" if region in high else "Lav nedgang"
    )
    return result


def _treatment_data(
    project_root: Path,
    raw_filename: str,
    panel: pd.DataFrame,
    adjusted_filename: str | None = None,
) -> pd.DataFrame:
    """Return raw and, where available, seasonally adjusted treatment series."""
    raw = pd.read_csv(project_root / "data" / "input" / "tiltak" / raw_filename)
    raw = raw.melt(
        id_vars="aarmnd",
        value_vars=[
            column for column in raw.columns if column not in {"aarmnd", "TOTAL"}
        ],
        var_name="region",
        value_name="tiltak",
    )
    raw["region"] = raw["region"].str.strip("'")
    raw["aarmnd"] = pd.to_datetime(raw["aarmnd"])
    dates = pd.to_datetime(panel["aarmnd"])
    start, end = dates.min(), dates.max()
    frames = [raw.loc[raw.aarmnd.between(start, end)].assign(model="Rådata")]
    if adjusted_filename:
        adjusted = pd.read_csv(
            project_root / "data" / "processed" / "tiltak-sa" / adjusted_filename
        )
        adjusted["aarmnd"] = pd.to_datetime(adjusted["aarmnd"], format="%Y%m")
        frames.append(
            adjusted.loc[adjusted.aarmnd.between(start, end)].assign(
                model="Sesongkorrigert (pre-periode)"
            )
        )
    return pd.concat(frames, ignore_index=True)


def _full_treatment_series(project_root: Path, filename: str) -> pd.DataFrame:
    """Read total treatment participation from January 2021 onward."""
    data = pd.read_csv(project_root / "data" / "input" / "tiltak" / filename)
    if "TOTAL" not in data:
        raise ValueError(f"Tiltekksfilen mangler TOTAL-kolonnen: {filename}")
    result = data.loc[:, ["aarmnd", "TOTAL"]].rename(columns={"TOTAL": "tiltak"})
    result["aarmnd"] = pd.to_datetime(result["aarmnd"])
    return result.loc[result.aarmnd >= "2021-01-01"].copy()


def _generate_treatment_summary(
    outputs_dir: Path,
    processed_dir: Path,
    summary_dir: Path,
    base: str,
    treatment_label: str,
    raw_filename: str,
    appendix: pd.DataFrame,
    adjusted_filename: str | None = None,
) -> list[Path]:
    """Generate every summary table and figure for one treatment definition."""
    indicator_results = pd.concat(
        [
            pd.read_csv(_result_path(outputs_dir, f"{base}--diskret")),
            pd.read_csv(_result_path(outputs_dir, f"{base}--kontinuerlig")),
        ],
        ignore_index=True,
    )
    actual_results = pd.concat(
        [
            pd.read_csv(_result_path(outputs_dir, f"{base}--faktisk-diskret")),
            pd.read_csv(_result_path(outputs_dir, f"{base}--faktisk-kontinuerlig")),
        ],
        ignore_index=True,
    )
    continuous_panel = pd.read_csv(
        _panel_path(processed_dir, f"{base}--kontinuerlig", "atid3")
    )
    paths = [
        write_workbook(
            indicator_results,
            actual_results,
            appendix,
            summary_dir / "summary.xlsx",
            treatment_label=treatment_label,
        ),
        figure1(
            _full_treatment_series(PROJECT_ROOT, "alle-tiltak.csv"),
            _full_treatment_series(PROJECT_ROOT, "midl.-lønnstilskudd.csv"),
            summary_dir,
        ),
        figure2(continuous_panel, summary_dir, treatment_label),
    ]
    for indicator, function in (("atid3", figure3), ("jobb3", figure4)):
        panel = pd.read_csv(
            _panel_path(processed_dir, f"{base}--kontinuerlig", indicator)
        )
        paths.append(function(_grouped_panel(panel), summary_dir))

    main_runs = (f"{base}--diskret", f"{base}--kontinuerlig")
    paths.extend(
        [
            figure5(
                _diagnostics(outputs_dir, main_runs, "event_study_results.csv"),
                summary_dir,
            ),
            figure6(
                _diagnostics(outputs_dir, main_runs, "pretrend_tests.csv"),
                summary_dir,
            ),
            figure7(
                _diagnostics(outputs_dir, main_runs, "leave_one_out_results.csv"),
                summary_dir,
            ),
        ]
    )

    groups = {
        "alle": "Alle",
        "situasjonsbestemt": "Gode muligheter",
        "spesielt-tilpasset": "Trenger veiledning",
        "standard": "Nedsatt arbeidsevne",
        "veiledning-kombinert": "Veiledning kombinert",
    }
    treatment_slug = base.split("--")[1]
    subgroup_runs = {
        "Alle": f"{base}--kontinuerlig",
        **{
            label: f"did--{treatment_slug}--{slug}--enheter--kontinuerlig"
            for slug, label in groups.items()
            if slug != "alle"
        },
    }
    if treatment_slug == "alle-tiltak":
        subgroup_runs.update(
            {
                "Alle – diskret": f"{base}--diskret",
                "Personer i tiltak": (
                    "did--alle-tiltak--personer_i_tiltak--enheter--kontinuerlig"
                ),
            }
        )
    paths.append(
        figure8(
            _results(outputs_dir, subgroup_runs),
            summary_dir,
        )
    )
    paths.extend(
        [
            figure9(
                _treatment_data(
                    PROJECT_ROOT,
                    raw_filename,
                    continuous_panel,
                    adjusted_filename,
                ),
                summary_dir,
                treatment_label,
            ),
            figure10(continuous_panel, summary_dir, treatment_label),
            figure11(
                _results(
                    outputs_dir,
                    {
                        "Alle – diskret": f"{base}--diskret",
                        "Alle – kontinuerlig": f"{base}--kontinuerlig",
                        "Trenger veiledning – kontinuerlig": (
                            f"did--{treatment_slug}--spesielt-tilpasset"
                            "--enheter--kontinuerlig"
                        ),
                        "Veiledning kombinert – kontinuerlig": (
                            f"did--{treatment_slug}--veiledning-kombinert"
                            "--enheter--kontinuerlig"
                        ),
                    },
                ),
                summary_dir,
                treatment_label,
            ),
        ]
    )

    expected = f"{base}--forventet-kontinuerlig"
    expected_panels = []
    for indicator in ("atid3", "jobb3"):
        panel = _grouped_panel(
            pd.read_csv(_panel_path(processed_dir, expected, indicator))
        )
        panel["indicator"] = indicator
        panel["outcome"] = "forventet"
        panel["value"] = panel["indikator"]
        expected_panels.append(panel)
    figure13_runs = {
        "Alle – diskret": f"{base}--forventet-diskret",
        "Alle – kontinuerlig": expected,
    }
    if treatment_slug == "alle-tiltak":
        figure13_runs.update(
            {
                "Personer i tiltak – diskret": (
                    "did--alle-tiltak--personer_i_tiltak--enheter--forventet-diskret"
                ),
                "Personer i tiltak – kontinuerlig": (
                    "did--alle-tiltak--personer_i_tiltak--enheter"
                    "--forventet-kontinuerlig"
                ),
            }
        )
    paths.extend(
        [
            figure12(pd.concat(expected_panels, ignore_index=True), summary_dir),
            figure13(
                _results(
                    outputs_dir,
                    figure13_runs,
                ),
                summary_dir,
                treatment_label,
            ),
        ]
    )
    return paths


def generate_summary(
    outputs_dir: Path = PROJECT_ROOT / "outputs",
    processed_dir: Path = PROJECT_ROOT / "data" / "processed",
) -> list[Path]:
    """Generate parallel summary folders for both treatment definitions."""
    wage_base = "did--midl-lonnstilskudd--alle--enheter"
    all_base = "did--alle-tiltak--personer_i_tiltak--enheter"
    appendix = appendix_table(
        pd.read_csv(_panel_path(processed_dir, f"{wage_base}--kontinuerlig", "atid3")),
        pd.read_csv(
            _panel_path(
                processed_dir, "did--alle-tiltak--alle--enheter--kontinuerlig", "atid3"
            )
        ),
        pd.read_csv(_panel_path(processed_dir, f"{wage_base}--diskret", "atid3")),
        pd.read_csv(
            _panel_path(
                processed_dir, "did--alle-tiltak--alle--enheter--diskret", "atid3"
            )
        ),
    )
    summary_dir = outputs_dir / "summary"
    return [
        *_generate_treatment_summary(
            outputs_dir,
            processed_dir,
            summary_dir / "midlertidig-lonnstilskudd",
            wage_base,
            "Midlertidig lønnstilskudd",
            "midl.-lønnstilskudd.csv",
            appendix,
        ),
        *_generate_treatment_summary(
            outputs_dir,
            processed_dir,
            summary_dir / "alle-tiltak",
            all_base,
            "Alle tiltak",
            "alle-tiltak.csv",
            appendix,
            adjusted_filename="alle-tiltak-sa-pre.csv",
        ),
    ]


def main() -> None:
    """Generate the summary artifacts from command-line paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    args = parser.parse_args()
    for path in generate_summary(args.outputs_dir, args.processed_dir):
        print(path)


if __name__ == "__main__":
    main()
