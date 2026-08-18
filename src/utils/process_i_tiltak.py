"""Process the i_tiltak extract into indicator CSVs for people in tiltak.

The source contains one row for each organisation, month, and binary
``NEDBRYTNING_VERDI``. Value ``1`` represents people in tiltak; value ``0`` is
the complementary population. The resulting files follow the same layouts as
the existing indicator groups:

  data/input/indikatorer/landet/personer_i_tiltak/<utfall>.csv
  data/input/indikatorer/nedbrytning/personer_i_tiltak/<utfall>.csv
  data/input/indikatorer/enhet/nedbrytning/personer_i_tiltak/<utfall>.csv
  data/input/personer/<organisation level>/personer_i_tiltak/antall_personer.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = _ROOT / "data" / "støttedata" / "i_tiltak.csv"
_GROUP = "personer_i_tiltak"
_TILTAK_VALUE = 1

_REGIONS = [
    "Nav Agder",
    "Nav Innlandet",
    "Nav Møre og Romsdal",
    "Nav Nordland",
    "Nav Oslo",
    "Nav Rogaland",
    "Nav Troms og Finnmark",
    "Nav Trøndelag",
    "Nav Vest-Viken",
    "Nav Vestfold og Telemark",
    "Nav Vestland",
    "Nav Øst-Viken",
]

_MEASURES = {
    "atid3": {
        "indikator": "INDI_ATID3_AVG",
        "forventet": "YHAT_ATID3_AVG",
        "faktisk": "ATID3_AVG",
    },
    "jobb3": {
        "indikator": "INDI_JOBB3_AVG",
        "forventet": "YHAT_JOBB3_AVG",
        "faktisk": "JOBB3_AVG",
    },
}

_REQUIRED_COLUMNS = {
    "AARMND_DATO",
    "NEDBRYTNING_VERDI",
    "ORG_NIVAA",
    "ORG_STED",
    "ANTALL_PERSONER",
    *(column for measure in _MEASURES.values() for column in measure.values()),
}


def _load_source(path: Path = _SOURCE_PATH) -> pd.DataFrame:
    """Load and validate rows for people in tiltak."""
    df = pd.read_csv(path, sep=";", decimal=",")
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Source is missing required columns: {', '.join(sorted(missing))}."
        )

    df = df[df["NEDBRYTNING_VERDI"] == _TILTAK_VALUE].copy()
    if df.empty:
        raise ValueError(
            f"Source contains no rows with NEDBRYTNING_VERDI={_TILTAK_VALUE}."
        )

    df["aarmnd"] = pd.to_datetime(
        df["AARMND_DATO"], format="%d.%m.%Y %H:%M:%S"
    ).dt.strftime("%Y%m")
    for source_column in _MEASURES["jobb3"].values():
        df[source_column] *= 100
    return df


def _write_landet(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write one country-level value per month."""
    saved: list[Path] = []
    for utfall, columns in _MEASURES.items():
        for kind, source_column in columns.items():
            output = (
                df[["aarmnd", source_column]]
                .rename(columns={source_column: kind})
                .sort_values("aarmnd")
            )
            path = output_dir / f"{'' if kind == 'indikator' else f'{kind}_'}{utfall}.csv"
            output.to_csv(path, index=False)
            saved.append(path)
    return saved


def _write_nedbrytning(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write region-level values in wide format."""
    saved: list[Path] = []
    for utfall, columns in _MEASURES.items():
        for kind, source_column in columns.items():
            output = (
                df.pivot(index="aarmnd", columns="ORG_STED", values=source_column)
                .sort_index()
                .reset_index()
            )
            output.columns.name = None
            output = output[["aarmnd", *_REGIONS]]
            path = output_dir / (
                f"{'' if kind == 'indikator' else f'{kind}_'}{utfall}.csv"
            )
            output.to_csv(path, index=False)
            saved.append(path)
    return saved


def _write_enhet(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write unit-level values in long format."""
    saved: list[Path] = []
    for utfall, columns in _MEASURES.items():
        for kind, source_column in columns.items():
            output = (
                df[["aarmnd", "ORG_STED", source_column]]
                .rename(columns={"ORG_STED": "enhet", source_column: kind})
                .sort_values(["aarmnd", "enhet"])
            )
            path = output_dir / f"{'' if kind == 'indikator' else f'{kind}_'}{utfall}.csv"
            output.to_csv(path, index=False)
            saved.append(path)
    return saved


def _write_personer_landet(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write country-level participant counts."""
    output = df[["aarmnd", "ANTALL_PERSONER"]].rename(
        columns={"ANTALL_PERSONER": "antall_personer"}
    )
    path = output_dir / "antall_personer.csv"
    output.sort_values("aarmnd").to_csv(path, index=False)
    return path


def _write_personer_nedbrytning(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write regional participant counts in wide format."""
    output = (
        df.pivot(index="aarmnd", columns="ORG_STED", values="ANTALL_PERSONER")
        .sort_index()
        .reset_index()
    )
    output.columns.name = None
    output = output[["aarmnd", *_REGIONS]]
    path = output_dir / "antall_personer.csv"
    output.to_csv(path, index=False)
    return path


def _write_personer_enhet(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write unit-level participant counts in long format."""
    output = (
        df[["aarmnd", "ORG_STED", "ANTALL_PERSONER"]]
        .rename(
            columns={
                "ORG_STED": "enhet",
                "ANTALL_PERSONER": "antall_personer",
            }
        )
        .sort_values(["aarmnd", "enhet"])
    )
    path = output_dir / "antall_personer.csv"
    output.to_csv(path, index=False)
    return path


def process_i_tiltak(
    source_path: Path = _SOURCE_PATH, output_root: Path = _ROOT
) -> list[Path]:
    """Process *source_path* and return the generated indicator and count paths."""
    df = _load_source(source_path)
    indikator_dir = output_root / "data" / "input" / "indikatorer"
    personer_dir = output_root / "data" / "input" / "personer"
    level_outputs = {
        1: (
            indikator_dir / "landet" / _GROUP,
            personer_dir / "landet" / _GROUP,
            _write_landet,
            _write_personer_landet,
        ),
        2: (
            indikator_dir / "nedbrytning" / _GROUP,
            personer_dir / "nedbrytning" / _GROUP,
            _write_nedbrytning,
            _write_personer_nedbrytning,
        ),
        3: (
            indikator_dir / "enhet" / "nedbrytning" / _GROUP,
            personer_dir / "enhet" / "nedbrytning" / _GROUP,
            _write_enhet,
            _write_personer_enhet,
        ),
    }

    saved: list[Path] = []
    for level, (indikator_dir, personer_dir, indikator_writer, personer_writer) in (
        level_outputs.items()
    ):
        level_df = df[df["ORG_NIVAA"] == level]
        if level == 2:
            level_df = level_df[level_df["ORG_STED"].isin(_REGIONS)]
        if level_df.empty:
            raise ValueError(f"Source contains no ORG_NIVAA={level} rows in tiltak.")
        if level_df.duplicated(["aarmnd", "ORG_STED"]).any():
            raise ValueError(
                f"Source contains duplicate month and organisation rows at ORG_NIVAA={level}."
            )
        if level == 2 and set(level_df["ORG_STED"]) != set(_REGIONS):
            raise ValueError("Source does not contain all expected Nav regions.")
        indikator_dir.mkdir(parents=True, exist_ok=True)
        personer_dir.mkdir(parents=True, exist_ok=True)
        saved.extend(indikator_writer(level_df, indikator_dir))
        saved.append(personer_writer(level_df, personer_dir))

    return saved


def main() -> None:
    """Process the default i_tiltak source file."""
    saved = process_i_tiltak()
    print(f"Saved {len(saved)} indicator and participant-count files:")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()