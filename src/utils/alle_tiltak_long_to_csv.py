"""Convert the long all-tiltak extract to the analysis pipeline's wide CSV.

data/input/tiltak/alle-tiltak-long.csv -> data/input/tiltak/alle-tiltak.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = _ROOT / "data" / "input" / "tiltak" / "alle-tiltak-long.csv"
_OUTPUT_PATH = _ROOT / "data" / "input" / "tiltak" / "alle-tiltak.csv"

_REGIONS = [
    "Nav Agder",
    "Nav Innlandet",
    "Nav Møre og Romsdal",
    "Nav Nordland",
    "Nav Oslo",
    "Nav Rogaland",
    "Nav Troms og Finnmark",
    "Nav Trøndelag",
    "Nav Vestfold og Telemark",
    "Nav Vestland",
    "Nav Vest-Viken",
    "Nav Øst-Viken",
]
_REQUIRED_COLUMNS = {"AARMND", "REGION", "COUNT(*)"}


def convert_alle_tiltak(
    source_path: Path = _SOURCE_PATH, output_path: Path = _OUTPUT_PATH
) -> Path:
    """Pivot the long source into the existing all-tiltak wide-file format."""
    source = pd.read_csv(source_path)
    missing = _REQUIRED_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(
            f"Source is missing required columns: {', '.join(sorted(missing))}."
        )

    source["aarmnd"] = pd.to_datetime(
        source["AARMND"].astype(str), format="%Y%m"
    ).dt.strftime("%Y-%m-%d")
    source["COUNT(*)"] = pd.to_numeric(source["COUNT(*)"], errors="raise")

    regional = source[source["REGION"].isin(_REGIONS)].copy()
    if regional.duplicated(["aarmnd", "REGION"]).any():
        raise ValueError("Source contains duplicate month and region rows.")

    missing_regions = set(_REGIONS) - set(regional["REGION"])
    if missing_regions:
        raise ValueError(
            f"Source is missing expected regions: {', '.join(sorted(missing_regions))}."
        )

    wide = (
        regional.pivot(index="aarmnd", columns="REGION", values="COUNT(*)")
        .reindex(columns=_REGIONS)
        .sort_index()
        .reset_index()
    )
    if wide[_REGIONS].isna().any().any():
        raise ValueError("Source is missing a region count for one or more months.")

    wide[_REGIONS] = wide[_REGIONS].astype(int)
    wide["TOTAL"] = wide[_REGIONS].sum(axis=1)
    wide.columns = ["aarmnd", *(f"'{region}'" for region in _REGIONS), "TOTAL"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    """Convert the default all-tiltak long source."""
    print(f"Saved {convert_alle_tiltak()}")


if __name__ == "__main__":
    main()
