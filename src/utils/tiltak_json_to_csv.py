"""Convert tiltak JSON source files to the wide-format CSVs used by the analysis pipeline.

  data/støttedata/tiltak-region.json       → data/input/tiltak/alle-tiltak.csv
  data/støttedata/tiltak-region-type.json  → data/input/tiltak/midl.-lønnstilskudd.csv

The output format matches the existing CSVs exactly:
  - aarmnd as first-of-month YYYY-MM-DD
  - one column per Nav-region (header wrapped in single quotes)
  - TOTAL column (row sum) as the last column
  - rows sorted ascending by aarmnd
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]

_TILTAK_REGION_JSON = _ROOT / "data" / "støttedata" / "tiltak-region.json"
_TILTAK_TYPE_JSON = _ROOT / "data" / "støttedata" / "tiltak-region-type.json"
_ALLE_TILTAK_CSV = _ROOT / "data" / "input" / "tiltak" / "alle-tiltak.csv"
_MIDL_CSV = _ROOT / "data" / "input" / "tiltak" / "midl.-lønnstilskudd.csv"

_MIDL_NAVN = "Midlertidig lønnstilskudd"

_REGION_ORDER = [
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


def _load_items(path: Path) -> pd.DataFrame:
    """Extract items list from nested JSON structure."""
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return pd.DataFrame(payload["results"][0]["items"])


def _parse_to_first_of_month(series: pd.Series) -> pd.Series:
    """Convert end-of-month Oracle timestamps to first-of-month YYYY-MM-DD strings."""
    dt = pd.to_datetime(series, format="%d.%m.%Y %H:%M:%S")
    return dt.dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")


def _to_wide_csv(items: pd.DataFrame, output_path: Path) -> None:
    """Pivot long items to wide CSV format and write to disk."""
    items = items.copy()
    items["aarmnd"] = _parse_to_first_of_month(items["aarmnd_dato"])

    wide = (
        items.pivot_table(index="aarmnd", columns="nav_region_navn", values="cnt", aggfunc="sum")
        .sort_index()
        .reset_index()
    )
    wide.columns.name = None

    # Ensure consistent region column order; fill any missing region with 0
    for region in _REGION_ORDER:
        if region not in wide.columns:
            wide[region] = 0
    wide = wide[["aarmnd"] + _REGION_ORDER]
    wide[_REGION_ORDER] = wide[_REGION_ORDER].fillna(0).astype(int)
    wide["TOTAL"] = wide[_REGION_ORDER].sum(axis=1)

    # Write with single-quoted region column names to match existing CSV format
    quoted_cols = (
        ["aarmnd"]
        + [f"'{r}'" for r in _REGION_ORDER]
        + ["TOTAL"]
    )
    wide.columns = quoted_cols
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False)


def main() -> None:
    # alle-tiltak: all rows from tiltak-region.json
    alle_items = _load_items(_TILTAK_REGION_JSON)
    _to_wide_csv(alle_items, _ALLE_TILTAK_CSV)
    print(f"Saved {_ALLE_TILTAK_CSV}")

    # midl. lønnstilskudd: filter to one tiltak type
    type_items = _load_items(_TILTAK_TYPE_JSON)
    midl_items = type_items[type_items["tiltaksnavn"] == _MIDL_NAVN].copy()
    if midl_items.empty:
        raise ValueError(f"No rows found for tiltaksnavn='{_MIDL_NAVN}'")
    _to_wide_csv(midl_items, _MIDL_CSV)
    print(f"Saved {_MIDL_CSV}")


if __name__ == "__main__":
    main()
