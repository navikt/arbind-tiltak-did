"""Reads in two JSON files and writes them to an Excel file with two sheets.

Sheet 1 – tiltak-region: wide format (aarmnd as rows, region as columns, cnt as values)
Sheet 2 – tiltak-region-type: long format sorted by date, region, tiltaksnavn
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
INPUT1 = _ROOT / "data" / "støttedata" / "tiltak-region.json"
INPUT2 = _ROOT / "data" / "støttedata" / "tiltak-region-type.json"
OUTPUT = _ROOT / "outputs" / "tiltak_region.xlsx"


def _load_items(path: Path) -> pd.DataFrame:
    """Extract the items list from the nested JSON structure."""
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return pd.DataFrame(payload["results"][0]["items"])


def _parse_date(series: pd.Series) -> pd.Series:
    """Parse Oracle-style date strings (e.g. '31.01.2024 00:00:00') to YYYYMM."""
    return pd.to_datetime(series, format="%d.%m.%Y %H:%M:%S").dt.strftime("%Y%m")


def main() -> None:
    # ── Sheet 1: tiltak-region, pivoted wide ──────────────────────────────────
    df1 = _load_items(INPUT1)
    df1["aarmnd"] = _parse_date(df1["aarmnd_dato"])
    df1_wide = (
        df1.pivot_table(index="aarmnd", columns="nav_region_navn", values="cnt", aggfunc="sum")
        .sort_index()
        .reset_index()
    )
    df1_wide.columns.name = None

    # ── Sheet 2: tiltak-region-type, long format sorted ───────────────────────
    df2 = _load_items(INPUT2)
    df2["aarmnd"] = _parse_date(df2["aarmnd_dato"])
    df2 = (
        df2.drop(columns=["aarmnd_dato"])
        .rename(columns={"nav_region_navn": "region", "tiltaksnavn": "tiltak", "cnt": "antall"})
        [["aarmnd", "region", "tiltak", "antall"]]
        .sort_values(["aarmnd", "region", "tiltak"])
        .reset_index(drop=True)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        df1_wide.to_excel(writer, sheet_name="tiltak-region", index=False)
        df2.to_excel(writer, sheet_name="tiltak-region-type", index=False)

    print(f"Saved {OUTPUT}")
    print(f"  tiltak-region: {len(df1_wide)} months × {len(df1_wide.columns) - 1} regions")
    print(f"  tiltak-region-type: {len(df2)} rows")


if __name__ == "__main__":
    main()
