"""Export treatment variables per region to Excel.

Produces one Excel file with four sheets — one per tiltak definition:
  - midl-lonnstilskudd        : midl. lønnstilskudd, no seasonal adjustment
  - alle-tiltak-ukorrigert    : alle tiltak, no seasonal adjustment
  - alle-tiltak-korrigert-full: alle tiltak, STL on full series
  - alle-tiltak-korrigert-pre : alle tiltak, STL fitted on pre-period only

Each sheet contains post-treatment rows only (aarmnd >= TREATMENT_START) with
columns: region, aarmnd, tiltak, peak_tiltak, tiltaksnedgang.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "diff-in-diff"))

from data_io import _load_tiltak_wide_to_long  # noqa: E402

TREATMENT_START = "202506"
MIDL_PATH = _ROOT / "data" / "input" / "tiltak" / "midl.-lønnstilskudd.csv"
ALLE_PATH = _ROOT / "data" / "input" / "tiltak" / "alle-tiltak.csv"
OUTPUT_PATH = _ROOT / "outputs" / "treatment_region.xlsx"


def _compute_treatment(tiltak_long: pd.DataFrame, treatment_start: str) -> pd.DataFrame:
    """Add peak_tiltak and tiltaksnedgang columns; filter to post-treatment rows."""
    pre_mask = tiltak_long["aarmnd"] < treatment_start
    peak = (
        tiltak_long.loc[pre_mask, ["region", "tiltak"]]
        .groupby("region")["tiltak"]
        .max()
        .rename("peak_tiltak")
    )
    df = tiltak_long.merge(peak, on="region", how="left")

    post_mask = df["aarmnd"] >= treatment_start
    df["tiltaksnedgang"] = np.nan
    valid = post_mask & (df["peak_tiltak"] > 0)
    df.loc[valid, "tiltaksnedgang"] = np.clip(
        (df.loc[valid, "peak_tiltak"] - df.loc[valid, "tiltak"]) / df.loc[valid, "peak_tiltak"],
        0.0,
        1.0,
    )

    return (
        df.loc[post_mask, ["region", "aarmnd", "tiltak", "peak_tiltak", "tiltaksnedgang"]]
        .sort_values(["region", "aarmnd"])
        .reset_index(drop=True)
    )


def main() -> None:
    sheets: dict[str, pd.DataFrame] = {}

    sheets["midl-lonnstilskudd"] = _compute_treatment(
        _load_tiltak_wide_to_long(MIDL_PATH),
        TREATMENT_START,
    )
    sheets["alle-tiltak-ukorrigert"] = _compute_treatment(
        _load_tiltak_wide_to_long(ALLE_PATH),
        TREATMENT_START,
    )
    sheets["alle-tiltak-korrigert-full"] = _compute_treatment(
        _load_tiltak_wide_to_long(ALLE_PATH, seasonal_adjust=True, seasonal_adjust_pre_only=False),
        TREATMENT_START,
    )
    sheets["alle-tiltak-korrigert-pre"] = _compute_treatment(
        _load_tiltak_wide_to_long(
            ALLE_PATH,
            seasonal_adjust=True,
            seasonal_adjust_pre_only=True,
            treatment_start=TREATMENT_START,
        ),
        TREATMENT_START,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved {OUTPUT_PATH}")
    for name, df in sheets.items():
        print(f"  {name}: {len(df)} rows")


if __name__ == "__main__":
    main()
