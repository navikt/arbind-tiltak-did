"""Standalone utility: convert a directory of CSVs into a multi-sheet Excel file.

Excel workbooks are normally produced automatically by ``src/get_data.py``
(via ``QuerySpec.excel_workbook``).  Use this script if you need to rebuild
a workbook manually from existing CSVs without re-fetching from BigQuery.

Usage
-----
    uv run python csv_to_excel.py <input_dir> <output_file>

    # Defaults (backwards-compatible):
    uv run python csv_to_excel.py
    #  reads  data/raw/indikatorer/landet/*.csv
    #  writes outputs/indikatorer_landet.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_DEFAULT_INPUT = Path("data/raw/indikatorer/landet")
_DEFAULT_OUTPUT = Path("outputs/indikatorer_landet.xlsx")


def csvs_to_excel(input_dir: Path, output_file: Path) -> None:
    """Read all CSVs in *input_dir* and write them as sheets in *output_file*."""
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df.to_excel(writer, sheet_name=csv_file.stem, index=False)
            print(f"  {csv_file.name} → sheet '{csv_file.stem}'")

    print(f"\nSaved {len(csv_files)} sheet(s) to {output_file}")


def main() -> None:
    """Parse CLI args and invoke :func:`csvs_to_excel`."""
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_INPUT
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_OUTPUT
    csvs_to_excel(input_dir, output_file)


if __name__ == "__main__":
    main()
