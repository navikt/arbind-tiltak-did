"""Fetch national Nav indicator data for one or more nedbrytning groups.

For each nedbrytning, one BigQuery query is issued and the results are saved as:

  data/input/indikatorer/landet/<group>/<utfall>.csv          (indikator)
  data/input/indikatorer/landet/<group>/forventet_<utfall>.csv
  data/input/indikatorer/landet/<group>/faktisk_<utfall>.csv
  data/input/personer/landet/<group>/antall_personer.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from bq_client import _ALIAS_MAP, _TABLE_URI, _TARGET_UTFALL, _slugify, run_query
from data_validation import collapse_consistent_values, ensure_unique_keys
from google.cloud import bigquery

_INDIKATOR_DIR = Path(__file__).resolve().parents[2] / "data" / "input" / "indikatorer"
_PERSONER_DIR = Path(__file__).resolve().parents[2] / "data" / "input" / "personer"


def _query_nedbrytning(nedbrytning: str) -> list[dict[str, Any]]:
    """Fetch all indicator columns for one nedbrytning in a single query."""
    query = f"""
        SELECT
            aarmnd_dato AS aarmnd,
            UTFALL AS utfall,
            INDIKATOR AS indikator,
            estimert AS forventet,
            faktisk AS faktisk,
            ANTALL_PERSONER AS antall_personer,
            NEDBRYTNING AS nedbrytning
        FROM `{_TABLE_URI}`
        WHERE ORG_NIVAA = 1
          AND NEDBRYTNING = @nedbrytning
          AND UTFALL IN UNNEST(@utfall)
        ORDER BY aarmnd
    """
    return run_query(
        query,
        [
            bigquery.ScalarQueryParameter("nedbrytning", "STRING", nedbrytning),
            bigquery.ArrayQueryParameter("utfall", "STRING", list(_TARGET_UTFALL)),
        ],
    )


def _pivot_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot national long-format data to one value column per month."""
    out = df.set_index("aarmnd")[[value_col]].sort_index().reset_index()
    out.columns.name = None
    return out


def fetch_and_save(nedbrytning: str) -> list[Path]:
    """Fetch all data for *nedbrytning* and save to the appropriate CSV files.

    *nedbrytning* may be a display alias (e.g. "Situasjonsbestemt") — the alias
    is resolved to the API name for the query, but the display name is used for
    the local folder slug.
    """
    api_nedbrytning = _ALIAS_MAP.get(nedbrytning, nedbrytning)
    records = _query_nedbrytning(api_nedbrytning)
    if not records:
        raise ValueError(f"No rows returned for nedbrytning='{nedbrytning}'.")

    df = pd.DataFrame(records)
    required = {
        "aarmnd",
        "utfall",
        "indikator",
        "forventet",
        "faktisk",
        "antall_personer",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Query result is missing required columns: {', '.join(sorted(missing))}."
        )
    df["aarmnd"] = pd.to_datetime(df["aarmnd"], utc=True).dt.strftime("%Y%m")
    ensure_unique_keys(
        df, ["aarmnd", "utfall"], "National indicator query result"
    )

    slug = _slugify(nedbrytning)
    indikator_dir = _INDIKATOR_DIR / "landet" / slug
    personer_dir = _PERSONER_DIR / "landet" / slug
    indikator_dir.mkdir(parents=True, exist_ok=True)
    personer_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # Per-utfall indicator files
    for utfall in _TARGET_UTFALL:
        sub = df[df["utfall"] == utfall].copy()
        if sub.empty:
            continue
        for col, prefix in [
            ("indikator", ""),
            ("forventet", "forventet_"),
            ("faktisk", "faktisk_"),
        ]:
            out_path = indikator_dir / f"{prefix}{utfall}.csv"
            _pivot_wide(sub, col).to_csv(out_path, index=False)
            saved.append(out_path)

    # antall_personer: one national value per month regardless of utfall
    personer_df = _pivot_wide(
        collapse_consistent_values(
            df,
            ["aarmnd"],
            "antall_personer",
            "National person-count query result",
        ),
        "antall_personer",
    )
    personer_path = personer_dir / "antall_personer.csv"
    personer_df.to_csv(personer_path, index=False)
    saved.append(personer_path)

    return saved


def main() -> None:
    """CLI: fetch all data for one or more nedbrytning groups.

    Usage:
      uv run python src/fetch_data/get_landet_data.py 'Alle'
      uv run python src/fetch_data/get_landet_data.py 'Alle' 'Situasjonsbestemt'
    """
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: uv run python src/fetch_data/get_landet_data.py "
            "<nedbrytning> [<nedbrytning> ...]\n"
            "Example: uv run python src/fetch_data/get_landet_data.py 'Alle'"
        )

    groups = sys.argv[1:]
    all_saved: list[Path] = []
    for group in groups:
        print(f"Fetching '{group}'...")
        saved = fetch_and_save(group)
        all_saved.extend(saved)
        for path in saved:
            print(f"  - {path}")

    print(f"\nDone. Saved {len(all_saved)} file(s) across {len(groups)} group(s).")


if __name__ == "__main__":
    main()
