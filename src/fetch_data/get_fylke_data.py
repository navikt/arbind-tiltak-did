"""Fetch all regional Nav indicator data for one or more nedbrytning groups.

For each nedbrytning, one BigQuery query is issued and the results are saved as:

  data/raw/indikatorer/nedbrytning/<group>/<utfall>.csv          (indikator)
  data/raw/indikatorer/nedbrytning/<group>/forventet_<utfall>.csv
  data/raw/indikatorer/nedbrytning/<group>/faktisk_<utfall>.csv
  data/raw/personer/nedbrytning/<group>/antall_personer.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

_TABLE_URI = "arbeidsindikator-prod-51bc.arbeidsindikator.agg_indikator_siste_pub"
_INDIKATOR_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "indikatorer"
_PERSONER_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "personer"
_TARGET_UTFALL = ("atid3", "jobb3")


def _slugify(value: str) -> str:
    """Return a filesystem-safe lower-case slug."""
    return re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("_")


def _query_nedbrytning(nedbrytning: str) -> list[dict[str, Any]]:
    """Fetch all indicator columns for one nedbrytning in a single query."""
    client = bigquery.Client()
    query = f"""
        SELECT
            CAST(BEHOLDNINGSMAANED AS STRING) AS aarmnd,
            org_sted AS org_sted,
            UTFALL AS utfall,
            INDIKATOR AS indikator,
            forventet AS forventet,
            faktisk AS faktisk,
            ANTALL_PERSONER AS antall_personer,
            NEDBRYTNING AS nedbrytning
        FROM `{_TABLE_URI}`
        WHERE ORG_NIVAA = 2
          AND NEDBRYTNING = @nedbrytning
          AND org_sted != 'Nasjonal oppfølgingsenhet'
          AND UTFALL IN UNNEST(@utfall)
        ORDER BY BEHOLDNINGSMAANED
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("nedbrytning", "STRING", nedbrytning),
            bigquery.ArrayQueryParameter("utfall", "STRING", list(_TARGET_UTFALL)),
        ]
    )
    results = client.query(query, job_config=job_config).result()
    return [dict(row) for row in results]


def _pivot_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot long dataframe to wide format with regions as columns."""
    out = (
        df.pivot_table(
            index="aarmnd",
            columns="org_sted",
            values=value_col,
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )
    out.columns.name = None
    return out


def fetch_and_save(nedbrytning: str) -> list[Path]:
    """Fetch all data for *nedbrytning* and save to the appropriate CSV files."""
    records = _query_nedbrytning(nedbrytning)
    if not records:
        raise ValueError(f"No rows returned for nedbrytning='{nedbrytning}'.")

    df = pd.DataFrame(records)
    required = {
        "aarmnd",
        "org_sted",
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
    df["aarmnd"] = pd.to_datetime(df["aarmnd"], errors="raise").dt.strftime("%Y%m")

    slug = _slugify(nedbrytning)
    indikator_dir = _INDIKATOR_DIR / "nedbrytning" / slug
    personer_dir = _PERSONER_DIR / "nedbrytning" / slug
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

    # antall_personer: one value per region per month regardless of utfall
    personer_df = df.drop_duplicates(subset=["aarmnd", "org_sted"]).pipe(
        _pivot_wide, "antall_personer"
    )
    personer_path = personer_dir / "antall_personer.csv"
    personer_df.to_csv(personer_path, index=False)
    saved.append(personer_path)

    return saved


def main() -> None:
    """CLI: fetch all data for one or more nedbrytning groups.

    Usage:
      uv run python src/fetch_data/get_fylke_data.py 'Alle'
      uv run python src/fetch_data/get_fylke_data.py 'Alle' 'Trenger veiledning'
    """
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: uv run python src/fetch_data/get_fylke_data.py "
            "<nedbrytning> [<nedbrytning> ...]\n"
            "Example: uv run python src/fetch_data/get_fylke_data.py 'Alle'"
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
