"""Fetch number of people by region for one or more Nav groups (nedbrytning)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

_TABLE_URI = "arbeidsindikator-prod-51bc.arbeidsindikator.agg_indikator_siste_pub"
_BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "personer"


def _slugify(value: str) -> str:
    """Return a filesystem-safe lower-case slug."""
    return re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("_")


def _output_dir_for_group(group_name: str) -> Path:
    """Return output directory for one nedbrytning group value."""
    return _BASE_DIR / "nedbrytning" / _slugify(group_name)


def get_antall_personer_fylke(group_name: str) -> list[dict[str, Any]]:
    """Fetch rows for one nedbrytning group value."""
    client = bigquery.Client()
    query = f"""
        SELECT
            CAST(BEHOLDNINGSMAANED AS STRING) AS aarmnd,
            ORG_STED AS org_sted,
            ANTALL_PERSONER AS antall_personer,
            NEDBRYTNING AS nedbrytning
        FROM `{_TABLE_URI}`
        WHERE ORG_NIVAA = 2
          AND NEDBRYTNING = @group_name
          AND ORG_STED != 'Nasjonal oppfølgingsenhet'
        ORDER BY BEHOLDNINGSMAANED
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("group_name", "STRING", group_name),
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    return [dict(row.items()) for row in rows]


def fetch_and_save_group(group_name: str) -> Path:
    """Fetch one group and save a wide CSV with one column per region."""
    records = get_antall_personer_fylke(group_name=group_name)
    if not records:
        raise ValueError(f"No rows returned for nedbrytning='{group_name}'.")

    df = pd.DataFrame(records)
    required = {"aarmnd", "org_sted", "antall_personer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Query result is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )

    out_df = (
        df.pivot_table(
            index="aarmnd",
            columns="org_sted",
            values="antall_personer",
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )
    out_df.columns.name = None

    out_dir = _output_dir_for_group(group_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "antall_personer.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    """CLI: fetch one or more groups.

    Usage:
      uv run python src/fetch_data/get_antall_personer_fylke.py "Alle"
      uv run python src/fetch_data/get_antall_personer_fylke.py "Alle" "Unge"
    """
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: uv run python src/fetch_data/get_antall_personer_fylke.py "
            "<nedbrytning> [<nedbrytning> ...]"
        )

    groups = sys.argv[1:]
    saved: list[Path] = []
    for group_name in groups:
        saved.append(fetch_and_save_group(group_name))

    print(f"Saved {len(saved)} file(s):")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
