"""Fetch Nav indikator data for a chosen nedbrytning."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

_TABLE_URI = "arbeidsindikator-prod-51bc.arbeidsindikator.agg_indikator_siste_pub"
_BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "indikatorer"
_TARGET_UTFALL = ("atid3", "jobb3")


def _slugify(value: str) -> str:
    """Return a filesystem-safe lower-case slug."""
    return re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("_")


def _output_dir_for_nedbrytning(nedbrytning: str) -> Path:
    """Return output directory for one nedbrytning value."""
    return _BASE_DIR / "nedbrytning" / _slugify(nedbrytning)


def get_indikator_fylke(nedbrytning: str) -> list[dict[str, Any]]:
    """Fetch all rows for one nedbrytning and target utfall values."""
    client = bigquery.Client()
    query = f"""
        SELECT
            CAST(BEHOLDNINGSMAANED AS STRING) AS aarmnd,
            org_sted AS org_sted,
            UTFALL AS utfall,
            INDIKATOR AS indikator,
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
    query_job = client.query(query, job_config=job_config)
    results = query_job.result()
    return [dict(row) for row in results]


def fetch_and_save_for_nedbrytning(nedbrytning: str) -> list[Path]:
    """Fetch both atid3 and jobb3 for *nedbrytning* and save wide CSV files."""
    records = get_indikator_fylke(nedbrytning=nedbrytning)
    if not records:
        raise ValueError(f"No rows returned for nedbrytning='{nedbrytning}'.")

    df = pd.DataFrame(records)
    required = {"aarmnd", "org_sted", "utfall", "indikator"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Query result is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )
    df["aarmnd"] = pd.to_datetime(df["aarmnd"], errors="raise").dt.strftime("%Y%m")

    out_dir = _output_dir_for_nedbrytning(nedbrytning)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for utfall in _TARGET_UTFALL:
        sub = df[df["utfall"] == utfall].copy()
        if sub.empty:
            continue

        out_df = (
            sub.pivot_table(
                index="aarmnd",
                columns="org_sted",
                values="indikator",
                aggfunc="first",
            )
            .sort_index()
            .reset_index()
        )
        out_df.columns.name = None

        out_path = out_dir / f"{utfall}.csv"
        out_df.to_csv(out_path, index=False)
        saved.append(out_path)
    if not saved:
        raise ValueError(
            "No rows found for target utfall values "
            f"{_TARGET_UTFALL!r} and nedbrytning='{nedbrytning}'."
        )
    return saved


def main() -> None:
    """CLI: uv run python src/fetch_data/get_indikator_fylke.py <nedbrytning>."""
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: uv run python src/fetch_data/get_indikator_fylke.py <nedbrytning>\n"
            "Example: uv run python src/fetch_data/get_indikator_fylke.py 'Alle'"
        )
    nedbrytning = sys.argv[1]
    saved = fetch_and_save_for_nedbrytning(nedbrytning)
    print(f"Saved {len(saved)} file(s) for nedbrytning='{nedbrytning}':")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
