"""Fetch all unit-level (enhet) Nav indicator data for one or more nedbrytning groups.

Mirrors get_fylke_data.py but queries ORG_NIVAA = 3 (enheter) and saves
results in **long format** (aarmnd, enhet, value) rather than wide, since
there are many units and the triple-diff pipeline expects long format.

Output structure:
  data/input/indikatorer/enhet/nedbrytning/<group>/<utfall>.csv
  data/input/indikatorer/enhet/nedbrytning/<group>/forventet_<utfall>.csv
  data/input/indikatorer/enhet/nedbrytning/<group>/faktisk_<utfall>.csv
  data/input/personer/enhet/nedbrytning/<group>/antall_personer.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from bq_client import _ALIAS_MAP, _TABLE_URI, _TARGET_UTFALL, _slugify, run_query
from google.cloud import bigquery

_INDIKATOR_DIR = Path(__file__).resolve().parents[2] / "data" / "input" / "indikatorer"
_PERSONER_DIR = Path(__file__).resolve().parents[2] / "data" / "input" / "personer"


def _query_nedbrytning_enhet(nedbrytning: str) -> list[dict[str, Any]]:
    """Fetch all indicator columns for one nedbrytning at enhet level."""
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
        WHERE ORG_NIVAA = 3
          AND NEDBRYTNING = @nedbrytning
          AND UTFALL IN UNNEST(@utfall)
        ORDER BY BEHOLDNINGSMAANED
    """
    return run_query(
        query,
        [
            bigquery.ScalarQueryParameter("nedbrytning", "STRING", nedbrytning),
            bigquery.ArrayQueryParameter("utfall", "STRING", list(_TARGET_UTFALL)),
        ],
    )


def _to_long(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Extract a long-format DataFrame with columns: aarmnd, enhet, <value_col>."""
    return df[["aarmnd", "enhet", value_col]].copy()


def fetch_and_save_enhet(nedbrytning: str) -> list[Path]:
    """Fetch all enhet-level data for *nedbrytning* and save long-format CSVs.

    *nedbrytning* may be a display alias (e.g. "Situasjonsbestemt") — the alias
    is resolved to the API name for the query, but the display name is used for
    the local folder slug.
    """
    api_nedbrytning = _ALIAS_MAP.get(nedbrytning, nedbrytning)
    records = _query_nedbrytning_enhet(api_nedbrytning)
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
    df = df.rename(columns={"org_sted": "enhet"})

    slug = _slugify(nedbrytning)
    indikator_dir = _INDIKATOR_DIR / "enhet" / "nedbrytning" / slug
    personer_dir = _PERSONER_DIR / "enhet" / "nedbrytning" / slug
    indikator_dir.mkdir(parents=True, exist_ok=True)
    personer_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    for utfall in _TARGET_UTFALL:
        sub = df[df["utfall"] == utfall].copy()
        if sub.empty:
            continue
        for col, prefix in [
            ("indikator", ""),
            ("forventet", "forventet_"),
            ("faktisk", "faktisk_"),
        ]:
            long_df = _to_long(sub, col).sort_values(["aarmnd", "enhet"])
            out_path = indikator_dir / f"{prefix}{utfall}.csv"
            long_df.to_csv(out_path, index=False)
            saved.append(out_path)

    # antall_personer: one value per enhet per month
    personer_df = df.drop_duplicates(subset=["aarmnd", "enhet"])[
        ["aarmnd", "enhet", "antall_personer"]
    ].sort_values(["aarmnd", "enhet"])
    personer_path = personer_dir / "antall_personer.csv"
    personer_df.to_csv(personer_path, index=False)
    saved.append(personer_path)

    return saved


def main() -> None:
    """CLI: fetch enhet-level data for one or more nedbrytning groups.

    Usage:
      uv run python src/fetch_data/get_enhet_data.py 'Alle'
      uv run python src/fetch_data/get_enhet_data.py 'Alle' 'Situasjonsbestemt'
    """
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: uv run python src/fetch_data/get_enhet_data.py "
            "<nedbrytning> [<nedbrytning> ...]\n"
            "Example: uv run python src/fetch_data/get_enhet_data.py 'Alle'"
        )

    groups = sys.argv[1:]
    all_saved: list[Path] = []
    for group in groups:
        print(f"Fetching enhet-level data for '{group}'...")
        saved = fetch_and_save_enhet(group)
        all_saved.extend(saved)
        for path in saved:
            print(f"  - {path}")

    print(f"\nDone. Saved {len(all_saved)} file(s) across {len(groups)} group(s).")


if __name__ == "__main__":
    main()
