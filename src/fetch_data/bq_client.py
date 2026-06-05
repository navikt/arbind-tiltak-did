"""Shared BigQuery client utilities for Nav indicator data fetching.

Provides the common table URI, target outcomes, slugify helper, and a
generic query runner used by :mod:`get_fylke_data` and :mod:`get_enhet_data`.
"""

from __future__ import annotations

import re
from typing import Any

from google.cloud import bigquery

_TABLE_URI = "arbeidsindikator-prod-51bc.arbeidsindikator.agg_indikator_siste_pub"
_TARGET_UTFALL = ("atid3", "jobb3")

# Maps new display names (old NAV innsatsgruppe scheme) to the API query strings.
_ALIAS_MAP: dict[str, str] = {
    "Situasjonsbestemt": "Innsatsgruppe - Trenger veiledning",
    "Spesielt tilpasset": "Innsatsgruppe - Trenger veiledning, nedsatt arbeidsevne",
    "Standard": "Innsatsgruppe - Gode muligheter",
}


def _slugify(value: str) -> str:
    """Return a filesystem-safe lower-case slug."""
    return re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("_")


def run_query(
    query: str,
    params: list[
        bigquery.query.ScalarQueryParameter | bigquery.query.ArrayQueryParameter
    ],
) -> list[dict[str, Any]]:
    """Execute *query* with *params* against BigQuery and return rows as dicts."""
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    results = client.query(query, job_config=job_config).result()
    return [dict(row) for row in results]
