"""Validering av nøkkelunikhet for datauttrekk og sammenslåinger."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def ensure_unique_keys(
    df: pd.DataFrame, keys: Sequence[str], context: str
) -> None:
    """Reis feil når flere rader har samme naturlige nøkkel."""
    duplicate_rows = df[df.duplicated(subset=list(keys), keep=False)]
    if duplicate_rows.empty:
        return
    examples = duplicate_rows.loc[:, list(keys)].drop_duplicates().head(5).to_dict(
        orient="records"
    )
    raise ValueError(
        f"{context} contains duplicate rows for keys {list(keys)}. "
        f"Examples: {examples}"
    )


def collapse_consistent_values(
    df: pd.DataFrame,
    keys: Sequence[str],
    value_column: str,
    context: str,
) -> pd.DataFrame:
    """Slå sammen gjentatte nøkler bare når alle verdier er like."""
    key_columns = list(keys)
    value_counts = df.groupby(key_columns, dropna=False)[value_column].nunique(
        dropna=False
    )
    conflicts = value_counts[value_counts > 1]
    if not conflicts.empty:
        examples = [
            dict(zip(key_columns, key))
            for key in conflicts.index.tolist()[:5]
        ]
        raise ValueError(
            f"{context} contains conflicting '{value_column}' values for keys "
            f"{key_columns}. Examples: {examples}"
        )
    return df.drop_duplicates(subset=key_columns).copy()
