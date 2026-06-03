"""File-loading helpers for the DiD analysis pipeline.

These functions handle raw CSV / JSON IO only. Panel construction and feature
engineering live in :mod:`prep_data`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _convert_aarmnd_format(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convert *date_col* in *df* to YYYYMM format if it's in a recognised date format."""
    if date_col in df.columns:
        for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
            try:
                df[date_col] = pd.to_datetime(df[date_col], format=fmt).dt.strftime(
                    "%Y%m"
                )
                break
            except ValueError, TypeError:
                continue
    return df


def _load_enhet_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load the enhet→region mapping from a JSON file.

    Expected format: ``{"results": [{"items": [{"nav_enhet_navn": ..., "nav_region_navn": ...}]}]}``.

    Returns a DataFrame with columns ``enhet`` and ``region``.
    """
    with open(mapping_path, encoding="utf-8") as f:
        payload = json.load(f)
    items = payload["results"][0]["items"]
    df = pd.DataFrame(items).rename(
        columns={"nav_enhet_navn": "enhet", "nav_region_navn": "region"}
    )
    df = df.dropna(subset=["enhet", "region"]).drop_duplicates(subset=["enhet"])
    return df[["enhet", "region"]].copy()


def _load_indicator_long(path: Path) -> pd.DataFrame:
    """Load an indicator CSV that is already in long format.

    Expected columns: ``aarmnd``, ``enhet`` (or entity column), ``indikator``.
    """
    df = pd.read_csv(path)
    expected = {"aarmnd", "enhet", "indikator"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(
            f"Long-format indicator file {path} must have columns {expected}, "
            f"found {set(df.columns)}"
        )
    return df


def _load_indicator_wide_to_long(path: Path) -> pd.DataFrame:
    """Load a wide-format indicator CSV and melt to long format.

    Wide format: rows are months (aarmnd), columns are entity names.
    Returns DataFrame with columns ``aarmnd``, ``region``, ``indikator``.
    """
    df = pd.read_csv(path)
    df = _convert_aarmnd_format(df, "aarmnd")
    df["aarmnd"] = df["aarmnd"].astype(str)
    return df.melt(id_vars=["aarmnd"], var_name="region", value_name="indikator")


def _load_tiltak_wide_to_long(path: Path) -> pd.DataFrame:
    """Load a wide-format tiltak CSV, clean it, and melt to long format.

    Returns DataFrame with columns ``aarmnd``, ``region``, ``tiltak``.
    """
    tiltak_df = pd.read_csv(path)
    tiltak_df.columns = [c.strip("'") for c in tiltak_df.columns]
    tiltak_df = tiltak_df.drop(columns=["TOTAL"], errors="ignore")
    tiltak_df = tiltak_df[tiltak_df["aarmnd"] != "aarmnd"].reset_index(drop=True)
    for col in tiltak_df.columns:
        if col != "aarmnd":
            tiltak_df[col] = pd.to_numeric(tiltak_df[col], errors="coerce")
    tiltak_df = _convert_aarmnd_format(tiltak_df, "aarmnd")
    tiltak_df["aarmnd"] = tiltak_df["aarmnd"].astype(str)
    return tiltak_df.melt(id_vars=["aarmnd"], var_name="region", value_name="tiltak")
