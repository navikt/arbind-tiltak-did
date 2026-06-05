"""File-loading helpers for the DiD analysis pipeline.

These functions handle raw CSV / JSON IO only. Panel construction and feature
engineering live in :mod:`prep_data`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
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


def _seasonal_adjust_tiltak(
    tiltak_wide: pd.DataFrame,
    pre_only: bool = False,
    treatment_start: str = "202506",
) -> pd.DataFrame:
    """Apply multiplicative seasonal adjustment to a wide-format tiltak DataFrame.

    The alle-tiltak series exhibits multiplicative seasonality: the seasonal
    amplitude grows proportionally with the level (~6% in 2020, ~31% in 2023–25).
    Additive STL on raw counts therefore over-corrects early years and
    under-corrects later ones.  This function applies STL to the log of the
    national total and back-transforms the seasonal factor per month to obtain
    a proportional correction that is applied uniformly to each region.

    Method
    ------
    1. Sum across all region columns to get the national total.
    2. Fit STL on ``log(national_total)`` with ``period=12``, ``seasonal=13``,
       ``robust=True``.  When ``pre_only=True``, STL is fit only on pre-treatment
       observations; the estimated seasonal factors (averaged by calendar month)
       are then applied to the full series.
    3. For each observation, compute the multiplicative seasonal factor:
       ``factor = exp(seasonal_component)``.
    4. Divide every region's raw count by the corresponding factor.

    Parameters
    ----------
    tiltak_wide:
        Wide-format DataFrame with an ``aarmnd`` column (YYYYMM string) and one
        numeric column per region.  Must not contain the ``TOTAL`` column.
    pre_only:
        If ``True``, fit STL on the pre-treatment period only and extrapolate
        seasonal factors to post-treatment months using the mean factor per
        calendar month estimated from the pre-period.
    treatment_start:
        First treatment month in YYYYMM format.  Used only when ``pre_only=True``.

    Returns
    -------
    DataFrame with the same shape and column names as *tiltak_wide* but with
    seasonally adjusted region counts.
    """
    from statsmodels.tsa.seasonal import STL  # lazy import — not needed for midl.

    region_cols = [c for c in tiltak_wide.columns if c != "aarmnd"]
    total = tiltak_wide[region_cols].sum(axis=1).values.astype(float)
    log_total = pd.Series(np.log(total))

    if pre_only:
        pre_mask = tiltak_wide["aarmnd"] < treatment_start
        if pre_mask.sum() < 24:
            raise ValueError(
                f"Too few pre-treatment observations ({pre_mask.sum()}) for STL."
            )
        res_pre = STL(log_total[pre_mask], period=12, seasonal=13, robust=True).fit()
        # Average seasonal factor by calendar month from pre-period
        months_pre = pd.to_datetime(
            tiltak_wide.loc[pre_mask, "aarmnd"], format="%Y%m"
        ).dt.month
        seasonal_by_month = (
            pd.Series(np.asarray(res_pre.seasonal), index=months_pre.values)
            .groupby(level=0)
            .mean()
        )
        # Apply to full series by calendar month
        all_months = pd.to_datetime(tiltak_wide["aarmnd"], format="%Y%m").dt.month
        seasonal_factor = np.exp(all_months.map(seasonal_by_month).values)
    else:
        res = STL(log_total, period=12, seasonal=13, robust=True).fit()
        seasonal_factor = np.exp(res.seasonal)

    out = tiltak_wide.copy()
    for col in region_cols:
        out[col] = out[col].values / seasonal_factor

    return out


def _load_tiltak_wide_to_long(
    path: Path,
    seasonal_adjust: bool = False,
    seasonal_adjust_pre_only: bool = False,
    treatment_start: str = "202506",
) -> pd.DataFrame:
    """Load a wide-format tiltak CSV, clean it, and melt to long format.

    Parameters
    ----------
    path:
        Path to the wide-format tiltak CSV.
    seasonal_adjust:
        If ``True``, apply multiplicative STL seasonal adjustment before melting.
        Use for alle-tiltak data which has a strong multiplicative seasonal
        component.  Leave ``False`` (default) for midlertidig lønnstilskudd.
    seasonal_adjust_pre_only:
        If ``True`` (and ``seasonal_adjust=True``), fit the STL model on the
        pre-treatment period only and extrapolate seasonal factors to post-treatment
        months using calendar-month averages from the pre-period.  Default is
        ``False`` (full-series fit).
    treatment_start:
        First treatment month in YYYYMM format.  Used only when
        ``seasonal_adjust_pre_only=True``.

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
    if seasonal_adjust:
        tiltak_df = _seasonal_adjust_tiltak(
            tiltak_df,
            pre_only=seasonal_adjust_pre_only,
            treatment_start=treatment_start,
        )
    return tiltak_df.melt(id_vars=["aarmnd"], var_name="region", value_name="tiltak")
