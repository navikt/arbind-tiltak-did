"""Tests for src/diff-in-diff/data_io.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import (
    ALL_MONTHS,
    ENHETER,
    REGIONS,
    write_enhet_mapping_json,
    write_indicator_wide_csv,
    write_tiltak_csv,
)
from data_io import (
    _convert_aarmnd_format,
    _load_enhet_mapping,
    _load_indicator_long,
    _load_indicator_wide_to_long,
    _load_tiltak_wide_to_long,
    _seasonal_adjust_tiltak,
)


def test_convert_aarmnd_format_dd_mm_yyyy(tmp_path: Path) -> None:
    """dd-mm-yyyy strings are converted to YYYYMM."""
    df = pd.DataFrame({"aarmnd": ["01-01-2021", "15-06-2022"]})
    result = _convert_aarmnd_format(df, "aarmnd")
    assert list(result["aarmnd"]) == ["202101", "202206"]


def test_convert_aarmnd_format_yyyy_mm_dd(tmp_path: Path) -> None:
    """yyyy-mm-dd strings are converted to YYYYMM."""
    df = pd.DataFrame({"aarmnd": ["2021-01-01", "2022-06-15"]})
    result = _convert_aarmnd_format(df, "aarmnd")
    assert list(result["aarmnd"]) == ["202101", "202206"]


def test_convert_aarmnd_format_passthrough() -> None:
    """Already-YYYYMM values are left unchanged (no recognised date format)."""
    df = pd.DataFrame({"aarmnd": ["202101", "202206"]})
    result = _convert_aarmnd_format(df, "aarmnd")
    assert list(result["aarmnd"]) == ["202101", "202206"]


def test_load_enhet_mapping(tmp_path: Path) -> None:
    """enhetsmapping.json is loaded as a DataFrame with enhet/region columns."""
    p = tmp_path / "map.json"
    write_enhet_mapping_json(p)
    df = _load_enhet_mapping(p)

    assert set(df.columns) == {"enhet", "region"}
    expected_n = sum(len(v) for v in ENHETER.values())
    assert len(df) == expected_n
    # Each enhet maps to its correct region
    mapping = dict(zip(df["enhet"], df["region"]))
    for region, enheter_list in ENHETER.items():
        for enhet in enheter_list:
            assert mapping[enhet] == region


def test_load_indicator_long_missing_columns(tmp_path: Path) -> None:
    """_load_indicator_long raises ValueError when required columns are absent."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"aarmnd": ["202101"], "enhet": ["Nav X"]}).to_csv(
        bad_csv, index=False
    )
    with pytest.raises(ValueError, match="indikator"):
        _load_indicator_long(bad_csv)


def test_load_indicator_wide_to_long(tmp_path: Path) -> None:
    """Wide indicator CSV is melted to long format with correct shape and columns."""
    p = tmp_path / "ind.csv"
    write_indicator_wide_csv(p)
    df = _load_indicator_wide_to_long(p)

    assert set(df.columns) >= {"aarmnd", "region", "indikator"}
    expected_rows = len(ALL_MONTHS) * len(REGIONS)
    assert len(df) == expected_rows
    assert set(df["region"].unique()) == set(REGIONS)


def test_load_tiltak_wide_to_long(tmp_path: Path) -> None:
    """Wide tiltak CSV is cleaned and melted: TOTAL dropped, aarmnd normalised."""
    p = tmp_path / "tiltak.csv"
    write_tiltak_csv(p)
    df = _load_tiltak_wide_to_long(p)

    assert set(df.columns) >= {"aarmnd", "region", "tiltak"}
    # TOTAL column should not appear as a region
    assert "TOTAL" not in df["region"].values
    expected_rows = len(ALL_MONTHS) * len(REGIONS)
    assert len(df) == expected_rows
    # Pre-period tiltak: flat at TILTAK_BASE for months earlier than -12
    pre_early = df[df["aarmnd"] < "202201"]
    assert (pre_early["tiltak"] == 1000).all()


def _make_seasonal_tiltak_wide(tmp_path: Path) -> Path:
    """Write a wide tiltak CSV with strong multiplicative seasonality (36 months)."""
    # Base level grows from 1000 to ~1300, with a July dip of 30%
    months = [(pd.Period("202001", freq="M") + i).strftime("%Y%m") for i in range(36)]
    rows = []
    for i, m in enumerate(months):
        p = pd.Period(m, freq="M")
        level = 1000 + i * 10
        month_num = p.month
        # Multiplicative seasonal factor: July = 0.70, Dec = 1.05, others linear
        seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * (month_num - 1) / 12) - 0.3 * (month_num == 7)
        row: dict = {"aarmnd": p.to_timestamp().strftime("%Y-%m-%d")}
        for j, r in enumerate(REGIONS):
            row[f"'{r}'"] = round(level * seasonal * (1 + 0.05 * j), 1)
        row["TOTAL"] = sum(row[f"'{r}'"] for r in REGIONS)
        rows.append(row)
    p_out = tmp_path / "seasonal_tiltak.csv"
    pd.DataFrame(rows).to_csv(p_out, index=False)
    return p_out


def test_seasonal_adjust_tiltak_preserves_shape(tmp_path: Path) -> None:
    """_seasonal_adjust_tiltak returns DataFrame with same shape and columns."""
    p = tmp_path / "tiltak.csv"
    write_tiltak_csv(p)
    raw_long = _load_tiltak_wide_to_long(p, seasonal_adjust=False)
    sa_long = _load_tiltak_wide_to_long(p, seasonal_adjust=True)
    assert raw_long.shape == sa_long.shape
    assert set(raw_long.columns) == set(sa_long.columns)


def test_seasonal_adjust_tiltak_reduces_amplitude(tmp_path: Path) -> None:
    """SA reduces the seasonal amplitude (month-of-year variance in national total)."""
    p = _make_seasonal_tiltak_wide(tmp_path)
    raw_long = _load_tiltak_wide_to_long(p, seasonal_adjust=False)
    sa_long = _load_tiltak_wide_to_long(p, seasonal_adjust=True)

    def month_var(df: pd.DataFrame) -> float:
        total = df.groupby("aarmnd")["tiltak"].sum()
        months = pd.to_datetime(total.index, format="%Y%m").month
        return float(pd.Series(total.values).groupby(months).mean().var())

    assert month_var(sa_long) < month_var(raw_long)


def test_seasonal_adjust_tiltak_differs_from_raw(tmp_path: Path) -> None:
    """SA output is numerically different from raw for data with seasonality."""
    p = _make_seasonal_tiltak_wide(tmp_path)
    raw_long = _load_tiltak_wide_to_long(p, seasonal_adjust=False)
    sa_long = _load_tiltak_wide_to_long(p, seasonal_adjust=True)
    # The two should not be equal
    assert not np.allclose(raw_long["tiltak"].values, sa_long["tiltak"].values)
