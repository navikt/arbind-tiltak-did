"""Tests for src/diff-in-diff/data_io.py."""

from __future__ import annotations

from pathlib import Path

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
