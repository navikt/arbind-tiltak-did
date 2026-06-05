"""Tests for src/diff-in-diff/prep_data.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from conftest import (
    ALL_MONTHS,
    BETA,
    ENHETER,
    REGIONS,
    TILTAK_DROPS,
    TREATMENT_START,
    write_enhet_mapping_json,
    write_indicator_long_csv,
    write_indicator_wide_csv,
    write_tiltak_csv,
)
from prep_data import (
    _add_time_features,
    _flatten_indicator_seasonally,
    build_treatment_variable,
    prepare_panel,
    prepare_triple_diff_panel,
)

# ── _add_time_features ──────────────────────────────────────────────────────────


def test_add_time_features_relative_month() -> None:
    """relative_month is 0 at treatment_start; negative before, positive after."""
    months = ["202212", "202301", "202302"]
    df = pd.DataFrame(
        {
            "aarmnd": months * len(REGIONS),
            "region": [r for r in REGIONS for _ in months],
            "indikator": 0.0,
        }
    )
    df = _add_time_features(df, TREATMENT_START)

    assert set(df.columns) >= {
        "relative_month",
        "post_treatment",
        "month_of_year",
        "year",
    }
    assert (
        df.loc[df["aarmnd"] == pd.Timestamp("2023-01-01"), "relative_month"].iloc[0]
        == 0
    )
    assert (
        df.loc[df["aarmnd"] == pd.Timestamp("2022-12-01"), "relative_month"].iloc[0]
        == -1
    )
    assert (
        df.loc[df["aarmnd"] == pd.Timestamp("2023-02-01"), "relative_month"].iloc[0]
        == 1
    )
    # post_treatment is True at relative_month >= 0
    assert df.loc[df["aarmnd"] == pd.Timestamp("2023-01-01"), "post_treatment"].iloc[0]
    assert not df.loc[
        df["aarmnd"] == pd.Timestamp("2022-12-01"), "post_treatment"
    ].iloc[0]


# ── build_treatment_variable ────────────────────────────────────────────────────


def _make_base_panel() -> pd.DataFrame:
    """Build a minimal panel DataFrame with tiltak and time features."""
    rows = []
    for m in ALL_MONTHS:
        p = pd.Period(m, freq="M")
        ts = p.to_timestamp()
        year = ts.year
        month = ts.month
        rel = (year - int(TREATMENT_START[:4])) * 12 + (
            month - int(TREATMENT_START[4:])
        )
        post = rel >= 0
        for i, r in enumerate(REGIONS):
            tiltak = 1000 * (1 - TILTAK_DROPS[i]) if post else 1000.0
            rows.append(
                {
                    "region": r,
                    "entity": r,
                    "aarmnd": ts,
                    "relative_month": rel,
                    "post_treatment": post,
                    "month_of_year": month,
                    "year": year,
                    "tiltak": tiltak,
                    "indikator": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_treatment_variable_continuous_peak() -> None:
    """Continuous tiltaksnedgang is 0 in the pre-period and correct in the post."""
    panel = _make_base_panel()
    panel = build_treatment_variable(
        panel, treatment_type="continuous", denominator="peak"
    )

    pre = panel[panel["relative_month"] < 0]
    post = panel[panel["relative_month"] >= 0]

    assert (pre["tiltaksnedgang"] == 0.0).all()
    # Region 0 drops 0% → tiltaksnedgang=0; region 4 drops 80% → tiltaksnedgang=0.8
    for i, r in enumerate(REGIONS):
        region_post = post[post["region"] == r]
        expected = TILTAK_DROPS[i]
        assert np.allclose(region_post["tiltaksnedgang"], expected, atol=1e-9), (
            f"Region {r}: expected tiltaksnedgang={expected}"
        )


def test_build_treatment_variable_discrete() -> None:
    """Discrete treatment: tiltaksnedgang=1 only for treated regions in post-period."""
    control_regions = [REGIONS[0], REGIONS[1]]
    panel = _make_base_panel()
    panel = build_treatment_variable(
        panel,
        treatment_type="discrete",
        control_regions=control_regions,
    )

    assert set(panel.columns) >= {"tiltaksnedgang", "treated"}
    # Control regions in post: tiltaksnedgang=0
    post_control = panel[
        panel["post_treatment"] & panel["region"].isin(control_regions)
    ]
    assert (post_control["tiltaksnedgang"] == 0.0).all()
    # Treated regions in post: tiltaksnedgang=1
    post_treated = panel[
        panel["post_treatment"] & ~panel["region"].isin(control_regions)
    ]
    assert (post_treated["tiltaksnedgang"] == 1.0).all()


# ── _flatten_indicator_seasonally ───────────────────────────────────────────────


def test_flatten_indicator_seasonally_removes_seasonal_variation() -> None:
    """After flattening, pre-period means are the same across months-of-year per entity."""
    panel = _make_base_panel()
    # Add a seasonal indicator
    panel["indikator"] = panel["month_of_year"].astype(float) * 2.0
    result = _flatten_indicator_seasonally(panel)

    # The seasonally flattened pre-period should have constant indikator per entity
    pre = result[result["relative_month"] < 0]
    for r in REGIONS:
        entity_pre = pre[pre["entity"] == r]["indikator"]
        assert entity_pre.std() < 1e-9, (
            f"Entity {r} still has seasonal variation after flattening"
        )


# ── prepare_panel ───────────────────────────────────────────────────────────────


def test_prepare_panel_region_level(tmp_path: Path) -> None:
    """prepare_panel returns correct columns, shape, and tiltaksnedgang values."""
    tiltak_p = tmp_path / "tiltak.csv"
    ind_p = tmp_path / "ind.csv"
    write_tiltak_csv(tiltak_p)
    write_indicator_wide_csv(ind_p)

    panel = prepare_panel(
        indicator_path=ind_p,
        tiltak_path=tiltak_p,
        indicator_name="indikator",
        treatment_start=TREATMENT_START,
        treatment_type="continuous",
        denominator="peak",
        flatten=False,
    )

    required_cols = {
        "region",
        "aarmnd",
        "indikator",
        "tiltak",
        "tiltaksnedgang",
        "relative_month",
        "post_treatment",
        "month_of_year",
        "year",
        "entity",
    }
    assert required_cols.issubset(set(panel.columns))
    assert len(panel) == len(ALL_MONTHS) * len(REGIONS)
    # tiltaksnedgang in [0, 1]
    assert panel["tiltaksnedgang"].between(0, 1).all()
    # entity == region for region-level analysis
    assert (panel["entity"] == panel["region"]).all()


def test_prepare_panel_enhet_level(tmp_path: Path) -> None:
    """prepare_panel with analysis_level='enhet' merges enhet mapping correctly."""
    tiltak_p = tmp_path / "tiltak.csv"
    ind_p = tmp_path / "ind_long.csv"
    mapping_p = tmp_path / "mapping.json"
    write_tiltak_csv(tiltak_p)
    write_indicator_long_csv(ind_p)
    write_enhet_mapping_json(mapping_p)

    panel = prepare_panel(
        indicator_path=ind_p,
        tiltak_path=tiltak_p,
        indicator_name="indikator",
        treatment_start=TREATMENT_START,
        treatment_type="continuous",
        denominator="peak",
        flatten=False,
        analysis_level="enhet",
        enhet_mapping_path=mapping_p,
    )

    n_enheter = sum(len(v) for v in ENHETER.values())
    assert len(panel) == len(ALL_MONTHS) * n_enheter
    assert "enhet" in panel.columns
    assert (panel["entity"] == panel["enhet"]).all()


# ── prepare_triple_diff_panel ───────────────────────────────────────────────────


def test_prepare_triple_diff_panel(tmp_path: Path) -> None:
    """Triple-diff panel has required columns and two distinct groups."""
    tiltak_p = tmp_path / "tiltak.csv"
    treated_p = tmp_path / "treated.csv"
    control_p = tmp_path / "control.csv"
    write_tiltak_csv(tiltak_p)
    write_indicator_wide_csv(treated_p, beta=BETA)
    write_indicator_wide_csv(control_p, beta=0.0)

    panel = prepare_triple_diff_panel(
        treated_indicator_path=treated_p,
        control_indicator_path=control_p,
        tiltak_path=tiltak_p,
        indicator_name="indikator",
        treatment_start=TREATMENT_START,
        treatment_type="continuous",
        denominator="last_pre",
        flatten=False,
    )

    required_cols = {
        "group",
        "treated",
        "tiltaksnedgang",
        "treatment_x_group",
        "region",
        "entity",
    }
    assert required_cols.issubset(set(panel.columns))
    assert set(panel["group"].unique()) == {"treated", "control"}
    # treatment_x_group = tiltaksnedgang × treated
    assert np.allclose(
        panel["treatment_x_group"],
        panel["tiltaksnedgang"] * panel["treated"],
        atol=1e-9,
    )
