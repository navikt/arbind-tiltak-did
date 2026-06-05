"""Tests for src/diff-in-diff/regression_triple_diff.py."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import BETA, REGIONS
from regression import RegressionResult
from regression_triple_diff import (
    run_triple_diff_baseline,
    run_triple_diff_leave_one_out,
    run_triple_diff_placebo,
    run_triple_diff_preferred,
)

# Tolerance for coefficient recovery
COEFF_TOL = 0.2


def test_run_triple_diff_baseline_coefficient(
    synthetic_triple_diff_panel: pd.DataFrame,
) -> None:
    """Baseline triple-diff model recovers treatment_x_group ≈ beta."""
    result = run_triple_diff_baseline(synthetic_triple_diff_panel)

    assert isinstance(result, RegressionResult)
    assert abs(result.coefficient - BETA) < COEFF_TOL, (
        f"Expected treatment_x_group ≈ {BETA}, got {result.coefficient:.4f}"
    )
    # Two groups × regions × months
    assert result.n_clusters == len(REGIONS)


def test_run_triple_diff_preferred_coefficient(
    synthetic_triple_diff_panel_flat: pd.DataFrame,
) -> None:
    """Preferred (flattened) triple-diff model recovers treatment_x_group ≈ beta."""
    result = run_triple_diff_preferred(synthetic_triple_diff_panel_flat)

    assert isinstance(result, RegressionResult)
    assert abs(result.coefficient - BETA) < COEFF_TOL, (
        f"Expected treatment_x_group ≈ {BETA}, got {result.coefficient:.4f}"
    )


def test_run_triple_diff_placebo_near_zero(
    synthetic_triple_diff_panel: pd.DataFrame,
) -> None:
    """Triple-diff placebo coefficient is near zero (no pre-trend in DGP)."""
    result = run_triple_diff_placebo(
        synthetic_triple_diff_panel, placebo_relative_month=-12
    )

    if result is None:
        pytest.skip("Placebo returned None (window too short)")

    assert isinstance(result, RegressionResult)
    assert abs(result.coefficient) < COEFF_TOL, (
        f"Placebo coefficient too large: {result.coefficient:.4f}"
    )


def test_run_triple_diff_leave_one_out_completeness(
    synthetic_triple_diff_panel: pd.DataFrame,
    synthetic_triple_diff_panel_flat: pd.DataFrame,
) -> None:
    """Triple-diff LOO produces one row per droppable region."""
    preferred = run_triple_diff_preferred(synthetic_triple_diff_panel_flat)
    loo = run_triple_diff_leave_one_out(synthetic_triple_diff_panel_flat, preferred)

    assert len(loo.rows) == len(REGIONS)
    assert set(loo.rows["dropped_region"]) == set(REGIONS)
    assert loo.full_coefficient == preferred.coefficient
