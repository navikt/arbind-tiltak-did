"""Tests for src/diff-in-diff/regression_did.py."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import BETA, REGIONS
from regression import RegressionResult
from regression_did import (
    run_baseline_model,
    run_leave_one_out,
    run_placebo_test,
    run_preferred_model,
)

# Tolerance for coefficient recovery: no noise in DGP so OLS should recover
# beta closely, but clustered SE can cause some deviation.
COEFF_TOL = 0.2


def test_run_baseline_model_coefficient(synthetic_panel: pd.DataFrame) -> None:
    """Baseline model recovers the known beta coefficient."""
    result = run_baseline_model(synthetic_panel)

    assert isinstance(result, RegressionResult)
    assert abs(result.coefficient - BETA) < COEFF_TOL, (
        f"Expected coeff ≈ {BETA}, got {result.coefficient:.4f}"
    )
    assert result.n_obs == len(synthetic_panel)
    assert result.n_clusters == len(REGIONS)


def test_run_preferred_model_coefficient(synthetic_panel_flat: pd.DataFrame) -> None:
    """Preferred (seasonally flattened) model recovers the known beta coefficient."""
    result = run_preferred_model(synthetic_panel_flat)

    assert isinstance(result, RegressionResult)
    assert abs(result.coefficient - BETA) < COEFF_TOL, (
        f"Expected coeff ≈ {BETA}, got {result.coefficient:.4f}"
    )


def test_run_placebo_test_near_zero(synthetic_panel: pd.DataFrame) -> None:
    """Placebo test coefficient is near zero (no pre-trend in the DGP)."""
    result = run_placebo_test(synthetic_panel, placebo_relative_month=-12)

    # The placebo may return None if the window is too short; skip in that case
    if result is None:
        pytest.skip("Placebo returned None (window too short)")

    assert isinstance(result, RegressionResult)
    # In our DGP tiltaksnedgang is exactly 0 in the pre-period, so the placebo
    # treatment has zero variation → coefficient should be 0 or near-zero.
    assert abs(result.coefficient) < COEFF_TOL, (
        f"Placebo coefficient too large: {result.coefficient:.4f}"
    )


def test_run_leave_one_out_completeness(
    synthetic_panel: pd.DataFrame,
    synthetic_panel_flat: pd.DataFrame,
) -> None:
    """LOO produces one row per droppable region and records full-sample reference."""
    preferred = run_preferred_model(synthetic_panel_flat)
    loo = run_leave_one_out(synthetic_panel_flat, preferred)

    # With 5 regions and a minimum of 3 remaining, all 5 should be droppable
    # (dropping any one leaves 4 clusters which is ≥ 3).
    assert len(loo.rows) == len(REGIONS)
    assert set(loo.rows["dropped_region"]) == set(REGIONS)
    assert loo.full_coefficient == preferred.coefficient
