"""Typed result containers for the DID analysis pipeline.

Replaces the loosely-typed ``dict[str, Any]`` bags previously passed between
:mod:`run_analysis` and :mod:`report`.  Using a dataclass makes the pipeline
contract explicit and enables IDE/type-checker support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from cluster_bootstrap import BootstrapResult
    from event_study import EventStudyResult
    from regression import LeaveOneOutResult, RegressionResult


@dataclass
class IndicatorResult:
    """All analysis outputs for a single indicator."""

    indicator_name: str

    # ── Regression results ────────────────────────────────────────────────────
    baseline: RegressionResult
    preferred: RegressionResult

    # ── Bootstrap results ─────────────────────────────────────────────────────
    bootstrap_baseline: BootstrapResult
    bootstrap_preferred: BootstrapResult

    # ── Event study ───────────────────────────────────────────────────────────
    event_study: EventStudyResult
    event_study_baseline: EventStudyResult

    # ── Placebo tests ─────────────────────────────────────────────────────────
    placebo: RegressionResult | None
    placebo_baseline: RegressionResult | None

    # ── Leave-one-out ─────────────────────────────────────────────────────────
    leave_one_out: LeaveOneOutResult
    leave_one_out_baseline: LeaveOneOutResult

    # ── Summary statistics ────────────────────────────────────────────────────
    mde: float
    baseline_mean: float
    baseline_mean_by_region: dict[str, float] = field(default_factory=dict)

    # ── Panels ────────────────────────────────────────────────────────────────
    #: Seasonally flattened panel (preferred model input).
    panel: pd.DataFrame = field(repr=False)
    #: Regular (non-flattened) panel (baseline model input).
    panel_regular: pd.DataFrame = field(repr=False)

    # ── Convenience ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict for backward-compatible callers."""
        return {
            "indicator_name": self.indicator_name,
            "baseline": self.baseline,
            "preferred": self.preferred,
            "bootstrap_baseline": self.bootstrap_baseline,
            "bootstrap_preferred": self.bootstrap_preferred,
            "event_study": self.event_study,
            "event_study_baseline": self.event_study_baseline,
            "placebo": self.placebo,
            "placebo_baseline": self.placebo_baseline,
            "leave_one_out": self.leave_one_out,
            "leave_one_out_baseline": self.leave_one_out_baseline,
            "mde": self.mde,
            "baseline_mean": self.baseline_mean,
            "baseline_mean_by_region": self.baseline_mean_by_region,
            "panel": self.panel,
            "panel_regular": self.panel_regular,
        }
