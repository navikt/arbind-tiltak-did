"""Shared regression types, utilities, and summary helpers.

Design-specific estimation code lives in:
  - :mod:`regression_did` — DiD baseline/preferred/placebo/leave-one-out
  - :mod:`regression_triple_diff` — triple-diff variants of the same

Standard errors are clustered at the region level using statsmodels'
``get_robustcov_results(cov_type='cluster')`` which applies the
CR1 small-sample correction:  ``G / (G − 1) × (N − 1) / (N − K)``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class RegressionResult:
    """Summary of a single panel regression."""

    model_name: str
    coefficient: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_obs: int
    n_clusters: int
    fixed_effects: list[str] = field(default_factory=list)
    r_squared_within: float = float("nan")
    r_squared_adjusted: float = float("nan")
    #: The fitted statsmodels result (with clustered SE already applied).
    result_obj: Any = field(default=None, repr=False)


def _compute_within_r2(y: pd.Series, X: pd.DataFrame, panel: pd.DataFrame) -> float:
    """Return FE-consistent within-R² using the fitted transformed equation."""
    fe_cols = [c for c in X.columns if c != "const" and c != "tiltaksnedgang"]
    y_arr = y.astype(float).to_numpy()
    x_arr = X["tiltaksnedgang"].astype(float).to_numpy()

    if not fe_cols:
        y_centered = y_arr - y_arr.mean()
        x_centered = x_arr - x_arr.mean()
    else:
        Z = X[fe_cols].to_numpy(dtype=float)
        z_coef_y, *_ = np.linalg.lstsq(Z, y_arr, rcond=None)
        z_coef_x, *_ = np.linalg.lstsq(Z, x_arr, rcond=None)
        y_centered = y_arr - Z @ z_coef_y
        x_centered = x_arr - Z @ z_coef_x

    denom = float(y_centered @ y_centered)
    if denom <= 1e-12:
        return float("nan")
    q = float(x_centered @ x_centered)
    if q <= 1e-12:
        return float("nan")
    beta = float(x_centered @ y_centered) / q
    resid = y_centered - beta * x_centered
    sse = float(resid @ resid)
    return float(max(0.0, 1.0 - sse / denom))


# ── Coefficient extraction ────────────────────────────────────────────────────


def _classify_coefficient(name: str) -> str:
    """Return a human-readable coefficient type label for *name*."""
    if name == "const":
        return "Konstantledd"
    if name == "treatment_x_group":
        return "Behandling × Gruppe"
    if name == "tiltaksnedgang":
        return "Behandling"
    if name == "treated":
        return "Gruppe"
    if name.startswith("r_"):
        return "Region FE"
    if name.startswith("e_"):
        return "Entitet FE"
    if name.startswith("t_"):
        return "Tidspunkt FE"
    return "Annet"


def extract_all_coefficients(result: RegressionResult) -> pd.DataFrame:
    """Return a tidy DataFrame with every coefficient from *result*.

    Each row contains the coefficient name, its type (region FE, time FE, etc.),
    and the standard regression statistics (estimate, SE, t-stat, p-value, CI).

    Parameters
    ----------
    result:
        A :class:`RegressionResult` with a non-``None`` ``result_obj``.

    Returns:
    -------
    DataFrame with columns:
        ``koeffisient_navn``, ``koeffisient_type``, ``modell``,
        ``estimat``, ``std_feil``, ``t_stat``, ``p_verdi``,
        ``ki_nedre``, ``ki_ovre``.
    """
    fit = result.result_obj
    ci = fit.conf_int()
    rows = []
    for name in fit.params.index:
        rows.append(
            {
                "koeffisient_navn": name,
                "koeffisient_type": _classify_coefficient(name),
                "modell": result.model_name,
                "estimat": float(fit.params[name]),
                "std_feil": float(fit.bse[name]),
                "t_stat": float(fit.tvalues[name]),
                "p_verdi": float(fit.pvalues[name]),
                "ki_nedre": float(ci.loc[name, 0]),
                "ki_ovre": float(ci.loc[name, 1]),
            }
        )
    return pd.DataFrame(rows)


# ── Summary table ─────────────────────────────────────────────────────────────


def format_results_table(
    baseline: RegressionResult,
    preferred: RegressionResult,
) -> pd.DataFrame:
    """Return a DataFrame summarising both regression models.

    The table contains one row per model with key statistics.  It is intended
    to be saved as a CSV and referenced in the markdown report.

    Parameters
    ----------
    baseline:
        Result from :func:`run_baseline_model`.
    preferred:
        Result from :func:`run_preferred_model`.

    Returns:
    -------
    DataFrame with columns for the coefficient, SE, t-stat, p-value, CI,
    observations, clusters, and fixed effects.
    """
    rows = []
    for res in (baseline, preferred):
        rows.append(
            {
                "Modell": res.model_name,
                "Koeffisient": round(res.coefficient, 4),
                "Std.feil (CR1)": round(res.std_error, 4),
                "t-statistikk": round(res.t_stat, 3),
                "p-verdi": round(res.p_value, 4),
                "95% KI nedre": round(res.ci_lower, 4),
                "95% KI øvre": round(res.ci_upper, 4),
                "Observasjoner": res.n_obs,
                "Clustere (regioner)": res.n_clusters,
                "Faste effekter": "; ".join(res.fixed_effects),
            }
        )
    return pd.DataFrame(rows)


# ── Leave-one-out result ──────────────────────────────────────────────────────


@dataclass
class LeaveOneOutResult:
    """Collection of leave-one-out estimates."""

    #: columns: dropped_region, coefficient, std_error, ci_lower, ci_upper, p_value
    rows: pd.DataFrame
    full_coefficient: float
    full_ci_lower: float
    full_ci_upper: float


# ── Minimum detectable effect ─────────────────────────────────────────────────


def compute_mde(
    preferred_result: RegressionResult,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Compute the minimum detectable effect (MDE) from the preferred model SE.

    Uses the standard closed-form formula with t-distribution:
        MDE = (t_{α/2, G-1} + t_{β, G-1}) × SE

    Parameters
    ----------
    preferred_result:
        Result from the preferred model; its SE is used.
    alpha:
        Two-sided significance level (default 0.05).
    power:
        Desired statistical power (default 0.80).

    Returns:
    -------
    MDE in the same units as the treatment coefficient (percentage points).
    """
    df = preferred_result.n_clusters - 1
    t_alpha = float(scipy_stats.t.ppf(1 - alpha / 2, df=df))
    t_beta = float(scipy_stats.t.ppf(power, df=df))
    return (t_alpha + t_beta) * preferred_result.std_error
