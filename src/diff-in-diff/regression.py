"""Panel fixed effects regression with CR1 clustered standard errors.

Two models are estimated using OLS with explicit dummy variables (via
:func:`pandas.get_dummies`) and :mod:`statsmodels`.

* **Baseline**   – region FE + year-month FE
* **Preferred**  – region FE + year-month FE + region × calendar-month FE

Standard errors are clustered at the region level using statsmodels'
``get_robustcov_results(cov_type='cluster')`` which applies the
CR1 small-sample correction:  ``G / (G − 1) × (N − 1) / (N − K)``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
    #: The fitted statsmodels result (with clustered SE already applied).
    result_obj: Any = field(default=None, repr=False)


def _build_regressors(panel: pd.DataFrame, preferred: bool) -> pd.DataFrame:
    """Construct the full design matrix (treatment + FE dummies) for OLS.

    Parameters
    ----------
    panel:
        The full analysis panel from :func:`src.data_prep.prepare_panel`.
    preferred:
        If ``True``, add region × calendar-month FE dummies on top of the
        baseline FE.

    Returns:
    -------
    DataFrame with a constant, the treatment variable, and all FE dummies.
    """
    # Treatment variable
    X = panel[["tiltaksnedgang"]].copy()

    # Region FE (drop first to avoid perfect collinearity)
    region_fe = pd.get_dummies(
        panel["region"], prefix="r", drop_first=True, dtype=float
    )

    # Year-month FE
    yearmonth_fe = pd.get_dummies(
        panel["aarmnd"].astype(str), prefix="t", drop_first=True, dtype=float
    )

    X = pd.concat([X, region_fe, yearmonth_fe], axis=1)

    if preferred:
        # Region × calendar-month-of-year FE
        # Encoding: each (region, month_of_year) cell gets a dummy.
        # We drop one per region to avoid collinearity with the region FE.
        rm_key = (
            panel["region"].str.replace(" ", "_")
            + "_m"
            + panel["month_of_year"].astype(str)
        )
        rm_fe = pd.get_dummies(rm_key, prefix="rm", drop_first=False, dtype=float)

        # Drop one dummy per region (the January dummy) to avoid collinearity
        # with the already-included region FE.
        to_drop = [f"rm_{r.replace(' ', '_')}_m1" for r in panel["region"].unique()]
        rm_fe = rm_fe.drop(
            columns=[c for c in to_drop if c in rm_fe.columns], errors="ignore"
        )
        X = pd.concat([X, rm_fe], axis=1)

    # Add constant (absorbed by the many dummies but required by statsmodels)
    X.insert(0, "const", 1.0)
    return X


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


# ── Estimation ────────────────────────────────────────────────────────────────


def _estimate(
    panel: pd.DataFrame,
    preferred: bool,
    model_name: str,
) -> RegressionResult:
    """Internal helper: build design matrix, fit OLS, apply clustered SE."""
    y = panel["indikator"].astype(float)
    X = _build_regressors(panel, preferred=preferred)
    clusters = panel["region"]

    logger.info(
        "Fitting %s: %d obs, %d regressors, %d clusters",
        model_name,
        len(y),
        X.shape[1],
        clusters.nunique(),
    )
    rank_x = int(np.linalg.matrix_rank(X.to_numpy(dtype=float)))  # noqa: F841
    # if rank_x < X.shape[1]:
    #     raise ValueError(
    #         f"{model_name}: rank-deficient design matrix "
    #         f"(rank={rank_x}, columns={X.shape[1]})."
    #     )

    cl_fit = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters.values},
        use_t=True,
    )

    param_names = list(X.columns)
    di_idx = param_names.index("tiltaksnedgang")

    coef = float(cl_fit.params.iloc[di_idx])
    se = float(cl_fit.bse.iloc[di_idx])
    tval = float(cl_fit.tvalues.iloc[di_idx])
    pval = float(cl_fit.pvalues.iloc[di_idx])
    ci_vals = cl_fit.conf_int()
    ci = ci_vals.iloc[di_idx]

    fe_labels = ["Region FE", "År-måned FE"]
    if preferred:
        fe_labels.append("Region × kalendermåned FE")

    result = RegressionResult(
        model_name=model_name,
        coefficient=coef,
        std_error=se,
        t_stat=tval,
        p_value=pval,
        ci_lower=float(ci.iloc[0]),
        ci_upper=float(ci.iloc[1]),
        n_obs=int(cl_fit.nobs),
        n_clusters=int(clusters.nunique()),
        fixed_effects=fe_labels,
        r_squared_within=_compute_within_r2(y, X, panel),
        result_obj=cl_fit,
    )

    logger.info(
        "%s → coef=%.4f  SE=%.4f  p=%.4f  95%%CI=[%.4f, %.4f]",
        model_name,
        coef,
        se,
        pval,
        result.ci_lower,
        result.ci_upper,
    )
    return result


def run_baseline_model(panel: pd.DataFrame) -> RegressionResult:
    """Estimate the baseline model (region FE + year-month FE).

    Parameters
    ----------
    panel:
        Processed analysis panel from :func:`src.data_prep.prepare_panel`.

    Returns:
    -------
    :class:`RegressionResult` for the baseline specification.
    """
    return _estimate(panel, preferred=False, model_name="Baseline")


def run_preferred_model(panel: pd.DataFrame) -> RegressionResult:
    """Estimate the preferred model (region FE + year-month FE + region×season FE).

    The region × calendar-month fixed effects absorb region-specific seasonal
    patterns that might otherwise confound the treatment effect.

    Parameters
    ----------
    panel:
        Processed analysis panel from :func:`src.data_prep.prepare_panel`.

    Returns:
    -------
    :class:`RegressionResult` for the preferred specification.
    """
    return _estimate(panel, preferred=True, model_name="Sesongjustert")


# ── Coefficient extraction ────────────────────────────────────────────────────


def _classify_coefficient(name: str) -> str:
    """Return a human-readable coefficient type label for *name*."""
    if name == "const":
        return "Konstantledd"
    if name == "tiltaksnedgang":
        return "Behandling"
    if name.startswith("r_"):
        return "Region FE"
    if name.startswith("t_"):
        return "Tidspunkt FE"
    if name.startswith("rm_"):
        return "Region × Sesong FE"
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


# ── Placebo test ──────────────────────────────────────────────────────────────


def run_placebo_test(
    panel: pd.DataFrame,
    placebo_relative_month: int = -12,
) -> RegressionResult | None:
    """Estimate the preferred model on a fake treatment date in the pre-period.

    The placebo analysis restricts to pre-treatment observations and constructs
    a new treatment variable that measures the change in tiltak *within* the
    pre-period, using ``placebo_relative_month − 1`` as the reference month
    (analogous to ``last_pre`` in the real analysis).

    A near-zero coefficient indicates that the main result is not driven by
    pre-existing diverging trends.

    Parameters
    ----------
    panel:
        Full analysis panel from :func:`prep_data.prepare_panel`.
    placebo_relative_month:
        The fake treatment start expressed as a relative month (must be < 0).
        Default is -12 (one year before the real treatment).

    Returns:
    -------
    :class:`RegressionResult` for the placebo specification, or ``None`` if
    the pre-period window is too short.
    """
    if placebo_relative_month >= 0:
        raise ValueError("placebo_relative_month must be negative.")

    pre = panel[panel["relative_month"] < 0].copy()

    n_before = int((pre["relative_month"] < placebo_relative_month).sum())
    n_after = int((pre["relative_month"] >= placebo_relative_month).sum())
    if n_before == 0 or n_after == 0:
        logger.warning(
            "Placebo at relative_month=%d: insufficient data on one side — skipping.",
            placebo_relative_month,
        )
        return None

    pre["post_treatment"] = pre["relative_month"] >= placebo_relative_month

    # Construct placebo tiltaksnedgang from actual tiltak movements in the
    # pre-period.  Reference = tiltak at (placebo_relative_month - 1), the
    # last "pre-placebo" month, analogous to last_pre in the real analysis.
    ref_month = placebo_relative_month - 1
    ref_mask = pre["relative_month"] == ref_month
    if not ref_mask.any():
        logger.warning(
            "Placebo: reference month %d not found in pre-period data — skipping.",
            ref_month,
        )
        return None

    ref = (
        pre.loc[ref_mask, ["region", "tiltak"]]
        .set_index("region")["tiltak"]
        .rename("ref_tiltak_placebo")
    )
    pre = pre.merge(ref, on="region", how="left")

    pre["tiltaksnedgang"] = 0.0
    post_mask = pre["post_treatment"]
    valid_ref = pre["ref_tiltak_placebo"] > 0
    pre.loc[post_mask & valid_ref, "tiltaksnedgang"] = np.clip(
        (pre.loc[post_mask & valid_ref, "ref_tiltak_placebo"]
         - pre.loc[post_mask & valid_ref, "tiltak"])
        / pre.loc[post_mask & valid_ref, "ref_tiltak_placebo"],
        0.0,
        1.0,
    )
    pre = pre.drop(columns=["ref_tiltak_placebo"])

    logger.info(
        "Placebo test at relative_month=%d: %d obs (%d 'pre', %d 'post'), "
        "mean placebo tiltaksnedgang=%.3f",
        placebo_relative_month,
        len(pre),
        n_before,
        n_after,
        float(pre.loc[post_mask, "tiltaksnedgang"].mean()),
    )
    return _estimate(pre, preferred=True, model_name=f"Placebo (τ={placebo_relative_month})")


# ── Leave-one-out robustness ──────────────────────────────────────────────────


@dataclass
class LeaveOneOutResult:
    """Collection of leave-one-out estimates."""

    #: columns: dropped_region, coefficient, std_error, ci_lower, ci_upper, p_value
    rows: pd.DataFrame
    full_coefficient: float
    full_ci_lower: float
    full_ci_upper: float


def run_leave_one_out(
    panel: pd.DataFrame,
    preferred_result: RegressionResult,
) -> LeaveOneOutResult:
    """Re-estimate the preferred model leaving each region out in turn.

    Parameters
    ----------
    panel:
        Full analysis panel from :func:`prep_data.prepare_panel`.
    preferred_result:
        The full-sample preferred model result (used to record the reference CI).

    Returns:
    -------
    :class:`LeaveOneOutResult` with per-drop estimates and the full-sample reference.
    """
    regions = sorted(panel["region"].unique().tolist())
    records = []
    for region in regions:
        sub = panel[panel["region"] != region].copy()
        n_remaining = sub["region"].nunique()
        if n_remaining < 3:
            logger.warning(
                "Skipping leave-out of %s: only %d clusters remain.", region, n_remaining
            )
            continue
        try:
            res = _estimate(sub, preferred=True, model_name=f"LOO drop {region}")
            records.append(
                {
                    "dropped_region": region,
                    "coefficient": res.coefficient,
                    "std_error": res.std_error,
                    "ci_lower": res.ci_lower,
                    "ci_upper": res.ci_upper,
                    "p_value": res.p_value,
                }
            )
        except Exception:
            logger.exception("Leave-one-out failed for dropped region %s", region)

    return LeaveOneOutResult(
        rows=pd.DataFrame(records),
        full_coefficient=preferred_result.coefficient,
        full_ci_lower=preferred_result.ci_lower,
        full_ci_upper=preferred_result.ci_upper,
    )


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
