"""
Panel fixed effects regression with CR1 clustered standard errors.

Two models are estimated using OLS with explicit dummy variables (via
:func:`pandas.get_dummies`) and :mod:`statsmodels`.

* **Baseline**   – region FE + year-month FE
* **Preferred**  – region FE + year-month FE + region × calendar-month FE

Standard errors are clustered at the region level using statsmodels'
``get_robustcov_results(cov_type='cluster')`` which applies the
CR1 small-sample correction:  ``G / (G − 1) × (N − 1) / (N − K)``.
"""

import logging
from typing import Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
    """Return within-R² (variation explained after demeaning by region mean)."""
    region_means = y.groupby(panel["region"].values).transform("mean")
    y_within = y - region_means
    x_within = X["tiltaksnedgang"] - X["tiltaksnedgang"].groupby(
        panel["region"].values
    ).transform("mean")
    corr = np.corrcoef(y_within, x_within)[0, 1]
    return float(corr**2)


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

    Returns
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
