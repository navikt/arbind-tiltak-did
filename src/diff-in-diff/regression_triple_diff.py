"""Triple-diff-specific panel regression helpers.

Builds on the shared types and utilities in :mod:`regression`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from regression import LeaveOneOutResult, RegressionResult, _compute_within_r2

logger = logging.getLogger(__name__)


# ── Design matrix ─────────────────────────────────────────────────────────────


def _build_triple_diff_regressors(panel: pd.DataFrame) -> pd.DataFrame:
    """Construct the design matrix for a triple-diff OLS model.

    Includes: ``treatment_x_group`` (main estimand), ``tiltaksnedgang``,
    ``treated`` (group indicator), entity FE, and year-month FE.

    Parameters
    ----------
    panel:
        Triple-diff panel from :func:`prep_data.prepare_triple_diff_panel`.
        Must have columns ``treatment_x_group``, ``tiltaksnedgang``,
        ``treated``, ``entity``, ``aarmnd``, and ``region``.
    """
    X = panel[["treatment_x_group", "tiltaksnedgang", "treated"]].copy()

    # Entity FE (region or enhet, depending on analysis level)
    entity_fe = pd.get_dummies(
        panel["entity"], prefix="e", drop_first=True, dtype=float
    )

    # Year-month FE
    yearmonth_fe = pd.get_dummies(
        panel["aarmnd"].astype(str), prefix="t", drop_first=True, dtype=float
    )

    X = pd.concat([X, entity_fe, yearmonth_fe], axis=1)
    X.insert(0, "const", 1.0)
    return X


# ── Estimation ────────────────────────────────────────────────────────────────


def _estimate_triple_diff(
    panel: pd.DataFrame,
    model_name: str,
) -> RegressionResult:
    """Fit a triple-diff OLS model with region-level clustered SE.

    The coefficient of interest is ``treatment_x_group``.
    """
    X = _build_triple_diff_regressors(panel)
    y = panel["indikator"].astype(float)
    clusters = panel["region"]

    # Drop rows with NaN in y or X to avoid silent NaN propagation in OLS
    valid = y.notna() & X.notna().all(axis=1)
    if not valid.all():
        n_drop = int((~valid).sum())
        logger.warning(
            "%s: dropping %d/%d rows with NaN before fitting.",
            model_name,
            n_drop,
            len(y),
        )
        y = y.loc[valid].reset_index(drop=True)
        X = X.loc[valid].reset_index(drop=True)
        clusters = clusters.loc[valid].reset_index(drop=True)

    logger.info(
        "Fitting %s (triple-diff): %d obs, %d regressors, %d clusters",
        model_name,
        len(y),
        X.shape[1],
        clusters.nunique(),
    )
    rank_x = int(np.linalg.matrix_rank(X.to_numpy(dtype=float)))
    if rank_x < X.shape[1]:
        raise ValueError(
            f"{model_name}: rank-deficient design matrix "
            f"(rank={rank_x}, columns={X.shape[1]}). "
            "Check for collinear fixed effects or insufficient variation."
        )

    cl_fit = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters.values},
        use_t=True,
    )

    param_names = list(X.columns)
    di_idx = param_names.index("treatment_x_group")

    coef = float(cl_fit.params.iloc[di_idx])
    se = float(cl_fit.bse.iloc[di_idx])
    tval = float(cl_fit.tvalues.iloc[di_idx])
    pval = float(cl_fit.pvalues.iloc[di_idx])
    ci_vals = cl_fit.conf_int()
    ci = ci_vals.iloc[di_idx]

    fe_labels = ["Entitet FE", "År-måned FE"]

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
        r_squared_adjusted=float(cl_fit.rsquared_adj),
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


# ── Public estimation API ─────────────────────────────────────────────────────


def run_triple_diff_baseline(panel: pd.DataFrame) -> RegressionResult:
    """Estimate the triple-diff model on the regular (non-flattened) panel."""
    return _estimate_triple_diff(panel, model_name="Trippel-diff basis")


def run_triple_diff_preferred(panel: pd.DataFrame) -> RegressionResult:
    """Estimate the triple-diff model on the seasonally flattened panel."""
    return _estimate_triple_diff(panel, model_name="Trippel-diff sesongjustert")


# ── Placebo test ──────────────────────────────────────────────────────────────


def run_triple_diff_placebo(
    panel: pd.DataFrame,
    placebo_relative_month: int = -12,
) -> RegressionResult | None:
    """Placebo test for the triple-diff model.

    Restricts to pre-period, constructs a fake treatment variable, and
    re-estimates the triple-diff specification.  A near-zero
    ``treatment_x_group`` coefficient supports the identifying assumption.

    Parameters
    ----------
    panel:
        Full triple-diff panel from :func:`prep_data.prepare_triple_diff_panel`.
    placebo_relative_month:
        Fake treatment start (must be < 0).  Default is -12.
    """
    if placebo_relative_month >= 0:
        raise ValueError("placebo_relative_month must be negative.")

    pre = panel[panel["relative_month"] < 0].copy()

    n_before = int((pre["relative_month"] < placebo_relative_month).sum())
    n_after = int((pre["relative_month"] >= placebo_relative_month).sum())
    if n_before == 0 or n_after == 0:
        logger.warning(
            "Triple-diff placebo at relative_month=%d: insufficient data — skipping.",
            placebo_relative_month,
        )
        return None

    pre["post_treatment"] = pre["relative_month"] >= placebo_relative_month

    ref_month = placebo_relative_month - 1
    ref_mask = pre["relative_month"] == ref_month
    if not ref_mask.any():
        logger.warning(
            "Triple-diff placebo: reference month %d not found — skipping.",
            ref_month,
        )
        return None

    # Build per-region reference from tiltak at (placebo_month - 1)
    ref = (
        pre.loc[ref_mask]
        .drop_duplicates(subset=["region"])[["region", "tiltak"]]
        .set_index("region")["tiltak"]
        .rename("ref_tiltak_placebo")
    )
    pre = pre.merge(ref, on="region", how="left")

    pre["tiltaksnedgang"] = 0.0
    post_mask = pre["post_treatment"]
    valid_ref = pre["ref_tiltak_placebo"] > 0
    pre.loc[post_mask & valid_ref, "tiltaksnedgang"] = np.clip(
        (
            pre.loc[post_mask & valid_ref, "ref_tiltak_placebo"]
            - pre.loc[post_mask & valid_ref, "tiltak"]
        )
        / pre.loc[post_mask & valid_ref, "ref_tiltak_placebo"],
        0.0,
        1.0,
    )
    pre = pre.drop(columns=["ref_tiltak_placebo"])
    pre["treatment_x_group"] = pre["tiltaksnedgang"] * pre["treated"]

    logger.info(
        "Triple-diff placebo at relative_month=%d: %d obs",
        placebo_relative_month,
        len(pre),
    )
    return _estimate_triple_diff(
        pre, model_name=f"Trippel-diff placebo (τ={placebo_relative_month})"
    )


# ── Leave-one-out robustness ──────────────────────────────────────────────────


def run_triple_diff_leave_one_out(
    panel: pd.DataFrame,
    preferred_result: RegressionResult,
) -> LeaveOneOutResult:
    """Re-estimate the triple-diff model leaving each region out in turn.

    Drops all observations for the region (both groups, all units).
    """
    regions = sorted(panel["region"].unique().tolist())
    records = []
    for region in regions:
        sub = panel[panel["region"] != region].copy()
        n_remaining = sub["region"].nunique()
        if n_remaining < 3:
            logger.warning(
                "Skipping triple-diff leave-out of %s: only %d clusters remain.",
                region,
                n_remaining,
            )
            continue
        try:
            res = _estimate_triple_diff(
                sub, model_name=f"Trippel-diff LOO drop {region}"
            )
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
        except ValueError, np.linalg.LinAlgError:
            logger.exception(
                "Triple-diff leave-one-out failed for dropped region %s", region
            )

    return LeaveOneOutResult(
        rows=pd.DataFrame(records),
        full_coefficient=preferred_result.coefficient,
        full_ci_lower=preferred_result.ci_lower,
        full_ci_upper=preferred_result.ci_upper,
    )
