"""DiD-specific panel regression helpers.

Builds on the shared types and utilities in :mod:`regression`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from regression import LeaveOneOutResult, RegressionResult, _compute_within_r2

logger = logging.getLogger(__name__)


# ── Design matrix ────────────────────────────────────────────────────────────


def build_regressors(panel: pd.DataFrame) -> pd.DataFrame:
    """Construct the full design matrix (treatment + FE dummies) for OLS.

    Parameters
    ----------
    panel:
        The full analysis panel from :func:`src.data_prep.prepare_panel`.

    Returns:
    -------
    DataFrame with a constant, the treatment variable, region FE dummies, and
    year-month FE dummies.
    """
    # Treatment variable
    X = panel[["tiltaksnedgang"]].copy()

    # Entity FE (region or enhet depending on analysis_level — drop first to avoid collinearity)
    entity_fe = pd.get_dummies(
        panel["entity"], prefix="e", drop_first=True, dtype=float
    )

    # Year-month FE
    yearmonth_fe = pd.get_dummies(
        panel["aarmnd"].astype(str), prefix="t", drop_first=True, dtype=float
    )

    X = pd.concat([X, entity_fe, yearmonth_fe], axis=1)

    # Add constant (absorbed by the many dummies but required by statsmodels)
    X.insert(0, "const", 1.0)
    return X


# ── Estimation ────────────────────────────────────────────────────────────────


def _estimate(
    panel: pd.DataFrame,
    model_name: str,
) -> RegressionResult:
    """Internal helper: build design matrix, fit OLS, apply clustered SE."""
    y = panel["indikator"].astype(float)
    X = build_regressors(panel)
    clusters = panel["region"]

    # Drop rows with NaN in y or X to avoid silent NaN propagation in OLS.
    # This can happen with enhet-level panels where some enheter have missing
    # indicator values for certain months.
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

    n_entities = panel["entity"].nunique()
    n_regions = panel["region"].nunique()
    entity_label = "Enhet FE" if n_entities > n_regions else "Region FE"

    logger.info(
        "Fitting %s: %d obs, %d regressors, %d clusters",
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
            "Check for collinear fixed effects or insufficient variation in treatment."
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

    fe_labels = [entity_label, "År-måned FE"]

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


def run_baseline_model(panel: pd.DataFrame) -> RegressionResult:
    """Estimate the model on the regular (non-flattened) panel.

    Uses region FE + year-month FE.

    Parameters
    ----------
    panel:
        Regular (non-flattened) analysis panel from :func:`src.data_prep.prepare_panel`.

    Returns:
    -------
    :class:`RegressionResult` for the baseline specification.
    """
    return _estimate(panel, model_name="Basis")


def run_preferred_model(panel: pd.DataFrame) -> RegressionResult:
    """Estimate the model on the seasonally flattened panel.

    Uses region FE + year-month FE.  The pre-treatment seasonal pattern is
    removed from the indicator in data preparation, so the regression only
    needs region and time fixed effects.

    Parameters
    ----------
    panel:
        Seasonally flattened analysis panel from :func:`src.data_prep.prepare_panel`
        (``flatten=True``).

    Returns:
    -------
    :class:`RegressionResult` for the preferred specification.
    """
    return _estimate(panel, model_name="Sesongjustert")


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
        (
            pre.loc[post_mask & valid_ref, "ref_tiltak_placebo"]
            - pre.loc[post_mask & valid_ref, "tiltak"]
        )
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
    return _estimate(pre, model_name=f"Placebo (τ={placebo_relative_month})")


# ── Leave-one-out robustness ──────────────────────────────────────────────────


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
                "Skipping leave-out of %s: only %d clusters remain.",
                region,
                n_remaining,
            )
            continue
        try:
            res = _estimate(sub, model_name=f"LOO drop {region}")
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
            logger.exception("Leave-one-out failed for dropped region %s", region)

    return LeaveOneOutResult(
        rows=pd.DataFrame(records),
        full_coefficient=preferred_result.coefficient,
        full_ci_lower=preferred_result.ci_lower,
        full_ci_upper=preferred_result.ci_upper,
    )
