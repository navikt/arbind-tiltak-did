"""Event study analysis for the arbeidsmarkedstiltak DID.

The event study interacts period indicators with a time-invariant, office-level
measure of tiltaksnedgang intensity.  This serves two purposes:

1. **Pre-trend test** – coefficients in the months before May 2025 should be
   near zero if the parallel-trends assumption holds.
2. **Dynamic effects** – post-treatment coefficients reveal whether the impact
   is immediate, builds over time, or fades.

Specification
-------------
Office-level intensity ``s_i`` is defined as the maximum tiltaksnedgang intensity
achieved by region *i* over the post-treatment period.

For each event-time ``τ`` in the window ``[−24, +4]`` (omitting ``τ = −1`` as
the base period):

    Y_{it} = Σ_τ β_τ · s_i · 1{relative_period_t = τ} + α_i + γ_t + ε_{it}

where ``α_i`` are region FE and ``γ_t`` are year-month FE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Event-study window (inclusive, months relative to treatment start)
ES_WINDOW_PRE: int = -24
ES_WINDOW_POST: int = 4
BASE_PERIOD: int = -1  # omitted as reference period


# ── Result containers ──────────────────────────────────────────────────────────


@dataclass
class EventStudyCoef:
    """Point estimate and uncertainty for a single event-time coefficient."""

    tau: int
    coefficient: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float


@dataclass
class EventStudyResult:
    """Full event study output."""

    coefs: list[EventStudyCoef]
    pretrend_f_stat: float
    pretrend_p_value: float
    pretrend_df_num: int
    pretrend_df_denom: int
    #: Time-invariant intensity score per region used as the continuous treatment.
    region_intensity: pd.Series | None = field(default=None, repr=False)


# ── Helpers ────────────────────────────────────────────────────────────────────


def compute_region_intensity(panel: pd.DataFrame) -> pd.Series:
    """Compute a time-invariant tiltaksnedgang intensity score per region.

    Defined as the **maximum** tiltaksnedgang fraction observed for each
    region over the post-treatment period.  Regions with no post-treatment
    data receive intensity 0.

    Parameters
    ----------
    panel:
        Analysis panel with columns ``region``, ``tiltaksnedgang``, and
        ``post_treatment``.

    Returns:
    -------
    Series indexed by region name.
    """
    post = panel[panel["post_treatment"]]
    intensity = (
        post.groupby("region")["tiltaksnedgang"]
        .max()
        .reindex(panel["region"].unique(), fill_value=0.0)
    )
    logger.info(
        "Region intensity — mean: %.3f  max: %.3f  zero-intensity regions: %d",
        float(intensity.mean()),
        float(intensity.max()),
        int((intensity == 0).sum()),
    )
    return intensity


def _build_event_study_regressors(
    panel: pd.DataFrame,
    intensity: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the design matrix for the event study regression.

    Parameters
    ----------
    panel:
        Analysis panel (output of :func:`prep_data.prepare_panel`).
    intensity:
        Time-invariant intensity score per region from
        :func:`compute_region_intensity`.

    Returns:
    -------
    Tuple of (design matrix X, list of event-study interaction column names).
    """
    panel = panel.copy()
    panel["_intensity"] = panel["region"].map(intensity)

    all_taus = [t for t in range(ES_WINDOW_PRE, ES_WINDOW_POST + 1) if t != BASE_PERIOD]
    observed_taus = set(panel["relative_month"].astype(int).unique().tolist())
    taus = [t for t in all_taus if t in observed_taus]
    if not taus:
        raise ValueError(
            "Event-study window has no overlap with observed relative_month values."
        )
    es_cols: dict[str, pd.Series] = {}
    for tau in taus:
        col = f"es_tau_{tau:+d}".replace("+", "p").replace("-", "m")
        es_cols[col] = (panel["relative_month"] == tau).astype(float) * panel[
            "_intensity"
        ]

    X_es = pd.DataFrame(es_cols, index=panel.index)

    # Region FE (drop first to avoid perfect collinearity)
    region_fe = pd.get_dummies(
        panel["region"], prefix="r", drop_first=True, dtype=float
    )
    # Year-month FE (drop first)
    yearmonth_fe = pd.get_dummies(
        panel["aarmnd"].astype(str), prefix="t", drop_first=True, dtype=float
    )

    X = pd.concat([X_es, region_fe, yearmonth_fe], axis=1)
    # Drop structurally zero columns to reduce avoidable rank deficiency.
    nonzero_cols = X.columns[(X != 0.0).any(axis=0)]
    X = X.loc[:, nonzero_cols]
    es_cols_kept = [c for c in es_cols.keys() if c in X.columns]
    if not es_cols_kept:
        raise ValueError(
            "No identified event-study interaction columns after removing empty columns."
        )
    X.insert(0, "const", 1.0)
    return X, es_cols_kept


# ── Estimation ────────────────────────────────────────────────────────────────


def run_event_study(panel: pd.DataFrame) -> EventStudyResult:
    """Estimate event-study coefficients with CR1 clustered standard errors.

    Uses OLS with region and year-month fixed effects.  Standard errors are
    clustered at the region level.  The reference period is ``τ = −1``.

    Parameters
    ----------
    panel:
        Analysis panel from :func:`prep_data.prepare_panel`.

    Returns:
    -------
    :class:`EventStudyResult` with per-period coefficients and the joint
    pre-trend Wald F-test.
    """
    intensity = compute_region_intensity(panel)
    X, es_col_names = _build_event_study_regressors(panel, intensity)
    y = panel["indikator"].astype(float)
    clusters = panel["region"]

    logger.info(
        "Fitting event study: %d obs, %d regressors (%d ES interactions), %d clusters",
        len(y),
        X.shape[1],
        len(es_col_names),
        clusters.nunique(),
    )

    cl_fit = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters.values},
        use_t=True,
    )

    taus = [
        int(col.split("_")[-1].replace("p", "+").replace("m", "-"))
        for col in es_col_names
    ]
    param_names = list(X.columns)
    ci_vals = cl_fit.conf_int()
    coefs: list[EventStudyCoef] = []
    for tau, col in zip(taus, es_col_names):
        idx = param_names.index(col)
        coefs.append(
            EventStudyCoef(
                tau=tau,
                coefficient=float(cl_fit.params.iloc[idx]),
                std_error=float(cl_fit.bse.iloc[idx]),
                ci_lower=float(ci_vals.iloc[idx, 0]),
                ci_upper=float(ci_vals.iloc[idx, 1]),
                p_value=float(cl_fit.pvalues.iloc[idx]),
            )
        )

    # Joint pre-trend Wald F-test: β_τ = 0 for all τ ∈ [−24, −2]
    pre_cols = [col for col, tau in zip(es_col_names, taus) if tau < BASE_PERIOD]
    pretrend = _joint_pretrend_test(
        cl_fit, pre_cols, n_clusters=int(clusters.nunique())
    )

    logger.info(
        "Joint pre-trend test: F(%d, %d) = %.3f  p = %.4f",
        pretrend["df_num"],
        pretrend["df_denom"],
        pretrend["f_stat"],
        pretrend["p_value"],
    )

    return EventStudyResult(
        coefs=coefs,
        pretrend_f_stat=float(pretrend["f_stat"]),
        pretrend_p_value=float(pretrend["p_value"]),
        pretrend_df_num=int(pretrend["df_num"]),
        pretrend_df_denom=int(pretrend["df_denom"]),
        region_intensity=intensity,
    )


def _joint_pretrend_test(
    cl_fit: Any,
    pre_cols: list[str],
    n_clusters: int,
) -> dict[str, float]:
    """Wald F-test that all pre-period event-study coefficients are jointly zero.

    Uses the cluster-robust covariance matrix.  The denominator degrees of
    freedom are ``G − 1`` (clusters minus one), the standard choice for
    cluster-robust inference.  When the number of pre-period lags exceeds
    ``G − 1``, the covariance matrix is rank-deficient and a pseudoinverse
    is used with the effective rank as the numerator degrees of freedom.

    Parameters
    ----------
    cl_fit:
        Statsmodels result object with clustered SE applied.
    pre_cols:
        Names of pre-period interaction columns to jointly test.
    n_clusters:
        Number of clusters (regions).

    Returns:
    -------
    Dict with keys ``f_stat``, ``p_value``, ``df_num``, ``df_denom``.
    """
    k = len(pre_cols)
    if k == 0:
        return {
            "f_stat": float("nan"),
            "p_value": float("nan"),
            "df_num": 0,
            "df_denom": 0,
        }

    param_names = list(cl_fit.params.index)
    pre_idxs = [param_names.index(c) for c in pre_cols]

    beta = np.array([float(cl_fit.params.iloc[i]) for i in pre_idxs])
    cov_arr = np.asarray(cl_fit.cov_params())
    V = cov_arr[np.ix_(pre_idxs, pre_idxs)]

    # Handle potential rank deficiency (common when k > G − 1)
    eigvals, eigvecs = np.linalg.eigh(V)
    tol = max(1e-10, float(np.abs(eigvals).max()) * k * np.finfo(float).eps * 1e4)
    pos_mask = eigvals > tol
    rank_V = int(pos_mask.sum())

    if rank_V == 0:
        logger.warning(
            "Pre-trend covariance matrix is numerically zero; F-test unavailable."
        )
        return {
            "f_stat": float("nan"),
            "p_value": float("nan"),
            "df_num": k,
            "df_denom": n_clusters - 1,
        }

    V_pinv = (
        eigvecs[:, pos_mask] @ np.diag(1.0 / eigvals[pos_mask]) @ eigvecs[:, pos_mask].T
    )
    f_stat = float(beta @ V_pinv @ beta) / rank_V
    df_num = rank_V
    df_denom = n_clusters - 1
    p_value = float(1.0 - scipy_stats.f.cdf(f_stat, df_num, df_denom))

    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "df_num": df_num,
        "df_denom": df_denom,
    }
