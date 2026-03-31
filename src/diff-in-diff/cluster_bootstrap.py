"""Wild cluster bootstrap with Webb (6-point) weights.

With only G = 12 clusters, asymptotic cluster-robust inference is unreliable.
The wild cluster bootstrap (Cameron, Gelbach & Miller, 2008; Webb, 2014; Mackinnon & Webb, 2018) is the
recommended solution.  Webb's six-point weight distribution is specifically
designed for small cluster counts.

Algorithm (imposing the null H₀: β_treatment = 0)
--------------------------------------------------
1. Project all fixed effects out of both Y and X using the Frisch–Waugh–Lovell
   (FWL) theorem.  This step is performed **once** for efficiency.
2. The null residuals are the FWL-transformed outcome ``Ỹ`` (because under H₀
   the restricted model fits only FEs, so the within-transformed residual
   equals the within-transformed Y).
3. For each bootstrap replicate b:
   a. Draw a Webb weight ``w_g ∈ {−√(3/2), −1, −√(1/2), √(1/2), 1, √(3/2)}``
      independently for each cluster g.
   b. Construct ``Ỹ*_b = w_g · Ỹ`` (broadcast by cluster membership).
   c. Estimate ``β*_b = (X̃ᵀX̃)⁻¹ X̃ᵀ Ỹ*_b``.
   d. Compute the cluster-robust t-statistic ``t*_b = β*_b / SE*_b``
      using CR1 small-sample correction.
4. Bootstrap p-value = share of |t*| ≥ |t_obs|.

The FWL approach makes the bootstrap fast: after the one-time projection step
each iteration requires only O(N·G) operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Webb (2014) six-point weight distribution, designed for small G
WEBB_WEIGHTS: np.ndarray = np.array(
    [-np.sqrt(3 / 2), -1.0, -np.sqrt(1 / 2), np.sqrt(1 / 2), 1.0, np.sqrt(3 / 2)]
)

DEFAULT_N_BOOT: int = 4_999
DEFAULT_SEED: int = 42


# ── Result container ───────────────────────────────────────────────────────────


@dataclass
class BootstrapResult:
    """Output of the wild cluster bootstrap."""

    observed_coefficient: float
    observed_t_stat: float
    observed_se: float
    bootstrap_p_value: float
    n_boot: int
    seed: int
    #: Full distribution of bootstrap t-statistics (length = n_boot).
    bootstrap_t_stats: np.ndarray


# ── FWL helpers ────────────────────────────────────────────────────────────────


def _build_fe_matrix(panel: pd.DataFrame, preferred: bool = True) -> np.ndarray:
    """Build the matrix of fixed-effect dummies Z (float, no column names).

    Includes region FE and year-month FE; when ``preferred=True`` also adds
    region × calendar-month FE.  One dummy per group is dropped to prevent
    exact collinearity.

    Parameters
    ----------
    panel:
        Analysis panel from :func:`prep_data.prepare_panel`.
    preferred:
        Whether to include region × calendar-month seasonal FE.

    Returns:
    -------
    2-D float array of shape (N, K_fe) including an intercept column.
    """
    region_fe = pd.get_dummies(
        panel["region"], prefix="r", drop_first=True, dtype=float
    )
    yearmonth_fe = pd.get_dummies(
        panel["aarmnd"].astype(str), prefix="t", drop_first=True, dtype=float
    )
    parts: list[np.ndarray] = [
        np.ones((len(panel), 1)),
        region_fe.values,
        yearmonth_fe.values,
    ]

    if preferred:
        rm_key = (
            panel["region"].str.replace(" ", "_")
            + "_m"
            + panel["month_of_year"].astype(str)
        )
        rm_fe = pd.get_dummies(rm_key, prefix="rm", drop_first=False, dtype=float)
        to_drop = [f"rm_{r.replace(' ', '_')}_m1" for r in panel["region"].unique()]
        rm_fe = rm_fe.drop(
            columns=[c for c in to_drop if c in rm_fe.columns], errors="ignore"
        )
        parts.append(rm_fe.values)

    return np.hstack(parts)


def _partial_out(Z: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove the column-space of Z from vector v (FWL annihilator M_Z v).

    Uses :func:`numpy.linalg.lstsq` which handles rank-deficient Z gracefully.

    Parameters
    ----------
    Z:
        Matrix whose column-space is to be projected out (shape N × K).
    v:
        Vector to be partialled (shape N,).

    Returns:
    -------
    Residual vector ``v − Z(ZᵀZ)⁻¹Zᵀv`` (shape N,).
    """
    coef, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ coef


# ── CR1 cluster-robust SE (single-regressor case after FWL) ───────────────────


def _cr1_se(
    x_tilde: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
    G: int,
) -> float:
    """Compute the CR1 clustered standard error for a single FWL regressor.

    CR1 small-sample correction: G/(G−1) × (N−1)/(N−K), with K = 1 after
    FWL partialling.

    Parameters
    ----------
    x_tilde:
        FWL-projected treatment vector (length N).
    residuals:
        OLS residuals from regressing ``y_tilde`` on ``x_tilde`` (length N).
    cluster_ids:
        Integer cluster index per observation (length N, values in 0…G−1).
    G:
        Number of clusters.

    Returns:
    -------
    Standard error (scalar ≥ 0).
    """
    N = len(x_tilde)
    K = 1
    Q = float(x_tilde @ x_tilde)

    meat = 0.0
    for g in range(G):
        mask = cluster_ids == g
        S_g = float(x_tilde[mask] @ residuals[mask])
        meat += S_g * S_g

    cr1_factor = (G / (G - 1)) * ((N - 1) / (N - K))
    return float(np.sqrt(cr1_factor * meat / (Q * Q)))


# ── Main bootstrap function ────────────────────────────────────────────────────


def wild_cluster_bootstrap(
    panel: pd.DataFrame,
    preferred: bool = True,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Run the wild cluster bootstrap for the treatment coefficient.

    Tests H₀: β_tiltaksnedgang = 0 using Webb (2014) weights.  The null is
    imposed by using the FWL-transformed outcome as the null residual (step 2
    of the algorithm in the module docstring).

    Parameters
    ----------
    panel:
        Processed analysis panel from :func:`prep_data.prepare_panel`.
    preferred:
        If ``True`` (default) use the preferred model specification (region FE
        + year-month FE + region × calendar-month FE).  If ``False``, use the
        baseline specification.
    n_boot:
        Number of bootstrap replications.  Minimum recommended: 4 999.
    seed:
        Random seed for reproducibility.

    Returns:
    -------
    :class:`BootstrapResult` containing the observed statistics, the full
    bootstrap t-distribution, and the two-sided bootstrap p-value.
    """
    rng = np.random.default_rng(seed)

    y = panel["indikator"].astype(float).values
    x_raw = panel["tiltaksnedgang"].astype(float).values
    regions = panel["region"].values
    unique_clusters = np.unique(regions)
    G = len(unique_clusters)

    # Map region strings → contiguous integer cluster indices
    cluster_map = {r: i for i, r in enumerate(unique_clusters)}
    cluster_ids = np.array([cluster_map[r] for r in regions], dtype=int)

    # ── Step 1: FWL projection (one-time cost) ────────────────────────────────
    logger.info("Building FE matrix for bootstrap (preferred=%s, G=%d) …", preferred, G)
    Z = _build_fe_matrix(panel, preferred=preferred)

    logger.info("Projecting Y and X out of FE space …")
    y_tilde = _partial_out(Z, y)
    x_tilde = _partial_out(Z, x_raw)

    # ── Step 2: Observed statistics ───────────────────────────────────────────
    Q = float(x_tilde @ x_tilde)
    beta_obs = float(x_tilde @ y_tilde) / Q
    resid_obs = y_tilde - x_tilde * beta_obs
    se_obs = _cr1_se(x_tilde, resid_obs, cluster_ids, G)
    t_obs = beta_obs / se_obs

    logger.info(
        "Observed: β = %.4f  SE = %.4f  t = %.3f",
        beta_obs,
        se_obs,
        t_obs,
    )

    # ── Step 3: Vectorised bootstrap ──────────────────────────────────────────
    # Webb weights: shape (n_boot, G)
    all_weights = rng.choice(WEBB_WEIGHTS, size=(n_boot, G))

    # Broadcast cluster weights to observation level: shape (n_boot, N)
    all_cluster_weights = all_weights[:, cluster_ids]

    # Null bootstrap outcomes Ỹ* = w_g · Ỹ: shape (n_boot, N)
    y_tilde_boot = all_cluster_weights * y_tilde[np.newaxis, :]

    # Bootstrap coefficients β* = X̃ᵀỸ* / Q: shape (n_boot,)
    beta_boots = (y_tilde_boot @ x_tilde) / Q

    # Bootstrap residuals ẽ* = Ỹ* − β* · X̃: shape (n_boot, N)
    resid_boots = y_tilde_boot - beta_boots[:, np.newaxis] * x_tilde[np.newaxis, :]

    # Cluster-level scores: shape (n_boot, G)
    N = len(y_tilde)
    scores = np.zeros((n_boot, G), dtype=float)
    for g in range(G):
        mask = cluster_ids == g
        scores[:, g] = resid_boots[:, mask] @ x_tilde[mask]

    # CR1 variances and SEs for each bootstrap replicate
    K = 1
    cr1_factor = (G / (G - 1)) * ((N - 1) / (N - K))
    var_boots = cr1_factor * np.sum(scores**2, axis=1) / (Q * Q)
    se_boots = np.sqrt(np.maximum(var_boots, 0.0))

    # Bootstrap t-statistics (guarded against degenerate SE)
    t_boots = np.where(se_boots > 1e-15, beta_boots / se_boots, 0.0)

    # ── Step 4: Two-sided p-value ─────────────────────────────────────────────
    p_value = float(np.mean(np.abs(t_boots) >= np.abs(t_obs)))

    logger.info(
        "Bootstrap complete: n_boot=%d  p_bootstrap=%.4f",
        n_boot,
        p_value,
    )

    return BootstrapResult(
        observed_coefficient=beta_obs,
        observed_t_stat=t_obs,
        observed_se=se_obs,
        bootstrap_p_value=p_value,
        n_boot=n_boot,
        seed=seed,
        bootstrap_t_stats=t_boots,
    )
