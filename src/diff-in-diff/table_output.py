"""Table-saving helpers for the DiD analysis pipeline.

Writes summary and full-coefficient CSV tables from regression results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _save_regression_table(
    all_results: dict[str, dict[str, Any] | None], tables_dir: Path
) -> None:
    """Save a summary regression table (one row per model) for all indicators.

    Parameters
    ----------
    all_results:
        Dict mapping ``indicator_name`` to a result dict (or ``None`` if skipped).
    """
    rows = []
    for ind, res in all_results.items():
        if res is None:
            continue
        indicator_base = str(res.get("indicator_name", ind))
        for model_name, result, boot_key in [
            ("baseline", res["baseline"], "bootstrap_baseline"),
            ("preferred", res["preferred"], "bootstrap_preferred"),
        ]:
            boot = res.get(boot_key)
            rows.append(
                {
                    "indicator": ind,
                    "indicator_base": indicator_base,
                    "model": model_name,
                    "baseline_mean": res.get("baseline_mean"),
                    "coefficient": result.coefficient,
                    "std_error": result.std_error,
                    "t_stat": result.t_stat,
                    "p_value_asymp": result.p_value,
                    "p_value_bootstrap": boot.bootstrap_p_value if boot else None,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "n_obs": result.n_obs,
                    "n_clusters": result.n_clusters,
                    "n_boot": boot.n_boot if boot else None,
                }
            )
    df = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "regression_results.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    logger.info("Regression table saved to %s", out)


def _save_coefficients_table(
    all_results: dict[str, dict[str, Any] | None], tables_dir: Path
) -> None:
    """Save a tidy table with every coefficient from all models and indicators.

    Each row represents one coefficient and includes a ``koeffisient_type``
    column classifying it as treatment, region FE, time FE, etc.

    Parameters
    ----------
    all_results:
        Dict mapping ``indicator_name`` to a result dict (or ``None`` if skipped).
    """
    from regression import extract_all_coefficients

    frames = []
    for ind, res in all_results.items():
        if res is None:
            continue
        indicator_base = str(res.get("indicator_name", ind))
        for result in (res["baseline"], res["preferred"]):
            df = extract_all_coefficients(result)
            df.insert(0, "indikator_base", indicator_base)
            df.insert(0, "indikator", ind)
            frames.append(df)

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "alle_koeffisienter.csv"
    combined.to_csv(out, index=False, float_format="%.6f")
    logger.info("Full coefficients table saved to %s", out)
