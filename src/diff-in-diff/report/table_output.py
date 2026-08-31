"""Table-saving helpers for the DiD analysis pipeline.

Writes summary and full-coefficient CSV tables from regression results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisRunMetadata:
    """Run-level fields written alongside every regression summary row."""

    config_slug: str
    config_path: str
    analysis_design: str
    treatment_type: str
    analysis_level: str
    treatment_start: str
    outcome: str = "indikator"
    weighting: str = "unweighted"


def _period_bounds(panel: pd.DataFrame) -> tuple[str, str]:
    """Return the first and last observed months in ISO date format."""
    months = pd.to_datetime(panel["aarmnd"], errors="raise")
    return (
        months.min().strftime("%Y-%m-%d"),
        months.max().strftime("%Y-%m-%d"),
    )


def _diagnostic_metadata(
    metadata: AnalysisRunMetadata,
    indicator: str,
    result: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Return fields shared by every persisted diagnostic observation."""
    return {
        "config_slug": metadata.config_slug,
        "config_path": metadata.config_path,
        "analysis_design": metadata.analysis_design,
        "treatment_type": metadata.treatment_type,
        "analysis_level": metadata.analysis_level,
        "treatment_start": metadata.treatment_start,
        "weighting": metadata.weighting,
        "outcome": metadata.outcome,
        "indicator": indicator,
        "indicator_base": str(result.get("indicator_name", indicator)),
        "model": model,
        "seasonally_adjusted": model == "preferred",
    }


def _save_regression_table(
    all_results: dict[str, dict[str, Any] | None],
    tables_dir: Path,
    metadata: AnalysisRunMetadata,
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
        for model_name, result, boot_key, panel_key in [
            ("baseline", res["baseline"], "bootstrap_baseline", "panel_regular"),
            ("preferred", res["preferred"], "bootstrap_preferred", "panel"),
        ]:
            boot = res.get(boot_key)
            period_start, period_end = _period_bounds(res[panel_key])
            rows.append(
                {
                    "config_slug": metadata.config_slug,
                    "config_path": metadata.config_path,
                    "analysis_design": metadata.analysis_design,
                    "treatment_type": metadata.treatment_type,
                    "analysis_level": metadata.analysis_level,
                    "treatment_start": metadata.treatment_start,
                    "weighting": metadata.weighting,
                    "indicator": ind,
                    "indicator_base": indicator_base,
                    "model": model_name,
                    "seasonally_adjusted": model_name == "preferred",
                    "entity_fixed_effects": any(
                        effect in {"Enhet FE", "Region FE"}
                        for effect in result.fixed_effects
                    ),
                    "time_fixed_effects": "År-måned FE" in result.fixed_effects,
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
                    "r_squared_adjusted": result.r_squared_adjusted,
                    "period_start": period_start,
                    "period_end": period_end,
                }
            )
    df = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "regression_results.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    logger.info("Regression table saved to %s", out)


def _save_diagnostic_tables(
    all_results: dict[str, dict[str, Any] | None],
    tables_dir: Path,
    metadata: AnalysisRunMetadata,
) -> None:
    """Save event-study, pre-trend, and leave-one-out data in tidy CSVs."""
    event_rows: list[dict[str, Any]] = []
    pretrend_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []

    for indicator, result in all_results.items():
        if result is None:
            continue
        for model, event_key, loo_key in (
            ("baseline", "event_study_baseline", "leave_one_out_baseline"),
            ("preferred", "event_study", "leave_one_out"),
        ):
            common = _diagnostic_metadata(metadata, indicator, result, model)
            event_study = result.get(event_key)
            if event_study is not None:
                pre_periods = sorted(
                    coefficient.tau
                    for coefficient in event_study.coefs
                    if coefficient.tau < -1
                )
                pretrend_rows.append(
                    {
                        **common,
                        "pretrend_f_stat": event_study.pretrend_f_stat,
                        "pretrend_p_value": event_study.pretrend_p_value,
                        "pretrend_df_num": event_study.pretrend_df_num,
                        "pretrend_df_denom": event_study.pretrend_df_denom,
                        "n_pre_periods": len(pre_periods),
                        "pre_periods": json.dumps(pre_periods),
                    }
                )
                for coefficient in event_study.coefs:
                    event_rows.append(
                        {
                            **common,
                            "relative_month": coefficient.tau,
                            "coefficient": coefficient.coefficient,
                            "std_error": coefficient.std_error,
                            "ci_lower": coefficient.ci_lower,
                            "ci_upper": coefficient.ci_upper,
                            "p_value": coefficient.p_value,
                        }
                    )

            leave_one_out = result.get(loo_key)
            if leave_one_out is not None:
                for row in leave_one_out.rows.to_dict(orient="records"):
                    leave_one_out_rows.append(
                        {
                            **common,
                            **row,
                            "full_coefficient": leave_one_out.full_coefficient,
                            "full_ci_lower": leave_one_out.full_ci_lower,
                            "full_ci_upper": leave_one_out.full_ci_upper,
                        }
                    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    files = (
        (
            "event_study_results.csv",
            event_rows,
            [
                "config_slug",
                "config_path",
                "analysis_design",
                "treatment_type",
                "analysis_level",
                "treatment_start",
                "outcome",
                "indicator",
                "indicator_base",
                "model",
                "seasonally_adjusted",
                "relative_month",
                "coefficient",
                "std_error",
                "ci_lower",
                "ci_upper",
                "p_value",
            ],
        ),
        (
            "pretrend_tests.csv",
            pretrend_rows,
            [
                "config_slug",
                "config_path",
                "analysis_design",
                "treatment_type",
                "analysis_level",
                "treatment_start",
                "outcome",
                "indicator",
                "indicator_base",
                "model",
                "seasonally_adjusted",
                "pretrend_f_stat",
                "pretrend_p_value",
                "pretrend_df_num",
                "pretrend_df_denom",
                "n_pre_periods",
                "pre_periods",
            ],
        ),
        (
            "leave_one_out_results.csv",
            leave_one_out_rows,
            [
                "config_slug",
                "config_path",
                "analysis_design",
                "treatment_type",
                "analysis_level",
                "treatment_start",
                "outcome",
                "indicator",
                "indicator_base",
                "model",
                "seasonally_adjusted",
                "dropped_region",
                "coefficient",
                "std_error",
                "ci_lower",
                "ci_upper",
                "p_value",
                "full_coefficient",
                "full_ci_lower",
                "full_ci_upper",
            ],
        ),
    )
    for filename, rows, columns in files:
        out = tables_dir / filename
        pd.DataFrame(rows, columns=columns).to_csv(
            out, index=False, float_format="%.6f"
        )
        logger.info("Diagnostic table saved to %s", out)


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
