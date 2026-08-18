"""Persist and load the analysis results required for report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cluster_bootstrap import BootstrapResult
from event_study import EventStudyCoef, EventStudyResult
from regression import LeaveOneOutResult, RegressionResult

REPORT_DATA_FILE = "report_data.json"
REPORT_DATA_VERSION = 1


def _regression_to_dict(result: RegressionResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        name: getattr(result, name)
        for name in (
            "model_name",
            "coefficient",
            "std_error",
            "t_stat",
            "p_value",
            "ci_lower",
            "ci_upper",
            "n_obs",
            "n_clusters",
            "fixed_effects",
            "r_squared_within",
            "r_squared_adjusted",
        )
    }


def _regression_from_dict(data: dict[str, Any] | None) -> RegressionResult | None:
    if data is None:
        return None
    return RegressionResult(**data)


def _bootstrap_to_dict(result: BootstrapResult) -> dict[str, Any]:
    return {
        "observed_coefficient": result.observed_coefficient,
        "observed_t_stat": result.observed_t_stat,
        "observed_se": result.observed_se,
        "bootstrap_p_value": result.bootstrap_p_value,
        "n_boot": result.n_boot,
        "seed": result.seed,
        "bootstrap_t_stats": result.bootstrap_t_stats.tolist(),
    }


def _bootstrap_from_dict(data: dict[str, Any]) -> BootstrapResult:
    return BootstrapResult(
        **{
            **data,
            "bootstrap_t_stats": np.asarray(data["bootstrap_t_stats"], dtype=float),
        }
    )


def _event_study_to_dict(result: EventStudyResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "coefs": [
            {
                "tau": coef.tau,
                "coefficient": coef.coefficient,
                "std_error": coef.std_error,
                "ci_lower": coef.ci_lower,
                "ci_upper": coef.ci_upper,
                "p_value": coef.p_value,
            }
            for coef in result.coefs
        ],
        "pretrend_f_stat": result.pretrend_f_stat,
        "pretrend_p_value": result.pretrend_p_value,
        "pretrend_df_num": result.pretrend_df_num,
        "pretrend_df_denom": result.pretrend_df_denom,
    }


def _event_study_from_dict(data: dict[str, Any] | None) -> EventStudyResult | None:
    if data is None:
        return None
    return EventStudyResult(
        coefs=[EventStudyCoef(**coef) for coef in data["coefs"]],
        pretrend_f_stat=data["pretrend_f_stat"],
        pretrend_p_value=data["pretrend_p_value"],
        pretrend_df_num=data["pretrend_df_num"],
        pretrend_df_denom=data["pretrend_df_denom"],
    )


def _leave_one_out_to_dict(result: LeaveOneOutResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "rows": result.rows.to_dict(orient="records"),
        "full_coefficient": result.full_coefficient,
        "full_ci_lower": result.full_ci_lower,
        "full_ci_upper": result.full_ci_upper,
    }


def _leave_one_out_from_dict(data: dict[str, Any] | None) -> LeaveOneOutResult | None:
    if data is None:
        return None
    return LeaveOneOutResult(
        rows=pd.DataFrame(data["rows"]),
        full_coefficient=data["full_coefficient"],
        full_ci_lower=data["full_ci_lower"],
        full_ci_upper=data["full_ci_upper"],
    )


def _result_to_dict(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        key: serializer(result[key])
        for key, serializer in {
            "baseline": _regression_to_dict,
            "preferred": _regression_to_dict,
            "bootstrap_baseline": _bootstrap_to_dict,
            "bootstrap_preferred": _bootstrap_to_dict,
            "event_study": _event_study_to_dict,
            "event_study_baseline": _event_study_to_dict,
            "placebo": _regression_to_dict,
            "placebo_baseline": _regression_to_dict,
            "leave_one_out": _leave_one_out_to_dict,
            "leave_one_out_baseline": _leave_one_out_to_dict,
        }.items()
    } | {
        key: value
        for key, value in result.items()
        if key
        in {
            "indicator_name",
            "analysis_level",
            "treated_group",
            "control_group",
            "mde",
            "baseline_mean",
            "baseline_mean_treated",
            "baseline_mean_control",
            "baseline_mean_by_region",
        }
    }


def _result_from_dict(
    data: dict[str, Any] | None, processed_dir: Path, result_name: str
) -> dict[str, Any] | None:
    if data is None:
        return None
    result = {
        key: deserializer(data[key])
        for key, deserializer in {
            "baseline": _regression_from_dict,
            "preferred": _regression_from_dict,
            "bootstrap_baseline": _bootstrap_from_dict,
            "bootstrap_preferred": _bootstrap_from_dict,
            "event_study": _event_study_from_dict,
            "event_study_baseline": _event_study_from_dict,
            "placebo": _regression_from_dict,
            "placebo_baseline": _regression_from_dict,
            "leave_one_out": _leave_one_out_from_dict,
            "leave_one_out_baseline": _leave_one_out_from_dict,
        }.items()
    }
    result.update(
        {
            key: value
            for key, value in data.items()
            if key
            in {
                "indicator_name",
                "analysis_level",
                "treated_group",
                "control_group",
                "mde",
                "baseline_mean",
                "baseline_mean_treated",
                "baseline_mean_control",
                "baseline_mean_by_region",
            }
        }
    )
    for panel_name in ("panel", "panel_regular"):
        path = (
            processed_dir
            / f"panel_{result_name}_{'flattened' if panel_name == 'panel' else 'regular'}.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Mangler rapportdata-panelet: {path}")
        result[panel_name] = pd.read_csv(path, parse_dates=["aarmnd"])
    return result


def save_report_data(
    all_results: dict[str, dict[str, Any] | None],
    report_data_dir: Path,
    cfg: dict[str, Any],
) -> None:
    """Write report-only inputs after successful analysis estimation."""
    report_data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": REPORT_DATA_VERSION,
        "analysis": cfg["analysis"],
        "data": cfg["data"],
        "results": {
            name: _result_to_dict(result) for name, result in all_results.items()
        },
    }
    (report_data_dir / REPORT_DATA_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def load_report_data(
    report_data_dir: Path, processed_dir: Path, cfg: dict[str, Any]
) -> dict[str, dict[str, Any] | None]:
    """Load report inputs and reject outputs from different effective settings."""
    path = report_data_dir / REPORT_DATA_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Mangler analyseresultater for rapportgenerering: {path}"
        ) from exc
    if payload.get("version") != REPORT_DATA_VERSION:
        raise ValueError(f"Ukjent rapportdataformat i {path}.")
    if payload.get("analysis") != cfg["analysis"] or payload.get("data") != cfg["data"]:
        raise ValueError(f"Rapportdata i {path} matcher ikke katalogkonfigurasjonen.")
    return {
        name: _result_from_dict(result, processed_dir, name)
        for name, result in payload["results"].items()
    }
