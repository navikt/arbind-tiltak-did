"""Run the full DID analysis for Nav employment indicators.

All configuration is read from ``analysis-config.yml`` in the same directory.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ── Project paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = Path(__file__).parent / "analysis-config.yml"
CONFIGS_DIR = Path(__file__).parent / "configs"
DATA_PROCESSED_BASE = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DID_BASE = PROJECT_ROOT / "outputs" / "did"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_analysis")


# ── Config ─────────────────────────────────────────────────────────────────────


def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the YAML analysis configuration."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run full DID analysis pipeline.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(CONFIG_PATH),
        help=(
            "Path or filename of the YAML config file. "
            "A bare filename (e.g. alle-discrete.yml) is resolved first relative "
            "to the current directory, then relative to diff-in-diff/configs/. "
            f"Default: {CONFIG_PATH.name}"
        ),
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help="Path or filename of the YAML config file (overrides positional argument).",
    )
    return parser.parse_args()


def _config_slug(cfg_path: Path) -> str:
    """Return a filesystem-safe slug based on config filename."""
    return re.sub(r"[^a-z0-9._-]+", "_", cfg_path.stem.lower()).strip("_")


# ── Pipeline ───────────────────────────────────────────────────────────────────


def _check_inputs(cfg: dict[str, Any]) -> bool:
    """Verify all required input files exist; return False on any missing file."""
    missing = []
    tiltak_path = PROJECT_ROOT / cfg["data"]["tiltak_file"]
    if not tiltak_path.exists():
        missing.append(f"  tiltak: {tiltak_path}")
    for ind in cfg["data"]["indikatorer"]:
        p = PROJECT_ROOT / ind["file"]
        if not p.exists():
            missing.append(f"  {ind['name']}: {p}")
    if missing:
        logger.error("Missing input files:\n%s", "\n".join(missing))
        return False
    return True


def _run_indicator(
    result_name: str,
    indicator_name: str,
    indicator_path: Path,
    tiltak_path: Path,
    treatment_start: str,
    treatment_type: str,
    denominator: str,
    flatten: bool,
    processed_dir: Path,
    controll_regions: list[str] | None = None,
) -> dict[str, Any] | None:
    """Prepare data, run regression, event study, and bootstrap for one indicator; return result dict."""
    from cluster_bootstrap import wild_cluster_bootstrap
    from event_study import run_event_study
    from prep_data import prepare_panel
    from regression import (
        compute_mde,
        run_baseline_model,
        run_leave_one_out,
        run_placebo_test,
        run_preferred_model,
    )

    logger.info("── Preparing panel for %s (%s) ──", indicator_name, result_name)
    panel = prepare_panel(
        indicator_path=indicator_path,
        tiltak_path=tiltak_path,
        indicator_name=indicator_name,
        treatment_start=treatment_start,
        treatment_type=treatment_type,
        denominator=denominator,
        flatten=flatten,
        controll_regions=controll_regions,
        processed_path=processed_dir / f"panel_{result_name}.csv",
    )

    n_post_obs = int(panel["post_treatment"].sum())
    if n_post_obs == 0:
        logger.warning(
            "%s (%s): skipping — no post-treatment months available.",
            indicator_name,
            result_name,
        )
        return None

    n_obs = len(panel)
    n_months = panel["aarmnd"].nunique()
    n_regions = panel["region"].nunique()
    logger.info(
        "%s: %d obs (%d months × %d regions)",
        result_name,
        n_obs,
        n_months,
        n_regions,
    )

    logger.info("Running regression models for %s", result_name)
    baseline = run_baseline_model(panel)
    preferred = run_preferred_model(panel)

    logger.info("Running wild cluster bootstrap for %s", result_name)
    bootstrap_baseline = wild_cluster_bootstrap(panel, preferred=False)
    bootstrap_preferred = wild_cluster_bootstrap(panel, preferred=True)

    logger.info("Running event study for %s", result_name)
    event_study = run_event_study(panel)

    logger.info("Running placebo test for %s", result_name)
    placebo = run_placebo_test(panel, placebo_relative_month=-12)

    logger.info("Running leave-one-out for %s", result_name)
    leave_one_out = run_leave_one_out(panel, preferred_result=preferred)

    mde = compute_mde(preferred)
    logger.info("MDE for %s: %.4f pp", result_name, mde)

    # Baseline mean: pre-period average of indikator across all treated regions
    pre_panel = panel[panel["relative_month"] < 0]
    baseline_mean = float(pre_panel["indikator"].mean())
    baseline_mean_by_region = (
        pre_panel.groupby("region")["indikator"].mean().to_dict()
    )

    return {
        "baseline": baseline,
        "preferred": preferred,
        "bootstrap_baseline": bootstrap_baseline,
        "bootstrap_preferred": bootstrap_preferred,
        "event_study": event_study,
        "placebo": placebo,
        "leave_one_out": leave_one_out,
        "mde": mde,
        "baseline_mean": baseline_mean,
        "baseline_mean_by_region": baseline_mean_by_region,
        "panel": panel,
        "indicator_name": indicator_name,
        "prep_variant": "flattened" if flatten else "regular",
    }


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
        prep_variant = str(res.get("prep_variant", "regular"))
        for model_name, result, boot_key in [
            ("baseline", res["baseline"], "bootstrap_baseline"),
            ("preferred", res["preferred"], "bootstrap_preferred"),
        ]:
            boot = res.get(boot_key)
            rows.append(
                {
                    "indicator": ind,
                    "indicator_base": indicator_base,
                    "prep_variant": prep_variant,
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
        prep_variant = str(res.get("prep_variant", "regular"))
        for result in (res["baseline"], res["preferred"]):
            df = extract_all_coefficients(result)
            df.insert(0, "prep_variant", prep_variant)
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


def main() -> int:
    """Entry point: run full pipeline for all indicators defined in the config."""
    args = _parse_args()
    cfg_path = Path(args.config_flag or args.config)
    if not cfg_path.is_absolute():
        cwd_candidate = (Path.cwd() / cfg_path).resolve()
        configs_candidate = (CONFIGS_DIR / cfg_path).resolve()
        # Prefer cwd-relative path if it exists; fall back to configs/ directory.
        if cwd_candidate.exists():
            cfg_path = cwd_candidate
        elif configs_candidate.exists():
            cfg_path = configs_candidate
        else:
            cfg_path = cwd_candidate  # will fail with a clear error below

    logger.info("═══ Nav DID analysis ═══")
    logger.info("Using config: %s", cfg_path)

    cfg = _load_config(cfg_path)
    if not _check_inputs(cfg):
        return 1

    config_slug = _config_slug(cfg_path)
    output_root = OUTPUTS_DID_BASE / config_slug
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    report_dir = output_root / "report"
    processed_dir = DATA_PROCESSED_BASE / config_slug

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    analysis = cfg["analysis"]
    treatment_start = str(analysis["treatment_start"])
    treatment_type = analysis["treatment_type"]
    # Use the first denominator definition from config, falling back to "peak".
    denom_defs = analysis.get("denominator_definitions", [])
    denominator = denom_defs[0]["id"] if denom_defs else "peak"
    prep_setups = analysis.get("prep_setups", [{"id": "regular", "flatten": False}])
    # For discrete treatment: list of control regions (required by the discrete path).
    controll_regions: list[str] | None = analysis.get("controll_regions", None)
    if treatment_type == "discrete" and not controll_regions:
        logger.error(
            "analysis.controll_regions must be a non-empty list for treatment_type='discrete'."
        )
        return 1

    tiltak_path = PROJECT_ROOT / cfg["data"]["tiltak_file"]

    all_results: dict[str, dict[str, Any] | None] = {}
    failed: list[str] = []

    valid_setup_ids = {"regular", "flattened"}
    normalized_setups: list[dict[str, Any]] = []
    for i, setup in enumerate(prep_setups):
        if not isinstance(setup, dict):
            raise ValueError(f"analysis.prep_setups[{i}] must be a mapping.")
        setup_id = str(setup.get("id", "")).strip()
        flatten = bool(setup.get("flatten", False))
        if setup_id == "":
            setup_id = "flattened" if flatten else "regular"
        if setup_id not in valid_setup_ids:
            raise ValueError(
                f"analysis.prep_setups[{i}].id must be one of "
                f"{sorted(valid_setup_ids)}; got '{setup_id}'."
            )
        normalized_setups.append({"id": setup_id, "flatten": flatten})

    for ind in cfg["data"]["indikatorer"]:
        name = ind["name"]
        path = PROJECT_ROOT / ind["file"]
        for setup in normalized_setups:
            result_name = (
                name if setup["id"] == "regular" else f"{name}__{setup['id']}"
            )
            try:
                result = _run_indicator(
                    result_name=result_name,
                    indicator_name=name,
                    indicator_path=path,
                    tiltak_path=tiltak_path,
                    treatment_start=treatment_start,
                    treatment_type=treatment_type,
                    denominator=denominator,
                    flatten=setup["flatten"],
                    processed_dir=processed_dir,
                    controll_regions=controll_regions,
                )
                all_results[result_name] = result
                if result is None:
                    logger.info("○ %s skipped (no post-treatment data)", result_name)
                else:
                    logger.info("✓ %s complete", result_name)
            except Exception:
                logger.exception("Failed to process %s", result_name)
                failed.append(result_name)
                all_results[result_name] = None

    if failed:
        logger.error("Indicators that failed: %s", ", ".join(failed))
        if all(v is None for v in all_results.values()):
            return 1

    if any(v is not None for v in all_results.values()):
        _save_regression_table(all_results, tables_dir=tables_dir)
        _save_coefficients_table(all_results, tables_dir=tables_dir)

        logger.info("Generating report")
        from report import generate_report

        report_path = report_dir / f"report_{config_slug}.qmd"
        generate_report(
            all_results=all_results,
            cfg=cfg,
            output_path=report_path,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
        )
        logger.info("Report written to %s", report_path)

    n_done = sum(1 for v in all_results.values() if v is not None)
    n_total = len(cfg["data"]["indikatorer"]) * len(normalized_setups)
    logger.info("═══ Done (%d/%d indicators) ═══", n_done, n_total)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
