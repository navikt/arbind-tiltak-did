"""Run the full DID analysis for Nav employment indicators.

All configuration is read from ``analysis-config.yml`` in the same directory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ── Project paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = Path(__file__).parent / "analysis-config.yml"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "did" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "did" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "did" / "report"

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
    indicator_name: str,
    indicator_path: Path,
    tiltak_path: Path,
    treatment_start: str,
    treatment_type: str,
    denominator: str,
) -> dict[str, Any] | None:
    """Prepare data, run regression models for one indicator; return result dict."""
    from prep_data import prepare_panel
    from regression import run_baseline_model, run_preferred_model

    logger.info("── Preparing panel for %s ──", indicator_name)
    panel = prepare_panel(
        indicator_path=indicator_path,
        tiltak_path=tiltak_path,
        indicator_name=indicator_name,
        treatment_start=treatment_start,
        treatment_type=treatment_type,
        denominator=denominator,
        processed_path=DATA_PROCESSED / f"panel_{indicator_name}.csv",
    )

    n_post = int(panel["post_treatment"].sum() // panel["region"].nunique())
    if n_post == 0:
        logger.warning(
            "%s: skipping — no post-treatment months available.", indicator_name
        )
        return None

    n_obs = len(panel)
    n_months = panel["aarmnd"].nunique()
    n_regions = panel["region"].nunique()
    logger.info(
        "%s: %d obs (%d months × %d regions)",
        indicator_name,
        n_obs,
        n_months,
        n_regions,
    )

    logger.info("Running regression models for %s", indicator_name)
    baseline = run_baseline_model(panel)
    preferred = run_preferred_model(panel)

    return {
        "baseline": baseline,
        "preferred": preferred,
        "panel": panel,
    }


def _save_regression_table(all_results: dict[str, dict[str, Any] | None]) -> None:
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
        for model_name, result in [
            ("baseline", res["baseline"]),
            ("preferred", res["preferred"]),
        ]:
            rows.append(
                {
                    "indicator": ind,
                    "model": model_name,
                    "coefficient": result.coefficient,
                    "std_error": result.std_error,
                    "t_stat": result.t_stat,
                    "p_value": result.p_value,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "n_obs": result.n_obs,
                    "n_clusters": result.n_clusters,
                }
            )
    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "regression_results.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    logger.info("Regression table saved to %s", out)


def _save_coefficients_table(all_results: dict[str, dict[str, Any] | None]) -> None:
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
        for result in (res["baseline"], res["preferred"]):
            df = extract_all_coefficients(result)
            df.insert(0, "indikator", ind)
            frames.append(df)

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    out = TABLES_DIR / "alle_koeffisienter.csv"
    combined.to_csv(out, index=False, float_format="%.6f")
    logger.info("Full coefficients table saved to %s", out)


def main() -> int:
    """Entry point: run full pipeline for all indicators defined in the config."""
    logger.info("═══ Nav DID analysis ═══")

    cfg = _load_config()
    if not _check_inputs(cfg):
        return 1

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    analysis = cfg["analysis"]
    treatment_start = str(analysis["treatment_start"])
    treatment_type = analysis["treatment_type"]
    # Use the first denominator definition from config, falling back to "peak".
    denom_defs = analysis.get("denominator_definitions", [])
    denominator = denom_defs[0]["id"] if denom_defs else "peak"

    tiltak_path = PROJECT_ROOT / cfg["data"]["tiltak_file"]

    all_results: dict[str, dict[str, Any] | None] = {}
    failed: list[str] = []

    for ind in cfg["data"]["indikatorer"]:
        name = ind["name"]
        path = PROJECT_ROOT / ind["file"]
        try:
            result = _run_indicator(
                indicator_name=name,
                indicator_path=path,
                tiltak_path=tiltak_path,
                treatment_start=treatment_start,
                treatment_type=treatment_type,
                denominator=denominator,
            )
            all_results[name] = result
            if result is None:
                logger.info("○ %s skipped (no post-treatment data)", name)
            else:
                logger.info("✓ %s complete", name)
        except Exception:
            logger.exception("Failed to process %s", name)
            failed.append(name)
            all_results[name] = None

    if failed:
        logger.error("Indicators that failed: %s", ", ".join(failed))
        if all(v is None for v in all_results.values()):
            return 1

    if any(v is not None for v in all_results.values()):
        _save_regression_table(all_results)
        _save_coefficients_table(all_results)

        logger.info("Generating report")
        from report import generate_report

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        generate_report(
            all_results=all_results,
            cfg=cfg,
            output_path=REPORT_DIR / "report.qmd",
            figures_dir=FIGURES_DIR,
            tables_dir=TABLES_DIR,
        )
        logger.info("Report written to %s", REPORT_DIR / "report.qmd")

    n_done = sum(1 for v in all_results.values() if v is not None)
    n_total = len(cfg["data"]["indikatorer"])
    logger.info("═══ Done (%d/%d indicators) ═══", n_done, n_total)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
