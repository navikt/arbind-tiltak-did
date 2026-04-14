"""Run the full DID analysis for Nav employment indicators.

Pass a config file as the first argument (name or path).  A bare filename is
resolved first relative to the current directory, then relative to the
``configs/`` subdirectory.  Run without arguments to use the default config.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ── Project paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = Path(__file__).parent / "configs"
CONFIG_PATH = CONFIGS_DIR / "alle-kontinuerlig.yml"
DATA_PROCESSED_BASE = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DID_BASE = PROJECT_ROOT / "outputs" / "did"
QUARTO_DIR = PROJECT_ROOT / "quarto"

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
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Available configs in configs/:\n"
            + "\n".join(f"  {p.name}" for p in sorted(CONFIGS_DIR.glob("*.yml")))
        )
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
            f"Default: {CONFIG_PATH.name} (from configs/)"
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


def _variation_from_cfg(cfg: dict[str, Any]) -> str:
    """Return the treatment variation folder name (default: 'regioner')."""
    return str(cfg["analysis"].get("variation", "regioner"))


def _update_quarto_chapters(quarto_dir: Path, variation: str) -> None:
    """Scan quarto/<variation>/*/ for report QMDs and update _quarto.yml chapters."""
    quarto_yml = quarto_dir / "_quarto.yml"
    if not quarto_yml.exists():
        logger.warning("_quarto.yml not found at %s, skipping chapter update", quarto_yml)
        return

    variation_dir = quarto_dir / variation
    report_qmds = sorted(
        p.relative_to(quarto_dir).as_posix()
        for p in variation_dir.glob("*/report_*.qmd")
    )

    with open(quarto_yml, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    part_path = f"{variation}/intro.qmd"
    chapters: list = cfg_yaml["book"]["chapters"]

    part_entry = next(
        (ch for ch in chapters if isinstance(ch, dict) and ch.get("part") == part_path),
        None,
    )
    if part_entry is None:
        part_entry = {"part": part_path, "chapters": []}
        chapters.append(part_entry)

    part_entry["chapters"] = report_qmds

    with open(quarto_yml, "w", encoding="utf-8") as f:
        yaml.dump(cfg_yaml, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info("Updated _quarto.yml: %d chapters for '%s'", len(report_qmds), variation)


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
    processed_dir: Path,
    control_regions: list[str] | None = None,
) -> dict[str, Any] | None:
    """Prepare regular and flattened panels, run regression and supporting analyses.

    The baseline model is estimated on the regular (non-flattened) panel.
    The preferred model is estimated on the seasonally flattened panel.
    Both use region FE + year-month FE only.

    Returns an :class:`IndicatorResult` serialised to a plain dict for
    backward-compatible downstream use, or ``None`` if skipped.
    """
    from cluster_bootstrap import wild_cluster_bootstrap
    from event_study import run_event_study
    from models import IndicatorResult
    from prep_data import prepare_panel
    from regression import (
        compute_mde,
        run_baseline_model,
        run_leave_one_out,
        run_placebo_test,
        run_preferred_model,
    )

    shared_kwargs = dict(
        indicator_path=indicator_path,
        tiltak_path=tiltak_path,
        indicator_name=indicator_name,
        treatment_start=treatment_start,
        treatment_type=treatment_type,
        denominator=denominator,
        control_regions=control_regions,
    )

    logger.info("── Preparing regular panel for %s ──", indicator_name)
    panel_regular = prepare_panel(
        **shared_kwargs,
        flatten=False,
        processed_path=processed_dir / f"panel_{result_name}_regular.csv",
    )

    logger.info("── Preparing flattened panel for %s ──", indicator_name)
    panel_flattened = prepare_panel(
        **shared_kwargs,
        flatten=True,
        processed_path=processed_dir / f"panel_{result_name}_flattened.csv",
    )

    n_post_obs = int(panel_regular["post_treatment"].sum())
    if n_post_obs == 0:
        logger.warning(
            "%s: skipping — no post-treatment months available.",
            result_name,
        )
        return None

    n_obs = len(panel_regular)
    n_months = panel_regular["aarmnd"].nunique()
    n_regions = panel_regular["region"].nunique()
    logger.info(
        "%s: %d obs (%d months × %d regions)",
        result_name,
        n_obs,
        n_months,
        n_regions,
    )

    logger.info("Running baseline model (regular) for %s", result_name)
    baseline = run_baseline_model(panel_regular)

    logger.info("Running preferred model (flattened) for %s", result_name)
    preferred = run_preferred_model(panel_flattened)

    # Model registry: run all downstream analyses uniformly for both panels.
    models: dict[str, Any] = {
        "basis": dict(
            panel=panel_regular,
            main=baseline,
            run_model=run_baseline_model,
        ),
        "flattet": dict(
            panel=panel_flattened,
            main=preferred,
            run_model=run_preferred_model,
        ),
    }

    for key, m in models.items():
        panel = m["panel"]
        main_result = m["main"]
        logger.info("Bootstrap (%s) for %s", key, result_name)
        m["bootstrap"] = wild_cluster_bootstrap(panel)
        logger.info("Event study (%s) for %s", key, result_name)
        m["event_study"] = run_event_study(panel)
        logger.info("Placebo (%s) for %s", key, result_name)
        m["placebo"] = run_placebo_test(panel, placebo_relative_month=-12)
        logger.info("Leave-one-out (%s) for %s", key, result_name)
        m["leave_one_out"] = run_leave_one_out(panel, preferred_result=main_result)

    mde = compute_mde(preferred)
    logger.info("MDE for %s: %.4f pp", result_name, mde)

    pre_panel = panel_flattened[panel_flattened["relative_month"] < 0]
    baseline_mean = float(pre_panel["indikator"].mean())
    baseline_mean_by_region = pre_panel.groupby("region")["indikator"].mean().to_dict()

    return IndicatorResult(
        indicator_name=indicator_name,
        baseline=baseline,
        preferred=preferred,
        bootstrap_baseline=models["basis"]["bootstrap"],
        bootstrap_preferred=models["flattet"]["bootstrap"],
        event_study=models["flattet"]["event_study"],
        event_study_baseline=models["basis"]["event_study"],
        placebo=models["flattet"]["placebo"],
        placebo_baseline=models["basis"]["placebo"],
        leave_one_out=models["flattet"]["leave_one_out"],
        leave_one_out_baseline=models["basis"]["leave_one_out"],
        mde=mde,
        baseline_mean=baseline_mean,
        baseline_mean_by_region=baseline_mean_by_region,
        panel=panel_flattened,
        panel_regular=panel_regular,
    ).to_dict()


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
    variation = _variation_from_cfg(cfg)
    output_root = OUTPUTS_DID_BASE / config_slug
    staging_root = output_root / "_staging"
    # figures_dir lives inside report_dir so relative paths in the QMD are
    # preserved correctly after the report is promoted to quarto/.
    report_dir = staging_root / "report"
    figures_dir = report_dir / "figures"
    tables_dir = staging_root / "tables"
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
    # For discrete treatment: list of control regions (required by the discrete path).
    control_regions: list[str] | None = analysis.get("control_regions", None)
    if treatment_type == "discrete" and not control_regions:
        logger.error(
            "analysis.control_regions must be a non-empty list for treatment_type='discrete'."
        )
        return 1

    tiltak_path = PROJECT_ROOT / cfg["data"]["tiltak_file"]

    all_results: dict[str, dict[str, Any] | None] = {}
    failed: list[str] = []

    for ind in cfg["data"]["indikatorer"]:
        name = ind["name"]
        path = PROJECT_ROOT / ind["file"]
        try:
            result = _run_indicator(
                result_name=name,
                indicator_name=name,
                indicator_path=path,
                tiltak_path=tiltak_path,
                treatment_start=treatment_start,
                treatment_type=treatment_type,
                denominator=denominator,
                processed_dir=processed_dir,
                control_regions=control_regions,
            )
            all_results[name] = result
            if result is None:
                logger.info("○ %s skipped (no post-treatment data)", name)
            else:
                logger.info("✓ %s complete", name)
        except ValueError, np.linalg.LinAlgError, KeyError, pd.errors.MergeError:
            logger.exception("Failed to process %s", name)
            failed.append(name)
            all_results[name] = None

    n_done = sum(1 for v in all_results.values() if v is not None)
    n_total = len(cfg["data"]["indikatorer"])

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

    # Promote staging → final destinations only when all indicators succeeded.
    # On partial failure keep staging in place so prior complete outputs survive.
    if failed:
        logger.error(
            "Run incomplete — %d/%d indicators failed: %s.  Staged outputs kept at: %s",
            len(failed),
            n_total,
            ", ".join(failed),
            staging_root,
        )
        exit_code = 1 if n_done == 0 else 2
    else:
        # Tables → outputs/did/<slug>/tables/
        final_tables = output_root / "tables"
        if final_tables.exists():
            shutil.rmtree(final_tables)
        shutil.copytree(tables_dir, final_tables)

        # Report + figures → quarto/<variation>/<slug>/
        quarto_output_dir = QUARTO_DIR / variation / config_slug
        if quarto_output_dir.exists():
            shutil.rmtree(quarto_output_dir)
        quarto_output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(report_dir, quarto_output_dir)

        shutil.rmtree(staging_root)
        logger.info("Tables promoted to %s", final_tables)
        logger.info("Report promoted to %s", quarto_output_dir)

        _update_quarto_chapters(QUARTO_DIR, variation)
        exit_code = 0

    logger.info("═══ Done (%d/%d indicators) ═══", n_done, n_total)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
