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
from quarto_utils import _update_quarto_chapters, _update_quarto_triple_diff_chapters
from report.table_output import _save_coefficients_table, _save_regression_table

# ── Project paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG = (
    CONFIGS_DIR / "did" / "midl-lonnstilskudd" / "alle" / "kontinuerlig.yml"
)
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


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and return the YAML analysis configuration."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Available configs in configs/:\n"
            + "\n".join(
                f"  {p.relative_to(CONFIGS_DIR)}"
                for p in sorted(CONFIGS_DIR.rglob("*.yml"))
            )
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run full DID analysis pipeline.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help=(
            "Path to a YAML config file, or a directory containing YAML config files. "
            "A bare name is resolved first relative to the current directory, then "
            "relative to the configs/ directory. "
            f"Default: {DEFAULT_CONFIG.relative_to(CONFIGS_DIR)} (from configs/)"
        ),
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help="Path or name of the YAML config file or directory (overrides positional argument).",
    )
    return parser.parse_args()


def _config_slug(cfg_path: Path) -> str:
    """Return a filesystem-safe slug derived from the config path relative to CONFIGS_DIR.

    Uses the path components relative to ``CONFIGS_DIR``, joined with ``--``, so that
    short filenames like ``kontinuerlig.yml`` remain unique across different subdirectories.

    Example: ``did/midl-lonnstilskudd/alle/kontinuerlig.yml``
             → ``did--midl-lonnstilskudd--alle--kontinuerlig``

    Falls back to the bare stem if the path is not under ``CONFIGS_DIR``.
    """
    try:
        rel = cfg_path.with_suffix("").relative_to(CONFIGS_DIR)
        raw = "--".join(rel.parts)
    except ValueError:
        raw = cfg_path.stem
    return re.sub(r"[^a-z0-9._-]+", "_", raw.lower()).strip("_")


def _variation_from_cfg(cfg: dict[str, Any]) -> str:
    """Return the treatment variation folder name (default: 'regioner')."""
    return str(cfg["analysis"].get("variation", "regioner"))


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
    analysis_level = cfg["analysis"].get("analysis_level", "region")
    design = cfg["analysis"].get("design", "did")
    if analysis_level == "enhet" and design != "triple_diff":
        mapping_file = cfg["data"].get("enhet_mapping_file")
        if not mapping_file:
            missing.append("  enhet_mapping_file: not specified in data config")
        else:
            mp = PROJECT_ROOT / mapping_file
            if not mp.exists():
                missing.append(f"  enhet_mapping_file: {mp}")
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
    analysis_level: str = "region",
    enhet_mapping_path: Path | None = None,
    seasonal_adjust: bool = False,
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
    from regression import compute_mde
    from regression_did import (
        run_baseline_model,
        run_leave_one_out,
        run_placebo_test,
        run_preferred_model,
    )

    logger.info("── Preparing regular panel for %s ──", indicator_name)
    panel_regular = prepare_panel(
        indicator_path=indicator_path,
        tiltak_path=tiltak_path,
        indicator_name=indicator_name,
        treatment_start=treatment_start,
        treatment_type=treatment_type,
        denominator=denominator,
        control_regions=control_regions,
        analysis_level=analysis_level,
        enhet_mapping_path=enhet_mapping_path,
        flatten=False,
        seasonal_adjust=seasonal_adjust,
        processed_path=processed_dir / f"panel_{result_name}_regular.csv",
    )

    logger.info("── Preparing flattened panel for %s ──", indicator_name)
    panel_flattened = prepare_panel(
        indicator_path=indicator_path,
        tiltak_path=tiltak_path,
        indicator_name=indicator_name,
        treatment_start=treatment_start,
        treatment_type=treatment_type,
        denominator=denominator,
        control_regions=control_regions,
        analysis_level=analysis_level,
        enhet_mapping_path=enhet_mapping_path,
        flatten=True,
        seasonal_adjust=seasonal_adjust,
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
    n_entities = panel_regular["entity"].nunique()
    n_regions = panel_regular["region"].nunique()
    if n_entities > n_regions:
        logger.info(
            "%s: %d obs (%d months × %d enheter in %d regions)",
            result_name,
            n_obs,
            n_months,
            n_entities,
            n_regions,
        )
    else:
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


def _run_triple_diff_indicator(
    result_name: str,
    indicator_name: str,
    treated_indicator_path: Path,
    control_indicator_path: Path,
    tiltak_path: Path,
    treatment_start: str,
    treatment_type: str,
    denominator: str,
    analysis_level: str,
    treated_group: str,
    control_group: str,
    processed_dir: Path,
    control_regions: list[str] | None = None,
    enhet_mapping_path: Path | None = None,
    seasonal_adjust: bool = False,
) -> dict[str, Any] | None:
    """Run the full triple-diff analysis for a single indicator.

    Returns a :class:`TripleDiffResult` serialised to a plain dict, or
    ``None`` if skipped.
    """
    from cluster_bootstrap import wild_cluster_bootstrap_triple_diff
    from event_study import run_triple_diff_event_study
    from models import TripleDiffResult
    from prep_data import prepare_triple_diff_panel
    from regression import compute_mde
    from regression_triple_diff import (
        run_triple_diff_baseline,
        run_triple_diff_leave_one_out,
        run_triple_diff_placebo,
        run_triple_diff_preferred,
    )

    logger.info("── Preparing regular triple-diff panel for %s ──", indicator_name)
    panel_regular = prepare_triple_diff_panel(
        treated_indicator_path=treated_indicator_path,
        control_indicator_path=control_indicator_path,
        tiltak_path=tiltak_path,
        indicator_name=indicator_name,
        treatment_start=treatment_start,
        treatment_type=treatment_type,
        analysis_level=analysis_level,
        denominator=denominator,
        flatten=False,
        control_regions=control_regions,
        enhet_mapping_path=enhet_mapping_path,
        seasonal_adjust=seasonal_adjust,
        processed_path=processed_dir / f"panel_{result_name}_regular.csv",
    )

    logger.info("── Preparing flattened triple-diff panel for %s ──", indicator_name)
    panel_flattened = prepare_triple_diff_panel(
        treated_indicator_path=treated_indicator_path,
        control_indicator_path=control_indicator_path,
        tiltak_path=tiltak_path,
        indicator_name=indicator_name,
        treatment_start=treatment_start,
        treatment_type=treatment_type,
        analysis_level=analysis_level,
        denominator=denominator,
        flatten=True,
        control_regions=control_regions,
        enhet_mapping_path=enhet_mapping_path,
        seasonal_adjust=seasonal_adjust,
        processed_path=processed_dir / f"panel_{result_name}_flattened.csv",
    )

    n_post_obs = int(panel_regular["post_treatment"].sum())
    if n_post_obs == 0:
        logger.warning(
            "%s: skipping — no post-treatment months available.", result_name
        )
        return None

    n_obs = len(panel_regular)
    n_entities = panel_regular["entity"].nunique()
    n_regions = panel_regular["region"].nunique()
    logger.info(
        "%s: %d obs (%d entities, %d regions, 2 groups)",
        result_name,
        n_obs,
        n_entities,
        n_regions,
    )

    logger.info("Running triple-diff baseline for %s", result_name)
    baseline = run_triple_diff_baseline(panel_regular)

    logger.info("Running triple-diff preferred for %s", result_name)
    preferred = run_triple_diff_preferred(panel_flattened)

    models: dict[str, Any] = {
        "basis": dict(
            panel=panel_regular,
            main=baseline,
            run_model=run_triple_diff_baseline,
        ),
        "flattet": dict(
            panel=panel_flattened,
            main=preferred,
            run_model=run_triple_diff_preferred,
        ),
    }

    for key, m in models.items():
        panel = m["panel"]
        main_result = m["main"]
        logger.info("Triple-diff bootstrap (%s) for %s", key, result_name)
        m["bootstrap"] = wild_cluster_bootstrap_triple_diff(panel)
        logger.info("Triple-diff event study (%s) for %s", key, result_name)
        m["event_study"] = run_triple_diff_event_study(panel)
        logger.info("Triple-diff placebo (%s) for %s", key, result_name)
        m["placebo"] = run_triple_diff_placebo(panel, placebo_relative_month=-12)
        logger.info("Triple-diff leave-one-out (%s) for %s", key, result_name)
        m["leave_one_out"] = run_triple_diff_leave_one_out(
            panel, preferred_result=main_result
        )

    mde = compute_mde(preferred)
    logger.info("MDE for %s: %.4f pp", result_name, mde)

    pre_panel = panel_flattened[panel_flattened["relative_month"] < 0]
    baseline_mean = float(pre_panel["indikator"].mean())
    baseline_mean_treated = float(
        pre_panel.loc[pre_panel["treated"] == 1.0, "indikator"].mean()
    )
    baseline_mean_control = float(
        pre_panel.loc[pre_panel["treated"] == 0.0, "indikator"].mean()
    )
    baseline_mean_by_region = pre_panel.groupby("region")["indikator"].mean().to_dict()

    return TripleDiffResult(
        indicator_name=indicator_name,
        analysis_level=analysis_level,
        treated_group=treated_group,
        control_group=control_group,
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
        baseline_mean_treated=baseline_mean_treated,
        baseline_mean_control=baseline_mean_control,
        baseline_mean_by_region=baseline_mean_by_region,
        panel=panel_flattened,
        panel_regular=panel_regular,
    ).to_dict()


def _run_single_config(cfg_path: Path) -> int:
    """Run the full pipeline for a single config file.  Returns 0 on success."""
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
    seasonal_adjust: bool = cfg["data"].get("tiltak_seasonal_adjust", False)
    design = analysis.get("design", "did")

    all_results: dict[str, dict[str, Any] | None] = {}
    failed: list[str] = []

    if design == "triple_diff":
        # Triple-diff: pair treated/control indicators by name
        analysis_level = analysis.get("analysis_level", "region")
        treated_group = analysis.get("treated_group", "treated")
        control_group = analysis.get("control_group", "control")

        enhet_mapping_path: Path | None = None
        if analysis_level == "enhet":
            mapping_file = cfg["data"].get("enhet_mapping_file")
            if not mapping_file:
                logger.error(
                    "data.enhet_mapping_file required when analysis_level='enhet'."
                )
                return 1
            enhet_mapping_path = PROJECT_ROOT / mapping_file

        # Group indicators by name, expecting one 'treated' and one 'control' per name
        from collections import defaultdict

        indicator_groups: dict[str, dict[str, Any]] = defaultdict(dict)
        for ind in cfg["data"]["indikatorer"]:
            group = ind.get("group", "treated")
            indicator_groups[ind["name"]][group] = ind

        for name, groups in indicator_groups.items():
            if "treated" not in groups or "control" not in groups:
                logger.warning(
                    "Indicator %s missing treated or control group — skipping.", name
                )
                continue
            treated_path = PROJECT_ROOT / groups["treated"]["file"]
            control_path = PROJECT_ROOT / groups["control"]["file"]
            try:
                result = _run_triple_diff_indicator(
                    result_name=name,
                    indicator_name=name,
                    treated_indicator_path=treated_path,
                    control_indicator_path=control_path,
                    tiltak_path=tiltak_path,
                    treatment_start=treatment_start,
                    treatment_type=treatment_type,
                    denominator=denominator,
                    analysis_level=analysis_level,
                    treated_group=treated_group,
                    control_group=control_group,
                    processed_dir=processed_dir,
                    control_regions=control_regions,
                    enhet_mapping_path=enhet_mapping_path,
                    seasonal_adjust=seasonal_adjust,
                )
                all_results[name] = result
                if result is None:
                    logger.info("○ %s skipped (no post-treatment data)", name)
                else:
                    logger.info("✓ %s complete (triple-diff)", name)
            except (
                ValueError,
                np.linalg.LinAlgError,
                KeyError,
                pd.errors.MergeError,
            ):
                logger.exception("Failed to process %s (triple-diff)", name)
                failed.append(name)
                all_results[name] = None
    else:
        # Standard DiD
        analysis_level = analysis.get("analysis_level", "region")
        enhet_mapping_path_did: Path | None = None
        if analysis_level == "enhet":
            mapping_file = cfg["data"].get("enhet_mapping_file")
            if not mapping_file:
                logger.error(
                    "data.enhet_mapping_file required when analysis_level='enhet'."
                )
                return 1
            enhet_mapping_path_did = PROJECT_ROOT / mapping_file

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
                    analysis_level=analysis_level,
                    enhet_mapping_path=enhet_mapping_path_did,
                    seasonal_adjust=seasonal_adjust,
                )
                all_results[name] = result
                if result is None:
                    logger.info("○ %s skipped (no post-treatment data)", name)
                else:
                    logger.info("✓ %s complete", name)
            except (
                ValueError,
                np.linalg.LinAlgError,
                KeyError,
                pd.errors.MergeError,
            ):
                logger.exception("Failed to process %s", name)
                failed.append(name)
                all_results[name] = None

    n_done = sum(1 for v in all_results.values() if v is not None)
    n_total = len(
        indicator_groups if design == "triple_diff" else cfg["data"]["indikatorer"]
    )

    if any(v is not None for v in all_results.values()):
        _save_regression_table(all_results, tables_dir=tables_dir)
        _save_coefficients_table(all_results, tables_dir=tables_dir)

        if design == "triple_diff":
            logger.info("Generating triple-diff report")
            from report.triple_diff import generate_triple_diff_report

            generate_triple_diff_report(
                all_results=all_results,
                cfg=cfg,
                report_dir=report_dir,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
            )
            logger.info("Triple-diff report chapters written to %s", report_dir)
        else:
            logger.info("Generating report")
            from report.did import generate_report

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

        # Report + figures → quarto/<variation>/<slug>/ (standard DiD)
        # or quarto/<variation>/ (triple-diff, variation encodes the full section path)
        if design == "triple_diff":
            quarto_output_dir = QUARTO_DIR / variation
        else:
            quarto_output_dir = QUARTO_DIR / variation / config_slug
        if quarto_output_dir.exists():
            shutil.rmtree(quarto_output_dir)
        quarto_output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(report_dir, quarto_output_dir)

        shutil.rmtree(staging_root)
        logger.info("Tables promoted to %s", final_tables)
        logger.info("Report promoted to %s", quarto_output_dir)

        if design == "triple_diff":
            _update_quarto_triple_diff_chapters(QUARTO_DIR, variation)
        else:
            _update_quarto_chapters(QUARTO_DIR, variation)
        exit_code = 0

    logger.info("═══ Done (%d/%d indicators) ═══", n_done, n_total)
    return exit_code


def _resolve_path(raw: str) -> Path:
    """Resolve a config path or folder, checking cwd then configs/ directory."""
    p = Path(raw)
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    configs_candidate = (CONFIGS_DIR / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if configs_candidate.exists():
        return configs_candidate
    return cwd_candidate  # will fail with a clear error downstream


def main() -> int:
    """Entry point: run pipeline for one config file or all configs in a folder."""
    args = _parse_args()
    resolved = _resolve_path(args.config_flag or args.config)

    if resolved.is_dir():
        cfg_paths = sorted(resolved.rglob("*.yml"))
        if not cfg_paths:
            logger.error("No .yml config files found in %s", resolved)
            return 1
        logger.info("Running %d config(s) from folder: %s", len(cfg_paths), resolved)
    elif resolved.is_file():
        cfg_paths = [resolved]
    else:
        logger.error("Config path not found: %s", resolved)
        return 1

    overall_exit = 0
    for cfg_path in cfg_paths:
        exit_code = _run_single_config(cfg_path)
        if exit_code != 0:
            overall_exit = exit_code

    return overall_exit


if __name__ == "__main__":
    sys.exit(main())
