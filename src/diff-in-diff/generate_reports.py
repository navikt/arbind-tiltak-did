"""Generate Quarto report chapters from persisted analysis outputs."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
from config_matrix import CONFIGS, GeneratedConfig, get_config
from generate_data_report import generate_data_report
from quarto_utils import _update_quarto_chapters, _update_quarto_triple_diff_chapters
from report_data import load_report_data

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DID_BASE = PROJECT_ROOT / "outputs" / "did"
DATA_PROCESSED_BASE = PROJECT_ROOT / "data" / "processed"
QUARTO_DIR = PROJECT_ROOT / "quarto"
REPORT_DATA_DIRNAME = "report-data"

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reports from persisted DID analysis outputs."
    )
    parser.add_argument("--all", action="store_true", help="Generate every report.")
    parser.add_argument("--config", help="Generate one catalog configuration.")
    args = parser.parse_args()
    if args.all and args.config:
        parser.error("--all cannot be combined with --config.")
    return args


def _processed_dir_for_config(generated: GeneratedConfig) -> Path:
    return DATA_PROCESSED_BASE / generated.storage_path


def _variation(cfg: dict[str, object]) -> str:
    analysis = cfg["analysis"]
    assert isinstance(analysis, dict)
    return str(analysis.get("variation", "regioner"))


def _generate_config(generated: GeneratedConfig) -> None:
    cfg = generated.config
    output_root = OUTPUTS_DID_BASE / generated.id
    report_dir = output_root / "_report_staging"
    figures_dir = report_dir / "figures"
    tables_dir = output_root / "tables"
    all_results = load_report_data(
        output_root / REPORT_DATA_DIRNAME,
        _processed_dir_for_config(generated),
        cfg,
    )
    coefficient_path = tables_dir / "alle_koeffisienter.csv"
    if not coefficient_path.is_file():
        raise FileNotFoundError(
            f"Mangler koeffisientdata for rapporten: {coefficient_path}"
        )
    coefficients = pd.read_csv(coefficient_path)
    for indicator, result in all_results.items():
        if result is not None:
            result["coefficients"] = coefficients.loc[
                coefficients["indikator"].eq(indicator)
            ].copy()
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True)

    analysis = cfg["analysis"]
    design = analysis.get("design", "did")
    variation = _variation(cfg)
    if design == "triple_diff":
        from report.triple_diff import generate_triple_diff_report

        generate_triple_diff_report(
            all_results, cfg, report_dir, figures_dir, tables_dir
        )
        target = QUARTO_DIR / variation
    else:
        from report.did import generate_report

        generate_report(
            all_results,
            cfg,
            report_dir / f"report_{generated.id}.qmd",
            figures_dir,
            tables_dir,
        )
        target = QUARTO_DIR / variation / generated.id

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(report_dir, target)
    shutil.rmtree(report_dir)
    if design == "triple_diff":
        _update_quarto_triple_diff_chapters(QUARTO_DIR, variation)
    else:
        _update_quarto_chapters(QUARTO_DIR, variation)


def main() -> int:
    """Generate all requested analysis reports and the data report chapters."""
    args = _parse_args()
    configs = CONFIGS if args.all or args.config is None else (get_config(args.config),)
    for generated in configs:
        logger.info("Generating report for %s", generated.id)
        _generate_config(generated)
    if args.all or args.config is None:
        generate_data_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
