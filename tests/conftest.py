"""Shared pytest fixtures and synthetic dataset helpers.

All synthetic data is generated deterministically (no random state needed) so
tests are reproducible and coefficient values are exactly known.

Data-generating process (DGP)
------------------------------
- 5 regions (R0–R4), 2 enheter per region
- 36 months: 24 pre-treatment (202101–202212) + 12 post (202301–202312)
- Treatment start: 202301
- Tiltak: 1000 in all pre-months; in post-months region i → 1000×(1−0.2×i)
  → tiltaksnedgang ∈ {0.0, 0.2, 0.4, 0.6, 0.8}
- Indicator: region_FE + seasonal_FE + BETA × tiltaksnedgang
  (no noise → BETA is exactly recoverable)
- Triple-diff: treated group uses BETA=-1.0, control group uses BETA=0.0
  → treatment_x_group coefficient ≈ BETA
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make diff-in-diff source modules importable from tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "diff-in-diff"))

# ── Synthetic dataset constants ────────────────────────────────────────────────

REGIONS = [f"Nav R{i}" for i in range(5)]
ENHETER = {r: [f"{r} E1", f"{r} E2"] for r in REGIONS}
TREATMENT_START = "202301"
PRE_MONTHS = 24
POST_MONTHS = 12
TOTAL_MONTHS = PRE_MONTHS + POST_MONTHS

# Each region i loses fraction TILTAK_DROPS[i] of its tiltak in the post-period
TILTAK_DROPS = [0.0, 0.2, 0.4, 0.6, 0.8]
TILTAK_BASE = 1000

# Known treatment effect (used to validate regression coefficients)
BETA = -1.0

# Region fixed effects (just an offset per region)
REGION_FE = {r: float(i) for i, r in enumerate(REGIONS)}

# Month-of-year seasonal effect: 0.1 × month_of_year (1–12)
SEASONAL_SCALE = 0.1


def _gen_months(start_yyyymm: str, n: int) -> list[str]:
    start = pd.Period(start_yyyymm, freq="M")
    return [(start + i).strftime("%Y%m") for i in range(n)]


# All month strings (YYYYMM) for the synthetic panel
_treatment_period = pd.Period(TREATMENT_START, freq="M")
_data_start = _treatment_period - PRE_MONTHS
ALL_MONTHS: list[str] = _gen_months(str(_data_start), TOTAL_MONTHS)


def _relative_month(yyyymm: str) -> int:
    """Compute the relative month (0 at treatment start) for a YYYYMM string."""
    t_year, t_month = int(TREATMENT_START[:4]), int(TREATMENT_START[4:])
    return (int(yyyymm[:4]) - t_year) * 12 + (int(yyyymm[4:]) - t_month)


def _is_post(month_yyyymm: str) -> bool:
    return _relative_month(month_yyyymm) >= 0


def _month_of_year(yyyymm: str) -> int:
    return int(yyyymm[4:6])


def _tiltak(region_idx: int, month_yyyymm: str) -> float:
    """Synthetic tiltak for a given region and month.

    - Months earlier than -12: flat at TILTAK_BASE (1000).
    - Months -12 to -1 (placebo window): small decline (20 % of the post drop).
      This gives the placebo test non-zero treatment variation.
    - Post-treatment (months ≥ 0): large decline proportional to TILTAK_DROPS[i].

    The peak pre-treatment tiltak is always TILTAK_BASE (achieved in months ≤ -13),
    so "peak" denominator yields tiltaksnedgang = TILTAK_DROPS[i] in the post-period,
    exactly matching the DGP.
    """
    rel = _relative_month(month_yyyymm)
    if rel >= 0:
        return TILTAK_BASE * (1 - TILTAK_DROPS[region_idx])
    elif rel >= -12:
        # Small pre-period decline so the placebo test has non-zero variation.
        return TILTAK_BASE * (1 - 0.2 * TILTAK_DROPS[region_idx])
    else:
        return float(TILTAK_BASE)


def _tiltaksnedgang(region_idx: int, month_yyyymm: str) -> float:
    """Compute the synthetic tiltaksnedgang (0 in pre, varies in post)."""
    if not _is_post(month_yyyymm):
        return 0.0
    return TILTAK_DROPS[region_idx]


def _indicator(region: str, month_yyyymm: str, beta: float = BETA) -> float:
    """Compute the synthetic indicator value for a region-month pair."""
    region_idx = REGIONS.index(region)
    seasonal = SEASONAL_SCALE * _month_of_year(month_yyyymm)
    treatment = beta * _tiltaksnedgang(region_idx, month_yyyymm)
    return REGION_FE[region] + seasonal + treatment


# ── CSV / JSON writers ─────────────────────────────────────────────────────────


def write_tiltak_csv(path: Path) -> None:
    """Write synthetic tiltak CSV in wide format (quoted column names, TOTAL column).

    Mimics the real data format: date as %Y-%m-%d, quoted region names, TOTAL col.
    """
    rows = []
    for m in ALL_MONTHS:
        p = pd.Period(m, freq="M")
        date_str = p.to_timestamp().strftime("%Y-%m-%d")
        row: dict = {"aarmnd": date_str}
        total = 0.0
        for i, r in enumerate(REGIONS):
            v = _tiltak(i, m)
            row[f"'{r}'"] = v
            total += v
        row["TOTAL"] = total
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_indicator_wide_csv(path: Path, beta: float = BETA) -> None:
    """Write synthetic wide-format indicator CSV (rows=months, cols=regions)."""
    rows = []
    for m in ALL_MONTHS:
        row: dict = {"aarmnd": m}
        for r in REGIONS:
            row[r] = _indicator(r, m, beta=beta)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_indicator_long_csv(path: Path, beta: float = BETA) -> None:
    """Write synthetic long-format indicator CSV (aarmnd, enhet, indikator)."""
    rows = []
    for m in ALL_MONTHS:
        for r, enheter_list in ENHETER.items():
            val = _indicator(r, m, beta=beta)
            for enhet in enheter_list:
                rows.append({"aarmnd": m, "enhet": enhet, "indikator": val})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_enhet_mapping_json(path: Path) -> None:
    """Write synthetic enhet→region mapping JSON in the expected format."""
    items = []
    for r, enheter_list in ENHETER.items():
        for enhet in enheter_list:
            items.append({"nav_enhet_navn": enhet, "nav_region_navn": r})
    payload = {"results": [{"items": items}]}
    path.write_text(json.dumps(payload), encoding="utf-8")


# ── Pytest fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def tiltak_csv(tmp_path: Path) -> Path:
    """Path to a synthetic tiltak CSV file."""
    p = tmp_path / "tiltak.csv"
    write_tiltak_csv(p)
    return p


@pytest.fixture()
def indicator_wide_csv(tmp_path: Path) -> Path:
    """Path to a synthetic wide-format indicator CSV (default beta=-1.0)."""
    p = tmp_path / "indicator.csv"
    write_indicator_wide_csv(p)
    return p


@pytest.fixture()
def indicator_wide_csv_zero_beta(tmp_path: Path) -> Path:
    """Wide-format indicator CSV with beta=0 (control group for triple-diff)."""
    p = tmp_path / "indicator_control.csv"
    write_indicator_wide_csv(p, beta=0.0)
    return p


@pytest.fixture()
def indicator_long_csv(tmp_path: Path) -> Path:
    """Path to a synthetic long-format (enhet-level) indicator CSV."""
    p = tmp_path / "indicator_long.csv"
    write_indicator_long_csv(p)
    return p


@pytest.fixture()
def indicator_long_csv_zero_beta(tmp_path: Path) -> Path:
    """Long-format indicator CSV with beta=0 (control group for triple-diff)."""
    p = tmp_path / "indicator_long_control.csv"
    write_indicator_long_csv(p, beta=0.0)
    return p


@pytest.fixture()
def enhet_mapping_json(tmp_path: Path) -> Path:
    """Path to a synthetic enhetsmapping.json file."""
    p = tmp_path / "enhetsmapping.json"
    write_enhet_mapping_json(p)
    return p


@pytest.fixture()
def synthetic_panel(tiltak_csv: Path, indicator_wide_csv: Path) -> "pd.DataFrame":
    """A prepared (region-level, continuous) panel ready for regression."""
    from prep_data import prepare_panel

    return prepare_panel(
        indicator_path=indicator_wide_csv,
        tiltak_path=tiltak_csv,
        indicator_name="indikator",
        treatment_start=TREATMENT_START,
        treatment_type="continuous",
        denominator="peak",
        flatten=False,
    )


@pytest.fixture()
def synthetic_panel_flat(tiltak_csv: Path, indicator_wide_csv: Path) -> "pd.DataFrame":
    """A prepared, seasonally flattened region-level panel."""
    from prep_data import prepare_panel

    return prepare_panel(
        indicator_path=indicator_wide_csv,
        tiltak_path=tiltak_csv,
        indicator_name="indikator",
        treatment_start=TREATMENT_START,
        treatment_type="continuous",
        denominator="peak",
        flatten=True,
    )


@pytest.fixture()
def synthetic_triple_diff_panel(
    tiltak_csv: Path,
    indicator_wide_csv: Path,
    indicator_wide_csv_zero_beta: Path,
) -> "pd.DataFrame":
    """A prepared triple-diff panel (treated beta=-1.0, control beta=0).

    Uses "peak" denominator so tiltaksnedgang matches the DGP exactly.
    """
    from prep_data import prepare_triple_diff_panel

    return prepare_triple_diff_panel(
        treated_indicator_path=indicator_wide_csv,
        control_indicator_path=indicator_wide_csv_zero_beta,
        tiltak_path=tiltak_csv,
        indicator_name="indikator",
        treatment_start=TREATMENT_START,
        treatment_type="continuous",
        denominator="peak",
        flatten=False,
    )


@pytest.fixture()
def synthetic_triple_diff_panel_flat(
    tiltak_csv: Path,
    indicator_wide_csv: Path,
    indicator_wide_csv_zero_beta: Path,
) -> "pd.DataFrame":
    """A prepared, seasonally flattened triple-diff panel.

    Uses "peak" denominator so tiltaksnedgang matches the DGP exactly.
    """
    from prep_data import prepare_triple_diff_panel

    return prepare_triple_diff_panel(
        treated_indicator_path=indicator_wide_csv,
        control_indicator_path=indicator_wide_csv_zero_beta,
        tiltak_path=tiltak_csv,
        indicator_name="indikator",
        treatment_start=TREATMENT_START,
        treatment_type="continuous",
        denominator="peak",
        flatten=True,
    )
