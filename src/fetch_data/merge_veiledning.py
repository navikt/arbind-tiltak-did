"""Merge veiledning groups into one weighted indikator series per region."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_IND_BASE = _ROOT / "data" / "raw" / "indikatorer" / "nedbrytning"
_PERS_BASE = _ROOT / "data" / "raw" / "personer" / "nedbrytning"
_OUT_DIR = _IND_BASE / "veiledning"

_GROUP_A = "Innsatsgruppe - Trenger veiledning"
_GROUP_B = "Innsatsgruppe - Trenger veiledning, nedsatt arbeidsevne"
_OUTCOMES = ("atid3", "jobb3")


def _slugify(value: str) -> str:
    """Return a filesystem-safe lower-case slug."""
    return re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("_")


def _read_wide(path: Path) -> pd.DataFrame:
    """Read a wide CSV and validate expected structure."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    if "aarmnd" not in df.columns:
        raise ValueError(f"Missing 'aarmnd' column in {path}")
    aarmnd_str = df["aarmnd"].astype(str).str.strip()
    is_yyyymm = aarmnd_str.str.fullmatch(r"\d{6}").fillna(False)
    df["aarmnd"] = aarmnd_str
    if (~is_yyyymm).any():
        converted = pd.to_datetime(aarmnd_str[~is_yyyymm], errors="raise").dt.strftime(
            "%Y%m"
        )
        df.loc[~is_yyyymm, "aarmnd"] = converted
    return df


def _to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert wide dataframe with region columns into long format."""
    return df.melt(id_vars=["aarmnd"], var_name="region", value_name=value_name)


def _merge_one_outcome(outcome: str, group_a: str, group_b: str) -> pd.DataFrame:
    """Return weighted merged indikator for one outcome in wide format."""
    g1 = _slugify(group_a)
    g2 = _slugify(group_b)

    ind1 = _read_wide(_IND_BASE / g1 / f"{outcome}.csv")
    ind2 = _read_wide(_IND_BASE / g2 / f"{outcome}.csv")
    n1 = _read_wide(_PERS_BASE / g1 / "antall_personer.csv")
    n2 = _read_wide(_PERS_BASE / g2 / "antall_personer.csv")

    df = (
        _to_long(ind1, "indikator_1")
        .merge(_to_long(ind2, "indikator_2"), on=["aarmnd", "region"], how="inner")
        .merge(_to_long(n1, "n_1"), on=["aarmnd", "region"], how="inner")
        .merge(_to_long(n2, "n_2"), on=["aarmnd", "region"], how="inner")
    )

    denom = df["n_1"] + df["n_2"]
    invalid = denom <= 0
    if invalid.any():
        raise ValueError(
            f"Non-positive denominator in weighted merge for {outcome}: "
            f"{int(invalid.sum())} rows."
        )

    df["indikator"] = (
        df["indikator_1"] * df["n_1"] + df["indikator_2"] * df["n_2"]
    ) / denom

    wide = (
        df.pivot_table(
            index="aarmnd", columns="region", values="indikator", aggfunc="first"
        )
        .sort_index()
        .reset_index()
    )
    wide.columns.name = None
    return wide


def merge_veiledning(
    group_a: str = _GROUP_A,
    group_b: str = _GROUP_B,
    outcomes: tuple[str, ...] = _OUTCOMES,
) -> list[Path]:
    """Merge two veiledning groups and save one CSV per outcome."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for outcome in outcomes:
        merged = _merge_one_outcome(outcome=outcome, group_a=group_a, group_b=group_b)
        out_path = _OUT_DIR / f"{outcome}.csv"
        merged.to_csv(out_path, index=False)
        saved.append(out_path)
    return saved


def main() -> None:
    """CLI.

    Usage:
      uv run python src/fetch_data/merge_veiledning.py
      uv run python src/fetch_data/merge_veiledning.py "<group_a>" "<group_b>"
    """
    group_a = sys.argv[1] if len(sys.argv) > 1 else _GROUP_A
    group_b = sys.argv[2] if len(sys.argv) > 2 else _GROUP_B
    saved = merge_veiledning(group_a=group_a, group_b=group_b)
    print(f"Saved {len(saved)} merged file(s):")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
