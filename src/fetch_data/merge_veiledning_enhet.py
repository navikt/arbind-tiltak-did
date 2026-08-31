"""Merge veiledning groups into weighted indicator series per enhet.

Mirrors merge_veiledning.py but operates on long-format enhet-level data
produced by get_enhet_data.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_IND_BASE = _ROOT / "data" / "input" / "indikatorer" / "enhet" / "nedbrytning"
_PERS_BASE = _ROOT / "data" / "input" / "personer" / "enhet" / "nedbrytning"
_OUT_DIR = _IND_BASE / "veiledning_kombinert"
_PERSON_OUT_DIR = _PERS_BASE / "veiledning_kombinert"

_GROUP_A = "Situasjonsbestemt"
_GROUP_B = "Spesielt tilpasset"
_OUTCOMES = (
    "atid3",
    "jobb3",
    "faktisk_atid3",
    "faktisk_jobb3",
    "forventet_atid3",
    "forventet_jobb3",
)


def _slugify(value: str) -> str:
    """Return a filesystem-safe lower-case slug."""
    return re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("_")


def _read_long(path: Path) -> pd.DataFrame:
    """Read a long-format CSV and validate expected structure."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    if "aarmnd" not in df.columns or "enhet" not in df.columns:
        raise ValueError(f"Missing 'aarmnd' or 'enhet' column in {path}")
    aarmnd_str = df["aarmnd"].astype(str).str.strip()
    is_yyyymm = aarmnd_str.str.fullmatch(r"\d{6}").fillna(False)
    df["aarmnd"] = aarmnd_str
    if (~is_yyyymm).any():
        converted = pd.to_datetime(aarmnd_str[~is_yyyymm], errors="raise").dt.strftime(
            "%Y%m"
        )
        df.loc[~is_yyyymm, "aarmnd"] = converted
    return df


def _merge_one_outcome(outcome: str, group_a: str, group_b: str) -> pd.DataFrame:
    """Return a weighted merged series for one outcome in long format."""
    g1 = _slugify(group_a)
    g2 = _slugify(group_b)

    ind1 = _read_long(_IND_BASE / g1 / f"{outcome}.csv")
    ind2 = _read_long(_IND_BASE / g2 / f"{outcome}.csv")
    n1 = _read_long(_PERS_BASE / g1 / "antall_personer.csv")
    n2 = _read_long(_PERS_BASE / g2 / "antall_personer.csv")

    value_column = (
        "faktisk"
        if outcome.startswith("faktisk_")
        else "forventet"
        if outcome.startswith("forventet_")
        else "indikator"
    )
    expected = {"aarmnd", "enhet", value_column}
    for path, df in (
        (_IND_BASE / g1 / f"{outcome}.csv", ind1),
        (_IND_BASE / g2 / f"{outcome}.csv", ind2),
    ):
        if not expected.issubset(df.columns):
            raise ValueError(
                f"Long-format indicator file {path} must have columns {expected}, "
                f"found {set(df.columns)}"
            )

    # Standardize source value names so all series use the same weighted merge.
    ind1 = ind1.rename(columns={value_column: "indikator_1"})
    ind2 = ind2.rename(columns={value_column: "indikator_2"})
    n1 = n1.rename(columns={"antall_personer": "n_1"})
    n2 = n2.rename(columns={"antall_personer": "n_2"})

    df = (
        ind1.merge(ind2, on=["aarmnd", "enhet"], how="inner")
        .merge(n1, on=["aarmnd", "enhet"], how="inner")
        .merge(n2, on=["aarmnd", "enhet"], how="inner")
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

    return (
        df[["aarmnd", "enhet", "indikator"]]
        .sort_values(["aarmnd", "enhet"])
        .reset_index(drop=True)
    )


def _merge_person_counts(group_a: str, group_b: str) -> pd.DataFrame:
    """Return combined counts for the two source groups at unit level."""
    g1 = _slugify(group_a)
    g2 = _slugify(group_b)
    n1 = _read_long(_PERS_BASE / g1 / "antall_personer.csv").rename(
        columns={"antall_personer": "n_1"}
    )
    n2 = _read_long(_PERS_BASE / g2 / "antall_personer.csv").rename(
        columns={"antall_personer": "n_2"}
    )
    counts = n1.merge(n2, on=["aarmnd", "enhet"], how="outer", validate="one_to_one")
    # A missing group-month is an absent subgroup, not an unknown total.
    counts[["n_1", "n_2"]] = counts[["n_1", "n_2"]].fillna(0)
    if (counts[["n_1", "n_2"]] < 0).any().any():
        raise ValueError("Negative person counts while merging veiledning groups.")
    counts["antall_personer"] = counts["n_1"] + counts["n_2"]
    return (
        counts[["aarmnd", "enhet", "antall_personer"]]
        .sort_values(["aarmnd", "enhet"])
        .reset_index(drop=True)
    )


def merge_veiledning_enhet(
    group_a: str = _GROUP_A,
    group_b: str = _GROUP_B,
    outcomes: tuple[str, ...] = _OUTCOMES,
) -> list[Path]:
    """Merge veiledning groups and save indicator, faktisk, and forventet CSVs."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _PERSON_OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for outcome in outcomes:
        merged = _merge_one_outcome(outcome=outcome, group_a=group_a, group_b=group_b)
        out_path = _OUT_DIR / f"{outcome}.csv"
        merged.to_csv(out_path, index=False)
        saved.append(out_path)
    person_counts_path = _PERSON_OUT_DIR / "antall_personer.csv"
    _merge_person_counts(group_a, group_b).to_csv(person_counts_path, index=False)
    saved.append(person_counts_path)
    return saved


def main() -> None:
    """CLI.

    Usage:
      uv run python src/fetch_data/merge_veiledning_enhet.py
      uv run python src/fetch_data/merge_veiledning_enhet.py "<group_a>" "<group_b>"
    """
    group_a = sys.argv[1] if len(sys.argv) > 1 else _GROUP_A
    group_b = sys.argv[2] if len(sys.argv) > 2 else _GROUP_B
    saved = merge_veiledning_enhet(group_a=group_a, group_b=group_b)
    print(f"Saved {len(saved)} merged file(s):")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
