"""
Preps data for analysis. Mainly feature engineering. 
"""

from pathlib import Path

import pandas as pd
import yaml

# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "analysis-config.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the analysis config from *path*."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _convert_aarmnd_format(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convert *date_col* in *df* to YYYYMM format if it's in a recognised date format."""
    if date_col in df.columns:
        for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
            try:
                df[date_col] = pd.to_datetime(
                    df[date_col], format=fmt
                ).dt.strftime("%Y%m")
                break
            except (ValueError, TypeError):
                continue
    return df

def convert_to_long_format(df: pd.DataFrame, id_vars: list[str], value_vars_prefix: str) -> pd.DataFrame:
    """Convert *df* from wide to long format based on *id_vars* and *value_vars_prefix*."""
    value_vars = [col for col in df.columns if col.startswith(value_vars_prefix)]
    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="variable", value_name="value")
    return long_df

def load_and_merge(indikator_path: Path, tiltak_path: Path) -> pd.DataFrame:
    """Load indicator and tiltak data, standardize date formats, convert to long format, and merge on common keys."""
    indikator_df = pd.read_csv(indikator_path)
    tiltak_df = pd.read_csv(tiltak_path)

    indikator_df = _convert_aarmnd_format(indikator_df, "aarmnd")
    tiltak_df = _convert_aarmnd_format(tiltak_df, "aarmnd")

    indikator_df = convert_to_long_format(indikator_df, id_vars=["region", "aarmnd"], value_vars_prefix="indikator")
    tiltak_df = convert_to_long_format(tiltak_df, id_vars=["region", "aarmnd"], value_vars_prefix="tiltak")

    merged_df = pd.merge(indikator_df, tiltak_df, on=["region", "aarmnd"], how="left")
    return merged_df

def _add_time_features(df: pd.DataFrame, treatment_start: str) -> pd.DataFrame:
    """Add time-based features to *df*."""
    df["aarmnd"] = pd.to_datetime(df["aarmnd"], format="%Y%m")
    df["year"] = df["aarmnd"].dt.year
    df["month_of_year"] = df["aarmnd"].dt.month
    df["relative_month"] = (
        (df["aarmnd"].dt.year - int(treatment_start[:4])) * 12 + 
        (df["aarmnd"].dt.month - int(treatment_start[4:]))
    )
    df["post_treatment"] = df["relative_month"] > 0
    return df

def build_treatment_variable(
        df: pd.DataFrame,
        treatment_type: str,
        denominator: str = "peak",
    ) -> pd.DataFrame:
    """Create a continuous treatment variable (tiltaksnedgang).

    Parameters
    ----------
    df:
        Panel DataFrame with columns ``region``, ``relative_month``, ``tiltak``.
    treatment_type:
        Currently only ``"continuous"`` is implemented.
    denominator:
        How to compute the reference level per region:
        - ``"peak"``     – maximum tiltak count in the pre-period (relative_month < 0)
        - ``"last_pre"`` – tiltak count in the last pre-treatment month (relative_month == -1)
    """
    pre_mask = df["relative_month"] < 0
    post_mask = ~pre_mask
    if treatment_type == "continuous":
        if denominator == "peak":
            ref = (
                df.loc[pre_mask, ["region", "tiltak"]]
                .groupby("region")["tiltak"]
                .max()
                .rename("ref_tiltak")
            )
        elif denominator == "last_pre":
            ref = (
                df.loc[df["relative_month"] == -1, ["region", "tiltak"]]
                .set_index("region")["tiltak"]
                .rename("ref_tiltak")
            )
        else:
            raise ValueError(f"Unknown denominator '{denominator}'. Use 'peak' or 'last_pre'.")

        df = df.merge(ref, on="region", how="left")
        df["tiltaksnedgang"] = 0.0
        post_fraction = (
            df.loc[post_mask, "ref_tiltak"] - df.loc[post_mask, "tiltak"]
        ) / df.loc[post_mask, "ref_tiltak"]
        df.loc[post_mask, "tiltaksnedgang"] = post_fraction.clip(lower=0.0, upper=1.0)
        # Rename for clarity in the output
        df = df.rename(columns={"ref_tiltak": "peak_tiltak"})
    elif treatment_type == "discrete":
        # Implement discrete treatment logic here
        pass
    return df

def prepare_panel(
    indicator_path: Path,
    tiltak_path: Path,
    indicator_name: str,
    treatment_start: str,
    treatment_type: str,
    denominator: str = "peak",
    processed_path: Path | None = None,
) -> pd.DataFrame:
    """Prepare a panel DataFrame based on the specified indicator and tiltak data.

    Parameters
    ----------
    indicator_path, tiltak_path:
        Paths to the raw CSV files.
    indicator_name:
        Name used as the outcome column in the merged panel.
    treatment_start:
        First treatment month in YYYYMM format.
    treatment_type:
        ``"continuous"`` or ``"discrete"``.
    denominator:
        Reference level for tiltaksnedgang: ``"peak"`` or ``"last_pre"``.
    processed_path:
        If given, save the prepared panel as CSV at this path.
    """
    # Load data
    indicator_df = pd.read_csv(indicator_path)
    tiltak_df = pd.read_csv(tiltak_path)

    def _convert_aarmnd_format(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Convert *date_col* in *df* to YYYYMM format if it's in a recognised date format."""
        if date_col in df.columns:
            for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
                try:
                    df[date_col] = pd.to_datetime(
                        df[date_col], format=fmt
                    ).dt.strftime("%Y%m")
                    break
                except (ValueError, TypeError):
                    continue
        return df

    # Strip surrounding quotes from tiltak column names and drop TOTAL
    tiltak_df.columns = [c.strip("'") for c in tiltak_df.columns]
    tiltak_df = tiltak_df.drop(columns=["TOTAL"], errors="ignore")
    # Drop any duplicate header rows embedded in the CSV data, then re-cast numerics
    tiltak_df = tiltak_df[tiltak_df["aarmnd"] != "aarmnd"].reset_index(drop=True)
    for col in tiltak_df.columns:
        if col != "aarmnd":
            tiltak_df[col] = pd.to_numeric(tiltak_df[col], errors="coerce")

    # Standardize aarmnd to YYYYMM string format in both dataframes
    indicator_df = _convert_aarmnd_format(indicator_df, "aarmnd")
    tiltak_df = _convert_aarmnd_format(tiltak_df, "aarmnd")
    indicator_df["aarmnd"] = indicator_df["aarmnd"].astype(str)
    tiltak_df["aarmnd"] = tiltak_df["aarmnd"].astype(str)

    # Convert from wide to long format so region becomes a column
    indicator_df = indicator_df.melt(
        id_vars=["aarmnd"], var_name="region", value_name="indikator"
    )
    tiltak_df = tiltak_df.melt(
        id_vars=["aarmnd"], var_name="region", value_name="tiltak"
    )

    # Merge datasets on common keys (e.g., region, month)
    df = pd.merge(indicator_df, tiltak_df, on=["region", "aarmnd"], how="left")
    # Feature engineering: Add time features, lags, etc.
    df = _add_time_features(df, treatment_start)
    df = build_treatment_variable(df, treatment_type, denominator=denominator)
    # Add more feature engineering steps here as needed.

    if processed_path is not None:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["period"], errors="ignore").to_csv(
            processed_path, index=False
        )
    return df

if __name__ == "__main__":
    cfg = load_config()
    treatment_start = cfg["analysis"]["treatment_start"]
    treatment_type = cfg["analysis"]["treatment_type"]
    tiltak_path = Path(cfg["data"]["tiltak_file"])

    for ind in cfg["data"]["indikatorer"]:
        prepare_panel(
            indicator_path=Path(ind["file"]),
            tiltak_path=tiltak_path,
            indicator_name=ind["name"],
            treatment_start=treatment_start,
            treatment_type=treatment_type,
            denominator="peak",
            processed_path=Path(f"data/processed/panel_regioner_lønnstilskudd_{ind['name']}.csv"),
        )
