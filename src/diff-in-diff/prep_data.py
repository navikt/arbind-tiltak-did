"""Preps data for analysis. Mainly feature engineering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from data_io import (
    _convert_aarmnd_format,
    _load_enhet_mapping,
    _load_indicator_long,
    _load_indicator_wide_to_long,
    _load_tiltak_wide_to_long,
)

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "analysis-config.yml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the analysis config from *path*."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _add_time_features(df: pd.DataFrame, treatment_start: str) -> pd.DataFrame:
    """Add time-based features to *df*."""
    df["aarmnd"] = pd.to_datetime(df["aarmnd"], format="%Y%m")
    df["year"] = df["aarmnd"].dt.year
    df["month_of_year"] = df["aarmnd"].dt.month
    df["relative_month"] = (df["aarmnd"].dt.year - int(treatment_start[:4])) * 12 + (
        df["aarmnd"].dt.month - int(treatment_start[4:])
    )
    # Treatment starts in the configured treatment month (relative_month == 0).
    df["post_treatment"] = df["relative_month"] >= 0
    return df


def build_treatment_variable(
    df: pd.DataFrame,
    treatment_type: str,
    denominator: str = "peak",
    control_regions: list[str] | None = None,
) -> pd.DataFrame:
    """Create the treatment variable (``tiltaksnedgang``) for the panel.

    Parameters
    ----------
    df:
        Panel DataFrame with columns ``region``, ``relative_month``, ``tiltak``.
    treatment_type:
        ``"continuous"`` or ``"discrete"``.
    denominator:
        For ``"continuous"`` only.  How to compute the reference level per region:
        - ``"peak"``     – maximum tiltak count in the pre-period (relative_month < 0)
        - ``"last_pre"`` – tiltak count in the last pre-treatment month (relative_month == -1)
    control_regions:
        For ``"discrete"`` only.  List of region names that serve as the
        control group.  All other regions are treated.
    """
    df = df.reset_index(drop=True)
    pre_mask = df["relative_month"] < 0
    post_mask = df["relative_month"] >= 0
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
                .drop_duplicates(subset=["region"])
                .set_index("region")["tiltak"]
                .rename("ref_tiltak")
            )
        else:
            raise ValueError(
                f"Unknown denominator '{denominator}'. Use 'peak' or 'last_pre'."
            )

        df = df.merge(ref, on="region", how="left")

        invalid_ref = post_mask & (df["ref_tiltak"].isna() | (df["ref_tiltak"] <= 0))
        if invalid_ref.any():
            bad_regions = sorted(
                df.loc[invalid_ref, "region"].dropna().unique().tolist()
            )
            sample = ", ".join(bad_regions[:5])
            raise ValueError(
                "Cannot construct tiltaksnedgang: missing/non-positive denominator "
                f"for {int(invalid_ref.sum())} post-treatment observations "
                f"across {len(bad_regions)} regions (examples: {sample})."
            )

        df["tiltaksnedgang"] = 0.0
        post_num = (
            df.loc[post_mask, "ref_tiltak"].values - df.loc[post_mask, "tiltak"].values
        )
        post_den = df.loc[post_mask, "ref_tiltak"].values
        post_fraction = np.where(post_den > 0, post_num / post_den, np.nan)
        df.loc[post_mask, "tiltaksnedgang"] = np.clip(post_fraction, 0.0, 1.0)
        # Rename for clarity in the output
        df = df.rename(columns={"ref_tiltak": "peak_tiltak"})
    elif treatment_type == "discrete":
        if control_regions is None:
            raise ValueError(
                "control_regions must be provided for treatment_type='discrete'."
            )
        controll_set = set(control_regions)
        # Binary indicator: 1 for treated regions, 0 for control regions
        df["treated"] = (~df["region"].isin(controll_set)).astype(float)
        # Treatment variable is 1 only for treated regions in the post-period
        df["tiltaksnedgang"] = 0.0
        df.loc[post_mask & (df["treated"] == 1.0), "tiltaksnedgang"] = 1.0
    else:
        raise ValueError(
            f"Unknown treatment_type '{treatment_type}'. "
            "Use 'continuous' or 'discrete'."
        )
    return df


def _flatten_indicator_seasonally(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonally flatten ``indikator`` using pre-treatment months only.

    For each entity and month_of_year:
      indikator_flat = indikator - mean_pre(entity, month) + mean_pre(entity)

    Uses the ``entity`` column, which equals ``region`` for region-level panels
    and ``enhet`` for enhet-level panels.
    """
    pre = df[df["relative_month"] < 0]
    if pre.empty:
        raise ValueError(
            "Cannot flatten indikator: no pre-treatment observations available."
        )

    entity_mean = pre.groupby("entity")["indikator"].mean().rename("pre_entity_mean")
    entity_month_mean = (
        pre.groupby(["entity", "month_of_year"])["indikator"]
        .mean()
        .rename("pre_entity_month_mean")
    )

    out = df.merge(entity_mean, on="entity", how="left")
    out = out.merge(entity_month_mean, on=["entity", "month_of_year"], how="left")

    missing_means = out["pre_entity_mean"].isna() | out["pre_entity_month_mean"].isna()
    if missing_means.any():
        bad_entities = sorted(
            out.loc[missing_means, "entity"].dropna().unique().tolist()
        )
        sample = ", ".join(bad_entities[:5])
        logger.warning(
            "Dropping %d observations across %d entities with missing pre-period "
            "seasonal means (examples: %s).",
            int(missing_means.sum()),
            len(bad_entities),
            sample,
        )
        out = out[~missing_means]

    out["indikator"] = (
        out["indikator"] - out["pre_entity_month_mean"] + out["pre_entity_mean"]
    )
    return out.drop(columns=["pre_entity_mean", "pre_entity_month_mean"])


def prepare_panel(
    indicator_path: Path,
    tiltak_path: Path,
    indicator_name: str,
    treatment_start: str,
    treatment_type: str,
    denominator: str = "peak",
    flatten: bool = False,
    control_regions: list[str] | None = None,
    analysis_level: str = "region",
    enhet_mapping_path: Path | None = None,
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
        Only used when ``treatment_type="continuous"``.
    flatten:
        If ``True``, seasonally flatten ``indikator`` by subtracting each
        entity-month pre-treatment mean and adding the entity pre-treatment mean.
    control_regions:
        List of region names that serve as the control group.
        Required when ``treatment_type="discrete"``.
    analysis_level:
        ``"region"`` (default) or ``"enhet"``.  For ``"enhet"``, the indicator
        CSV is expected in long format (``aarmnd``, ``enhet``, ``indikator``)
        and treatment is assigned at the region level via *enhet_mapping_path*.
    enhet_mapping_path:
        Path to ``enhetsmapping.json`` (required when ``analysis_level="enhet"``).
    processed_path:
        If given, save the prepared panel as CSV at this path.

    Column contract for all downstream modules
    ------------------------------------------
    ``region``, ``aarmnd`` (datetime), ``indikator`` (float), ``tiltak`` (float),
    ``tiltaksnedgang`` (float 0–1), ``relative_month`` (int),
    ``post_treatment`` (bool), ``month_of_year`` (int), ``year`` (int),
    ``entity`` (str — equals ``region`` at region level, ``enhet`` at enhet level).
    """
    tiltak_long = _load_tiltak_wide_to_long(tiltak_path)

    if analysis_level == "enhet":
        if enhet_mapping_path is None:
            raise ValueError(
                "enhet_mapping_path is required when analysis_level='enhet'."
            )
        indicator_df = _load_indicator_long(indicator_path)
        indicator_df = _convert_aarmnd_format(indicator_df, "aarmnd")
        indicator_df["aarmnd"] = indicator_df["aarmnd"].astype(str)

        mapping = _load_enhet_mapping(enhet_mapping_path)
        indicator_df = indicator_df.merge(mapping, on="enhet", how="left")
        unmapped = indicator_df["region"].isna().sum()
        if unmapped > 0:
            logger.warning("%d rows have no region mapping; dropping them.", unmapped)
            indicator_df = indicator_df.dropna(subset=["region"])

        df = pd.merge(indicator_df, tiltak_long, on=["region", "aarmnd"], how="left")
        df["entity"] = df["enhet"]
    else:
        indicator_df = pd.read_csv(indicator_path)
        indicator_df = _convert_aarmnd_format(indicator_df, "aarmnd")
        indicator_df["aarmnd"] = indicator_df["aarmnd"].astype(str)
        indicator_df = indicator_df.melt(
            id_vars=["aarmnd"], var_name="region", value_name="indikator"
        )

        df = pd.merge(indicator_df, tiltak_long, on=["region", "aarmnd"], how="left")
        df["entity"] = df["region"]

    df = _add_time_features(df, treatment_start)
    if flatten:
        df = _flatten_indicator_seasonally(df)
    df = build_treatment_variable(
        df, treatment_type, denominator=denominator, control_regions=control_regions
    )

    if processed_path is not None:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["period"], errors="ignore").to_csv(processed_path, index=False)
    return df


# ── Triple-diff data preparation ──────────────────────────────────────────────


def prepare_triple_diff_panel(
    treated_indicator_path: Path,
    control_indicator_path: Path,
    tiltak_path: Path,
    indicator_name: str,
    treatment_start: str,
    treatment_type: str,
    analysis_level: str = "region",
    denominator: str = "last_pre",
    flatten: bool = False,
    control_regions: list[str] | None = None,
    enhet_mapping_path: Path | None = None,
    processed_path: Path | None = None,
) -> pd.DataFrame:
    """Prepare a triple-diff panel with treated and control groups stacked.

    Parameters
    ----------
    treated_indicator_path:
        Path to the indicator CSV for the treated group (e.g., veiledning).
    control_indicator_path:
        Path to the indicator CSV for the control group (e.g., gode muligheter).
    tiltak_path:
        Path to the tiltak CSV (region-level, wide format).
    indicator_name:
        Name used as the outcome column.
    treatment_start:
        First treatment month in YYYYMM format.
    treatment_type:
        ``"continuous"`` or ``"discrete"``.
    analysis_level:
        ``"region"`` or ``"enhet"``.
    denominator:
        Reference level for tiltaksnedgang (for continuous treatment).
    flatten:
        If ``True``, seasonally flatten the indicator.
    control_regions:
        For ``"discrete"`` treatment type only.
    enhet_mapping_path:
        Path to enhetsmapping.json (required when ``analysis_level="enhet"``).
    processed_path:
        If given, save the prepared panel as CSV at this path.

    Returns:
    -------
    DataFrame with columns including ``entity`` (region or enhet), ``region``,
    ``group``, ``treated``, ``tiltaksnedgang``, ``treatment_x_group``,
    ``indikator``, time features, etc.
    """
    if analysis_level == "enhet":
        if enhet_mapping_path is None:
            raise ValueError(
                "enhet_mapping_path is required when analysis_level='enhet'."
            )
        mapping = _load_enhet_mapping(enhet_mapping_path)

        treated_df = _load_indicator_long(treated_indicator_path)
        treated_df["group"] = "treated"
        control_df = _load_indicator_long(control_indicator_path)
        control_df["group"] = "control"

        indicator_df = pd.concat([treated_df, control_df], ignore_index=True)
        indicator_df = _convert_aarmnd_format(indicator_df, "aarmnd")
        indicator_df["aarmnd"] = indicator_df["aarmnd"].astype(str)

        # Join with mapping to get region
        indicator_df = indicator_df.merge(mapping, on="enhet", how="left")
        unmapped = indicator_df["region"].isna().sum()
        if unmapped > 0:
            logger.warning("%d rows have no region mapping; dropping them.", unmapped)
            indicator_df = indicator_df.dropna(subset=["region"])

        indicator_df["entity"] = indicator_df["enhet"]
    else:
        treated_df = _load_indicator_wide_to_long(treated_indicator_path)
        treated_df["group"] = "treated"
        control_df = _load_indicator_wide_to_long(control_indicator_path)
        control_df["group"] = "control"

        indicator_df = pd.concat([treated_df, control_df], ignore_index=True)
        indicator_df["entity"] = indicator_df["region"]

    tiltak_long = _load_tiltak_wide_to_long(tiltak_path)

    # Merge indicator with tiltak on region + aarmnd
    df = indicator_df.merge(tiltak_long, on=["region", "aarmnd"], how="left")

    # Treated indicator (1 for treated group, 0 for control)
    df["treated"] = (df["group"] == "treated").astype(float)

    # Time features
    df = _add_time_features(df, treatment_start)

    # Seasonal flattening (per entity × group combination)
    if flatten:
        df = _flatten_indicator_triple_diff(df)

    # Treatment variable (region-level)
    df = build_treatment_variable(
        df,
        treatment_type,
        denominator=denominator,
        control_regions=control_regions,
    )

    # Triple-diff interaction: treatment × treated_group
    df["treatment_x_group"] = df["tiltaksnedgang"] * df["treated"]

    if processed_path is not None:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["period"], errors="ignore").to_csv(processed_path, index=False)
    return df


def _flatten_indicator_triple_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonally flatten ``indikator`` per entity × group in a triple-diff panel.

    For each (entity, group) and month_of_year:
      indikator_flat = indikator - mean_pre(entity, group, month) + mean_pre(entity, group)
    """
    pre = df[df["relative_month"] < 0]
    if pre.empty:
        raise ValueError(
            "Cannot flatten indikator: no pre-treatment observations available."
        )

    group_key = ["entity", "group"]
    entity_group_mean = pre.groupby(group_key)["indikator"].mean().rename("pre_eg_mean")
    entity_group_month_mean = (
        pre.groupby([*group_key, "month_of_year"])["indikator"]
        .mean()
        .rename("pre_egm_mean")
    )

    out = df.merge(entity_group_mean, on=group_key, how="left")
    out = out.merge(
        entity_group_month_mean, on=[*group_key, "month_of_year"], how="left"
    )

    missing = out["pre_eg_mean"].isna() | out["pre_egm_mean"].isna()
    if missing.any():
        n_missing = int(missing.sum())
        logger.warning(
            "Dropping %d rows with missing seasonal means during flatten.", n_missing
        )
        out = out[~missing]

    out["indikator"] = out["indikator"] - out["pre_egm_mean"] + out["pre_eg_mean"]
    return out.drop(columns=["pre_eg_mean", "pre_egm_mean"])
