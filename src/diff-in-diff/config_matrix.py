"""Build runtime analysis configurations from the declarative YAML catalog."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml

CATALOG_PATH = Path(__file__).parent / "configs" / "catalog.yml"


def _label(key: str) -> str:
    return key.replace("-", " ").capitalize()


def _level_data(level: str) -> dict[str, str]:
    return {
        "region": {"slug": "regioner", "label": "Regioner", "directory": ""},
        "enhet": {"slug": "enheter", "label": "Enheter", "directory": "enhet/"},
    }[level]


def _measure_data(catalog: dict[str, Any], measure: str) -> dict[str, Any]:
    filename = catalog["measures"][measure]
    if measure == "midl-lonnstilskudd":
        return {
            "title": "Midl. lønnstilskudd",
            "label": "Midlertidig lønnstilskudd",
            "file": f"data/input/tiltak/{filename}",
        }
    return {
        "title": _label(measure),
        "label": _label(measure),
        "file": f"data/input/tiltak/{filename}",
        "seasonal_adjust_for_did": True,
    }


def _treatment_data(treatment: str) -> dict[str, str]:
    return {
        "diskret": {"type": "discrete", "title": "Diskret"},
        "kontinuerlig": {"type": "continuous", "title": "Kontinuerlig"},
    }[treatment]


def _run_variant(treatment: str, outcome: str) -> str:
    """Return the legacy-compatible run name for treatment and outcome."""
    return treatment if outcome == "indikator" else f"{outcome}-{treatment}"


def _period(catalog: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Return default period settings updated by a matrix-entry override."""
    return {**catalog["defaults"]["period"], **(override or {})}


@dataclass(frozen=True)
class GeneratedConfig:
    """A runtime configuration and its stable selection/output identifier."""

    id: str
    storage_path: Path
    config: dict[str, Any]


def _load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        catalog = yaml.safe_load(file)
    if not isinstance(catalog, dict):
        raise ValueError(f"Catalog must be a mapping: {path}")
    return catalog


def _indicators(
    catalog: dict[str, Any],
    level: str,
    group: str,
    outcome: str,
    indicator_group: str | None = None,
) -> list[dict[str, str]]:
    defaults = catalog["defaults"]
    level_data = _level_data(level)
    result = []
    for name, description in defaults["indicators"].items():
        indicator = {
            "name": name,
            "file": defaults["indicator_file_template"].format(
                level_directory=level_data["directory"],
                group_directory=group.replace("-", "_"),
                outcome_prefix="" if outcome == "indikator" else f"{outcome}_",
                indicator=name,
            ),
            "description": f"{description} ({_label(group)})"
            if indicator_group
            else description,
        }
        if indicator_group:
            indicator["group"] = indicator_group
        result.append(indicator)
    return result


def _build_did(
    catalog: dict[str, Any],
    measure_key: str,
    group: str,
    level: str,
    treatment: str,
    outcome: str,
    period_override: dict[str, Any] | None = None,
) -> GeneratedConfig:
    measure = _measure_data(catalog, measure_key)
    treatment_data = _treatment_data(treatment)
    level_data = _level_data(level)
    id_parts = [
        "did",
        measure_key,
        group,
        level_data["slug"],
        _run_variant(treatment, outcome),
    ]
    analysis: dict[str, Any] = {
        **_period(catalog, period_override),
        "title": " | ".join(
            part
            for part in (
                "DiD",
                measure["title"],
                _label(group),
                _label(outcome) if outcome != "indikator" else "",
                treatment_data["title"],
            )
            if part
        )
        + (" (Enheter)" if level == "enhet" else ""),
        "treatment_type": treatment_data["type"],
        "variation": f"did/{measure_key}/{level_data['slug']}",
        "analysis_level": level,
    }
    if treatment_data["type"] == "discrete":
        analysis["control_regions"] = list(catalog["defaults"]["control_regions"])
    else:
        analysis["denominator_definitions"] = list(
            catalog["defaults"]["denominator_definitions"]
        )
    data: dict[str, Any] = {
        "tiltak_file": measure["file"],
        "tiltak_label": measure["label"],
        "indikatorer": _indicators(catalog, level, group, outcome),
    }
    if measure.get("seasonal_adjust_for_did"):
        data["tiltak_seasonal_adjust"] = True
    if level == "enhet":
        data["enhet_mapping_file"] = catalog["defaults"]["enhet_mapping_file"]
    return GeneratedConfig(
        "--".join(id_parts), Path(*id_parts), {"analysis": analysis, "data": data}
    )


def _build_triple(
    catalog: dict[str, Any],
    measure_key: str,
    level: str,
    treatment: str,
    outcome: str,
    treated: str,
    control: str,
    period_override: dict[str, Any] | None = None,
) -> GeneratedConfig:
    measure, level_data = _measure_data(catalog, measure_key), _level_data(level)
    id_parts = [
        "triple-diff",
        measure_key,
        level_data["slug"],
        _run_variant(treatment, outcome),
    ]
    indicators = _indicators(catalog, level, treated, outcome, "treated") + _indicators(
        catalog, level, control, outcome, "control"
    )
    data: dict[str, Any] = {
        "tiltak_file": measure["file"],
        "tiltak_label": measure["label"],
        "indikatorer": indicators,
    }
    if level == "enhet":
        data["enhet_mapping_file"] = catalog["defaults"]["enhet_mapping_file"]
    return GeneratedConfig(
        "--".join(id_parts),
        Path(*id_parts),
        {
            "analysis": {
                **_period(catalog, period_override),
                "title": f"TrippelDiD | {measure['title']} | {level_data['label']} | {_treatment_data(treatment)['title']}",
                "treatment_type": _treatment_data(treatment)["type"],
                "variation": f"triple-diff/{measure_key}/{level_data['slug']}",
                "design": "triple_diff",
                "analysis_level": level,
                "treated_group": _label(treated),
                "control_group": _label(control),
                "denominator_definitions": list(
                    catalog["defaults"]["denominator_definitions"]
                ),
            },
            "data": data,
        },
    )


def build_configs() -> tuple[GeneratedConfig, ...]:
    """Expand and validate the explicitly permitted catalog matrix."""
    catalog, configs = _load_catalog(), []
    for entry in catalog["matrix"]:
        for measure in entry["measures"]:
            for level in entry["levels"]:
                for treatment, outcome in product(
                    entry["treatments"], entry["outcomes"]
                ):
                    if (
                        level not in catalog["levels"]
                        or outcome not in catalog["outcomes"]
                        or treatment not in catalog["treatments"]
                    ):
                        raise ValueError(f"Invalid catalog matrix entry: {entry}")
                    if entry["design"] == "did":
                        for group in entry["groups"]:
                            if group not in catalog["groups"]:
                                raise ValueError(
                                    f"Unknown group in catalog matrix: {group}"
                                )
                            configs.append(
                                _build_did(
                                    catalog,
                                    measure,
                                    group,
                                    level,
                                    treatment,
                                    outcome,
                                    entry.get("period"),
                                )
                            )
                    else:
                        configs.append(
                            _build_triple(
                                catalog,
                                measure,
                                level,
                                treatment,
                                outcome,
                                entry["treated_group"],
                                entry["control_group"],
                                entry.get("period"),
                            )
                        )
    ids = [config.id for config in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalog matrix contains duplicate configuration IDs.")
    return tuple(configs)


CONFIGS = build_configs()
CONFIGS_BY_ID = {config.id: config for config in CONFIGS}
DEFAULT_CONFIG_ID = "did--midl-lonnstilskudd--alle--regioner--kontinuerlig"


def get_config(run_id: str) -> GeneratedConfig:
    """Return a selected configuration with available IDs in invalid-selection errors."""
    try:
        return CONFIGS_BY_ID[run_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown configuration '{run_id}'. Available IDs:\n"
            + "\n".join(f"  {config.id}" for config in CONFIGS)
        ) from exc
