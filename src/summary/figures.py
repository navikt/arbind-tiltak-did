"""Generates figures based on the data from the diff-in-diff analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_GROUP_ORDER = (
    "Alle",
    "Gode muligheter",
    "Trenger veiledning",
    "Nedsatt arbeidsevne",
    "Veiledning kombinert",
)


def figure1(data, output_dir):
    """Generates average number of people per halfyear in tiltak and in midlertidig lønnstilskuddd."""
    pass


def figure2(data, output_dir):
    """Generates average number of people per halfyear in midlertidig lønnstilskuddd per fylke."""
    pass


def figure3(data, output_dir):
    """Arbeidsindikator (Atid3) over time in two groups: high and low drop in use of midlertidig lønnstilskuddd."""
    pass


def figure4(data, output_dir):
    """Arbeidsindikator (Jobb3) over time in two groups: high and low drop in use of midlertidig lønnstilskuddd."""
    pass


def figure5(data: pd.DataFrame, output_dir: Path) -> Path:
    """Write a forest plot of preferred DiD estimates by heterogeneous group.

    ``data`` must contain the persisted regression-summary columns
    ``group``, ``indicator``, ``model``, ``coefficient``, ``ci_lower``, and
    ``ci_upper``. One preferred continuous-model row is required for every
    group/outcome combination. The two panels show Atid3 and Jobb3.
    """
    required = {
        "group",
        "indicator",
        "model",
        "coefficient",
        "ci_lower",
        "ci_upper",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Figure 5 input is missing columns: {', '.join(sorted(missing))}."
        )

    selected = data.loc[
        data["group"].isin(_GROUP_ORDER)
        & data["indicator"].isin(("atid3", "jobb3"))
        & (data["model"] == "preferred")
    ].copy()
    expected = pd.MultiIndex.from_product(
        [_GROUP_ORDER, ("atid3", "jobb3")], names=["group", "indicator"]
    )
    selected = selected.set_index(["group", "indicator"])
    if selected.index.has_duplicates:
        raise ValueError("Figure 5 input has duplicate preferred-model group results.")
    missing_results = expected.difference(selected.index)
    if len(missing_results):
        raise ValueError(
            "Figure 5 input is missing results for: "
            + ", ".join(f"{group}/{indicator}" for group, indicator in missing_results)
            + "."
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    positions = np.arange(len(_GROUP_ORDER))
    for axis, indicator, title in zip(
        axes, ("atid3", "jobb3"), ("Atid3", "Jobb3"), strict=True
    ):
        results = selected.loc[pd.IndexSlice[:, indicator], :].reset_index()
        results["group"] = pd.Categorical(
            results["group"], categories=_GROUP_ORDER, ordered=True
        )
        results = results.sort_values("group")
        axis.errorbar(
            results["coefficient"],
            positions,
            xerr=[
                results["coefficient"] - results["ci_lower"],
                results["ci_upper"] - results["coefficient"],
            ],
            fmt="o",
            color="#003366",
            ecolor="#66A3C8",
            capsize=3,
        )
        axis.axvline(0, color="#666666", linestyle=":", linewidth=1)
        axis.set(title=title, xlabel="DiD estimate (95% confidence interval)")
        axis.set_yticks(positions, _GROUP_ORDER)
    axes[0].invert_yaxis()
    fig.suptitle("Preferred DiD estimates by heterogeneous group")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "figure5_heterogeneous_estimates.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_figures(data, output_dir):
    """Generate all summary figures."""
    pass
