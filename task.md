# Coding Agent Guide: Panel Fixed Effects Analysis of Downsizing on Employment

## Background

The Norwegian Labour and Welfare Administration (Nav) with **12 regions** underwent a significant reduction in use of "arbeidsmarkedstiltak" (referred to as "tiltaksnedgang") starting in **May 2025**, continuing gradually through the autumn. The reduction was distributed unevenly across regions. Regions operate independently.

There is prior evidence that **regions with larger downsizing were overperforming before the cuts**. This is an important analytical consideration: it means the treatment was not applied to already-weak offices, which strengthens the case for a causal interpretation of any post-treatment decline, and suggests estimates may be conservative.

## Objective

Determine whether the downsizing had a causal effect on Nav's indicator for effect on employment, and estimate its magnitude. Produce clean, well-documented code suitable for a professional setting, and write up the findings in a quarto report.

---

## Tooling

- **Package manager:** `uv`. Always use `uv run <tool>` — never activate the venv manually.
- **Task runner:** `just`. Key recipes: `just fix` (ruff auto-fix + format), `just lint` (full pre-commit suite including mypy), `just render` (Quarto report).
- **Python ≥ 3.12.** Use `from __future__ import annotations` at the top of every module.
- **Type annotations:** all public functions must be fully typed. Use `dict[str, Any]` not bare `dict`. When mypy requires stub packages (`types-*`), add them to both `[dependency-groups] dev` in `pyproject.toml` and to `additional_dependencies` in the mypy hook in `.pre-commit-config.yaml`.
- Always run `just fix && just lint` before finishing a task.

---

## Project structure

```
src/
├── diff-in-diff/        ★ Main analysis package — all new development goes here
├── exploratory/         Exploratory comparison of tiltaksnedgang definitions
├── fetch_data/          BigQuery extraction stubs
└── old-code/            Reference only — do NOT import from here
data/
├── raw/                 CSVs of the indikator and tiltak
└── processed/           Where processed panels are stored
outputs
└── did/                 Main results: figures, tables and report.
```


---

## Configuration

All analysis parameters live in `src/diff-in-diff/analysis-config.yml`. The pipeline reads this file at startup — **do not hardcode paths or parameters in Python**. To add an indicator, append an entry under `indikatorer`. To change the tiltak source, update `tiltak_file`.

---

## Module responsibilities

### `prep_data.py`
Loads raw CSVs (wide or long format), standardises `aarmnd` to YYYYMM, merges indicator and tiltak data, and engineers features. Column contract for all downstream modules: `region`, `aarmnd` (datetime), `indikator` (float), `tiltak` (float), `tiltaksnedgang` (float 0–1), `relative_month` (int), `post_treatment` (bool), `month_of_year` (int), `year` (int).

For triple-diff: `prepare_triple_diff_panel()` loads treated and control group indicators, stacks them with a `group` column, and adds `treated` (0/1) and `treatment_x_group` (interaction) columns. For unit-level analysis, reads long-format unit data and joins with `enhetsmapping.json`. Additional columns: `entity`, `group`, `treated`, `treatment_x_group`.

### `regression.py`
Two model specs — **Baseline** (region FE + year-month FE) and **Preferred** (+ region × calendar-month FE). CR1 small-sample correction: `G/(G−1) × (N−1)/(N−K)`. `extract_all_coefficients(result)` returns a tidy DataFrame with a `koeffisient_type` column classifying each parameter.

For triple-diff: `run_triple_diff_baseline()` and `run_triple_diff_preferred()` estimate the triple-diff model with `treatment_x_group` as the main estimand. Also includes `run_triple_diff_placebo()` and `run_triple_diff_leave_one_out()`.

### `event_study.py`
Specification: `Y_{it} = Σ_τ β_τ · s_i · 1{relative_month = τ} + α_i + γ_t + ε_{it}`, where `s_i = max post-period tiltaksnedgang` per region. Window τ ∈ [−24, +4], base period τ = −1 omitted. Includes a joint pre-trend Wald F-test using a pseudoinverse when the covariance matrix is rank-deficient (common since 23 pre-period lags > G−1 = 11).

For triple-diff: `run_triple_diff_event_study()` uses triple interactions `s_i · Treated_g · 1{τ}` as the coefficients of interest.

### `cluster_bootstrap.py`
Wild cluster bootstrap imposing H₀: β_tiltaksnedgang = 0, using Webb 6-point weights. Algorithm: FWL projection once, then vectorised bootstrap (B × N matrix ops), CR1 SE per replicate, two-sided p-value. Run on both baseline and preferred models. **Bootstrap p-values are the primary inferential result.**

For triple-diff: `wild_cluster_bootstrap_triple_diff()` tests H₀: β_{treatment_x_group} = 0 with region-level clustering.

### `report.py`
Writes `report.qmd` (Quarto, Norwegian). Significance stars in the regression table use bootstrap p-values; both bootstrap and asymptotic p-values are shown. Figures: trends, FE coefficients, event study, bootstrap distribution.

### `report_triple_diff.py`
Multi-chapter report generator for triple-diff. Produces four `.qmd` files per config: descriptive statistics, regression results, bootstrap inference, and assumption tests. Each chapter is self-contained with its own figures.

### `models.py`
Typed result containers: `IndicatorResult` for standard DiD, `TripleDiffResult` for triple-diff (adds `analysis_level`, `treated_group`, `control_group`, group-level baseline means).

### `run_analysis.py`
Pipeline entry point. Reads config, loops over indicators. Branches on `analysis.design`: `"did"` (default) runs the standard two-way DiD flow; `"triple_diff"` pairs treated/control indicators and runs the triple-diff flow. Outputs: `regression_results.csv`, `alle_koeffisienter.csv`, report QMD(s).

---

## Report

The report should be written in a professional but readable tone — intended for a non-technical audience familiar with the business context. Key numbers (the estimated effect, confidence interval, and bootstrap p-value) should be stated clearly in plain language. The pre-period overperformance finding and its implications for causal interpretation should be discussed explicitly. The report should be written in Norwegian.

---

## Methodological notes

**Why the treatment variable is time-varying:** the downsizing was gradual, not a single event. A ramp variable based on actual headcount changes is more appropriate than a binary post-indicator, and allows testing for a dose-response relationship between cut severity and employment impact.

**Why two regression models:** the second model adds office-specific seasonal fixed effects. With five years of pre-period data, there are sufficient degrees of freedom to estimate these. If seasonal exposure varies across offices, the baseline model could confound seasonality with treatment effects.

**Why wild cluster bootstrap:** with G = 12 clusters, the asymptotic approximation underlying standard cluster-robust inference is unreliable. Webb weights are the recommended choice for this cluster count. The bootstrap p-value is the headline inferential result.

**On causal interpretation:** the parallel trends assumption — that offices would have followed similar employment trajectories absent the downsizing — is the key identifying assumption. The event study pre-trend plot is the primary diagnostic. The prior evidence that heavily-cut offices were overperforming before the cuts is analytically favourable: it rules out the concern that those offices were already in decline, and if anything implies the estimated negative effect understates the true impact.
