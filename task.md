# Coding Agent Prompt: Panel Fixed Effects Analysis of Downsizing on Employment

## Background

The Norwegian Labour and Welfare Administration (Nav) with **12 regions** underwent a significant reduction in use of "arbeidsmarkedstiltak" (referred to as "downsizing") starting in **May 2025**, continuing gradually through the autumn. Nav experienced a general decline in their ability to help people get a job both before and after the downsizing. The downsizing was distributed unevenly across regions — some were largely unaffected, others had moderate reductions, and others very large ones. Regions operate independently and do not influence each other's ability to help people get a job.

There is prior evidence that **regions with larger downsizing were overperforming before the cuts**. This is an important analytical consideration: it means the treatment was not applied to already-weak offices, which strengthens the case for a causal interpretation of any post-treatment decline, and suggests estimates may be conservative.

## Objective

Determine whether the downsizing had a causal effect on Nav's indicator for effect on employment, and estimate its magnitude. Produce clean, well-documented code suitable for a professional setting, and write up the findings in a markdown report.

---

## Data

Two sets of data will be used:

- **Monthly indicator for Nav's effect on employment** per region
- **Monthly people on arbeidsmarkedstiltak** per region

Data spans January 2021 through December 2025 (60 months × 12 regions). The code should validate the inputs on load — checking for missing offices, duplicate records, and implausible values — and raise informative errors if validation fails.

The method of aquiring the data will be specified later. 

### Treatment variable

The primary treatment variable should be a **continuous, time-varying measure of downsizing intensity**: the fraction of people on arbeidsmarkedstiltak lost at each office relative to its pre-treatment peak, set to zero for all months before June 2025. This captures the gradual nature of the reduction and allows for a dose-response interpretation of results.

---

## Project Structure

Organise the project with a clear separation of concerns:

```
project/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   └── tables/
├── src/                  # individual modules for each analysis step
└── run_analysis.py       # single entry point that runs the full pipeline
```

---

## Analysis

Implement the following steps as separate, importable modules in `src/`. All functions should have docstrings and type hints. Use Python's `logging` module rather than bare `print` statements.

### 1. Data preparation
Load, validate, merge, and engineer all necessary features. Save the processed panel dataset before proceeding.

### 2. Panel regression with fixed effects

Estimate the effect of downsizing on Nav's employment outcome using OLS with:
- **Office fixed effects** — absorbs time-invariant differences between offices
- **Year-month fixed effects** — absorbs firm-wide trends and macro conditions, including the pre-existing employment outcome decline
- **Office × calendar-month fixed effects** — absorbs office-specific seasonal patterns

Implement the regression and clustered standard errors using external regression libraries. Apply **CR1 small-sample correction** to the clustered standard errors.

Run two models: one without the seasonal fixed effects (baseline) and one with (preferred specification). The treatment coefficient represents the estimated change in monthly employment outcomes per unit increase in downsizing intensity. Report results for both models.

### 3. Event study

Estimate a separate treatment coefficient for each month in a window of roughly 24 months before and 6 months after June 2025, omitting the month immediately before treatment as the base period. Each coefficient is estimated as an interaction between a period indicator and the office-level downsizing intensity.

This serves two purposes:
- **Pre-trend test:** coefficients in the months before June 2025 should be statistically indistinguishable from zero if the parallel trends assumption holds. Given the pre-period overperformance by heavily-cut offices, mildly positive pre-period coefficients are plausible and should be noted — they would imply the post-treatment estimates are conservative.
- **Dynamic effects:** reveals whether the employment impact is immediate, builds over time, or fades.

Perform a joint statistical test that all pre-period coefficients are zero, and report the result.

### 4. Wild cluster bootstrap

With only 12 offices, asymptotic cluster-robust standard errors can be unreliable. Implement a **wild cluster bootstrap using Webb (6-point) weights**, which is specifically recommended for small cluster counts. Use a sufficient number of replications (at minimum 4,999) with a fixed random seed for reproducibility.

The bootstrap p-value for the main treatment coefficient should be treated as the **primary inferential result**. Asymptotic p-values are secondary.

---

## Outputs

- **Three figures:** an exploratory overview (firm-wide trends, group comparisons, pre-period performance scatter, headcount chart), an event study plot, and a bootstrap distribution plot. Save as PNG files.
- **A results table** summarising both regression models. Save as CSV.
- **A markdown report** (`report.md`) written to the project root, structured as a proper analytical write-up with the following sections:
    - Background and research question
    - Data and methodology
    - Results (incorporating figures inline and referencing the key numbers from the regression and bootstrap output)
    - Interpretation and caveats
    - Conclusion

The report should be written in a professional but readable tone — intended for a non-technical audience familiar with the business context. Key numbers (the estimated effect, confidence interval, and bootstrap p-value) should be stated clearly in plain language. The pre-period overperformance finding and its implications for causal interpretation should be discussed explicitly. The report should be written in Norwegian. 

---

## Methodological Notes

**Why the treatment variable is time-varying:** the downsizing was gradual, not a single event. A ramp variable based on actual headcount changes is more appropriate than a binary post-indicator, and allows testing for a dose-response relationship between cut severity and employment impact.

**Why two regression models:** the second model adds office-specific seasonal fixed effects. With five years of pre-period data, there are sufficient degrees of freedom to estimate these. If seasonal exposure varies across offices, the baseline model could confound seasonality with treatment effects.

**Why wild cluster bootstrap:** with G = 12 clusters, the asymptotic approximation underlying standard cluster-robust inference is unreliable. Webb weights are the recommended choice for this cluster count. The bootstrap p-value is the headline inferential result.

**On causal interpretation:** the parallel trends assumption — that offices would have followed similar employment trajectories absent the downsizing — is the key identifying assumption. The event study pre-trend plot is the primary diagnostic. The prior evidence that heavily-cut offices were overperforming before the cuts is analytically favourable: it rules out the concern that those offices were already in decline, and if anything implies the estimated negative effect understates the true impact.
