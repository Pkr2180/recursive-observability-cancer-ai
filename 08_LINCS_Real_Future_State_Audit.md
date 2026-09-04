# LINCS Real Perturbational Future-State Audit

**Modal profile:** `pradeepaiperio`  
**Source:** NCBI GEO GSE70138 / NIH LINCS Phase II  
**Policy:** measured public data only; no simulated biological trajectories

## Data execution

- Downloaded Level-5 matrix: 5,365,179,698 bytes.
- NCBI-published SHA-512: verified.
- Full GCTX layout: 118,050 signatures × 12,328 genes.
- Cancer extraction: 89,192 signatures × 978 landmark genes.
- Cancer representation: 24 measured cell lines and eight primary sites.
- Compact quintile state-token cells: 87,229,776.

## Matched-time design

Signatures were matched only when cell identifier, perturbation identifier, dose and
perturbation type were identical. This produced 3,674 conditions and 3,675 consecutive
transitions:

- 3,638 conditions with 3→24 h measurements;
- 35 conditions with 6→24 h measurements;
- one condition with 3→6→24 h measurements.

These are matched experimental conditions, not repeated measurements from patients.

## Frozen holdout

The split unit was the complete perturbation identifier. Twenty-four perturbations were frozen
as holdout and were absent from model fitting and calibration. The holdout contained 712 real
transitions. Ridge, random-forest and nearest-neighbour agents predicted the later state; their
calibration-only reliability weights were 0.303, 0.350 and 0.347.

## Recursive observability outcome

| Channel | Holdout measurement |
|---|---:|
| Transition events | 2,136 |
| Trace completeness | 1.000 |
| State reconstructability | 0.223 |
| Recursive consistency | 0.568 |
| Master holdout RMSE | 0.871 |
| Master uncertainty MAE | 0.267 |
| Uncertainty–error correlation | 0.615 |
| OSIC | 0.895 |

The high OSIC means that large observed biological state changes coincide with larger Master
forecast errors. It exposes a real limitation and possible falsification axis. It must not be
presented as proof that the current Master predicts patient cancer evolution.

## Instability separation

Every holdout transition retains five non-collapsed channels: observed cancer-state change,
individual agent forecast error, inter-agent disagreement, Master forecast error, and the
Master's calibrated self-uncertainty/meta-calibration error. Counterfactual removal records the
influence of every agent.

## Frozen nonlinear and null controls

Before null execution, the analysis froze nine channel pairs, three metrics, 1,000 permutations,
seed `20260824`, alpha 0.05 and Benjamini–Hochberg correction. Permutations only rearranged the
measured right-hand channel and preserved its marginal distribution.

Twenty-six of 27 exploratory tests survived FDR correction. For observed cancer-state change
versus Master error:

| Metric | Observed | Empirical p | FDR |
|---|---:|---:|---:|
| Pearson | 0.895 | 0.001 | 0.001 |
| Spearman | 0.799 | 0.001 | 0.001 |
| Quantile normalized mutual information | 0.265 | 0.001 | 0.001 |

The frozen specification SHA-256 is
`2a15c469b43cabda83a912d492c1a3a9cce143acca25df35f10f0b7f3dfc1b35`.
