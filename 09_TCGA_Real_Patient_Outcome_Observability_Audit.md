# TCGA Real Patient-Outcome Observability Audit

**Modal profile:** `pradeepaiperio`  
**Sources:** NCI GDC TCGA-CDR and open-access GDC STAR-count RNA-seq  
**Policy:** observed patient data only; no simulated patients or molecular trajectories

## Official clinical data and RNA linkage

The official TCGA Pan-Cancer Clinical Data Resource workbook was downloaded through its NCI
GDC file identifier and stored only in the `pradeepaiperio` raw-data Volume. Its recorded
SHA-256 is `ea594c0fbb6731477c7ac511fab449ca9c38b0d42d269591ed9f5c4090e75a5a`.

The workbook contains 11,160 patients across all 33 TCGA projects. All 330 previously
harmonised RNA-seq files linked to a TCGA-CDR case. Their specimen types were:

- 300 primary tumors;
- 10 primary blood-derived cancers;
- 13 solid-tissue normal samples;
- six metastases;
- one additional new primary.

Only the 310 baseline tumor or primary blood-cancer specimens were eligible for outcome
modeling. Normal, metastatic and additional-new-primary specimens were excluded before the
endpoint filter.

## Frozen two-year overall-survival endpoint

The endpoint was all-cause death by day 730. A patient was labeled as an event only when the
recorded overall-survival event occurred on or before day 730. A patient was labeled as a
non-event only when observation extended through day 730. Event-free records censored before
day 730 were excluded. One stable-sorted RNA file per TCGA case was retained.

This yielded 227 eligible patients spanning all 33 TCGA projects. Projects—not patients—were
assigned by a fixed SHA-256 ordering into 146 fit patients, 39 calibration patients and 42
test patients. Seven complete cancer projects were held out: BLCA, CESC, PRAD, SKCM, STAD,
UCS and UVM. The frozen specification SHA-256 is
`eeaea7657b46b84a8156cd39fd7581646255e501e13b790472bd3e4a79fa30e4`.

Feature selection, scaling and PCA were fitted on fit projects only. Linear transcriptome,
nonlinear forest and patient-neighborhood agents each used 24 bootstrap fits that resampled
only observed fit patients. The Master weights were learned from calibration-project Brier
scores. Bootstrap resampling and null permutations are statistical procedures; they did not
create artificial patient records.

## Held-out outcome results

The 42-patient test set contained 15 deaths by two years and 27 non-events.

| Measurement | Held-out value |
|---|---:|
| AUROC | 0.677 |
| Average precision | 0.607 |
| Brier score | 0.215 |
| Fit-prevalence baseline Brier | 0.239 |
| Accuracy at 0.5 | 0.643 |
| Five-bin calibration error | 0.112 |
| AUROC permutation p | 0.0270 |
| Brier permutation p | 0.0360 |

The Master outperformed the prevalence Brier baseline and both fixed observed-label permutation
tests crossed 0.05. This is evidence within the small, frozen seven-project holdout; it is not
external clinical validation and is secondary to the observability endpoint.

## Required recursive instability separation

The audit retained five non-collapsed streams for every held-out patient:

1. cancer future-state instability: entropy of observed two-year outcomes among the 15
   nearest fit patients;
2. individual-agent instability: within-agent probability standard deviation across 24
   observed-patient bootstrap fits;
3. inter-agent disagreement: standard deviation across the three agent probabilities;
4. Master-state instability: entropy of the weighted Master outcome probability;
5. uncertainty in Master self-observation: disagreement across three calibration-trained
   models estimating the Master's absolute error.

The 126 agent-integration transitions have trace completeness 1.0. State reconstructability
was 1.0 because the declared Master weighting is exactly recoverable from logged agent
probabilities and weights. Cancer–Master OSIC was 0.896, with an observed-value permutation
`p = 0.000999`. This is structured coupling between local outcome heterogeneity and Master
ambiguity; it is not evidence of causation or forecast accuracy.

The self-observer did not generalize well: its predicted-versus-actual error correlation was
-0.267, self-error MAE was 0.209 and recursive consistency was 0.436. This negative result is
preserved as a falsification-relevant weakness of the current Master.

## Scope boundary

This experiment forecasts a real future patient outcome from a baseline tumor transcriptome.
It does not contain repeated molecular sampling from the same patient and therefore does not
establish patient-specific molecular future-state prediction, treatment-response causality or
clinical utility.

## Remote artifacts

```text
/pancancer/raw/tcga_cdr/TCGA-CDR-SupplementalTableS1.xlsx
/pancancer/raw/tcga_cdr/tcga_cdr_download_manifest.json
/pancancer/processed/tcga_cdr/tcga_cdr_patients.parquet
/pancancer/processed/tcga_cdr/tcga_cdr_rnaseq_linked.parquet
/pancancer/processed/tcga_cdr/tcga_cdr_harmonisation_summary.json
/pancancer/results/tcga_outcome_observability/tcga_outcome_observability_report.json
/pancancer/results/tcga_outcome_observability/heldout_project_predictions.parquet
/pancancer/results/tcga_outcome_observability/recursive_observability_events.jsonl
/pancancer/results/tcga_outcome_observability/frozen_project_split.json
```
