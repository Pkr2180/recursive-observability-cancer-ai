# Modal Runbook

## Goal

Run the pan-cancer biological universe pipeline in the separate Modal `pradeepaiperio` workspace.

Required local Modal profile:

```powershell
modal profile activate pradeepaiperio
modal profile current
```

The second command must print `pradeepaiperio` before any data or compute action. The
`saveetha` and `pradeepkumar-sdc` workspaces are separate accounts and are not project targets.
This is also enforced by `modal_app.py`; commands fail before launching their remote action if
the selected profile is not `pradeepaiperio`.

Use pan-cancer data only. Do not run oral-resistome or unrelated project datasets through this workflow.

Local folder:

```text
D:\pending work\ai - view
```

Remote Modal app:

```text
pancancer-biological-universe
```

Remote Modal Volumes:

```text
pancancer-raw-data
pancancer-processed-data
pancancer-models
pancancer-results
```

No large datasets should be downloaded to the local machine.

---

## Safe First Commands

Run from:

```text
D:\pending work\ai - view
```

Check Modal execution and mounted volume layout:

```powershell
modal run modal_app.py --action status
```

Create a remote index of configured sources:

```powershell
modal run modal_app.py --action index
```

Dry-run the download plan:

```powershell
modal run modal_app.py --action download --dry-run
```

Dry-run a single source:

```powershell
modal run modal_app.py --action download --dry-run --source-key tcga_gdc_public_rnaseq_index
```

---

## First Real Remote Execution

This queries the GDC API remotely and saves only a small metadata index in the Modal raw-data Volume:

```powershell
modal run modal_app.py --action download --no-dry-run --source-key tcga_gdc_public_rnaseq_index
```

Then profile downloaded/indexed files remotely:

```powershell
modal run modal_app.py --action preprocess
```

Then create bootstrap pan-cancer state tokens remotely:

```powershell
modal run modal_app.py --action quantise
```

Do not run synthetic agent or distillation smoke actions for scientific analyses. They are
legacy engineering checks and their outputs are not evidence. Project analyses must consume
the provenance-bearing TCGA, DepMap and PRISM matrices.

---

## Executed Prototype Data Pipeline

Download a small batch of public GDC RNA-seq quantification files:

```powershell
modal run modal_app.py --action download-gdc-indexed --source-key tcga_gdc_public_rnaseq_index --max-files 5
```

The GDC index is generated independently for all 33 TCGA projects, and downloads are selected
round-robin by project. For the executed balanced real-data cohort, use 330 files (10 per
project):

```powershell
modal run modal_app.py --action download-gdc-indexed --source-key tcga_gdc_public_rnaseq_index --max-files 330
```

Harmonise GDC RNA-seq:

```powershell
modal run modal_app.py --action harmonise-gdc-rnaseq --source-key tcga_gdc_public_rnaseq_index --max-files 5
```

Index DepMap 24Q2 Figshare:

```powershell
modal run modal_app.py --action index-depmap-figshare --article-id 25880521
```

Download DepMap core files:

```powershell
modal run modal_app.py --action download-depmap-figshare --article-id 25880521 --file-names Model.csv,CRISPRGeneEffect.csv,OmicsExpressionProteinCodingGenesTPMLogp1.csv --max-files 3
```

Harmonise DepMap expression and CRISPR:

```powershell
modal run modal_app.py --action harmonise-depmap --top-n 500
```

Index PRISM Repurposing Public 24Q2:

```powershell
modal run modal_app.py --action index-depmap-figshare --article-id 25917643
```

Download first PRISM data package:

```powershell
modal run modal_app.py --action download-depmap-figshare --article-id 25917643 --file-names README.txt,Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv,Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv,Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv,Repurposing_Public_24Q2_Treatment_Meta_Data.csv --max-files 5
```

Harmonise PRISM:

```powershell
modal run modal_app.py --action harmonise-prism --top-n 500
```

Quantise all currently harmonised data:

```powershell
modal run modal_app.py --action quantise
```

Mine expression-dependency rule collapse:

```powershell
modal run modal_app.py --action mine-rule-collapse --top-n 200
```

Mine dependency-pharmacology inversion:

```powershell
modal run modal_app.py --action mine-dependency-pharmacology --top-n 200
```

Run real-data lineage, target-curation and observed-value permutation validation:

```powershell
modal run modal_app.py --action validate-real-data --top-n 100 --permutations 200 --min-lineage-models 20
```

Run the first real-data Master Pan-Cancer Brain with recursive observability:

```powershell
modal run modal_app.py --action master-real --agent-components 6 --master-components 8
```

The Master action consumes only the harmonised DepMap and PRISM observations. It writes agent
states, Master states, 2,169 transition records and the RMOI/OSIC/OOSC report under
`/pancancer/results/master_brain`.

Download and harmonise the official Sanger GDSC1/GDSC2 fitted-response releases, then run
independent replication of PRISM/CRISPR associations:

```powershell
modal run modal_app.py --action download-gdsc
modal run modal_app.py --action harmonise-gdsc
modal run modal_app.py --action replicate-gdsc --min-lineage-models 20
```

The replication outputs are written under
`/pancancer/results/validation/gdsc_external_replication`.

Download and process the official NCBI GEO LINCS Phase II post-perturbation expression data:

```powershell
modal run modal_app.py --action download-lincs-metadata
modal run modal_app.py --action harmonise-lincs-metadata
modal run modal_app.py --action download-lincs-level5
modal run modal_app.py --action prepare-lincs-level5
modal run modal_app.py --action extract-lincs-cancer
modal run modal_app.py --action profile-lincs-trajectories
modal run modal_app.py --action quantise-lincs
modal run modal_app.py --action lincs-future-state --master-components 16
modal run modal_app.py --action lincs-nonlinear-nulls --permutations 1000
```

The future-state action freezes whole perturbation identifiers into fit, calibration and holdout
sets. Its 712 holdout transitions come from 24 perturbations absent from model fitting. Outputs
are stored under `/pancancer/results/validation/lincs_future_state` and the fitted models under
`/pancancer/models/lincs_future_state`.

Train and validate the reward-conditioned recursively observable Master as a detached job:

```powershell
modal run --detach modal_app.py --action reinforcement-master --master-components 16 --permutations 2000
```

This action recreates the frozen LINCS source audit, learns the observer policy from calibration
perturbation rewards and evaluates only on the unseen observed holdout perturbations. It uses no
simulated biological trajectories. Results are written to
`/pancancer/results/reinforcement_master_brain` and models to
`/pancancer/models/reinforcement_master_brain`. A detached run continues if the local computer
disconnects or is switched off.

Monitor the current detached run:

```powershell
modal app list
modal app logs ap-HoejzlbpPgUgiQozkYB3Tt --timestamps
```

Download the official open-access TCGA Pan-Cancer Clinical Data Resource, link it to the
harmonised RNA cohort and execute the frozen cancer-project-holdout outcome audit:

```powershell
modal run modal_app.py --action download-tcga-clinical
modal run modal_app.py --action harmonise-tcga-clinical
modal run modal_app.py --action tcga-outcome-audit
```

The endpoint is censoring-aware two-year overall survival. Feature preparation uses fit cancer
projects only, Master weights use separate calibration projects and seven whole TCGA projects
remain untouched until test scoring. Outputs are under
`/pancancer/results/tcga_outcome_observability`.

Run the mandatory observability-first consolidation after any scientific Master trace changes:

```powershell
modal run modal_app.py --action observability-primary
```

This action fails if any required event field or scientific-state value is invalid. It writes
MSOI, RMOI and SAOI profiles, the metric dictionary, divergence/hypothesis/provenance/gate
artifacts and the dashboard under `/pancancer/results/observability_primary`. Prediction and
learning metrics appear only in the secondary section.

Run the observational, non-ablation verification after a successful primary audit:

```powershell
modal run modal_app.py --action observability-verify --permutations 2000
```

This verifies every Master architecture and all nine agents. Confidence intervals resample
independent episodes, patients or observed holdout transitions rather than individual agent
events. It also writes five-channel rank/dependence tests, uncertainty calibration,
selective-risk curves and component-level metrics under
`/pancancer/results/observability_primary/verification_non_ablation`. This action does not
convert logged or counterfactual influence into a causal-attribution claim.

Generate the complete publication results package after a successful primary audit:

```powershell
modal run modal_app.py --action publication-results
```

This writes 19 publication figures in PNG and SVG, 20 tables in CSV and HTML, an HTML index,
a Markdown results summary, and SHA-256 manifests under
`/pancancer/results/observability_primary/publication_package`. It also creates the compact
download archive
`/pancancer/results/observability_primary/primary_scientific_observability_results.zip`.

Summarise remote Modal Volume contents:

```powershell
modal run modal_app.py --action summary --max-files 160
```

---

## Dataset Policy

Use only pan-cancer sources for this project:

- TCGA/GDC pan-cancer profiles
- DepMap/CCLE cancer cell-line omics and dependency data
- PRISM/GDSC cancer pharmacology data
- CPTAC pan-cancer proteomics, when source access is settled
- LINCS perturbation data only when linked to cancer-state modeling
- selected cancer single-cell/spatial atlases only when added deliberately

Start with remote metadata/indexing.

Only enable large downloads after checking:

- database terms
- storage size
- whether credentials are required
- whether controlled-access data are involved
- Modal cost expectations

Controlled-access data must not be downloaded unless the Modal account has the required authorization.

---

## Next Implementation Milestones

1. Completed: exact public DepMap, PRISM and official GDSC downloads.
2. Completed: controlled GDC file download mode for selected TCGA files.
3. Completed: balanced 330-file RNA-seq cohort across all 33 TCGA projects.
4. Completed: TCGA-CDR patient/outcome linkage and baseline tumor specimen filtering.
5. Add further omic-level quantisation.
6. Completed: first real rule-collapse mining:
   - CRISPR dependency high but drug resistance high.
   - mRNA high but protein low.
   - oncogene amplification but inhibitor resistance.
7. Completed: first real-data Master Pan-Cancer Brain prototype with O0-O6 telemetry.
8. Completed: independent GDSC1/GDSC2 replication.
9. Completed: matched-time LINCS cancer-cell perturbational validation with an unseen-
   perturbation holdout.
10. Completed: real baseline-RNA-to-two-year-patient-outcome audit with whole-project holdout.
11. Still required for molecular future-state claims: repeated molecular sampling from the same
    patients and an external patient cohort.
12. Completed: fail-closed primary observability audit and dashboard across all 4,431 Master
    transition events.
13. Completed: non-ablation verification of all three Master architectures and all nine agents,
    with 2,000 biological-unit bootstrap/permutation replicates.
14. Completed: publication results package with all figures, tables and checksum manifests.
