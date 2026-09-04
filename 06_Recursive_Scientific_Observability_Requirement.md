# Recursive Scientific Observability — Core Project Requirement

## Non-negotiable scientific novelty

The primary hypothesis of this project is:

> A Master Reinforcement AI can become scientifically observable at multiple recursive levels, allowing instability of cancer future states to be separated from instability within individual agents, disagreement between agents, instability of the Master state, and uncertainty in the Master's own self-observation.

This is a core architectural and evaluation requirement, not an optional explainability layer. Prediction performance, reward, loss, AUROC and related measures are secondary engineering checks.

“GOD-Observability” may be used internally only as the acronym **Global Observability of Distributed Reinforcement States**. It is not a claim of consciousness, sentience or divinity.

## Required synchronized streams

Every experiment must retain five distinct, time-aligned streams:

1. cancer future-state instability, `U_C(t)`;
2. individual-agent instability, `IRI_i(t)`;
3. inter-agent disagreement, `D_ij(t)` / `ADI(t)`;
4. Master-state instability, `U_M(t)`;
5. uncertainty in the Master's self-observation, `U_Meta(t) = U(U_M(t))`.

They must never be prematurely averaged into a generic “AI uncertainty” score. Their separation is the testable scientific claim.

## Implemented real-data evidence

The requirement is now executed in three complementary real-data settings:

- a cross-sectional DepMap/PRISM Master with three biological modality agents and 2,169
  provenance-complete transitions;
- a matched-time LINCS perturbation audit with whole perturbations frozen as holdout, 712
  observed test transitions and 2,136 provenance-complete agent events.
- a real TCGA patient-outcome audit linking baseline tumor RNA to censoring-aware two-year
  overall survival, with seven complete cancer projects held out and 126 provenance-complete
  agent events.

The LINCS audit stores observed cancer-state change, each agent's error, agent disagreement,
Master error, Master self-uncertainty, meta-calibration error and counterfactual agent influence
as separate fields. Its low state reconstructability is retained as a falsification-relevant
result rather than being hidden by a composite score.

The LINCS error-coupling result was also tested with 1,000 observed-value permutation nulls,
nonlinear normalized mutual information and Benjamini–Hochberg correction under a frozen
exploratory specification. Surviving coupling is interpreted as structured forecast failure,
not as causal biological control or clinical validity.

The TCGA outcome audit likewise preserves all five channels. Its modest AUROC did not cross
the frozen observed-label permutation threshold, and its Master error self-observer had negative
held-out error correlation. Those results are retained because recursive observability is meant
to make such limitations measurable, not to convert them into a favorable composite.

## Observability hierarchy

| Level | Required observation | Primary evidence |
|---:|---|---|
| O0 | Inputs, outputs, actions, tools and evidence | trace completeness |
| O1 | Each agent's state transitions | IRI and agent reconstructability |
| O2 | Disagreement and influence between agents | divergence matrix and attribution |
| O3 | Master reinforcement state | independent state reconstruction |
| O4 | Master's uncertainty, policy and representation changes | self-observation trajectory |
| O5 | Reliability of the Master's self-observation | meta-uncertainty and recursive consistency |
| O6 | Coupling across cancer, agents, Master and meta-observer | OSIC and OOSC profile |

Observer depth is measured empirically. Recursion stops where useful information saturates or reconstruction becomes unreliable; infinite recursion is not assumed to be beneficial.

## Primary evaluation profile

RMOI is reported first as six separate, preregistered dimensions:

- state reconstructability (SR);
- trace completeness (TC);
- agent influence identifiability (AII);
- uncertainty visibility (UV);
- meta-uncertainty observability (MU);
- recursive consistency (RC).

No composite weights may be invented after observing results. A composite may be used only if weights are justified and frozen before confirmatory evaluation.

OSIC measures cancer–Master coupling. OOSC extends it across cancer instability, distributed agent instability, Master instability and meta-uncertainty. Correlation is acceptable only for smoke testing; confirmatory analyses should use preregistered lagged, nonlinear and null-controlled measures such as mutual information or transfer entropy.

## Required transition telemetry

Every scientific-state transition must contain the fields enforced in `src/observability/recursive.py`, including before/after agent and Master states, both uncertainty levels, evidence provenance, parent trace, considered futures, quantum-inspired state, applied gate and hypothesis transition.

A conclusion without its producing transition is an observability failure. Missing telemetry must be reported, never silently imputed.

The implementation now fails closed on missing required fields, empty or non-finite state
vectors, before/after dimension mismatches, invalid uncertainty values, empty provenance,
invalid future-state weights and quantum-inspired amplitudes that do not encode the declared
future probabilities.

## Implemented observability-first outputs

`src/observability/primary.py` makes observability the primary endpoint across every completed
real-data Master run. It generates separate MSOI, RMOI and SAOI profiles without arbitrary
weights, plus:

- empirical observability rank ratio;
- agentic-divergence matrix and agent-to-Master influence coverage;
- uncertainty and meta-uncertainty trajectories;
- hypothesis-transition traceability and graph;
- evidence-provenance completeness, depth and graph;
- OSIC, pairwise OOSC, multivariate OOSC and index-lag diagnostics;
- Master state switching and persistence diagnostics;
- quantum-inspired state observability and gate-influence traces;
- O0-O6 observer-depth availability;
- a ten-panel Master Scientific Observability Dashboard.

The consolidated audit validated 4,431 real-data transitions with strict validity 1.0. Observer-
depth saturation remains an explicitly unresolved intervention question; it is not inferred from
telemetry completeness.

## Falsification requirement

The Skeptic Agent must test whether apparent cancer–AI coupling is explained by noise, missing modalities, batch effects, calibration failure, under-training, seed sensitivity or architecture. Required null worlds include shuffled biological transitions, agent identities, histories and modality correspondence; random policies; and destroyed cross-modal structure with preserved marginals.

Only coupling that survives frozen null tests may be interpreted as biological information. Even then, informational coupling does not establish that the AI causally changes cancer.
