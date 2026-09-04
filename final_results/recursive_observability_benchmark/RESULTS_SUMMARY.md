# Recursive Scientific Observability Benchmark

## Operational definition

For a recorded distributed scientific-AI process, recursive scientific observability (RSO) is
defined as the conjunction `RSO = N1 AND N2 AND N3 AND N4 AND N5 AND N6`. These six necessary
conditions become operationally sufficient within the declared transition schema when the
architecture also satisfies `S1 AND S2 AND S3`: controlled failure sensitivity, recursive
non-redundancy and cross-system invariance. This is a falsifiable operational definition for
the recorded architecture, not a universal mathematical theorem for all dynamical systems.

## Formal result

9/9 operational necessary/sufficient conditions were demonstrated. These conditions establish recursive observability for the recorded AI process; they do not establish biological outcome validity or causal treatment efficacy.

## Controlled failures

The benchmark executed 90 graded disruption conditions over 4,431 real-data transition events. 18/18 system-by-failure detection tests passed the prespecified effect (>0.05) and monotonicity (Spearman < -0.70) criteria. Failed tests: none.

## Recursive depth

- depmap_prism: maximum marginal held-out reconstruction gain 0.2069.
- lincs: maximum marginal held-out reconstruction gain 0.4163.
- tcga: maximum marginal held-out reconstruction gain 0.3992.

Positive gain indicates that a higher observer level contains held-out information not present at lower levels. Negative or zero gain identifies redundant or destabilizing telemetry rather than being hidden by an aggregate score.

## Cross-system invariance

The median pairwise Spearman concordance of controlled failure signatures across DepMap/PRISM, LINCS and TCGA was 0.754. This evaluates invariance of observability failure detection, not invariance of biological outcomes.

## Claim boundary

The result supports recursive scientific observability as a measurable, disruptable and falsifiable property of the recorded distributed AI architecture. It does not prove that an observable system is biologically correct, clinically effective, conscious or causally self-aware.
