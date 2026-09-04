from __future__ import annotations


SKEPTIC_CHECKS = [
    "batch_effect",
    "missing_modality",
    "dataset_leakage",
    "cancer_type_imbalance",
    "drug_promiscuity",
    "crispr_quality",
    "random_seed_instability",
    "quantisation_artifact",
    "distillation_artifact",
    "known_pathway_explanation",
]


def planned_skeptic_checks() -> list[str]:
    return list(SKEPTIC_CHECKS)

