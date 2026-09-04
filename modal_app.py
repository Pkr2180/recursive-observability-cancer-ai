from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Optional

import modal


APP_NAME = "pancancer-biological-universe"
EXPECTED_MODAL_PROFILE = "pradeepaiperio"

REMOTE_ROOT = Path("/pancancer")
RAW_DIR = REMOTE_ROOT / "raw"
PROCESSED_DIR = REMOTE_ROOT / "processed"
MODEL_DIR = REMOTE_ROOT / "models"
RESULTS_DIR = REMOTE_ROOT / "results"

raw_volume = modal.Volume.from_name("pancancer-raw-data", create_if_missing=True)
processed_volume = modal.Volume.from_name("pancancer-processed-data", create_if_missing=True)
model_volume = modal.Volume.from_name("pancancer-models", create_if_missing=True)
results_volume = modal.Volume.from_name("pancancer-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26",
        "pandas>=2.2",
        "pyarrow>=16.0",
        "requests>=2.32",
        "PyYAML>=6.0",
        "tqdm>=4.66",
        "scikit-learn>=1.5",
        "networkx>=3.3",
        "joblib>=1.4",
        "openpyxl>=3.1",
        "h5py>=3.11",
        "matplotlib>=3.9",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)

app = modal.App(APP_NAME)


def _volumes() -> dict[str, modal.Volume]:
    return {
        str(RAW_DIR): raw_volume,
        str(PROCESSED_DIR): processed_volume,
        str(MODEL_DIR): model_volume,
        str(RESULTS_DIR): results_volume,
    }


def _assert_project_modal_profile() -> None:
    """Prevent this project from executing in a different Modal account."""
    active = os.environ.get("MODAL_PROFILE", "").strip()
    if not active:
        result = subprocess.run(
            ["modal", "profile", "current"],
            check=True,
            capture_output=True,
            text=True,
        )
        active = result.stdout.strip()
    if active != EXPECTED_MODAL_PROFILE:
        raise RuntimeError(
            f"Refusing project execution in Modal profile {active!r}. "
            f"Activate {EXPECTED_MODAL_PROFILE!r} first."
        )


@app.function(image=image, volumes=_volumes(), timeout=60)
def remote_status() -> dict:
    """Confirm Modal execution and mounted volume paths."""
    from src.common.paths import ensure_remote_layout

    layout = ensure_remote_layout(REMOTE_ROOT)
    return {
        "app": APP_NAME,
        "required_modal_profile": EXPECTED_MODAL_PROFILE,
        "remote_root": str(REMOTE_ROOT),
        "layout": {key: str(value) for key, value in layout.items()},
    }


@app.function(image=image, volumes=_volumes(), timeout=60 * 5)
def remote_summary(max_entries: int = 200) -> dict:
    """Summarize key files currently present in Modal Volumes."""
    from src.common.remote_summary import summarize_remote_tree

    return summarize_remote_tree(
        [RAW_DIR, PROCESSED_DIR, MODEL_DIR, RESULTS_DIR],
        max_entries=max_entries,
    )


@app.function(image=image, volumes=_volumes(), timeout=60 * 5)
def inspect_remote_csv(relative_path: str, nrows: int = 5) -> dict:
    """Inspect a small CSV preview inside Modal without copying the file locally."""
    from src.common.inspect import inspect_csv

    candidate = REMOTE_ROOT / relative_path.lstrip("/").replace("\\", "/")
    if not candidate.exists():
        for base in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
            alternative = base / relative_path.lstrip("/").replace("\\", "/")
            if alternative.exists():
                candidate = alternative
                break
    return inspect_csv(candidate, nrows=nrows)


@app.function(image=image, volumes=_volumes(), timeout=60 * 10, memory=8192)
def inspect_remote_artifact(relative_path: str, nrows: int = 5) -> dict:
    """Inspect a JSON, CSV, or Parquet artifact in mounted Modal Volumes."""
    from src.common.inspect import inspect_artifact

    normalized = relative_path.lstrip("/").replace("\\", "/")
    if ".." in Path(normalized).parts:
        raise ValueError("Artifact path must remain inside /pancancer")
    candidate = REMOTE_ROOT / normalized
    if not candidate.exists():
        for base in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, RESULTS_DIR]:
            alternative = base / normalized
            if alternative.exists():
                candidate = alternative
                break
    return inspect_artifact(candidate, nrows=nrows)


@app.function(image=image, volumes=_volumes(), timeout=60 * 30)
def build_source_index() -> dict:
    """Create a remote source index from the configured dataset manifest."""
    from src.downloaders.registry import build_source_index

    return build_source_index(config_path="configs/data_sources.yaml", raw_dir=RAW_DIR)


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 8)
def download_public_sources(
    source_keys: Optional[list[str]] = None,
    dry_run: bool = True,
) -> dict:
    """Download configured public sources inside Modal Volumes.

    Use dry_run=True first. Set dry_run=False only after reviewing the remote plan,
    storage footprint and database terms.
    """
    from src.downloaders.registry import download_sources

    result = download_sources(
        config_path="configs/data_sources.yaml",
        raw_dir=RAW_DIR,
        source_keys=source_keys,
        dry_run=dry_run,
    )
    raw_volume.commit()
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4)
def download_gdc_indexed_files(source_key: str, max_files: int = 5) -> dict:
    """Download a controlled number of files from an existing remote GDC index."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.gdc import download_indexed_gdc_files

    out_dir = safe_dataset_dir(RAW_DIR, source_key)
    result = download_indexed_gdc_files(out_dir=out_dir, max_files=max_files)
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=2, memory=4096)
def download_official_gdsc() -> dict:
    """Download official Sanger GDSC1/GDSC2 fitted-dose-response releases."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.gdsc import download_gdsc_fitted_releases

    result = download_gdsc_fitted_releases(safe_dataset_dir(RAW_DIR, "gdsc_fitted"))
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=2, memory=4096)
def download_lincs_metadata() -> dict:
    """Download official NCBI GEO LINCS Phase II metadata with checksums."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.lincs import download_lincs_phase2_metadata

    result = download_lincs_phase2_metadata(safe_dataset_dir(RAW_DIR, "lincs_gse70138"))
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 8, cpu=2, memory=4096)
def download_lincs_level5() -> dict:
    """Download the official LINCS Phase II Level-5 perturbation matrix."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.lincs import download_lincs_phase2_level5

    result = download_lincs_phase2_level5(safe_dataset_dir(RAW_DIR, "lincs_gse70138"))
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 30, cpu=2, memory=4096)
def download_tcga_clinical() -> dict:
    """Download the official open-access TCGA Pan-Cancer Clinical Data Resource."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.tcga_clinical import download_tcga_cdr

    result = download_tcga_cdr(safe_dataset_dir(RAW_DIR, "tcga_cdr"))
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=2, memory=8192)
def download_pcawg_external_cohort() -> dict:
    """Download open PCAWG donor expression and clinical matrices from UCSC Xena."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.pcawg_external import download_pcawg_external

    result = download_pcawg_external(safe_dataset_dir(RAW_DIR, "pcawg_external"))
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 30)
def index_depmap_files(
    release_filter: str = "25Q2",
    file_names: Optional[list[str]] = None,
) -> dict:
    """Fetch and filter the DepMap portal download manifest inside Modal."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.depmap import fetch_depmap_file_manifest

    out_dir = safe_dataset_dir(RAW_DIR, "depmap_portal")
    result = fetch_depmap_file_manifest(
        out_dir=out_dir,
        release_filter=release_filter,
        file_names=file_names,
    )
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 8)
def download_depmap_files(
    release_filter: str = "25Q2",
    file_names: Optional[list[str]] = None,
    max_files: int = 1,
) -> dict:
    """Download selected DepMap files inside Modal."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.depmap import download_depmap_selected_files

    out_dir = safe_dataset_dir(RAW_DIR, "depmap_portal")
    result = download_depmap_selected_files(
        out_dir=out_dir,
        release_filter=release_filter,
        file_names=file_names,
        max_files=max_files,
    )
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 30)
def index_depmap_figshare_files(
    article_id: int = 25880521,
    file_names: Optional[list[str]] = None,
    name_contains: str = "",
) -> dict:
    """Fetch and filter a public DepMap Figshare release manifest inside Modal."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.depmap import fetch_depmap_figshare_manifest

    out_dir = safe_dataset_dir(RAW_DIR, "depmap_figshare")
    result = fetch_depmap_figshare_manifest(
        out_dir=out_dir,
        article_id=article_id,
        file_names=file_names,
        name_contains=name_contains,
    )
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 8)
def download_depmap_figshare_files(
    article_id: int = 25880521,
    file_names: Optional[list[str]] = None,
    name_contains: str = "",
    max_files: int = 1,
) -> dict:
    """Download selected public DepMap Figshare files inside Modal."""
    from src.common.paths import safe_dataset_dir
    from src.downloaders.depmap import download_depmap_figshare_files

    out_dir = safe_dataset_dir(RAW_DIR, "depmap_figshare")
    result = download_depmap_figshare_files(
        out_dir=out_dir,
        article_id=article_id,
        file_names=file_names,
        name_contains=name_contains,
        max_files=max_files,
    )
    raw_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=8192)
def preprocess_sources(source_keys: Optional[list[str]] = None) -> dict:
    """Preprocess downloaded files into harmonised pan-cancer tables."""
    from src.preprocessing.pipeline import preprocess_sources

    result = preprocess_sources(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        source_keys=source_keys,
    )
    processed_volume.commit()
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=8192)
def harmonise_gdc_rnaseq(source_key: str, max_files: Optional[int] = None) -> dict:
    """Harmonise downloaded GDC RNA-seq STAR gene-count files into matrices."""
    from src.preprocessing.gdc_rnaseq import harmonise_gdc_rnaseq

    result = harmonise_gdc_rnaseq(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        source_key=source_key,
        max_files=max_files,
    )
    processed_volume.commit()
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=16384)
def harmonise_depmap_core(top_n: int = 1000) -> dict:
    """Harmonise DepMap model metadata, expression and CRISPR gene-effect matrices."""
    from src.preprocessing.depmap_core import harmonise_depmap_core

    result = harmonise_depmap_core(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        top_n=top_n,
    )
    processed_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=8192)
def harmonise_prism_primary(top_n: int = 1000) -> dict:
    """Harmonise PRISM Repurposing 24Q2 primary drug-response matrix."""
    from src.preprocessing.prism import harmonise_prism_primary

    result = harmonise_prism_primary(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        top_n=top_n,
    )
    processed_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=16384)
def harmonise_gdsc_fitted() -> dict:
    """Harmonise official GDSC1/GDSC2 fitted response tables."""
    from src.preprocessing.gdsc import harmonise_gdsc_releases

    result = harmonise_gdsc_releases(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
    processed_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=16384)
def harmonise_lincs_metadata() -> dict:
    """Harmonise LINCS metadata and identify real cancer-cell perturbation signatures."""
    from src.preprocessing.lincs import harmonise_lincs_phase2_metadata

    result = harmonise_lincs_phase2_metadata(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
    processed_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 30, cpu=4, memory=8192)
def harmonise_tcga_clinical() -> dict:
    """Harmonise TCGA-CDR endpoints and link them to GDC RNA-seq samples."""
    from src.preprocessing.tcga_clinical import harmonise_tcga_cdr

    result = harmonise_tcga_cdr(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
    processed_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=16384)
def harmonise_pcawg_external_cohort() -> dict:
    """Harmonise the independent non-TCGA PCAWG cohort to the frozen TCGA model."""
    from src.preprocessing.pcawg_external import harmonise_pcawg_external

    result = harmonise_pcawg_external(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        model_dir=MODEL_DIR,
    )
    processed_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=8192)
def prepare_lincs_level5_matrix() -> dict:
    """Decompress and inspect the checksum-verified LINCS Level-5 GCTX matrix."""
    from src.preprocessing.lincs_level5 import prepare_lincs_level5

    result = prepare_lincs_level5(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
    processed_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=16384)
def extract_lincs_cancer_matrix(batch_size: int = 512) -> dict:
    """Extract real cancer-cell LINCS Level-5 landmark-gene signatures."""
    from src.preprocessing.lincs_level5 import extract_lincs_cancer_landmarks

    result = extract_lincs_cancer_landmarks(PROCESSED_DIR, batch_size=batch_size)
    processed_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 30, cpu=2, memory=8192)
def profile_lincs_trajectories() -> dict:
    """Profile exact real LINCS cell/perturbation/dose matches across time."""
    from src.analysis.lincs_trajectories import profile_lincs_matched_trajectories

    result = profile_lincs_matched_trajectories(PROCESSED_DIR, RESULTS_DIR)
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 6, cpu=8, memory=32768)
def run_lincs_future_state(components: int = 16) -> dict:
    """Run the frozen unseen-perturbation future-state observability audit."""
    from src.validation.lincs_future_state import run_lincs_future_state_audit

    result = run_lincs_future_state_audit(
        processed_dir=PROCESSED_DIR,
        results_dir=RESULTS_DIR,
        model_dir=MODEL_DIR,
        components=components,
    )
    from src.observability.primary import run_primary_observability_for_run

    result["primary_observability"] = run_primary_observability_for_run(
        RESULTS_DIR, "lincs_future_state"
    )
    results_volume.commit()
    model_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 8, cpu=8, memory=32768)
def run_reinforcement_master_brain(
    components: int = 16,
    replicates: int = 2000,
) -> dict:
    """Train and validate the real-data reward-conditioned observable Master."""
    from src.validation.lincs_future_state import run_lincs_future_state_audit
    from src.validation.reinforcement_master import run_reinforcement_master_validation

    # Recreate the frozen source audit so the calibration reward partition and
    # untouched holdout are generated by the same mounted source snapshot.
    source_audit = run_lincs_future_state_audit(
        processed_dir=PROCESSED_DIR,
        results_dir=RESULTS_DIR,
        model_dir=MODEL_DIR,
        components=components,
    )
    reinforcement = run_reinforcement_master_validation(
        results_dir=RESULTS_DIR,
        model_dir=MODEL_DIR,
        replicates=replicates,
    )
    results_volume.commit()
    model_volume.commit()
    return {
        "status": "completed",
        "source_audit": source_audit,
        "reinforcement_master": reinforcement,
    }


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=8192)
def run_lincs_nonlinear_nulls(permutations: int = 1000) -> dict:
    """Run frozen nonlinear and permutation null controls on LINCS holdout telemetry."""
    from src.validation.nonlinear_nulls import run_lincs_nonlinear_null_audit

    result = run_lincs_nonlinear_null_audit(RESULTS_DIR, permutations=permutations)
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=8, memory=16384)
def run_tcga_outcome_audit() -> dict:
    """Run the frozen cancer-project-holdout TCGA patient-outcome audit."""
    from src.validation.tcga_outcomes import run_tcga_outcome_observability_audit

    result = run_tcga_outcome_observability_audit(
        processed_dir=PROCESSED_DIR,
        results_dir=RESULTS_DIR,
        model_dir=MODEL_DIR,
    )
    from src.observability.primary import run_primary_observability_for_run

    result["primary_observability"] = run_primary_observability_for_run(
        RESULTS_DIR, "tcga_patient_outcome"
    )
    results_volume.commit()
    model_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=16384)
def score_pcawg_external_cohort() -> dict:
    """Create and lock label-blind PCAWG predictions from the frozen TCGA bundle."""
    from src.validation.pcawg_external import score_pcawg_external

    result = score_pcawg_external(PROCESSED_DIR, MODEL_DIR, RESULTS_DIR)
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=16384)
def evaluate_pcawg_external_cohort() -> dict:
    """Evaluate locked PCAWG predictions against external two-year OS outcomes."""
    from src.validation.pcawg_external import evaluate_pcawg_external

    result = evaluate_pcawg_external(PROCESSED_DIR, RESULTS_DIR)
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=8192)
def run_primary_observability() -> dict:
    """Run the fail-closed observability-first audit over all real-data Master traces."""
    from src.observability.primary import run_primary_observability_audit

    result = run_primary_observability_audit(RESULTS_DIR)
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=8192)
def verify_observability_non_ablation(replicates: int = 2000) -> dict:
    """Verify the Master architecture and every agent without new ablation experiments."""
    from src.observability.verification import run_non_ablation_observability_verification

    result = run_non_ablation_observability_verification(
        RESULTS_DIR,
        replicates=replicates,
    )
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=8192)
def generate_results_package() -> dict:
    """Generate all static result figures and tables from completed real-data artifacts."""
    from src.reporting.observability_results import generate_observability_results_package

    result = generate_observability_results_package(RESULTS_DIR)
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=8192)
def quantise_states() -> dict:
    """Build first-pass pan-cancer quantised state tokens."""
    from src.quantisation.pipeline import quantise_processed_tables

    result = quantise_processed_tables(processed_dir=PROCESSED_DIR, results_dir=RESULTS_DIR)
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=8192)
def quantise_lincs_states() -> dict:
    """Quantise real LINCS cancer signatures into compact state-token codes."""
    from src.quantisation.lincs import quantise_lincs_cancer_states

    result = quantise_lincs_cancer_states(PROCESSED_DIR, RESULTS_DIR)
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=8192)
def mine_rule_collapse(top_n: int = 200) -> dict:
    """Mine first-pass expression-dependency rule-collapse candidates."""
    from src.analysis.rule_collapse import mine_expression_dependency_collapse

    result = mine_expression_dependency_collapse(
        processed_dir=PROCESSED_DIR,
        results_dir=RESULTS_DIR,
        top_n=top_n,
    )
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 2, cpu=4, memory=8192)
def mine_dependency_pharmacology(top_n: int = 200) -> dict:
    """Mine first-pass dependency-pharmacology inversion candidates."""
    from src.analysis.rule_collapse import mine_dependency_pharmacology_inversion

    result = mine_dependency_pharmacology_inversion(
        processed_dir=PROCESSED_DIR,
        results_dir=RESULTS_DIR,
        top_n=top_n,
    )
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=16384)
def validate_real_pan_cancer(
    top_genes: int = 50,
    top_drug_pairs: int = 100,
    permutations: int = 200,
    min_lineage_models: int = 20,
) -> dict:
    """Run lineage, curation and observed-value permutation validation."""
    from src.validation.real_data import run_real_data_validation

    result = run_real_data_validation(
        processed_dir=PROCESSED_DIR,
        results_dir=RESULTS_DIR,
        top_genes=top_genes,
        top_drug_pairs=top_drug_pairs,
        permutations=permutations,
        min_lineage_models=min_lineage_models,
    )
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=16384)
def replicate_prism_in_gdsc(min_models: int = 20) -> dict:
    """Externally compare PRISM/CRISPR associations with official GDSC observations."""
    from src.validation.gdsc_replication import run_gdsc_external_replication

    result = run_gdsc_external_replication(
        processed_dir=PROCESSED_DIR,
        results_dir=RESULTS_DIR,
        min_models=min_models,
    )
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 6, cpu=8, memory=32768)
def run_real_master_pan_cancer(
    agent_components: int = 6,
    master_components: int = 8,
) -> dict:
    """Run the first real-data Master Brain and recursive observability trace."""
    from src.agents.master import run_real_master_brain

    result = run_real_master_brain(
        processed_dir=PROCESSED_DIR,
        results_dir=RESULTS_DIR,
        agent_components=agent_components,
        master_components=master_components,
    )
    from src.observability.primary import run_primary_observability_for_run

    result["primary_observability"] = run_primary_observability_for_run(
        RESULTS_DIR, "depmap_prism_master"
    )
    results_volume.commit()
    return result


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=8192)
def run_agent_smoke_test() -> dict:
    """Disabled: project science must use provenance-bearing real data."""
    raise RuntimeError("Synthetic agent smoke execution is disabled for this project.")


@app.function(image=image, volumes=_volumes(), timeout=60 * 60 * 4, cpu=4, memory=8192)
def distil_smoke_test() -> dict:
    """Disabled: project science must use provenance-bearing real data."""
    raise RuntimeError("Synthetic distillation smoke execution is disabled for this project.")


@app.local_entrypoint()
def main(
    action: str = "status",
    dry_run: bool = True,
    source_key: str = "",
    max_files: int = 5,
    release_filter: str = "25Q2",
    file_names: str = "",
    article_id: int = 25880521,
    top_n: int = 1000,
    name_contains: str = "",
    permutations: int = 200,
    min_lineage_models: int = 20,
    agent_components: int = 6,
    master_components: int = 8,
):
    """Local CLI entrypoint that triggers remote Modal functions."""
    _assert_project_modal_profile()
    keys = [source_key] if source_key else None
    if action == "status":
        print(remote_status.remote())
    elif action == "summary":
        print(remote_summary.remote(max_entries=max_files))
    elif action == "inspect-csv":
        if not source_key:
            raise ValueError("--source-key must contain the relative CSV path for inspect-csv")
        print(inspect_remote_csv.remote(relative_path=source_key, nrows=max_files))
    elif action == "inspect-artifact":
        if not source_key:
            raise ValueError("--source-key must contain a relative path for inspect-artifact")
        print(inspect_remote_artifact.remote(relative_path=source_key, nrows=max_files))
    elif action == "index":
        print(build_source_index.remote())
    elif action == "download":
        print(download_public_sources.remote(source_keys=keys, dry_run=dry_run))
    elif action == "download-gdc-indexed":
        if not source_key:
            raise ValueError("--source-key is required for download-gdc-indexed")
        print(download_gdc_indexed_files.remote(source_key=source_key, max_files=max_files))
    elif action == "download-gdsc":
        print(download_official_gdsc.remote())
    elif action == "download-lincs-metadata":
        print(download_lincs_metadata.remote())
    elif action == "download-lincs-level5":
        print(download_lincs_level5.remote())
    elif action == "download-tcga-clinical":
        print(download_tcga_clinical.remote())
    elif action == "download-pcawg-external":
        print(download_pcawg_external_cohort.remote())
    elif action == "index-depmap":
        names = [name.strip() for name in file_names.split(",") if name.strip()] or None
        print(index_depmap_files.remote(release_filter=release_filter, file_names=names))
    elif action == "download-depmap":
        names = [name.strip() for name in file_names.split(",") if name.strip()] or None
        print(
            download_depmap_files.remote(
                release_filter=release_filter,
                file_names=names,
                max_files=max_files,
            )
        )
    elif action == "index-depmap-figshare":
        names = [name.strip() for name in file_names.split(",") if name.strip()] or None
        print(
            index_depmap_figshare_files.remote(
                article_id=article_id,
                file_names=names,
                name_contains=name_contains,
            )
        )
    elif action == "download-depmap-figshare":
        names = [name.strip() for name in file_names.split(",") if name.strip()] or None
        print(
            download_depmap_figshare_files.remote(
                article_id=article_id,
                file_names=names,
                name_contains=name_contains,
                max_files=max_files,
            )
        )
    elif action == "preprocess":
        print(preprocess_sources.remote(source_keys=keys))
    elif action == "harmonise-gdc-rnaseq":
        if not source_key:
            raise ValueError("--source-key is required for harmonise-gdc-rnaseq")
        print(harmonise_gdc_rnaseq.remote(source_key=source_key, max_files=max_files))
    elif action == "harmonise-depmap":
        print(harmonise_depmap_core.remote(top_n=top_n))
    elif action == "harmonise-prism":
        print(harmonise_prism_primary.remote(top_n=top_n))
    elif action == "harmonise-gdsc":
        print(harmonise_gdsc_fitted.remote())
    elif action == "harmonise-lincs-metadata":
        print(harmonise_lincs_metadata.remote())
    elif action == "harmonise-tcga-clinical":
        print(harmonise_tcga_clinical.remote())
    elif action == "harmonise-pcawg-external":
        print(harmonise_pcawg_external_cohort.remote())
    elif action == "prepare-lincs-level5":
        print(prepare_lincs_level5_matrix.remote())
    elif action == "extract-lincs-cancer":
        print(extract_lincs_cancer_matrix.remote())
    elif action == "profile-lincs-trajectories":
        print(profile_lincs_trajectories.remote())
    elif action == "lincs-future-state":
        print(run_lincs_future_state.remote(components=master_components))
    elif action == "reinforcement-master":
        print(
            run_reinforcement_master_brain.remote(
                components=master_components,
                replicates=permutations,
            )
        )
    elif action == "lincs-nonlinear-nulls":
        print(run_lincs_nonlinear_nulls.remote(permutations=permutations))
    elif action == "tcga-outcome-audit":
        print(run_tcga_outcome_audit.remote())
    elif action == "score-pcawg-external":
        print(score_pcawg_external_cohort.remote())
    elif action == "evaluate-pcawg-external":
        print(evaluate_pcawg_external_cohort.remote())
    elif action == "observability-primary":
        print(run_primary_observability.remote())
    elif action == "observability-verify":
        print(verify_observability_non_ablation.remote(replicates=permutations))
    elif action == "publication-results":
        print(generate_results_package.remote())
    elif action == "quantise":
        print(quantise_states.remote())
    elif action == "quantise-lincs":
        print(quantise_lincs_states.remote())
    elif action == "mine-rule-collapse":
        print(mine_rule_collapse.remote(top_n=top_n))
    elif action == "mine-dependency-pharmacology":
        print(mine_dependency_pharmacology.remote(top_n=top_n))
    elif action == "validate-real-data":
        print(
            validate_real_pan_cancer.remote(
                top_genes=min(top_n, 200),
                top_drug_pairs=top_n,
                permutations=permutations,
                min_lineage_models=min_lineage_models,
            )
        )
    elif action == "replicate-gdsc":
        print(replicate_prism_in_gdsc.remote(min_models=min_lineage_models))
    elif action == "master-real":
        print(
            run_real_master_pan_cancer.remote(
                agent_components=agent_components,
                master_components=master_components,
            )
        )
    elif action == "agents-smoke":
        print(run_agent_smoke_test.remote())
    elif action == "distil-smoke":
        print(distil_smoke_test.remote())
    else:
        raise ValueError(f"Unknown action: {action}")
