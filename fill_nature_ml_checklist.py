from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "manuscript_package" / "Nature_Machine_Intelligence_Submission_2026-09-04"
SOURCE = PACKAGE / "Nature_Forms" / "Nature_Machine_Learning_Checklist_v1.1.pdf"
OUTPUT = PACKAGE / "Nature_Forms" / "Nature_Machine_Learning_Checklist_COMPLETED.pdf"
REPOSITORY = "https://github.com/Pkr2180/recursive-observability-cancer-ai"


def yes(field: str) -> str:
    return f"/{field}_Yes_On"


def no(field: str) -> str:
    return f"/{field}_No_On"


def main() -> None:
    reader = PdfReader(SOURCE)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    values: dict[str, str] = {
        "Corresponding authors": "Pradeep Kumar Yadalam, BDS, MDS, PhD",
        "Code will be included in a CodeOcean capsule": "/Off",
        "The source code is included in the submission or a": "/On",
        "Textfield": REPOSITORY,
        "A compiled standalone version of the software is i": "/Off",
        "Textfield-0": "Not applicable",
        "A test dataset and instructionsscripts for replica": "/On",
        "Textfield-1": REPOSITORY,
        "A Readme file with instructions for installing and": "/On",
        "Textfield-2": REPOSITORY,
        "The code is made available to reviewers during rev": "/On",
        "Pretrained models are used in the study and access": "/Off",
        "Textfield-3": "Not applicable; no inaccessible pretrained model is used.",
        "Pretrained models are used in the study and are no": "/Off",
        "The paper contains information on how to obtain co": "/On",
    }

    radio_values = {
        "A All data sources are listed in the paper": "yes",
        "B The train test and validation datasets are publi": "yes",
        "C We have reported and discussed potential dataset": "yes",
        "D The data cleaning and preprocessing steps are cl": "yes",
        "E Instances of combining data from multiple source": "yes",
        "B A Model Card is provided1": "no",
        "C The model clearly splits data into different set": "yes",
        "D The method of data splitting eg random cluster o": "yes",
        "E The data splitting mimics anticipated realworld": "yes",
        "F The data splitting procedure has been chosen to": "yes",
        "G The interpretability of the model has been studi": "no",
        "A The performance metrics used are described and j": "yes",
        "B Crossvalidation of the results is included": "yes",
        "C Communityaccepted benchmark datasetstasks are us": "yes",
        "D Baseline comparisons to simpletrivial models for": "yes",
        "E Benchmarks with current stateoftheart are provid": "no",
        "F Ablation experiments are included": "no",
        "G The model has been tested on a fully independent": "no",
        "A The paper contains information on hardwarecomput": "yes",
        "B The paper includes information on the computatio": "no",
    }
    for field, answer in radio_values.items():
        values[field] = yes(field) if answer == "yes" else no(field)

    values.update(
        {
            "Yes": "Discussion and Nature ML response draft, section 2 (dataset limitations).",
            "No": "Not applicable.",
            "Yes-0": "Methods: Data sources and biological units; preprocessing source code.",
            "No-0": "Not applicable.",
            "Yes-1": "Methods: Data sources and distributed observer architectures.",
            "No-1": "Not applicable.",
            "A What model architecture is the current model bas": (
                "Heterogeneous multi-observer ensemble with a Master integration layer (see Methods)."
            ),
            "Yes-2": "Methods: Data sources and biological units; deterministic project/perturbation splits.",
            "No-2": "Not applicable.",
            "Yes-3": "Methods: held-out perturbations and held-out cancer projects mimic unseen scientific units.",
            "No-3": "Not applicable.",
            "Yes-4": "Methods: whole-unit split rules prevent perturbation/project leakage.",
            "No-4": "Not applicable.",
            "Yes-5": "Not applicable.",
            "No-5": "The primary endpoint is recursive observability, not post-hoc model interpretability.",
            "Yes-6": "Methods: Transition contract, observability metrics and Statistical analysis.",
            "No-6": "Not applicable.",
            "Yes-7": "Public DepMap, PRISM, GDSC, LINCS and TCGA resources; Methods and Data availability.",
            "No-7": "Not applicable.",
            "Yes-8": "Results and Supplementary Tables S12-S14 report prevalence, frozen-Master and equal-ensemble baselines.",
            "No-8": "Not applicable.",
            "Yes-9": "Not applicable.",
            "No-9": "RSO is a new audit endpoint; no directly equivalent state-of-the-art benchmark exists.",
            "Yes-10": "Not applicable.",
            "No-10": "Controlled observability disruptions are stress tests, not causal model-ablation experiments.",
            "DD-MM-YYYY": "04-09-2026",
        }
    )

    for page in writer.pages:
        writer.update_page_form_field_values(page, values, auto_regenerate=False)
    writer.set_need_appearances_writer(True)
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    print(OUTPUT)


if __name__ == "__main__":
    main()
