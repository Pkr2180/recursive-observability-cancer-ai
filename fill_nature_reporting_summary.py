"""Fill Nature Portfolio's XFA Reporting Summary smart form from the audited response draft.

The published form is a LiveCycle (XFA) document, so its answers live in the ``datasets``
packet rather than in AcroForm widgets. This script rewrites that packet with the answers
recorded in ``05_Nature_Reporting_Summary_Responses.docx`` and drops the stale ``form``
packet so that Adobe Reader/Acrobat rebuilds the presentation from template plus new data.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject

from build_nmi_submission_package import (
    AUTHOR_FULL,
    DATE,
    ETHICS_STATEMENT,
    FORMS_OUT,
    REPOSITORY_URL,
    ZENODO_STATUS,
)

DATA_NS = "http://www.xfa.org/schema/xfa-data/1.0/"
SOURCE = FORMS_OUT / "Nature_Portfolio_Reporting_Summary_Smart_Form.pdf"
OUTPUT = FORMS_OUT / "Nature_Portfolio_Reporting_Summary_COMPLETED.pdf"

# Exclusive-group on-values taken from the form template.
CONFIRMED = "2"          # statistics rows: the statement is reported
NOT_APPLICABLE = "1"     # statistics rows and method modules: n/a or not involved
INVOLVED = "2"           # method modules: system was involved in the study
LIFE_SCIENCES = "1"      # study-type selector

TEXT_ANSWERS: dict[str, str] = {
    "Header/author": AUTHOR_FULL,
    "Header/lastupdate": DATE,
    "Universal/Software/collectioninfo": (
        "No new data were collected. Public releases were retrieved programmatically with the "
        "project download modules (Python 3.12.10) from DepMap/PRISM, GDSC, GEO accession "
        "GSE70138 (LINCS) and the NCI Genomic Data Commons TCGA-CDR. Retrieval code is in "
        f"src/downloaders of Supplementary Software 1 and at {REPOSITORY_URL}."
    ),
    "Universal/Software/analysisinfo": (
        "Python 3.12.10 with NumPy 2.2.5, pandas 2.3.1, scikit-learn 1.8.0 and Matplotlib 3.10.1. "
        "The project specifies Python >=3.11, NumPy >=1.26, pandas >=2.2, scikit-learn >=1.5 and "
        "Matplotlib >=3.9. Remote execution used Modal (Debian-slim Python 3.12 container). All "
        "analysis, observability and reporting code is supplied in Supplementary Software 1 and at "
        f"{REPOSITORY_URL}. {ZENODO_STATUS}."
    ),
    "Universal/Dataavail/DAS/howselectedsamplesize": (
        "All data analysed here are public and available from their original repositories: DepMap "
        "and PRISM releases (depmap.org), GDSC (cancerrxgene.org), LINCS L1000 GEO accession "
        "GSE70138, and the NCI Genomic Data Commons with the TCGA-CDR clinical resource. No new or "
        "restricted data were generated. Every result table used in this manuscript is supplied as "
        "an original CSV file in Supplementary Data 1 and every source figure in Supplementary "
        "Data 2. Analysis code and tests are supplied in Supplementary Software 1 and at "
        f"{REPOSITORY_URL}. {ZENODO_STATUS}."
    ),
    "Humans/humandetails[0]": (
        "Sex and gender were not used as model inputs, stratification variables or subgroup "
        "endpoints. The TCGA-CDR records reanalysed here are de-identified and were used only to "
        "define held-out patient outcome units for the secondary observability check; no sex- or "
        "gender-based analysis was performed and no sex- or gender-based claim is made."
    ),
    "Humans/humandetails[1]": (
        "Race, ethnicity and other socially relevant groupings were not used as model inputs, "
        "stratification variables or subgroup endpoints, and no group-based claim is made. The "
        "reanalysed public records are de-identified and the reported endpoint is observability of "
        "the AI process rather than any patient- or group-level biological or clinical conclusion."
    ),
    "Humans/humandetails[2]": (
        "The human data are de-identified public TCGA and TCGA-CDR records. The secondary outcome "
        "check used 42 held-out patients drawn from 7 TCGA projects, giving 126 recorded "
        "patient-level observability events. Covariates such as age, sex, stage and treatment were "
        "not modelled; population characteristics are documented in the TCGA and TCGA-CDR source "
        "publications."
    ),
    "Humans/recruitment": (
        "No participants were recruited. This is a computational secondary reanalysis of already "
        "published, de-identified public datasets, so no recruitment, consent or selection "
        "procedure was carried out by the author. Any recruitment bias present in the original TCGA "
        "cohorts is inherited from the source studies; because the reported endpoint is "
        "observability of the AI process rather than a clinical or biological effect, such bias "
        "does not affect the claims made."
    ),
    "Humans/ethics": ETHICS_STATEMENT,
    "Life/howselectedsamplesize": (
        "No prospective power calculation was performed and no sample size was chosen by the "
        "author. All eligible observed units produced by the frozen public-data pipelines were "
        "analysed. The independent biological units were 723 DepMap models, 712 LINCS held-out "
        "transitions and 42 TCGA held-out patients, yielding 4,431 recorded observability events "
        "(2,169 DepMap/PRISM, 2,136 LINCS, 126 TCGA). This is a complete-case audit of the "
        "available public data rather than a sampled experiment."
    ),
    "Life/exclusions": (
        "Only the prespecified eligibility, linkage, sample-type and partition rules described in "
        "Methods were applied; no data were excluded post hoc on the basis of results. Missing or "
        "invalid transition telemetry failed closed at the observability layer rather than being "
        "imputed, so it is recorded as a detected failure instead of a silent exclusion."
    ),
    "Life/limitations": (
        "All findings reported here are computational and reproducible: the pipelines are "
        "deterministic given the frozen public inputs and the recorded seeds, and rerunning the "
        "supplied code reproduces the reported numbers. Architectural replication was built into "
        "the design: failure detection was repeated across three heterogeneous biological systems "
        "(DepMap/PRISM, LINCS, TCGA) and six controlled failure classes, with 9 of 9 declared "
        "conditions demonstrated and 18 of 18 controlled-disruption tests passing across 90 graded "
        "conditions. All replication attempts succeeded."
    ),
    "Life/methodrandomization": (
        "There were no biological intervention groups, so no allocation was randomized. Data "
        "partitioning was deterministic: held-out perturbations and held-out cancer projects were "
        "assigned by fixed hashing rules and seeded episode selection, so whole scientific units "
        "rather than individual rows were separated and no leakage occurs across splits. "
        "Statistical inference used seeded resampling (2,000 cluster-bootstrap replicates; "
        "1,000-permutation tests) at the level of the independent biological unit."
    ),
    "Life/blindinggroupallocation": (
        "Blinding was not performed and is not applicable. The analyses are deterministic "
        "computational audits of public data with prespecified metrics and no human outcome "
        "assessment, so investigator expectation cannot influence a measurement. Group labels "
        "(held-out unit, disruption class) are required inputs to the audit itself."
    ),
    "Celllines/celllinesource": (
        "No cell line was obtained, cultured or manipulated in this study. All cell-line "
        "measurements are secondary data from public releases: DepMap/PRISM cell-line models and "
        "GDSC drug-response profiles. Cell-line provenance is documented by DepMap, PRISM and GDSC "
        "in their source publications and release notes."
    ),
    "Celllines/celllineauthentication": (
        "No authentication was performed by the author because no cells were handled. "
        "Authentication of the underlying models, including STR profiling, is performed and "
        "reported by the DepMap, PRISM and GDSC projects that generated the public data reanalysed "
        "here."
    ),
    "Celllines/celllinemycoplasma": (
        "Not applicable; no cells were handled. Mycoplasma testing of the underlying models is the "
        "responsibility of the DepMap, PRISM and GDSC source projects and is reported by them."
    ),
    "Celllines/misidentified/rationaleICLAC": (
        "No cell line was selected by the author. The analysis uses whichever models are present in "
        "the frozen public DepMap/PRISM and GDSC releases; commonly misidentified lines were "
        "neither specifically included nor specifically excluded, because the endpoint is "
        "observability of the AI process rather than any biological conclusion about an individual "
        "line."
    ),
}

EXCL_ANSWERS: dict[str, str] = {
    # Statistics confirmations.
    "Universal/Statistics/questions/samplesize": CONFIRMED,
    "Universal/Statistics/questions/collection": CONFIRMED,
    "Universal/Statistics/questions/statstests": CONFIRMED,
    "Universal/Statistics/questions/covariates": NOT_APPLICABLE,
    "Universal/Statistics/questions/assumptions": CONFIRMED,
    "Universal/Statistics/questions/descstats": CONFIRMED,
    "Universal/Statistics/questions/nullhypothesis": CONFIRMED,
    "Universal/Statistics/questions/bayesian": NOT_APPLICABLE,
    "Universal/Statistics/questions/hierarchical": CONFIRMED,
    "Universal/Statistics/questions/effectsizes": CONFIRMED,
    # Study type.
    "SelectField/Type": LIFE_SCIENCES,
    # Materials and experimental systems.
    "MethodModules/MethodspecificReporting/materials/antibodies": NOT_APPLICABLE,
    "MethodModules/MethodspecificReporting/materials/celllines": INVOLVED,
    "MethodModules/MethodspecificReporting/materials/palaeontology": NOT_APPLICABLE,
    "MethodModules/MethodspecificReporting/materials/animals": NOT_APPLICABLE,
    "MethodModules/MethodspecificReporting/materials/clinical": NOT_APPLICABLE,
    "MethodModules/MethodspecificReporting/materials/durc": NOT_APPLICABLE,
    "MethodModules/MethodspecificReporting/materials/plants": NOT_APPLICABLE,
    # Methods.
    "MethodModules/MethodspecificReporting/methods/chip-seq": NOT_APPLICABLE,
    "MethodModules/MethodspecificReporting/methods/flowcytometry": NOT_APPLICABLE,
    "MethodModules/MethodspecificReporting/methods/mri": NOT_APPLICABLE,
}


# Human-readable captions, taken from the form template, so the Word response draft
# can be generated from exactly the values written into the official PDF.
LABELS: dict[str, str] = {
    "Header/author": "Corresponding author",
    "Header/lastupdate": "Last updated",
    "Universal/Statistics/questions/samplesize": "Exact sample size for each group/condition",
    "Universal/Statistics/questions/collection": "Distinct samples versus repeated measurement",
    "Universal/Statistics/questions/statstests": "Statistical tests used, one- or two-sided",
    "Universal/Statistics/questions/covariates": "Description of all covariates tested",
    "Universal/Statistics/questions/assumptions": "Assumptions and corrections, including multiple comparisons",
    "Universal/Statistics/questions/descstats": "Central tendency and variation or uncertainty",
    "Universal/Statistics/questions/nullhypothesis": "Test statistic, confidence intervals, effect sizes, degrees of freedom and exact P",
    "Universal/Statistics/questions/bayesian": "Bayesian priors and MCMC settings",
    "Universal/Statistics/questions/hierarchical": "Appropriate level for tests in hierarchical designs",
    "Universal/Statistics/questions/effectsizes": "Estimates of effect sizes and how they were calculated",
    "Universal/Software/collectioninfo": "Software: data collection",
    "Universal/Software/analysisinfo": "Software: data analysis",
    "Universal/Dataavail/DAS/howselectedsamplesize": "Data availability statement",
    "Humans/humandetails[0]": "Reporting on sex and gender",
    "Humans/humandetails[1]": "Reporting on race, ethnicity or other socially relevant groupings",
    "Humans/humandetails[2]": "Population characteristics",
    "Humans/recruitment": "Recruitment",
    "Humans/ethics": "Ethics oversight",
    "SelectField/Type": "Study type selected",
    "Life/howselectedsamplesize": "Life sciences: sample size",
    "Life/exclusions": "Life sciences: data exclusions",
    "Life/limitations": "Life sciences: replication",
    "Life/methodrandomization": "Life sciences: randomization",
    "Life/blindinggroupallocation": "Life sciences: blinding",
    "MethodModules/MethodspecificReporting/materials/antibodies": "Materials: antibodies",
    "MethodModules/MethodspecificReporting/materials/celllines": "Materials: eukaryotic cell lines",
    "MethodModules/MethodspecificReporting/materials/palaeontology": "Materials: palaeontology and archaeology",
    "MethodModules/MethodspecificReporting/materials/animals": "Materials: animals and other organisms",
    "MethodModules/MethodspecificReporting/materials/clinical": "Materials: clinical data",
    "MethodModules/MethodspecificReporting/materials/durc": "Materials: dual use research of concern",
    "MethodModules/MethodspecificReporting/materials/plants": "Materials: plants",
    "MethodModules/MethodspecificReporting/methods/chip-seq": "Methods: ChIP-seq",
    "MethodModules/MethodspecificReporting/methods/flowcytometry": "Methods: flow cytometry",
    "MethodModules/MethodspecificReporting/methods/mri": "Methods: MRI-based neuroimaging",
    "Celllines/celllinesource": "Cell lines: source",
    "Celllines/celllineauthentication": "Cell lines: authentication",
    "Celllines/celllinemycoplasma": "Cell lines: mycoplasma contamination",
    "Celllines/misidentified/rationaleICLAC": "Cell lines: commonly misidentified lines (ICLAC register)",
}

SELECTION_TEXT = {
    ("Universal/Statistics/questions", CONFIRMED): "confirmed as reported",
    ("Universal/Statistics/questions", NOT_APPLICABLE): "not applicable to this study",
    ("MethodModules", INVOLVED): "involved in the study",
    ("MethodModules", NOT_APPLICABLE): "not involved in the study",
    ("SelectField", LIFE_SCIENCES): "Life sciences",
}


def described_answers() -> list[tuple[str, str]]:
    """Return (label, human-readable answer) for every value written into the official form."""
    rows = []
    for path, label in LABELS.items():
        if path in TEXT_ANSWERS:
            rows.append((label, TEXT_ANSWERS[path]))
            continue
        value = EXCL_ANSWERS[path]
        prefix = next(key for key in ("Universal/Statistics/questions", "MethodModules", "SelectField") if path.startswith(key))
        rows.append((label, SELECTION_TEXT[(prefix, value)]))
    return rows


def read_packets(document) -> tuple[list, dict[str, int]]:
    root = document.trailer["/Root"] if hasattr(document, "trailer") else document.root_object
    array = root["/AcroForm"]["/XFA"].get_object()
    return array, {str(array[i]): i for i in range(0, len(array) - 1, 2)}


def normalise(raw: bytes) -> str:
    """Undo the form's newline-before-bracket serialisation so ElementTree can parse it."""
    return re.sub(r"\s*\n\s*(/?>)", r"\1", raw.decode("utf-8"))


def resolve(container: ET.Element, path: str) -> ET.Element:
    node = container
    for step in path.split("/"):
        match = re.fullmatch(r"(.+)\[(\d+)\]", step)
        if match:
            node = node.findall(match.group(1))[int(match.group(2))]
        else:
            found = node.find(step)
            if found is None:
                raise KeyError(f"missing datasets node: {path}")
            node = found
    return node


def maincontent(root: ET.Element) -> ET.Element:
    return resolve(root.find(f"{{{DATA_NS}}}data"), "form/Maincontent")


def build_datasets(raw: bytes) -> bytes:
    ET.register_namespace("xfa", DATA_NS)
    root = ET.fromstring(normalise(raw))
    main = maincontent(root)
    for path, value in {**TEXT_ANSWERS, **EXCL_ANSWERS}.items():
        resolve(main, path).text = value
    return ET.tostring(root, encoding="utf-8", xml_declaration=False)


def main() -> None:
    reader = PdfReader(SOURCE)
    array, index = read_packets(reader)
    datasets = build_datasets(array[index["datasets"] + 1].get_object().get_data())

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    out_array, out_index = read_packets(writer)

    stream = DecodedStreamObject()
    stream.set_data(datasets)
    out_array[out_index["datasets"] + 1] = writer._add_object(stream)

    # Drop the stale merged-form packet so Adobe rebuilds the presentation from
    # the template plus the new data instead of the empty-state field styling.
    form_at = out_index["form"]
    del out_array[form_at : form_at + 2]

    with OUTPUT.open("wb") as handle:
        writer.write(handle)

    verify()


def verify() -> None:
    check = PdfReader(OUTPUT)
    array, index = read_packets(check)
    if "form" in index:
        raise ValueError("stale form packet was not removed")
    root = ET.fromstring(normalise(array[index["datasets"] + 1].get_object().get_data()))
    main = maincontent(root)
    for path, value in {**TEXT_ANSWERS, **EXCL_ANSWERS}.items():
        written = resolve(main, path).text
        if written != value:
            raise ValueError(f"{path} did not round-trip")
    print(f"{OUTPUT} ({len(TEXT_ANSWERS)} text answers, {len(EXCL_ANSWERS)} selections)")


if __name__ == "__main__":
    main()
