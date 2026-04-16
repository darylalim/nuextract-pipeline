import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


@pytest.fixture
def sample_icd10_codes():
    """Small ICD-10-CM code subset for validation tests."""
    return {"E119", "I10", "J189", "J441", "E785", "F411", "J209"}


@pytest.fixture
def sample_3_chunk_text():
    """Clinical text long enough to split into ~3 chunks at small max_tokens."""
    return (
        "Chief Complaint: chest pain.\n"
        "HPI: patient presents with acute onset chest pain radiating to left arm.\n"
        "Duration two hours. Associated with shortness of breath and diaphoresis.\n"
        "PMH: hypertension, hyperlipidemia, type 2 diabetes mellitus.\n"
        "MEDICATIONS:\n"
        "lisinopril 20mg daily, atorvastatin 40mg nightly, metformin 1000mg twice daily.\n"
        "ASSESSMENT:\n"
        "Acute coronary syndrome, rule out STEMI. Troponin pending.\n"
        "PLAN:\n"
        "Admit to cardiology. Aspirin 325mg, heparin drip, serial ECGs.\n"
    )


@pytest.fixture
def sample_per_chunk_results():
    """Three per-chunk extraction results for merge testing."""
    return [
        {
            "chief_complaint": "chest pain",
            "hpi": "acute onset chest pain radiating to left arm",
            "medications": [],
            "assessment": [],
        },
        {
            "chief_complaint": "",
            "hpi": "",
            "medications": [{"name": "lisinopril", "dose": "20mg"}],
            "assessment": [],
        },
        {
            "chief_complaint": "",
            "hpi": "",
            "medications": [{"name": "atorvastatin", "dose": "40mg"}],
            "assessment": [
                {"diagnosis": "Acute coronary syndrome", "icd10_code": "I24.9"}
            ],
        },
    ]
