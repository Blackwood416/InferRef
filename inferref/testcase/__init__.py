"""Testcase extraction and deduplication (SPEC §23, §24)."""

from inferref.testcase.dedup import SignatureGroup, dedup_operators, operator_signature, summarise
from inferref.testcase.extract import (
    ExtractedTestcase,
    ExtractionError,
    extract_operator,
    extract_region,
)

__all__ = [
    "ExtractedTestcase",
    "ExtractionError",
    "SignatureGroup",
    "dedup_operators",
    "extract_operator",
    "extract_region",
    "operator_signature",
    "summarise",
]
