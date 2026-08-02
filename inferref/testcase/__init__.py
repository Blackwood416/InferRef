"""Testcase extraction and deduplication (SPEC §23, §24)."""

from inferref.testcase.dedup import (
    SignatureGroup,
    dedup_operators,
    operator_signature,
    summarise,
)
from inferref.testcase.extract import (
    ExtractedTestcase,
    ExtractionError,
    extract_operator,
    extract_region,
)
from inferref.testcase.validate import (
    TestcaseValidationError,
    TestcaseValidationIssue,
    TestcaseValidationResult,
    require_valid_testcase,
    validate_testcase,
)

__all__ = [
    "ExtractedTestcase",
    "ExtractionError",
    "SignatureGroup",
    "TestcaseValidationError",
    "TestcaseValidationIssue",
    "TestcaseValidationResult",
    "dedup_operators",
    "extract_operator",
    "extract_region",
    "operator_signature",
    "require_valid_testcase",
    "summarise",
    "validate_testcase",
]
