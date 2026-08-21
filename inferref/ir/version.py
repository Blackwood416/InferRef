"""Version constants for the InferRef trace format.

Trace IR §5: every trace MUST specify ``format`` and ``format_version``.
Tensor binary format versioning is independent of graph IR versioning.
"""

#: Magic string identifying an InferRef trace package (IR §5).
FORMAT = "inferref-trace"

#: Graph/manifest IR version (IR §5).
FORMAT_VERSION = "0.1"

#: ``.irtensor`` binary format version — independent of ``FORMAT_VERSION`` (IR §5).
TENSOR_FORMAT_VERSION = 1

#: Version of this InferRef implementation.
INFERREF_VERSION = "0.9.0"

#: Format string for a testcase package (IR §54).
TESTCASE_FORMAT = "inferref-testcase"

#: Testcase manifest version (IR §54).
TESTCASE_FORMAT_VERSION = "0.2"

# The 0.2 writer adds derived requirements; legacy 0.1 testcases remain
# readable and derive the same information at load time.
TESTCASE_READ_VERSIONS = ("0.1", "0.2", "0.3")


def check_format_version(fmt: str, version: str) -> None:
    """Reject a trace whose major IR semantics we do not support (IR §46).

    Readers must reject unsupported major IR semantics *cleanly* rather than
    silently misinterpreting them.
    """
    if fmt != FORMAT:
        raise ValueError(f"not an InferRef trace: format={fmt!r} (expected {FORMAT!r})")
    major = version.split(".", 1)[0]
    supported_major = FORMAT_VERSION.split(".", 1)[0]
    if major != supported_major:
        raise ValueError(
            f"unsupported trace format_version {version!r}; "
            f"this InferRef build reads {supported_major}.x"
        )
