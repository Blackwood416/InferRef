"""Framework-neutral Agent and engine-adapter API (SPEC §21, §42, Phase 7)."""

from inferref.agent.protocol import (
    AGENT_PROTOCOL_FORMAT,
    AGENT_PROTOCOL_VERSION,
    ENGINE_ADAPTER_FORMAT,
    ENGINE_ADAPTER_VERSION,
    AdapterCapabilities,
    AgentResponse,
    EngineAdapter,
)
from inferref.agent.summary import (
    AGENT_SUMMARY_FORMAT,
    AGENT_SUMMARY_FORMAT_VERSION,
    build_agent_summary,
    summarize_report,
)

_SERVICE_OPERATIONS = frozenset(
    {
        "capabilities",
        "compare_outputs",
        "context",
        "extract_testcase",
        "run_engine",
        "run_scenario",
    }
)

__all__ = [
    "AGENT_PROTOCOL_FORMAT",
    "AGENT_PROTOCOL_VERSION",
    "AGENT_SUMMARY_FORMAT",
    "AGENT_SUMMARY_FORMAT_VERSION",
    "ENGINE_ADAPTER_FORMAT",
    "ENGINE_ADAPTER_VERSION",
    "AdapterCapabilities",
    "AgentResponse",
    "EngineAdapter",
    "build_agent_summary",
    "capabilities",
    "compare_outputs",
    "context",
    "extract_testcase",
    "run_engine",
    "run_scenario",
    "summarize_report",
]


def __getattr__(name: str):
    """Lazily expose service operations to avoid import cycles with scenario."""

    if name in _SERVICE_OPERATIONS:
        from inferref.agent import service

        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
