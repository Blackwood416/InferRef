"""Framework-neutral Agent and engine-adapter API (SPEC §21, §42, Phase 7)."""

from inferref.agent.protocol import (
    AGENT_PROTOCOL_FORMAT,
    AGENT_PROTOCOL_VERSION,
    ENGINE_ADAPTER_FORMAT,
    ENGINE_ADAPTER_VERSION,
    AgentResponse,
    EngineAdapter,
)
from inferref.agent.service import (
    capabilities,
    compare_outputs,
    context,
    extract_testcase,
    run_engine,
)

__all__ = [
    "AGENT_PROTOCOL_FORMAT",
    "AGENT_PROTOCOL_VERSION",
    "ENGINE_ADAPTER_FORMAT",
    "ENGINE_ADAPTER_VERSION",
    "AgentResponse",
    "EngineAdapter",
    "capabilities",
    "compare_outputs",
    "context",
    "extract_testcase",
    "run_engine",
]
