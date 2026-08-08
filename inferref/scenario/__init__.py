"""Versioned scenario artifacts: ordered chains of testcases with state binding."""

from inferref.scenario.schema import (
    SCENARIO_FORMAT,
    SCENARIO_FORMAT_VERSION,
    Scenario,
    ScenarioError,
    ScenarioInput,
    ScenarioOutput,
    ScenarioState,
    ScenarioStep,
    load_scenario,
)
from inferref.scenario.validate import validate_scenario

_RUN_EXPORTS = frozenset(
    {"SCENARIO_RUN_FORMAT", "SCENARIO_RUN_FORMAT_VERSION", "STATE_MODES", "run_scenario"}
)

__all__ = [
    "SCENARIO_FORMAT",
    "SCENARIO_FORMAT_VERSION",
    "SCENARIO_RUN_FORMAT",
    "SCENARIO_RUN_FORMAT_VERSION",
    "STATE_MODES",
    "Scenario",
    "ScenarioError",
    "ScenarioInput",
    "ScenarioOutput",
    "ScenarioState",
    "ScenarioStep",
    "load_scenario",
    "run_scenario",
    "validate_scenario",
]


def __getattr__(name: str):
    """Lazily import the executor to avoid cycles with suite/agent packages."""

    if name in _RUN_EXPORTS:
        from inferref.scenario.run import (
            SCENARIO_RUN_FORMAT,
            SCENARIO_RUN_FORMAT_VERSION,
            STATE_MODES,
            run_scenario,
        )

        exports = {
            "SCENARIO_RUN_FORMAT": SCENARIO_RUN_FORMAT,
            "SCENARIO_RUN_FORMAT_VERSION": SCENARIO_RUN_FORMAT_VERSION,
            "STATE_MODES": STATE_MODES,
            "run_scenario": run_scenario,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
