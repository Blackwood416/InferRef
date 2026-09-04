# InferRef 0.10.0: Alignment Orchestrator and Evidence-Driven Debugging

> **Status:** Draft / implementation-ready 0.10.0 scope

## 1. Release decision

InferRef 0.9.2 can locate a mismatch once an Agent has selected a testcase or
region. 0.10 adds the missing control plane: a persistent session that selects
the next useful boundary, records immutable observations, narrows the failure,
and returns evidence-backed next actions.

The 0.10.0 release gate contains:

1. Alignment Session v0.1, immutable Evidence v0.1, and append-only events.
2. Deterministic boundary inventory, planning, execution, and narrowing.
3. Rule-based failure-domain candidates with explicit support and uncertainty.
4. Human CLI, Agent facade, MCP tools, compact summary, and InferRef-first skill.
5. Fixture-kind tracking, an evidence matrix, and measurable workflow metrics.

The following are deliberately deferred rather than coupled to the first
orchestrator release:

- Debug Probe Protocol and engine-observed input comparison: 0.11 candidate.
- `inferref artifact compare` and quantized artifact decoding: 0.11 candidate.
- graph visualization and cross-commit regression views: later 0.10.x/0.11.

The session and evidence formats reserve typed extension points for those
features. 0.10 must not claim that engine inputs match unless an engine-observed
input artifact exists; reference inputs being supplied to a standalone testcase
is not that observation.

## 2. Confirmed 0.9.2 baseline

The specification is based on these current implementation facts:

- Agent Protocol v0.1 exposes `context`, `extract_testcase`,
  `compare_outputs`, `run_engine`, and `run_scenario`.
- `execute_adapter()` validates a testcase and comparison policy before launch,
  checks Adapter v0.2 capabilities, creates a fresh run directory, and writes
  `inferref-run.json`.
- comparison reports already contain per-output status, layout diagnostics,
  numeric metrics, provenance, and `first_failure`.
- `region recommend` scores payload completeness, reproducibility, semantic
  confidence, boundary size, operator count, and mutation risk.
- regions may nest or overlap by node set, but Trace IR has no explicit parent
  relation. Module and operator boundaries are not stored as regions.
- there is no alignment artifact, evidence schema, fixture-kind vocabulary,
  event log, or failure classifier.
- Adapter v0.2 has no debug-probe or engine-input-observation contract.

0.10 composes the existing primitives. It does not create a second execution or
comparison implementation.

## 3. Product goal

One-sentence goal:

> InferRef 0.10.0 turns numerical validation from individually callable tools
> into a persistent, auditable workflow in which every next action is derived
> from current evidence and the failure boundary becomes monotonically more
> specific when the available trace permits it.

The standard loop becomes:

```text
init -> plan -> run -> record evidence -> classify -> narrow -> plan ...
```

## 4. Non-goals

0.10 does not:

- edit engine source or act as an autonomous coding agent;
- prohibit custom dumps or analysis scripts;
- infer causality from a numerical mismatch;
- assert that a candidate failure domain is the bug;
- mutate the source trace to manufacture hierarchy;
- replace model conversion, quantization, or artifact tooling;
- require an engine to expose an ATen-like graph;
- monitor arbitrary shell activity to police Agent behavior;
- calculate productivity claims from unrecorded work.

## 5. Terminology

| Term | Meaning |
| --- | --- |
| Alignment Session | Persistent state and provenance for one trace/adapter alignment effort |
| Boundary | A stored region, derived module boundary, or single operator boundary |
| Evidence | Immutable record of what was observed, with links and digests to source artifacts |
| Event | Append-only audit record of a session mutation |
| Fixture Kind | `synthetic`, `reference_derived`, `real_weight`, `end_to_end`, or `other` |
| Current Evidence | Latest comparable observation for one subject/fixture/policy/engine identity |
| Historical Evidence | Superseded or stale observations retained for audit |
| Smallest Confirmed Failing Boundary | A failing boundary with no comparable failing descendant yet observed |
| Classification Candidate | Deterministic rule output, not a causal conclusion |

`semantic` remains reserved for region detection. Alignment uses `boundary`,
`evidence`, and `classification` terminology.

## 6. User workflow and CLI

### 6.1 Command surface

```text
inferref align
├── init
├── plan
├── run
├── status
├── summary
├── narrow
├── note
└── report
```

`align` without a subcommand prints help and exits 2.

### 6.2 Initialize

```bash
inferref align init reference.irtrace \
  --adapter aila.json \
  --output alignment/ \
  --scope 'model.layers' \
  --target-model DiffusionGemma \
  --target-engine Aila
```

Rules:

- validate the trace and adapter before writing the session;
- require at least one in-scope boundary;
- store the adapter source path and a redacted normalized digest, never adapter
  environment values;
- fingerprint trace metadata and all payload files used by an extracted
  testcase; source traces are treated as immutable;
- `--scope` is repeatable. A value without glob metacharacters matches an exact
  canonical boundary name and descendants. Otherwise shell-style matching is
  applied to canonical region/module names;
- initialization uses a staging directory and atomic rename. Existing output is
  rejected; there is no destructive `--force` in 0.10.

### 6.3 Plan

```bash
inferref align plan alignment/ --limit 3 --json
```

Planning is read-only and deterministic. It combines:

- stored semantic regions;
- derived module and single-operator boundaries;
- `region recommend` scores and boundary details;
- Adapter capability assessment;
- source/testcase reproducibility;
- current verified, failing, unsupported, and stale states;
- the current causal narrowing frontier.

The planner does not extract testcases or launch an engine. Example candidate:

```json
{
  "boundary": "region:41",
  "name": "layer.4.moe.router",
  "priority_score": 126,
  "plan_confidence": "high",
  "runnability": {
    "status": "eligible",
    "basis": "estimated",
    "reasons": []
  },
  "fixture_kind": "reference_derived",
  "action": "extract_and_run",
  "reasons": [
    "immediate child of current failing boundary",
    "complete payload coverage",
    "adapter capabilities appear compatible"
  ]
}
```

`priority_score` orders work; it is not a probability. `plan_confidence` is
`high`, `medium`, or `low` and describes how complete the runnability and causal
information is.

### 6.4 Run

```bash
inferref align run alignment/ --limit 1
inferref align run alignment/ --boundary region:41
inferref align run alignment/ \
  --testcase fixtures/router-synthetic/ \
  --subject region:41 \
  --fixture-kind synthetic \
  --engine-revision aila-dev-20260903-1
```

Rules:

- with no explicit boundary/testcase, recompute the plan and select its first
  `--limit N` candidates; default `N=1`;
- `--boundary` is repeatable and accepts stable IDs from `plan`;
- `--testcase` associates an already valid testcase with exactly one
  `--subject` and requires an explicit fixture kind;
- automatic region/module/operator runs extract or reuse a content-addressed
  testcase under the session;
- delegate process execution to `execute_adapter()` and comparison to
  `compare_testcase()`; every engine execution still receives a fresh run
  directory;
- stop after the first numerical failure or execution error by default;
  `--continue-on-fail` is explicit;
- accept the existing comparator/tolerance/layout overrides and store the
  effective policy in Evidence;
- resolve a stable engine identity from Git revision plus dirty-tree digest, or
  from local executable/file-argument digests. Require `--engine-revision` when
  no stable identity can be derived;
- when engine identity changes, make it the active identity, retain older runs
  as historical evidence, and mark previously verified boundaries `stale` until
  they are rerun. Planner priority stays on the previously failing boundary
  before broad regression verification;
- write Evidence and an event, then refresh the materialized session state;
- a failed comparison returns exit 1; invalid session, policy, adapter, or
  execution setup returns exit 2. A successful selected batch returns 0.

The human CLI may use the adapter pinned by the session. The Agent and MCP
`alignment_run` operations require the adapter path on every call and verify its
normalized digest against the session before execution.

### 6.5 Narrow

```bash
inferref align narrow alignment/ --json
inferref align narrow alignment/ --from region:41 --json
```

`narrow` is read-only. It explains the current smallest confirmed failing
boundary, comparable failing descendants, verified exclusions, uncovered
causal nodes, and the next narrower candidates. `run` updates narrowing state as
new evidence is committed; `narrow` never launches an engine.

### 6.6 Status and summary

```bash
inferref align status alignment/
inferref align summary alignment/ --json
```

`status` is a human-oriented complete view. `summary` is a compact projection
for Agent context. Inspection succeeds with exit 0 even if the session contains
a numerical failure; load/validation failure exits 2.

Session statuses are:

- `new`: no engine observation exists;
- `aligning`: useful runnable work remains;
- `blocked`: an active failure has no runnable narrower candidate, or an
  explicit external blocker is active;
- `verified`: every required in-scope frontier is verified;
- `stale`: the trace, adapter definition, or required policy no longer matches;
- `error`: the session cannot be validated or replayed.

### 6.7 Escape-hatch notes and classification decisions

```bash
inferref align note alignment/ \
  --category missing-observation \
  --subject region:41 \
  --reason "Need expert-routing histogram not representable by a comparator" \
  --duration-minutes 18
```

Supported note categories are `missing-observation`, `custom-tool`,
`external-blocker`, `milestone`, `classification-reject`, and
`classification-override`.

- ordinary notes are evidence of workflow activity, not evidence that a tensor
  matched or a subsystem failed;
- a rejection targets a concrete classification candidate ID;
- an override requires a domain and reason and is displayed separately as a
  user decision; it never masquerades as rule-derived classification;
- milestone `first-correct-model` closes TFCM timing;
- optional duration enables measured ad-hoc-debugging time. Missing duration is
  reported as unknown, not zero.

### 6.8 Report

```bash
inferref align report alignment/ --output alignment/reports/latest.md
```

The 0.10 reporter writes Markdown and a JSON sidecar. It contains provenance,
the fixture matrix, current and historical evidence counts, narrowing path,
classification candidates/decisions, next actions, and available productivity
metrics. Graphical visualization is not part of 0.10.

## 7. Alignment Session v0.1

### 7.1 Directory layout

```text
alignment/
├── alignment.json
├── events.jsonl
├── evidence/
│   └── ev-<uuid>.json
├── testcases/
│   └── <boundary-key>-<digest>/
├── runs/
│   └── <fresh-run-id>/
└── reports/
```

All internal paths are relative POSIX paths and must resolve inside the session
root. External trace/adapter references store a display path plus digest and are
revalidated on every operation.

### 7.2 Snapshot shape

```json
{
  "format": "inferref-alignment",
  "format_version": "0.1",
  "session_id": "al-<uuid>",
  "created_at": "2026-09-03T00:00:00Z",
  "updated_at": "2026-09-03T00:30:00Z",
  "last_event_seq": 18,
  "source": {
    "trace": {
      "path": "D:/traces/diffusion-gemma.irtrace",
      "metadata_sha256": "..."
    }
  },
  "target": {
    "model": "DiffusionGemma",
    "engine": "Aila",
    "adapter": {
      "path": "D:/aila/inferref.adapter.json",
      "normalized_sha256": "..."
    }
  },
  "scope": ["model.layers"],
  "required_fixture_kinds": ["reference_derived"],
  "status": "aligning",
  "boundaries": {},
  "narrowing": {},
  "classification": {},
  "metrics": {},
  "extensions": {}
}
```

The snapshot is a materialized view. Evidence files and the event log are the
audit trail. A loader replays events after `last_event_seq` to recover from a
crash between event commit and snapshot replacement.

### 7.3 Boundary state

Boundary aggregate states are `untested`, `verified`, `failing`, `blocked`,
`unsupported`, and `stale`.

`verified` has a strict meaning:

- every required fixture kind has current `pass` evidence;
- evidence trace/testcase, adapter definition, engine identity, and effective
  comparison policy are comparable to the current session;
- no newer comparable `fail` or `error` supersedes it.

A synthetic pass alone does not verify a reference-derived or real-weight
boundary. Historical failures remain visible after a later pass.

### 7.4 Persistence and concurrency

- mutations acquire an exclusive session lock containing PID, hostname, and
  start time;
- a live lock is a stable `alignment_locked` error; stale lock recovery is an
  explicit `--recover-stale-lock` flag on mutating commands, never automatic;
- evidence is written to a same-directory temporary file and atomically
  renamed;
- event append is flushed before the snapshot is atomically replaced;
- loaders reject malformed JSONL sequence gaps, unknown required versions,
  path traversal, missing referenced evidence, and digest mismatches;
- unknown top-level fields are rejected except under `extensions`.

## 8. Evidence v0.1 and events

### 8.1 Evidence shape

```json
{
  "format": "inferref-evidence",
  "format_version": "0.1",
  "id": "ev-<uuid>",
  "created_at": "2026-09-03T00:20:00Z",
  "type": "numerical_comparison",
  "subject": {
    "boundary": "region:41",
    "name": "layer.1.expert.3.quant_linear"
  },
  "fixture": {
    "kind": "real_weight",
    "tags": ["quantized"]
  },
  "result": "fail",
  "observations": {
    "provided_inputs_valid": true,
    "engine_inputs_observed": false,
    "outputs_match": false,
    "layout_match": true,
    "first_failure": {
      "output": "y",
      "max_abs_error": 0.1824
    }
  },
  "provenance": {
    "trace_metadata_sha256": "...",
    "testcase": {"path": "testcases/...", "tree_sha256": "..."},
    "run_record": {"path": "runs/.../inferref-run.json", "sha256": "..."},
    "adapter_normalized_sha256": "...",
    "engine_identity": {"kind": "git", "revision": "...", "dirty": false},
    "effective_comparison_sha256": "..."
  }
}
```

Native 0.10 evidence types are `numerical_comparison` and `manual_note`.
`probe_comparison` and `artifact_comparison` are reserved but rejected until a
reader implements their schema.

Evidence records observations only. In particular:

- `provided_inputs_valid` means testcase validation passed;
- `engine_inputs_observed` remains false in 0.10 native runs;
- no `inputs_match` fact is emitted without engine-observed input data;
- classification is not embedded in Evidence.

### 8.2 Current versus historical evidence

Comparable evidence has the same subject, fixture kind/tags, testcase digest,
adapter digest, engine identity, and effective comparison digest. The latest
event sequence in that equivalence class is current. Everything else remains
historical or stale.

Cross-fixture classification deliberately compares different testcase digests.
It requires the same subject, active engine identity, output-role contract, and
effective comparison digest; the explicit fixture kind is the varying axis.

If engine identity cannot be determined, `align run` requires an explicit
`--engine-revision` token instead of letting an unknown build silently
supersede earlier evidence.

### 8.3 Event shape

Each UTF-8 JSONL record has:

```json
{
  "format": "inferref-alignment-event",
  "format_version": "0.1",
  "seq": 18,
  "timestamp": "2026-09-03T00:20:00Z",
  "event": "region_failed",
  "data": {"boundary": "region:41"},
  "evidence_ids": ["ev-<uuid>"]
}
```

Required events are `session_initialized`, `plan_selected`, `testcase_reused`,
`testcase_extracted`, `boundary_verified`, `boundary_failed`,
`boundary_unsupported`, `execution_error`, `first_divergence_changed`,
`failure_domain_changed`, `note_recorded`, `classification_rejected`,
`classification_overridden`, and `session_verified`.

No-op status/plan/summary calls emit no event.

## 9. Boundary graph, planning, and narrowing

### 9.1 Boundary inventory

Stable IDs are:

- `region:<region-id>` for stored regions;
- `module:<module-id>` for derived module boundaries;
- `op:<operator-id>` for operator leaves.

Module and operator boundaries are built in memory and never written back into
the trace. Identical node sets are canonicalized with priority
`region > module > op`; aliases remain available for lookup.

### 9.2 Hierarchy

Containment is a DAG because regions may overlap. For each boundary, the
canonical parent is the smallest strict node-set superset; ties are resolved by
type priority and stable ID. Other minimal parents are retained as
`alternate_parents`.

For a failing output with a trace value ID, reverse data dependency selects the
causally relevant descendants. If provenance is missing, narrowing falls back
to containment and marks confidence `low`.

### 9.3 Smallest confirmed failure

The algorithm reports the smallest currently observed failing boundary, not the
smallest possible bug location:

1. consider only current comparable failing evidence;
2. choose failures reachable below the current failure in the containment DAG;
3. prefer the smallest node set, then causal relevance, then stable ID;
4. if no failing descendant exists, keep the parent as the smallest confirmed
   failure and recommend runnable relevant children;
5. verified children exclude only their exact tested fixture/policy/engine
   equivalence class.

If all relevant children pass but their parent fails, report an interaction or
unobserved-boundary gap. Do not declare every child implementation correct for
other fixtures or end-to-end execution.

### 9.4 Planning score

The existing `recommend_regions()` result is the base score for stored regions.
Derived boundaries receive the same measurable features. Session modifiers are
then applied in this order:

- exclude current verified and definitively unsupported candidates;
- strongly prefer immediate relevant children of the current failure;
- prefer boundaries on the reverse dependency cone of the first failing output;
- prefer reusable validated testcases;
- penalize unknown capability compatibility and large boundary cost;
- deterministic tie break: score descending, node count ascending, stable ID.

Every score contribution is returned in `reasons`. Tests assert ordering and
reasons, not one magic total across unrelated traces.

## 10. Failure-domain classification

### 10.1 Stable domains

Wire IDs are lowercase:

```text
implementation
layout
input_preparation
model_artifact
quantization
reference
environment
unknown
```

Uppercase names may be rendered for humans but are not wire aliases.

### 10.2 Candidate shape and confidence

```json
{
  "id": "fc-<digest>",
  "domain": "model_artifact",
  "confidence": 0.7,
  "rule": "fixture_real_fails_after_synthetic_passes/v1",
  "supporting_evidence": ["ev-a", "ev-b"],
  "contradicting_evidence": [],
  "limitations": ["engine inputs were not observed"]
}
```

Confidence is a deterministic ranking value defined by the rule, not a
calibrated probability. A candidate ID is the digest of rule version and sorted
supporting/contradicting evidence IDs.

### 10.3 Required 0.10 rules

| Rule | Candidate | Confidence | Required explicit observations |
| --- | --- | ---: | --- |
| logical values pass but strict layout fails | `layout` | 0.95 | same output and run expose both facts |
| synthetic boundary fails after successful process execution | `implementation` | 0.60 | output mismatch; no engine-input observation claim |
| real-weight fails while comparable synthetic and reference-derived fixtures pass | `model_artifact` | 0.70 | all three fixture labels and current evidence |
| previous rule plus explicit `quantized` fixture tag | `quantization` | 0.75 | explicit tag, not dtype/name guessing |
| engine-observed input differs from reference | `input_preparation` | reserved | cannot fire from native 0.10 evidence |
| independent references disagree | `reference` | reserved | cannot fire without typed future evidence |
| same engine identity passes/fails only by environment | `environment` | reserved | cannot fire without comparable environment evidence |
| no rule has adequate support | `unknown` | 0.0 | always available fallback |

Multiple candidates may be returned. Rules list contradictory evidence and
reduce or suppress candidates deterministically. Rejections and overrides are
decisions layered over classifier output; they do not edit Evidence.

## 11. Agent facade, MCP, and summary

### 11.1 Agent operations

Add stable operations and matching CLI commands:

```text
alignment_init
alignment_plan
alignment_run
alignment_status
alignment_narrow
alignment_summary
alignment_note
```

All return Agent Protocol v0.1 envelopes. The response envelope shape is
unchanged, so the protocol version remains 0.1. `capabilities()` adds the
Alignment/Evidence/Event formats and new operations.

Envelope semantics:

- init/run/note: `pass` when the mutation completed, `fail` for a numerical
  mismatch, and `error` for structural/execution failure;
- plan/status/narrow/summary: `ok` when inspection succeeded. Alignment state is
  in `data.status`, so a known mismatch does not make an inspection call fail;
- `next_actions` always uses `{operation, reason, ...}` objects and includes
  exact session and boundary arguments where useful.

### 11.2 MCP tools

Expose:

```text
inferref_alignment_init
inferref_alignment_plan
inferref_alignment_run
inferref_alignment_status
inferref_alignment_narrow
inferref_alignment_summary
inferref_alignment_note
```

MCP path policy applies transitively. A session path being allowed does not
authorize trace, adapter, testcase, run, or report paths referenced inside it.
Alignment services accept an access-policy callback; the MCP layer revalidates
every resolved read/write target. `alignment_run` requires an explicit adapter
argument and checks it against the pinned digest.

### 11.3 Compact summary

`alignment_summary` returns at most the current state needed for the next turn:

```json
{
  "status": "blocked",
  "counts": {"verified": 47, "failing": 1, "untested": 13},
  "current_failure": {
    "boundary": "region:41",
    "name": "layer.1.expert.3.quant_linear",
    "engine_inputs": "not_observed",
    "fixtures": {
      "synthetic": "pass",
      "reference_derived": "pass",
      "real_weight": "fail"
    }
  },
  "classification_candidates": [
    {
      "id": "fc-...",
      "domain": "quantization",
      "confidence": 0.75,
      "supporting_evidence": ["ev-a", "ev-b", "ev-c"]
    }
  ],
  "next_actions": [
    {
      "operation": "alignment_note",
      "reason": "Record missing quantized artifact observation capability."
    }
  ],
  "artifacts": {"alignment": "alignment/alignment.json"}
}
```

Summary size is bounded by counts, one current failure, the top three
classification candidates, and the top three next actions. Full history belongs
in status/report.

## 12. Fixture matrix and productivity metrics

The session derives a matrix by subject and fixture kind. It never merges runs
whose engine identity or comparison policy is not comparable.

Automatically measurable:

- numerical alignment time: first current mismatch to session verified;
- InferRef validation steps: committed numerical-comparison evidence count;
- build/run cycles: distinct engine identities and run records;
- verified/failing/untested boundary counts.

User-recorded:

- TFCM: session initialization to `first-correct-model` milestone;
- ad-hoc debugging time: sum of note durations for custom-tool and
  missing-observation work;
- ad-hoc debugging steps: relevant note count.

Adoption ratio is:

```text
InferRef numerical validation steps
-----------------------------------------------
InferRef numerical validation + recorded ad-hoc steps
```

It is labeled `complete: false` unless the user explicitly marks activity
logging complete. An absent denominator is `unknown`, never 100%.

## 13. Compatibility and security

- Trace IR remains 0.1.
- Testcase write versions remain 0.2/0.3.
- Comparison Spec remains 0.1.
- Suite remains 0.2/0.3.
- Scenario remains 0.1.
- Engine Adapter remains 0.2.
- Agent response protocol remains 0.1.
- New Alignment, Evidence, and Event formats start at 0.1.

Alignment sessions contain trusted adapter references. Existing adapter process
limits, shell-free invocation, output truncation, artifact limits, and fresh run
directories remain mandatory. Reports and summaries use redacted adapter
records and never serialize configured environment values.

## 14. Acceptance criteria

### 14.1 Session and persistence

- [ ] `align init` creates a valid session atomically and refuses existing output.
- [ ] Evidence is immutable, content-linked to testcase/run artifacts, and replayable.
- [ ] Event sequence and snapshot recovery survive an injected mid-commit failure.
- [ ] Concurrent mutation is rejected with `alignment_locked`.
- [ ] Path traversal, missing evidence, digest mismatch, and unsupported versions fail closed.
- [ ] Trace/adapter definition drift produces `stale`; it never silently reuses verification.

### 14.2 Plan and run

- [ ] Planner inventories stored regions, derived modules, and operator leaves.
- [ ] Nested and overlapping regions produce a deterministic containment DAG.
- [ ] Plan excludes current verified/unsupported work and explains every score.
- [ ] Capability assessment occurs before engine process creation.
- [ ] `align run` reuses only digest-identical valid testcases.
- [ ] Every engine invocation uses a fresh directory and existing run-record format.
- [ ] Numerical pass/fail/error creates correct Evidence and events.
- [ ] Fixture kinds remain distinct in state and report.

### 14.3 Narrowing and classification

- [ ] Reverse dependency uses the first failing output value when provenance exists.
- [ ] The smallest confirmed failure never claims unobserved descendants failed.
- [ ] A standalone testcase input is not reported as an observed matching engine input.
- [ ] Every classification candidate names a versioned rule and supporting evidence.
- [ ] Required layout, implementation, model-artifact, quantization, and unknown rules pass fixtures.
- [ ] Reserved input/reference/environment rules cannot fire without their typed evidence.
- [ ] User rejection/override is auditable and does not mutate Evidence.

### 14.4 CLI, Agent, MCP, and skill

- [ ] All human commands support `--json`; Agent commands return v0.1 envelopes.
- [ ] Capabilities reports new formats, operations, and exact MCP tool names.
- [ ] Summary is bounded and supplies structured next actions.
- [ ] MCP revalidates transitive session paths and requires the adapter on run.
- [ ] Skill starts numerical debugging with alignment state before ad-hoc tooling.
- [ ] Skill records a missing capability whenever custom numerical tooling is required.
- [ ] Existing 0.9 CLI, Agent, comparator, suite, scenario, and C++ tests remain green.

### 14.5 End-to-end acceptance fixture

One deterministic fixture trace contains a parent region, two nested children,
an overlap, and a failing engine implementation. The acceptance test must prove:

1. init succeeds;
2. plan selects a runnable high-value boundary;
3. run records a failure;
4. narrow selects the causally relevant child;
5. a fixed engine identity rerun preserves old evidence and verifies the child;
6. summary reports the remaining parent interaction without inventing input evidence;
7. note records an escape-hatch reason and updates incomplete adoption metrics;
8. event replay reconstructs the same snapshot.

The external Diffusion Gemma/Aila-style A/B experiment is a release report
artifact, not a hermetic CI test. Productivity improvement claims require that
field evidence; the software release itself is gated by the deterministic
fixture above.

## 15. Implementation tasks and ownership

Each task has one integration owner. Workers may add task-local tests, but the
listed shared files belong to the named owner to reduce merge conflicts.

### L1: Session schemas, store, hashing, and recovery

- Owner: persistence worker.
- Files: `inferref/alignment/schema.py`, `store.py`, `hashing.py`,
  `tests/core/test_alignment_store.py`.
- Outcome: sections 7, 8.2, and persistence/security acceptance.
- Must expose stable APIs for loading, locking, committing evidence/events, and
  rebuilding the snapshot. It must not contain planning or classification logic.

### L2: Boundary graph and deterministic planner

- Owner: planning worker.
- Files: `inferref/alignment/boundaries.py`, `planner.py`,
  `tests/core/test_alignment_planner.py`.
- Outcome: sections 9.1, 9.2, and 9.4.
- Reuse `region.analysis`, `region.boundary`, `region.recommend`, and adapter
  capability assessment rather than cloning their rules.

### L3: Evidence normalization and fixture matrix

- Owner: evidence worker.
- Files: `inferref/alignment/evidence.py`, `matrix.py`,
  `tests/core/test_alignment_evidence.py`.
- Outcome: sections 8.1, 8.2, and 12 data projections.
- Depends on L1 schema/store contracts. It extracts facts from existing
  comparison/run records and never emits classification.

### L4: Alignment runner and testcase cache

- Owner: orchestration worker.
- Files: `inferref/alignment/runner.py`, `testcase_cache.py`,
  `tests/core/test_alignment_runner.py`.
- Outcome: sections 6.4 and 14.2.
- Depends on L1-L3. It delegates extraction/execution/comparison to existing
  functions and owns no numerical comparison implementation.

### L5: Narrowing and rule classifier

- Owner: diagnosis worker.
- Files: `inferref/alignment/narrowing.py`, `classification.py`,
  `tests/core/test_alignment_diagnosis.py`.
- Outcome: sections 9.3 and 10.
- Depends on L2-L3. Rule IDs and confidence values are fixtures and require code
  review when changed.

### L6: Human CLI and reports

- Owner: CLI worker; exclusive owner of `inferref/cli/main.py` for this release.
- Files: `inferref/cli/main.py`, `inferref/alignment/report.py`,
  `tests/core/test_alignment_cli.py`.
- Outcome: sections 6 and 6.8, including exit-code behavior.
- Depends on stable public APIs from L1-L5; it contains no business logic.

### L7: Agent facade, MCP, summary, and path policy

- Owner: Agent integration worker.
- Files: `inferref/agent/service.py`, `inferref/agent/mcp_server.py`,
  `inferref/agent/summary.py`, `inferref/agent/path_policy.py`,
  `inferref/agent/__init__.py`, `tests/agent/test_alignment_mcp.py`,
  `tests/core/test_alignment_agent.py`.
- Outcome: section 11 and transitive path-policy acceptance.
- Depends on L1-L5 and must not edit `inferref/cli/main.py`.

### L8: InferRef-first skill, workflow docs, and metrics semantics

- Owner: documentation/skill worker.
- Files: `skills/inferref/`, `docs/AGENT_WORKFLOW.md`, `README.md`,
  `docs/dev/status/`, existing skill validation tests.
- Outcome: policy and workflow adoption requirements plus metric definitions.
- Begins after L6-L7 freeze command/tool names.

### L9: Integration fixture and release gate

- Owner: release integration worker.
- Files: `tests/fixtures/alignment/`,
  `tests/core/test_alignment_e2e.py`, CI configuration only if required,
  `docs/dev/status/<date>-v0100-alignment-orchestrator.md`.
- Outcome: section 14.5 and full regression pass.
- This owner resolves package exports and cross-task test failures after L1-L8;
  feature workers should not make unrelated cleanup changes here.

### Deferred design tasks (not 0.10 release blockers)

- D1: Debug Probe Protocol and Adapter v0.3 compatibility proposal.
- D2: Model/quantized artifact comparison schema and comparator proposal.
- D3: alignment graph visualization and cross-commit regression report.

Each deferred task starts with a separate spec review. It must not land an
unversioned probe directory convention or make 0.10 infer `inputs_match` from
reference testcase inputs.

## 16. Dispatch waves

```text
Wave 1 (parallel)
  L1 Session store
  L2 Boundary graph/planner

Wave 2 (after L1 contracts; parallel)
  L3 Evidence/matrix
  L5 Narrowing/classifier skeleton (uses frozen L2 boundary API)

Wave 3
  L4 Runner/testcase cache (after L1-L3)
  L5 completes against real evidence projections

Wave 4 (parallel after L4/L5 public API freeze)
  L6 Human CLI/report
  L7 Agent/MCP/summary/path policy

Wave 5
  L8 Skill/docs
  L9 End-to-end fixture and integration review
```

Merge order is L1 -> L2 -> L3 -> L4 -> L5 -> L6/L7 -> L8 -> L9. Parallel work
uses separate branches/worktrees; only L9 resolves final cross-branch conflicts.

## 17. Review gates

Before implementation is declared complete:

1. schema review: formats, state transitions, replay, path containment;
2. algorithm review: containment DAG, causal narrowing, comparability classes;
3. epistemic review: every observation/classification statement is justified by
   available artifacts and never upgrades inference to fact;
4. security review: trusted adapter invocation, transitive MCP paths, secret
   redaction, lock recovery, and report paths;
5. independent end-to-end review using only documented CLI/Agent interfaces;
6. full Python and C++ regression gates on supported platforms.

## 18. Release theme

```text
0.9: "Does my engine reproduce the reference at tensor/task semantics?"

0.10: "Given everything already observed, what should I validate next, and
       which failure hypotheses has the evidence actually narrowed?"
```
