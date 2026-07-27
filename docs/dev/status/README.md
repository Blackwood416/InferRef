# Development status reports

Point-in-time snapshots of where the project stands: what works, what is deliberately deferred,
what is known to be broken or untested, and what to do next.

## Convention

One file per snapshot, named `YYYY-MM-DD-<slug>.md`. Snapshots are **not** edited after the fact —
write a new one instead, so the history stays readable. The newest entry at the top of the index
below is the current state.

Each report should cover:

- phase status against [SPEC §64](../../spec/InferRef_SPEC.md)
- acceptance criteria against [Trace IR §57](../../spec/InferRef_Trace_IR_v0.1.md)
- what exists, with enough detail to orient someone who has not read the code
- verified environment, and explicitly what is *not* verified
- implementation decisions that constrain future changes
- known gaps and limitations, including test coverage gaps
- suggested next steps
- commands to reproduce the claimed state

The gaps section matters most. A status report that only lists accomplishments is not useful for
planning.

## Index

| Date | Report | State |
| --- | --- | --- |
| 2026-07-27 | [MVP complete](2026-07-27-mvp-complete.md) | SPEC §64 Phases 0-3 done; 148 tests passing; Windows only |
