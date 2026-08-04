# Semantic detector plugins

Third-party detectors use the standard Python entry-point group
`inferref.semantic_detectors`:

```toml
[project.entry-points."inferref.semantic_detectors"]
project_gate = "project.detectors:create_detector"
```

The target must be a zero-argument factory returning an object that implements
`SemanticDetector`. The entry-point name and `detector.name` must match.
Installed plugins are discoverable through `inferref doctor` without importing
their modules or calling their factories. Use `inferref doctor --verify-plugins`
to explicitly import and instantiate them. A detector participates in semantic
analysis only when explicitly selected:

```bash
inferref region detect trace/ --detector project_gate
```

Plugin names cannot shadow built-in detectors, and duplicate names or import
failures are reported as structured errors.
