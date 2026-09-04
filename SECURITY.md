# Security Policy

## Supported versions

Only the current `main` branch is maintained.

## Reporting a vulnerability

Do not publish exploitable details in a public issue. Use GitHub's private vulnerability reporting interface when available. Include the affected commit, reproduction steps, impact, and any proposed mitigation.

## Security boundaries

The repository processes local JSON, JSONL, CSV, and Safetensors files. It does not require network access at runtime and does not execute model-provided code.

Safetensors is used instead of pickle for checkpoints. JSON input is validated before constructing domain objects. Corpus manifests are fingerprinted, and exact labels are recomputed on load. Paths supplied to the CLI are treated as local user-controlled paths; run the software with filesystem permissions appropriate to the data being processed.

## Untrusted artifacts

Even non-executable files may be crafted to consume excessive memory or computation. Apply external file-size limits before loading untrusted corpora or checkpoints. The branch-and-bound solver is exponential in the worst case; untrusted instances can cause deliberate denial of service through search complexity.

## Dependency policy

CI installs declared dependencies into an isolated hosted runner, checks dependency consistency, and uses pinned commit SHAs for GitHub Actions. Review dependency updates before merging them.
