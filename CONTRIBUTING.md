# Contributing

Contributions should preserve the distinction between approximate neural components and exact optimization components.

## Development setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
mypy --python-version 3.11 src
pytest
```

## Correctness requirements

Changes to dynamic programming, backtracking, bounds, pruning, dataset labels, or checkpoint loading require focused regression tests. An exact solver change must retain at least one independent verification path. Neural guidance must not be allowed to invalidate a bound or suppress a branch without a correctness-preserving proof.

## Experimental changes

Document new regimes, seeds, metrics, and claims boundaries. Do not replace disaggregated metrics with a single opaque score. Negative results and slower guidance modes should remain reportable.

## Style

Use type annotations, immutable domain records where practical, deterministic tie-breaking, explicit validation, and concise docstrings. Keep the public API small.
