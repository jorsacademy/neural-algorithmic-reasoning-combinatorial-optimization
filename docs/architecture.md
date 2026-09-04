# Architecture

## Data flow

```text
controlled generator
      │
      ├── exact dynamic program ──► canonical solution + Bellman trace
      │
      └── JSONL corpus ───────────► SHA-256 fingerprint + exact-label audit

training corpus
      │
      ├── trace cells ────────────► BellmanReasoner
      │
      └── terminal masks ─────────► DirectPolicy

held-out instance
      │
      ├── density heuristic
      ├── direct-policy candidate
      └── Bellman rollout candidate
              │
              ▼
exact branch-and-bound with fractional upper bound
              │
              ▼
DP verification + metrics
```

## Domain layer

`domain.py` is dependency-light and contains the exact reference semantics. It defines immutable instances and solutions, integer dynamic programming, exhaustive verification, feasibility audits, density greedy construction, and deterministic repair.

## Trace representation

For an instance with `n` items and capacity `C`, the exact trace contains two `(n + 1) × (C + 1)` matrices:

- integer Bellman values;
- Boolean take decisions, with skip chosen on equality.

Trace supervision is flattened into independent cells for teacher-forced training. Autoregressive evaluation reconstructs complete rows and accumulates approximation error across item steps.

## Bellman reasoner

The processor is an MLP shared over every item-capacity cell. A sigmoid gate interpolates between the valid skip and take candidates. This preserves feasibility of the local value estimate and makes the parameter count independent of item count and capacity.

## Direct policy

The terminal baseline embeds each item, pools mean and maximum embeddings, and decodes inclusion logits. It is permutation-equivariant but receives no recurrence state or trace labels.

## Exact-search interface

`HeuristicAdvice` contains:

- a source label;
- a preferred binary selection;
- one finite score per item;
- a feasible incumbent candidate.

`search.py` validates this boundary before search. Advice selects variable order, branch order, and an initial incumbent. It never alters the fractional upper bound, pruning rule, or final verification.

## Persistence

Datasets use human-readable JSONL with a manifest, exact labels, and a fingerprint. Models use Safetensors plus JSON metadata. Reports use JSON and optional CSV.
