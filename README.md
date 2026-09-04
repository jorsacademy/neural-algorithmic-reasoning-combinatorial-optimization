# Neural Algorithmic Reasoning for Combinatorial Optimization

[![CI](https://github.com/jorsacademy/neural-algorithmic-reasoning-combinatorial-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/neural-algorithmic-reasoning-combinatorial-optimization/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A verification-first research implementation of **neural algorithmic reasoning (NAR)** for 0-1 knapsack. The repository studies whether supervising a size-equivariant neural processor on the intermediate states of the exact Bellman recurrence produces better extrapolation and more useful exact-search guidance than supervising a neural policy only on final optimal selections.

The neural models never certify optimality. Exactness comes from dynamic programming and branch-and-bound. Learned outputs are used only as feasible incumbents and traversal advice, so a poor neural prediction may slow search but cannot change the returned optimum.

## Research question

> Does intermediate Bellman-trace supervision improve size and distribution generalization, and does that algorithmic knowledge reduce exact branch-and-bound search relative to terminal-only supervision and a classical density heuristic?

The benchmark separates three questions that are often conflated:

1. **Algorithm imitation:** can a shared processor reproduce the local Bellman update?
2. **Combinatorial solution quality:** does an autoregressive rollout reconstruct a good feasible subset on larger or shifted instances?
3. **Exact-solver utility:** does neural advice reduce branch-and-bound nodes while preserving exactness?

## Claims boundary

This is a compact methodology benchmark. It does **not** claim:

- a state-of-the-art knapsack solver;
- exact neural execution on arbitrary instance sizes;
- that trace supervision always beats direct supervision;
- that fewer search nodes necessarily imply lower wall-clock time at industrial scale;
- transfer to every dynamic program or NP-hard problem;
- reproduction of the full CLRS benchmark;
- commercial solver integration;
- optimality certification by a neural network.

The controlled synthetic protocol is intended to expose failure modes, not hide them behind one aggregate score.

## Optimization problem

For positive integer weights \(w_i\), values \(v_i\), and capacity \(C\), 0-1 knapsack is

\[
\max_{x\in\{0,1\}^n}
\sum_{i=1}^{n}v_i x_i
\quad\text{subject to}\quad
\sum_{i=1}^{n}w_i x_i\leq C.
\]

The exact dynamic program defines

\[
V_i(c)=
\max\left\{
V_{i-1}(c),
\;v_i+V_{i-1}(c-w_i)
\right\},
\]

with the take term disabled when \(w_i>c\). Ties deterministically choose **skip**, giving a reproducible canonical trace and backtracking target.

## Trace-supervised Bellman processor

The `BellmanReasoner` learns one shared local comparison operator. For every item and capacity cell it receives:

- normalized skip value;
- normalized take value;
- take-minus-skip difference;
- normalized capacity position;
- item-weight fraction;
- item-value fraction;
- feasibility indicator.

The network predicts a take logit \(z\). Its next-value estimate is constrained to the interval between the two Bellman candidates:

\[
\widehat V_i(c)
=
V_{i-1}(c)
+
\sigma(z)
\left[
\left(v_i+V_{i-1}(c-w_i)\right)-V_{i-1}(c)
\right].
\]

This inductive bias makes the learned operation size-equivariant: the same parameters are reused across capacities and item steps. Training combines take/skip classification with next-value regression. Evaluation is autoregressive; exact previous rows are not injected at test time.

```text
exact DP traces during training
          │
          ▼
shared local Bellman processor
          │ reused over items and capacities
          ▼
autoregressive value-table rollout
          │
          ▼
backtracked feasible candidate + item scores
```

## Terminal-only baseline

The `DirectPolicy` is a permutation-equivariant DeepSets-style network. It embeds each item, pools global mean and maximum representations, and predicts one inclusion logit per item from the original instance only. It receives no intermediate dynamic-programming supervision.

Both neural approaches use deterministic repair to produce a feasible candidate. Repair is reported separately from exact solving and cannot establish optimality.

## Exact branch-and-bound bridge

Three guidance modes are compared:

| Guidance | Initial incumbent | Variable/branch advice |
| --- | --- | --- |
| `density` | value-density greedy solution | density order and take-first traversal |
| `direct_policy` | repaired terminal-policy candidate | policy confidence and preferred decisions |
| `trace_reasoner` | backtracked Bellman-rollout candidate | trace confidence and preferred decisions |

Every branch-and-bound node uses a valid **fractional-knapsack upper bound** over the remaining undecided items. Neural scores affect only traversal order and the starting incumbent. The exact solver is independently checked against dynamic programming on every evaluated instance.

This design isolates a practically important criterion: a neural model can be useful even when its standalone solution is imperfect, provided it discovers strong incumbents or promising branches early.

## Exactness and reliability contract

The implementation fails closed when a correctness boundary is violated.

- Dynamic programming uses integer objective arithmetic.
- Small instances can enumerate all \(2^n\) subsets and compare the optimum.
- Stored corpus labels are recomputed and audited on load.
- Every candidate is checked for binary decisions, capacity feasibility, and objective consistency.
- Branch-and-bound upper bounds do not depend on neural validity.
- Every branch-and-bound result is compared with the dynamic-programming optimum.
- Checkpoints use Safetensors rather than pickle.
- Non-finite losses, scores, logits, and gradients raise errors.
- Seeds and corpus fingerprints are stored in reports.

See [`docs/exactness.md`](docs/exactness.md) for the formal boundary between exact and approximate components.

## Controlled generalization protocol

Training item counts are \(n\in\{8,10,12,14\}\). The frozen protocol evaluates disjoint seeds under:

1. `interpolation` — nominal 12-item instances;
2. `size_18` — nominal size extrapolation;
3. `size_24` — harder nominal size extrapolation;
4. `tight_capacity` — approximately 25% of total weight;
5. `loose_capacity` — approximately 65% of total weight;
6. `inverse_correlated` — high weights tend to have lower values;
7. `heavy_tail` — Pareto-like value distribution;
8. `clustered` — bimodal item weights;
9. `value_scale_shift` — values outside the training scale.

Training includes nominal, weakly correlated, and strongly correlated instances. Evaluation regimes and seeds are never used to fit either model.

## Metrics

### Algorithmic fidelity

- full-table value RMSE;
- terminal value relative error;
- take/skip trace accuracy;
- rollout candidate relative gap.

### Standalone combinatorial quality

- mean, median, and P90 relative objective gap;
- optimal-solution hit rate;
- feasibility rate;
- capacity utilization;
- deterministic bootstrap interval for mean relative gap.

### Exact-search utility

- mean, median, and P90 branch-and-bound node count;
- pruned nodes;
- initial incumbent relative gap;
- exact-solution verification rate;
- mean node reduction relative to density guidance;
- deterministic paired bootstrap interval for node reduction;
- wall-clock search time, reported without claiming hardware-independent speedup.

No weighted composite score merges trace fidelity, approximation quality, and exact-search efficiency.

## Installation

```bash
python -m pip install -e ".[dev]"
```

CPU-only PyTorch is sufficient.

## CLI

### Generate one exact-labeled instance

```bash
narcopt generate \
  --item-count 12 \
  --regime iid \
  --seed 42 \
  --output artifacts/example.json
```

### Build deterministic corpora

```bash
narcopt collect \
  --count 80 \
  --item-counts 8 10 12 14 \
  --regimes iid iid weakly_correlated strongly_correlated \
  --seed 3026 \
  --output artifacts/train.jsonl

narcopt collect \
  --count 24 \
  --item-counts 8 10 12 14 \
  --regimes iid weakly_correlated strongly_correlated \
  --seed 4026 \
  --output artifacts/validation.jsonl
```

### Verify dynamic programming by enumeration

```bash
narcopt oracle artifacts/validation.jsonl \
  --sample-index 0 \
  --output artifacts/oracle-check.json
```

### Train the trace reasoner

```bash
narcopt train artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --model trace_reasoner \
  --epochs 30 \
  --checkpoint artifacts/trace-reasoner.safetensors \
  --output-report artifacts/trace-training.json
```

### Train the terminal-only policy

```bash
narcopt train artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --model direct_policy \
  --epochs 40 \
  --checkpoint artifacts/direct-policy.safetensors \
  --output-report artifacts/policy-training.json
```

### Compare approximation and exact-search guidance

```bash
narcopt benchmark artifacts/test.jsonl \
  --reasoner-checkpoint artifacts/trace-reasoner.safetensors \
  --policy-checkpoint artifacts/direct-policy.safetensors \
  --scenario size_24 \
  --output-json artifacts/benchmark.json \
  --output-csv artifacts/benchmark.csv
```

### Run the frozen research protocol

```bash
narcopt research \
  --config configs/research_v1.json \
  --checkpoint-directory artifacts/checkpoints \
  --output-report artifacts/research-report.json
```

## Repository layout

```text
src/narcopt/
├── domain.py       # exact DP, brute force, solution audits, greedy repair
├── dataset.py      # controlled generators, JSONL corpora, SHA-256 fingerprints
├── models.py       # Bellman reasoner, terminal policy, Safetensors checkpoints
├── reasoning.py    # trace cells, autoregressive rollout, neural advice
├── search.py       # exact branch-and-bound with fractional upper bounds
├── training.py     # deterministic trace and terminal supervision
├── evaluation.py   # fidelity, approximation, exact-search metrics and bootstrap
├── experiment.py   # frozen nine-scenario transfer protocol
└── cli.py          # end-to-end workflows
```

Additional documentation:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/exactness.md`](docs/exactness.md)
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- [`docs/research_context.md`](docs/research_context.md)
- [`docs/model_card.md`](docs/model_card.md)

## Tests and CI

GitHub Actions runs on Python 3.11 and 3.12:

```text
package installation and dependency check
Ruff lint and formatting
strict mypy
branch-aware pytest coverage
collect → train both models → exact oracle → benchmark smoke
```

The regression suite covers dynamic-programming recurrence consistency, exhaustive verification, deterministic generation, corpus tamper detection, feature invariants, feasible rollout/backtracking, safe checkpoint round trips, exact branch-and-bound under arbitrary neural advice, both training modes, report serialization, the frozen protocol, and the CLI.

## Methodological limitations

0-1 knapsack has a pseudo-polynomial exact dynamic program, making it unusually suitable for dense intermediate supervision. Results should not be extrapolated mechanically to problems without tractable trace generation. The local processor is deliberately small and strongly biased toward learning a comparison operator; it is not a generic Transformer or graph neural executor.

Canonical skip-on-tie labels make supervision reproducible but select one representative among potentially many optimal solutions. A direct policy may be penalized for predicting a different optimal subset during training even though evaluation treats any optimal objective as a hit.

Branch-and-bound node counts depend on the chosen relaxation, branching implementation, and item distribution. Neural inference overhead may dominate on small instances. Accordingly, node reduction and runtime are both reported, and exact-solver claims are restricted to this implementation.

## Research context

The repository is positioned relative to:

- Veličković et al., [“The CLRS Algorithmic Reasoning Benchmark”](https://proceedings.mlr.press/v162/velickovic22a.html), ICML 2022, which standardizes intermediate-supervision tasks for classical algorithms;
- Georgiev et al., [“Neural Algorithmic Reasoning for Combinatorial Optimisation”](https://proceedings.mlr.press/v231/georgiev24a.html), LoG 2023/2024 proceedings, which studies algorithmic pretraining for combinatorial tasks;
- Požgaj et al., [“KNARsack: Teaching Neural Algorithmic Reasoners to Solve Pseudo-Polynomial Problems”](https://arxiv.org/abs/2509.15239), 2025, which directly studies DP-trace supervision for knapsack;
- Gasse et al., [“Exact Combinatorial Optimization with Graph Convolutional Neural Networks”](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html), NeurIPS 2019, which demonstrates learned branching inside exact search;
- He and Vitercik, [“Primal-Dual Neural Algorithmic Reasoning”](https://proceedings.mlr.press/v267/he25r.html), ICML 2025, which extends NAR toward approximation-algorithm structure;
- Hertrich and Skutella, [“Provably Good Solutions to the Knapsack Problem via Neural Networks of Bounded Size”](https://arxiv.org/abs/2005.14105), 2020, which analyzes neural representations of knapsack dynamic programs.

This implementation is not a reproduction of any one paper. Its narrower contribution is the **trace-to-search bridge**: evaluate the same algorithmic reasoner both as a standalone approximate solver and as non-binding guidance for an independently verified exact solver.

## License

PolyForm Noncommercial 1.0.0. The repository is source-available for noncommercial use; it is not offered under an OSI-approved open-source license.
