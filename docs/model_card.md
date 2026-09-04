# Model Card

## Intended use

The models are research baselines for studying trace supervision and exact-search guidance on synthetic 0-1 knapsack instances. They are suitable for controlled experiments, teaching, and regression testing.

## Out-of-scope use

- production optimization;
- financial, medical, safety-critical, or legal decisions;
- optimality certification without the exact solver;
- claims about arbitrary NP-hard problems;
- deployment on distributions not explicitly evaluated.

## Inputs and outputs

`BellmanReasoner` consumes local normalized Bellman-cell features and outputs take logits. Autoregressive rollout yields an approximate value table, item scores, and a feasible candidate.

`DirectPolicy` consumes normalized item features and outputs one inclusion logit per item. Deterministic repair produces a feasible candidate.

## Training data

All data are synthetic and generated from documented positive-integer regimes. Exact labels come from dynamic programming. There are no personal, copyrighted, or sensitive records.

## Evaluation

Reports separate trace fidelity, standalone objective gaps, and exact-search node counts. Branch-and-bound results are verified against dynamic programming.

## Limitations

The models are small, CPU-oriented baselines. Canonical tie-breaking introduces label arbitrariness when multiple optimal subsets exist. The local Bellman architecture has strong task-specific inductive bias and should not be interpreted as a generic reasoning model.
