# Exactness and Reliability Contract

## Exact components

For every declared 0-1 knapsack instance:

- item weights and values are positive integers;
- the dynamic program uses integer arithmetic and deterministic skip-on-tie backtracking;
- small instances can enumerate every subset and compare the optimal objective;
- solution audits recompute total weight and objective from the binary vector;
- corpus loading recomputes optimality rather than trusting stored labels;
- branch-and-bound uses a valid fractional-knapsack upper bound;
- every branch-and-bound result is compared with the dynamic-programming optimum.

## Approximate components

- Bellman gates are learned approximations;
- autoregressive rollout values can drift from the exact table;
- neural and density candidates may be suboptimal;
- neural score confidence is not calibrated;
- bootstrap intervals have Monte Carlo error;
- wall-clock timings depend on hardware and process noise.

## Non-binding neural guidance

Neural outputs affect only:

1. the feasible incumbent available before search;
2. the order in which variables are branched;
3. whether include or exclude is visited first.

They do not affect the upper bound. Therefore invalid or misleading advice cannot cause an incorrect optimum to be accepted. The implementation still validates advice shape, finiteness, feasibility, and objective consistency to avoid silently benchmarking malformed inputs.

## Independent verification

Dynamic programming and branch-and-bound have different computational structures. Agreement between them is a stronger regression check than testing either implementation against its own stored output. Exhaustive enumeration adds a third independent check for small instances.

## Failure policy

The code raises rather than silently repairs correctness failures involving:

- invalid instance domains;
- inconsistent exact labels;
- corpus fingerprint mismatch;
- non-finite neural outputs or gradients;
- infeasible search advice;
- branch-and-bound disagreement with dynamic programming;
- incompatible checkpoint schema or model type.
