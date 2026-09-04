# Frozen Experiment Protocol

## Hypothesis

Intermediate supervision of the Bellman recurrence should encourage size-equivariant computation. The primary hypothesis is not merely that the trace reasoner predicts better subsets, but that its scores and incumbents remain useful to an exact solver under size and distribution shift.

## Training data

- item counts: 8, 10, 12, and 14;
- regimes: IID, weakly correlated, and strongly correlated;
- deterministic disjoint train and validation seeds;
- exact canonical labels from dynamic programming;
- no evaluation instance is reused for training or model selection.

## Models

### Trace reasoner

Teacher-forced cell supervision combines binary take/skip cross-entropy and next-value regression. Validation combines teacher-forced loss with autoregressive candidate gap.

### Direct policy

Terminal-only binary cross-entropy uses the exact canonical selection. The architecture is permutation-equivariant and has a comparable hidden width, but it receives no intermediate trace targets.

## Evaluation scenarios

| Scenario | Item count | Shift |
| --- | ---: | --- |
| `interpolation` | 12 | none |
| `size_18` | 18 | size |
| `size_24` | 24 | larger size |
| `tight_capacity` | 18 | capacity ratio |
| `loose_capacity` | 18 | capacity ratio |
| `inverse_correlated` | 18 | value-weight relation |
| `heavy_tail` | 18 | value tail |
| `clustered` | 18 | bimodal weights |
| `value_scale_shift` | 18 | numeric scale |

## Comparisons

Standalone candidates:

- density greedy;
- direct policy plus deterministic repair;
- Bellman rollout plus deterministic repair.

Exact search:

- density-guided branch-and-bound;
- direct-policy-guided branch-and-bound;
- trace-reasoner-guided branch-and-bound.

All exact-search outputs must match dynamic programming.

## Statistical reporting

The report stores raw scenario aggregates and deterministic bootstrap intervals. Node reduction is paired by instance against density guidance. The configured seed and draw count are retained in metadata.

## Negative-result policy

The report does not suppress scenarios in which:

- trace rollout drifts;
- the direct policy generalizes better;
- neural guidance expands more nodes;
- inference overhead outweighs search savings;
- value scaling or capacity shift degrades performance.
