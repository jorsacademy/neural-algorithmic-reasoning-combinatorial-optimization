# Research Context

Neural algorithmic reasoning trains neural processors to imitate the state transitions of classical algorithms rather than learning only input-output mappings. The CLRS benchmark established a broad, standardized collection of such tasks and emphasized out-of-distribution generalization to larger inputs.

Combinatorial optimization creates a distinct evaluation pressure. A neural model may output a high-quality solution without learning the relevant algorithm, and a model that imitates an algorithm may still be less useful than a classical heuristic. This repository therefore evaluates algorithmic fidelity, standalone solution quality, and exact-search utility separately.

## Closest lines of work

- **CLRS algorithmic reasoning:** broad benchmark and intermediate hints across classical algorithms.
- **Algorithmic pretraining for CO:** pretraining on graph algorithms before learning NP-hard tasks.
- **KNARsack:** direct recent evidence that knapsack DP traces can improve extrapolation over direct prediction.
- **Learning to branch:** learned policies can improve exact branch-and-bound traversal while solver bounds preserve correctness.
- **Primal-dual NAR:** algorithmic supervision can incorporate approximation-algorithm structure for harder problems.
- **Expressivity analyses:** bounded neural networks can represent exact or approximate knapsack dynamic programs under explicit size assumptions.

## Scope of this repository

The implementation does not attempt to reproduce the architectures or headline results of those works. It provides a small auditable test bed for a narrower question: whether a learned Bellman processor supplies better **non-binding search advice** than a terminal-only policy under controlled extrapolation.

## References

1. Veličković et al. “The CLRS Algorithmic Reasoning Benchmark.” ICML 2022. <https://proceedings.mlr.press/v162/velickovic22a.html>
2. Georgiev et al. “Neural Algorithmic Reasoning for Combinatorial Optimisation.” Learning on Graphs proceedings, 2024. <https://proceedings.mlr.press/v231/georgiev24a.html>
3. Požgaj et al. “KNARsack: Teaching Neural Algorithmic Reasoners to Solve Pseudo-Polynomial Problems.” 2025. <https://arxiv.org/abs/2509.15239>
4. Gasse et al. “Exact Combinatorial Optimization with Graph Convolutional Neural Networks.” NeurIPS 2019. <https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html>
5. He and Vitercik. “Primal-Dual Neural Algorithmic Reasoning.” ICML 2025. <https://proceedings.mlr.press/v267/he25r.html>
6. Hertrich and Skutella. “Provably Good Solutions to the Knapsack Problem via Neural Networks of Bounded Size.” 2020. <https://arxiv.org/abs/2005.14105>
