# Test-set Benchmark Analysis (temporary)

## Runs analyzed

- `seeds2_test`
- `seeds3_test`
- `seeds5_test`

## Snapshot metrics

| Run | GraphRAG acc | FlatRAG acc | Delta | GraphRAG latency | FlatRAG latency | GraphRAG cost | FlatRAG cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seeds2_test | 20% | 40% | -20% | 2.108s | 2.130s | 0.000623 | 0.000993 |
| seeds3_test | 40% | 40% | 0% | 2.276s | 2.238s | 0.000808 | 0.001010 |
| seeds5_test | 40% | 60% | -20% | 2.932s | 2.097s | 0.001096 | 0.000985 |

Best test seed right now: `seeds=3` (highest stable GraphRAG accuracy, lower penalty than seeds=5).

## Why FlatRAG wins more right now

### 1) Test set is tiny (5 questions), so one question swings 20%
- With only 5 items, each miss/hit changes accuracy by 20%.
- FlatRAG only needs 1 extra pass to look "much better."

### 2) Question-set / corpus mismatch on key items (both systems fail)
- Q6 and Q12 are consistently wrong for both systems.
- This means benchmark is currently dominated by question difficulty/mismatch rather than retrieval architecture quality.

### 3) GraphRAG seed quality is noisy
- In failed cases, seed entities include generic or off-target nodes (example patterns: `human abilities`, `OpenAI, Inc.` duplicates).
- Noisy seeds lead BFS to retrieve subgraphs that miss the exact evidence chain needed by the ground truth.

### 4) Triple extraction quality introduces KG ambiguity
- Extraction created many broad/weak entities, which dilutes graph traversal precision.
- The graph summary from build step also showed many nodes under `source=None`, indicating provenance is weak for large portions of nodes.

### 5) LLM-judge leniency can favor verbose FlatRAG answers
- In `seeds5_test` Q20, FlatRAG got `True` even with partially hedged phrasing.
- This can inflate FlatRAG score on small test sets.

## Practical interpretation

- This result does **not** prove FlatRAG is fundamentally better.
- It shows current `test` subset is too small/noisy to separate methods reliably.
- GraphRAG is competitive at `seeds=3`, but not yet superior on this test pack.

## Immediate next actions

1. Keep `seeds=3` as candidate.
2. Run full set once:
   - `python scripts/05_benchmark.py --question-set full --seeds 3 --run-name seeds3_full`
3. In report, include failure modes with IDs:
   - FM-RetrievalMiss: Q6, Q12
   - FM-SeedNoise/BFSDrift: Q20 (GraphRAG)
   - FM-JudgeVariance: Q20 (FlatRAG pass despite partial answer)
4. If full-set still underperforms, tune:
   - question quality / ground truth strictness,
   - triple extraction schema quality,
   - seed filtering (prefer named entities over generic phrases).
