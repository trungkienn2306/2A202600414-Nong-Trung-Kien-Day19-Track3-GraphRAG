# Benchmark Report Folder

This folder contains official benchmark-report artifacts for Lab 19.

## Structure

```text
report/
├── README.md
├── benchmark_report.md
└── benchmark_runs/
    ├── .gitkeep
    └── run_YYYYMMDD_HHMMSS/
        ├── benchmark_details.csv
        ├── summary.json
        └── run_metadata.json
```

## How to run 3 top_k_seeds settings

Use the small test set first for quick iteration:

```bash
python scripts/05_benchmark.py --question-set test --seeds 2 --run-name seeds2_test
python scripts/05_benchmark.py --question-set test --seeds 3 --run-name seeds3_test
python scripts/05_benchmark.py --question-set test --seeds 5 --run-name seeds5_test
```

When you pick the best seed count, validate once on full set:

```bash
python scripts/05_benchmark.py --question-set full --seeds 3 --run-name seeds3_full
```

If you need a clean FlatRAG index rebuild:

```bash
python scripts/05_benchmark.py --question-set test --seeds 3 --force-reindex --run-name seeds3_test_rebuild
```

## How to complete the official report

1. Run the 3 seed configurations and collect each `summary.json`.
2. Fill tables in `benchmark_report.md`:
   - accuracy / latency / cost by seed setting
   - pass/fail for `GraphRAG - FlatRAG >= 20%`
3. Use `benchmark_details.csv` + `*_reason` columns to write failure modes.
4. Include final recommendation (best seed setting + tradeoffs).

## Submission checklist

- [ ] 3 test runs with `seeds=2/3/5` under `report/benchmark_runs/`
- [ ] 1 full run for selected seed
- [ ] `benchmark_report.md` filled with real numbers
- [ ] failure modes section includes concrete question IDs
- [ ] explicit conclusion for `>=20%` target
