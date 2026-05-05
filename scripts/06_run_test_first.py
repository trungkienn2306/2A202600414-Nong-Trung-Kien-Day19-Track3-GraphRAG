"""Run lab test-first flow with realtime streaming logs.

This orchestrator executes each step sequentially, streams output in realtime,
and stops immediately on any crash/non-zero exit code.

Usage:
    python scripts/06_run_test_first.py --skip-fetch
    python scripts/06_run_test_first.py --seeds 2,3,5 --run-prefix nightly
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def run_step(name: str, command: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"[STEP] {name}")
    print(f"[CMD ] {' '.join(command)}")
    print("=" * 80)
    t0 = time.time()

    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n[FAILED] {name} (exit={proc.returncode}, elapsed={elapsed:.1f}s)")
        sys.exit(proc.returncode)

    print(f"\n[DONE] {name} (elapsed={elapsed:.1f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GraphRAG lab test-first flow with realtime logs"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip scripts/01_fetch.py if corpus is already present",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="2,3,5",
        help="Comma-separated GraphRAG seed values for test benchmark runs (default: 2,3,5)",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="seeds",
        help="Prefix for benchmark run names (default: seeds)",
    )
    parser.add_argument(
        "--smoke-question",
        type=str,
        default="Who founded Anthropic?",
        help="Question for scripts/04_test_query.py smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PYTHON.exists():
        print(f"[ERROR] Python env not found at: {PYTHON}")
        print("Create/install venv first, then retry.")
        sys.exit(1)

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("[ERROR] --seeds is empty")
        sys.exit(1)

    if not args.skip_fetch:
        run_step("Fetch corpus", [str(PYTHON), "scripts/01_fetch.py", "--skip-if-exists"])
    else:
        print("\n[SKIP] Fetch step skipped by --skip-fetch")

    run_step("Extract triples", [str(PYTHON), "scripts/02_extract.py"])
    run_step("Build Neo4j graph", [str(PYTHON), "scripts/03_build_graph.py"])
    run_step(
        "Smoke test query",
        [str(PYTHON), "scripts/04_test_query.py", "-q", args.smoke_question],
    )

    for seed in seeds:
        run_name = f"{args.run_prefix}{seed}_test"
        run_step(
            f"Benchmark test set (seeds={seed})",
            [
                str(PYTHON),
                "scripts/05_benchmark.py",
                "--question-set",
                "test",
                "--seeds",
                seed,
                "--run-name",
                run_name,
            ],
        )

    print("\n" + "=" * 80)
    print("[ALL DONE] Test-first flow completed successfully.")
    print("Check artifacts under: report/benchmark_runs/")
    print("=" * 80)


if __name__ == "__main__":
    main()
