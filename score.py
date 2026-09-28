"""Grades samples against each problem's tests and prints how many problems each method solves.

    python score.py --samples samples.jsonl --problems problems.jsonl --out scored.jsonl

A sample is correct if its program passes every test. Each test runs the program in a fresh
Python process with a 10 second timeout and a 4 GB memory limit, as in the paper. This executes
model-written code, so run it in a container. OJBench problems are skipped: grade them with
OJBench's own judge (https://github.com/He-Ren/OJBench).
"""

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path

import prompts

TIMEOUT_SECONDS = 10
MEMORY_BYTES = 4 * 2**30
CALL_HARNESS = Path(__file__).parent / "call_harness.py"
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def limit_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))


def run(args: list, stdin: str, workdir: str) -> str | None:
    """Runs a Python script and returns its stdout, or None if it failed or timed out."""
    # PYTHONINTMAXSTRDIGITS=0 lets programs print integers longer than 4300 digits.
    try:
        result = subprocess.run(
            [sys.executable, *args],
            input=stdin,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            cwd=workdir,
            preexec_fn=limit_memory,
            env={**os.environ, "PYTHONINTMAXSTRDIGITS": "0"},
        )
    except subprocess.TimeoutExpired:
        return None
    return result.stdout if result.returncode == 0 else None


def numbers(line: str) -> list[Decimal] | None:
    tokens = line.split()
    if not all(NUMBER.fullmatch(token) for token in tokens):
        return None
    return [Decimal(token) for token in tokens]


def same_output(actual: str, expected: str) -> bool:
    """Compares line by line, ignoring surrounding whitespace, and compares numbers by value."""
    actual_lines = [line.strip() for line in actual.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]
    if len(actual_lines) != len(expected_lines):
        return False
    for a, e in zip(actual_lines, expected_lines, strict=True):
        if a != e and (numbers(a) is None or numbers(a) != numbers(e)):
            return False
    return True


def passes(code: str, ground_truth: dict) -> bool:
    tests = ground_truth["input_output"]
    with tempfile.TemporaryDirectory() as workdir:
        program = Path(workdir) / "program.py"
        program.write_text(code)
        for stdin, expected in zip(tests["inputs"], tests["outputs"], strict=True):
            if ground_truth["eval_type"] == "call":
                output = run([CALL_HARNESS, program, ground_truth["fn_name"]], stdin, workdir)
                correct = output is not None and (
                    json.loads(output.splitlines()[-1]) == json.loads(expected)
                )
            else:
                output = run([program], stdin, workdir)
                correct = output is not None and same_output(output, expected)
            if not correct:
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    problems = {}
    for line in args.problems.read_text().splitlines():
        problem = json.loads(line)
        problems[problem["problem_id"]] = problem
    samples = [json.loads(line) for line in args.samples.read_text().splitlines()]
    graded = [
        s for s in samples if "judge_backend" not in problems[s["problem_id"]]["ground_truth"]
    ]
    if len(graded) < len(samples):
        print(f"skipping {len(samples) - len(graded)} OJBench samples")

    codes = [prompts.extract_code(sample["output"]) for sample in graded]
    ground_truths = [problems[sample["problem_id"]]["ground_truth"] for sample in graded]
    with ProcessPoolExecutor(args.workers) as pool:
        verdicts = pool.map(passes, codes, ground_truths, chunksize=8)
        for sample, code, correct in zip(graded, codes, verdicts, strict=True):
            sample["code"] = code
            sample["correct"] = correct

    with args.out.open("w") as out:
        for sample in graded:
            out.write(json.dumps(sample) + "\n")

    by_method = defaultdict(list)
    for sample in graded:
        by_method[sample["method"]].append(sample)
    print(f"{'method':8} {'samples':>8} {'correct':>8} {'solved':>7}  of {len(problems)} problems")
    for method, rows in sorted(by_method.items()):
        solved = {row["problem_id"] for row in rows if row["correct"]}
        print(
            f"{method:8} {len(rows):8d} {sum(row['correct'] for row in rows):8d} {len(solved):7d}"
        )


if __name__ == "__main__":
    main()
