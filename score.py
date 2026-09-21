#!/usr/bin/env python3
r"""Grades sampled programs and reports how many problems each method solves.

    python score.py --samples samples.jsonl --problems problems.jsonl \\
        --out scored.jsonl

Each scored row is the sample plus ``correct``, ``tests_passed``,
``tests_run`` and ``status``. Byte-identical programs for the same problem are
judged once and the verdict is shared, which matters because a collapsed
sampler produces a lot of duplicates.

Which judge grades a sample is the problem's own choice: every row
declares it in ``ground_truth``. Problems that name an external judge
(OJBench) are graded by it when ``--ojbench-problem-dirs`` points at its test
data, and are written out unscored otherwise, rather than silently counted as
wrong.

WARNING: scoring runs model-written code. Use a container or a VM you are
willing to lose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import code_eval


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yields the records of a JSON Lines file, skipping blank lines."""
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sample_key(row: dict) -> tuple[str, str, int]:
    """Returns the identity of a sample, for resuming a scoring run."""
    return (row["problem_id"], row["method"], row["idx"])


def code_digest(code: str) -> str:
    """Returns a stable digest of a program, for sharing verdicts."""
    return hashlib.sha256((code or "").encode("utf-8", "ignore")).hexdigest()


def verdict(results: Sequence[int], details: Sequence[dict]) -> dict:
    """Collapses per-test codes into the fields written to the scored file.

    Args:
        results: Per-test codes from ``code_eval.run_tests``.
        details: The matching per-test details.

    Returns:
        The scored fields to merge into a sample record.
    """
    passed = sum(1 for result in results if result == code_eval.PASS)
    if code_eval.passes_all(results):
        status = "pass"
    elif details:
        status = details[-1].get("status", "error")
    else:
        status = "error"
    return {
        "correct": code_eval.passes_all(results),
        "tests_passed": passed,
        "tests_run": len(results),
        "status": status,
    }


def score_locally(
    samples: Sequence[dict],
    problems: dict[str, dict],
    *,
    timeout: float,
    memory_mb: int,
    max_tests: int | None,
    stop_on_failure: bool,
    workers: int,
) -> list[dict]:
    """Grades samples with the built-in executor.

    Args:
        samples: The samples to grade.
        problems: Problems by id, for their ground truth.
        timeout: Seconds allowed per test.
        memory_mb: Address-space cap for each candidate process.
        max_tests: Grade only the first N tests of each problem.
        stop_on_failure: Stop a problem at its first failing test.
        workers: How many candidates to run at once.

    Returns:
        The samples with their scored fields merged in.
    """
    cache: dict[tuple[str, str], dict] = {}
    lock = threading.Lock()
    counts = {"done": 0}

    def score_one(sample: dict) -> dict:
        key = (sample["problem_id"], code_digest(sample.get("code") or ""))
        with lock:
            cached = cache.get(key)
        if cached is None:
            results, details = code_eval.run_tests(
                sample.get("code") or "",
                problems[sample["problem_id"]]["ground_truth"],
                timeout=timeout,
                memory_mb=memory_mb,
                max_tests=max_tests,
                stop_on_failure=stop_on_failure,
            )
            cached = verdict(results, details)
            with lock:
                cache[key] = cached
        with lock:
            counts["done"] += 1
            if counts["done"] % 100 == 0:
                print(f"  {counts['done']}/{len(samples)}", file=sys.stderr)
        return {**sample, **cached}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(score_one, samples))


def score_with_ojbench(
    samples: Sequence[dict],
    problems: dict[str, dict],
    *,
    problem_dirs: Sequence[Path],
    workers: int,
) -> list[dict]:
    """Grades samples with OJBench's own DMOJ judge.

    OJBench problems come with no inline tests, and its verdicts are only
    comparable to published numbers when its judge produces them.

    Args:
        samples: The samples to grade.
        problems: Problems by id, for their judge ids.
        problem_dirs: OJBench test-data directories, one per dataset.
        workers: Judge worker processes.

    Returns:
        The samples with their scored fields merged in.

    Raises:
        SystemExit: If OJBench is not installed on this machine.
    """
    try:
        import ojbench
    except ImportError as exc:
        raise SystemExit(
            "OJBench problems need the OJBench package "
            "(https://github.com/He-Ren/OJBench) and its test data on this "
            f"machine: {exc}"
        ) from exc
    ojbench.init(problem_dirs=[str(path) for path in problem_dirs])

    # One entry per unique (problem, code). DMOJ is deterministic, so exact
    # duplicates can share a verdict.
    entries: list[dict] = []
    members: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        ground_truth = problems[sample["problem_id"]]["ground_truth"]
        code = (sample.get("code") or "").rstrip()
        key = f"{sample['problem_id']}:{code_digest(code)}"
        if key not in members:
            entries.append(
                {
                    "key": key,
                    "id": ground_truth["problem_id"],
                    "language": "python",
                    "content": f"```python\n{code}\n```",
                }
            )
        members[key].append(index)

    judged = ojbench.judge_jsonl_data(entries, num_workers=workers)

    scored = [dict(sample) for sample in samples]
    for entry in judged:
        fields = {
            "correct": bool(entry.get("is_passed")),
            "tests_passed": int(entry.get("tests_passed") or 0),
            "tests_run": int(entry.get("num_tests") or 0),
            "status": "pass" if entry.get("is_passed") else "wrong",
        }
        for index in members[entry["key"]]:
            scored[index].update(fields)
    return scored


def pass_at_k(n: int, c: int, k: int) -> float:
    """Returns the unbiased pass@k of Chen et al. (2021).

    Args:
        n: Samples drawn for the problem.
        c: How many of them were correct.
        k: The budget to estimate at.

    Returns:
        The probability that k draws from the n contain a correct one.
    """
    if n - c < k:
        return 1.0
    probability = 1.0
    for i in range(k):
        probability *= (n - c - i) / (n - i)
    return 1.0 - probability


def report(scored: Sequence[dict], k: int | None = None) -> str:
    """Summarises a scored run, one row per method.

    Args:
        scored: Scored samples. Rows with no verdict are ignored.
        k: Report unbiased pass@k at this budget. None reports the share of
            problems solved at least once, which is pass@n.

    Returns:
        A printable table.
    """
    by_method: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in scored:
        if "correct" in row:
            by_method[row["method"]][row["problem_id"]].append(row)

    header = (
        f"{'method':8s} {'probs':>6s} {'samples':>8s} {'sample pass':>12s} "
        f"{'solved':>7s} {'pass@k':>8s} {'leaked':>7s}"
    )
    lines = [header]
    for method in sorted(by_method):
        problems = by_method[method]
        rows = [row for group in problems.values() for row in group]
        correct = sum(1 for row in rows if row["correct"])
        solved = sum(
            1
            for group in problems.values()
            if any(row["correct"] for row in group)
        )
        guided = [row for row in rows if row.get("approach")]
        leaked = sum(1 for row in guided if row.get("leaked"))
        if k:
            estimates = [
                pass_at_k(
                    len(group),
                    sum(1 for row in group if row["correct"]),
                    k,
                )
                for group in problems.values()
                if len(group) >= k
            ]
            pass_k = (
                f"{sum(estimates) / len(estimates):.3f}" if estimates else "-"
            )
        else:
            pass_k = f"{solved / len(problems):.3f}" if problems else "-"
        lines.append(
            f"{method:8s} {len(problems):6d} {len(rows):8d} "
            f"{correct / max(1, len(rows)):12.3f} {solved:7d} {pass_k:>8s} "
            f"{(leaked / len(guided) if guided else 0.0):7.3f}"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--timeout",
        type=float,
        default=code_eval.DEFAULT_TIMEOUT,
        help="seconds per test, or per problem for call-based problems",
    )
    parser.add_argument(
        "--memory-mb", type=int, default=code_eval.DEFAULT_MEMORY_MB
    )
    parser.add_argument(
        "--max-tests",
        type=int,
        default=0,
        help="grade only the first N tests of each problem",
    )
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="keep going past the first failure, for full per-test counts",
    )
    parser.add_argument(
        "--ojbench-problem-dirs",
        nargs="*",
        type=Path,
        default=[],
        help="OJBench test data; supplying it opts into its judge",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="report unbiased pass@k instead of the solve rate",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Grades a samples file and prints the summary table."""
    args = parse_args(argv)

    problems = {row["problem_id"]: row for row in read_jsonl(args.problems)}
    samples = list(read_jsonl(args.samples))

    missing = {row["problem_id"] for row in samples} - problems.keys()
    if missing:
        raise SystemExit(
            f"{len(missing)} sampled problem(s) are not in {args.problems}, "
            f"e.g. {sorted(missing)[:3]}"
        )

    already = (
        {sample_key(row): row for row in read_jsonl(args.out)}
        if args.out.exists()
        else {}
    )
    todo = [row for row in samples if sample_key(row) not in already]
    if already:
        print(
            f"resuming: {len(already)} samples already scored", file=sys.stderr
        )

    external_ids = {
        row["problem_id"]
        for row in todo
        if problems[row["problem_id"]]["ground_truth"].get("judge_backend")
    }
    local = [row for row in todo if row["problem_id"] not in external_ids]
    external = [row for row in todo if row["problem_id"] in external_ids]

    scored: list[dict] = list(already.values())
    if local:
        print(f"scoring {len(local)} samples locally", file=sys.stderr)
        scored += score_locally(
            local,
            problems,
            timeout=args.timeout,
            memory_mb=args.memory_mb,
            max_tests=args.max_tests or None,
            stop_on_failure=not args.all_tests,
            workers=args.workers,
        )
    if external and args.ojbench_problem_dirs:
        print(f"scoring {len(external)} samples with OJBench", file=sys.stderr)
        scored += score_with_ojbench(
            external,
            problems,
            problem_dirs=args.ojbench_problem_dirs,
            workers=args.workers,
        )
    elif external:
        print(
            f"leaving {len(external)} samples unscored: they need OJBench's "
            "own judge, so pass --ojbench-problem-dirs",
            file=sys.stderr,
        )
        scored += external

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for row in scored:
            handle.write(json.dumps(row) + "\n")

    print(report(scored, args.k or None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
