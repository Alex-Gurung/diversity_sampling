#!/usr/bin/env python3
"""Turns a code benchmark into the one problem format this repo uses.

    python load.py --suite lcb --path lcb_test6.jsonl --out problems.jsonl

Every row that comes out looks the same, whatever the suite:

    {"problem_id", "suite", "language", "difficulty",
     "prompt",        the problem statement
     "instruction",   the output-format instruction, "" when the prompt has one
     "prompt_full",   prompt + instruction, which is what the model is shown
     "ground_truth"}  either inline tests, or a pointer to an external judge

``ground_truth`` takes one of two shapes:

    {"eval_type": "stdio"|"call", "fn_name": ...,
     "input_output": {"inputs": [...], "outputs": [...]}}
    {"judge_backend": "ojbench", "dataset": ..., "problem_id": ...}

``score.py`` grades the first shape directly. The second needs OJBench's own
DMOJ judge; see the README. Source files are not vendored here -- they run to
hundreds of megabytes -- so pass --path, or set $DIVERSITY_DATA_DIR and let the
defaults find them.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import sys
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path

SUITES = ("lcb", "cobalt", "ojbench")

DEFAULT_FILES = {
    "lcb": "lcb_test6.jsonl",
    "cobalt": "cobalt_test.jsonl",
    "ojbench": "ojbench_prompts_full.jsonl",
}

STDIO_INSTRUCTION = (
    "Read input from stdin and write output to stdout. Return your final "
    "program inside ```python``` fences.\n\n```python\n# YOUR CODE HERE\n```"
)

LCB_PREAMBLE = (
    "You will be given a question (problem specification) and will generate a "
    "correct Python program that matches the specification and passes all "
    "tests.\n\nQuestion:\n"
)


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yields the records of a JSON Lines file, skipping blank lines."""
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _full(prompt: str, instruction: str) -> str:
    """Joins a statement and its output-format instruction."""
    if not instruction:
        return prompt
    return f"{prompt}\n\n{instruction}".rstrip()


def _decode_private_tests(raw: str | list) -> list:
    """Decodes LiveCodeBench's private test cases.

    Newer releases store them as plain JSON; older ones base64-encode a
    zlib-compressed pickle, so only load a file you trust.

    Args:
        raw: The ``private_test_cases`` field of a LiveCodeBench row.

    Returns:
        The decoded test cases.
    """
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return pickle.loads(zlib.decompress(base64.b64decode(raw.encode())))


def load_lcb(path: Path) -> list[dict]:
    """Loads LiveCodeBench code_generation_lite.

    Public and private tests are concatenated, because the benchmark grades on
    all of them and a solution has to pass every one.

    Args:
        path: The suite's jsonl file.

    Returns:
        Problems in this repo's format.
    """
    problems = []
    for row in read_jsonl(path):
        starter = (row.get("starter_code") or "").strip()
        prompt = LCB_PREAMBLE + row["question_content"]
        if starter:
            instruction = (
                "You will use the following starter code to write the "
                "solution and enclose your code within delimiters.\n"
                f"```python\n{starter}\n```"
            )
        else:
            instruction = STDIO_INSTRUCTION

        tests = json.loads(row["public_test_cases"])
        tests += _decode_private_tests(row["private_test_cases"])
        metadata = json.loads(row.get("metadata") or "{}")
        functional = bool(tests) and tests[0]["testtype"] == "functional"

        problems.append(
            {
                "problem_id": f"lcb:{row.get('question_id')}",
                "suite": "lcb",
                "language": "python",
                "difficulty": row.get("difficulty"),
                "prompt": prompt,
                "instruction": instruction,
                "prompt_full": _full(prompt, instruction),
                "ground_truth": {
                    "eval_type": "call" if functional else "stdio",
                    "fn_name": metadata.get("func_name"),
                    "input_output": {
                        "inputs": [test["input"] for test in tests],
                        "outputs": [test["output"] for test in tests],
                    },
                },
            }
        )
    return problems


def load_cobalt(path: Path) -> list[dict]:
    """Loads Cobalt, a filtered split of osunlp/TACO-Cobalt.

    Args:
        path: The suite's jsonl file.

    Returns:
        Problems in this repo's format, all stdio with inline tests.
    """
    problems = []
    for row in read_jsonl(path):
        ground_truth = row.get("ground_truth", {})
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        prompt = str(row["prompt"])
        instruction = str(row.get("instruction") or "")
        problems.append(
            {
                "problem_id": str(row["problem_id"]),
                "suite": "cobalt",
                "language": "python",
                "difficulty": row.get("difficulty"),
                "prompt": prompt,
                "instruction": instruction,
                "prompt_full": str(
                    row.get("prompt_full") or _full(prompt, instruction)
                ),
                "ground_truth": ground_truth,
            }
        )
    return problems


def normalize_ojbench_dataset(value: str) -> str:
    """Returns an OJBench dataset name spelled the way its judge expects."""
    lowered = str(value).strip().lower()
    return {"noi": "NOI", "icpc": "ICPC"}.get(lowered, str(value).strip())


def load_ojbench(path: Path) -> list[dict]:
    """Loads OJBench, Python problems only.

    Correctness comes from OJBench's own judge rather than from inline tests,
    so the ground truth is a pointer to it.

    Args:
        path: The suite's prompts jsonl file.

    Returns:
        Problems in this repo's format.
    """
    problems = []
    for row in read_jsonl(path):
        if str(row.get("language", "")).lower() != "python":
            continue
        difficulty = str(row.get("difficulty"))
        source_id = str(row.get("id"))
        dataset = normalize_ojbench_dataset(row.get("dataset", ""))
        # The OJBench prompt is already complete, instruction included.
        prompt = str(row["prompt"])
        problems.append(
            {
                "problem_id": f"ojbench_{difficulty}:{source_id}",
                "suite": f"ojbench_{difficulty}",
                "language": "python",
                "difficulty": difficulty,
                "prompt": prompt,
                "instruction": "",
                "prompt_full": prompt,
                "ground_truth": {
                    "judge_backend": "ojbench",
                    "dataset": dataset,
                    # NOI problem directories are named loj-<id>; ICPC uses
                    # the bare id.
                    "problem_id": (
                        f"loj-{source_id}" if dataset == "NOI" else source_id
                    ),
                },
            }
        )
    return problems


LOADERS = {
    "lcb": load_lcb,
    "cobalt": load_cobalt,
    "ojbench": load_ojbench,
}


def resolve_path(suite: str, path: Path | None) -> Path:
    """Returns the source file for a suite.

    Args:
        suite: One of ``SUITES``.
        path: An explicit path, which wins when given.

    Returns:
        The file to read.

    Raises:
        SystemExit: If neither --path nor $DIVERSITY_DATA_DIR is set.
    """
    if path:
        return path
    data_dir = os.environ.get("DIVERSITY_DATA_DIR")
    if not data_dir:
        raise SystemExit(
            f"--path is required for --suite {suite}, "
            "or set $DIVERSITY_DATA_DIR"
        )
    return Path(data_dir) / DEFAULT_FILES[suite]


def main(argv: Sequence[str] | None = None) -> int:
    """Converts one suite into a problems file."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="source jsonl; defaults to $DIVERSITY_DATA_DIR/<suite file>",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--limit", type=int, default=0, help="keep only the first N problems"
    )
    args = parser.parse_args(argv)

    source = resolve_path(args.suite, args.path)
    if not source.exists():
        raise SystemExit(f"no such file: {source}")

    problems = LOADERS[args.suite](source)
    if args.limit:
        problems = problems[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for problem in problems:
            handle.write(json.dumps(problem) + "\n")
    print(f"{len(problems)} problems -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
