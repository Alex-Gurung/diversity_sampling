#!/usr/bin/env python3
"""Draws samples for code problems with IID, VS or GrOoT.

    python sample.py --problems problems.jsonl --method groot --out out.jsonl

IID draws n independent solutions. VS and GrOoT each make one extra planning
call that returns n approaches, then solve each approach in its own call, so a
budget of n costs n + 1 calls. The two differ only in how the planner is asked
to represent the space of approaches: VS enumerates a flat list with
probability estimates, GrOoT builds a decision tree and takes n root-to-leaf
paths through it.

Generation goes through any OpenAI-compatible endpoint -- vLLM, SGLang, or the
OpenAI API itself:

    export DIVERSITY_API_BASE=http://127.0.0.1:8000/v1
    export DIVERSITY_MODEL=Qwen/Qwen3-4B-Instruct-2507

Runs are resumable: re-running with the same --out skips the (problem, method)
pairs already in it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import prompts

# Defaults are the paper's code-sampling settings: budget 4, planner T=0.45,
# solver and IID T=0.85, top-p 0.95, 8192 max tokens throughout.
DEFAULT_N = 4
PLANNER_TEMPERATURE = 0.45
SOLVER_TEMPERATURE = 0.85
TOP_P = 0.95
MAX_TOKENS = 8192
PLANNER_RETRIES = 3

METHODS = ("iid", "vs", "groot")


@dataclass
class Client:
    """A thin wrapper over an OpenAI-compatible chat endpoint.

    Attributes:
        model: The model name the endpoint serves.
        top_p: Nucleus sampling cutoff, shared by both stages.
        max_tokens: Generation cap, shared by both stages.
    """

    model: str
    top_p: float = TOP_P
    max_tokens: int = MAX_TOKENS
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Builds the endpoint client from the environment."""
        from openai import OpenAI

        self._client = OpenAI(
            base_url=os.environ.get(
                "DIVERSITY_API_BASE", "http://127.0.0.1:8000/v1"
            ),
            api_key=os.environ.get("DIVERSITY_API_KEY", "EMPTY"),
            timeout=float(os.environ.get("DIVERSITY_TIMEOUT", "900")),
            max_retries=4,
        )

    def chat(self, prompt: str, temperature: float) -> tuple[str, str]:
        """Sends one user message and returns the reply.

        Args:
            prompt: The user message.
            temperature: Sampling temperature for this call.

        Returns:
            A (text, finish_reason) pair.
        """
        choice = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        ).choices[0]
        return choice.message.content or "", choice.finish_reason or ""


@dataclass
class Plan:
    """The result of one planning call.

    Attributes:
        approaches: The approaches that parsed out of the response.
        raw: The planner's full response, kept on every record.
        attempts: How many planning calls it took.
    """

    approaches: list[prompts.Approach]
    raw: str
    attempts: int


def plan(
    client: Client,
    problem: str,
    method: str,
    n: int,
    retries: int = PLANNER_RETRIES,
) -> Plan:
    """Runs the planning stage until it yields n approaches.

    A planner occasionally returns fewer than n: it runs out of tokens, or
    writes prose around the tags. Retrying is cheap relative to solving, and
    if no attempt reaches n we keep the fullest one rather than throwing the
    call away.

    Args:
        client: The endpoint to generate against.
        problem: The problem statement.
        method: Either "vs" or "groot".
        n: How many approaches to ask for.
        retries: Maximum planning calls before giving up.

    Returns:
        The best plan obtained, which may hold fewer than n approaches.
    """
    build = prompts.groot_planner if method == "groot" else prompts.vs_planner
    best = Plan(approaches=[], raw="", attempts=0)
    for attempt in range(1, retries + 1):
        raw, _ = client.chat(build(problem, n), PLANNER_TEMPERATURE)
        approaches = prompts.parse_approaches(raw, n)
        if len(approaches) > len(best.approaches):
            best = Plan(approaches=approaches, raw=raw, attempts=attempt)
        if len(approaches) >= n:
            return best
    best.attempts = retries
    return best


def sample_problem(
    client: Client,
    problem: dict,
    method: str,
    n: int,
    retries: int = PLANNER_RETRIES,
) -> list[dict]:
    """Draws all n samples for one problem under one method.

    Args:
        client: The endpoint to generate against.
        problem: One row from a problems file.
        method: One of ``METHODS``.
        n: The sampling budget.
        retries: Maximum planning calls before giving up.

    Returns:
        One record per sample, ready to serialise.
    """
    statement = problem.get("prompt_full") or problem["prompt"]
    if method == "iid":
        drawn = Plan(approaches=[], raw="", attempts=0)
        approaches: Sequence[prompts.Approach | None] = [None] * n
    else:
        drawn = plan(client, statement, method, n, retries)
        # No approach survived parsing: fall back to a single unguided solve so
        # the problem is still represented, and leave approach=None so the
        # shortfall stays visible downstream.
        approaches = list(drawn.approaches) or [None]

    planner_record = _planner_record(method, drawn)
    records = []
    for idx, approach in enumerate(approaches):
        if approach is None:
            prompt = prompts.iid_prompt(statement)
        else:
            prompt = prompts.solver(statement, approach.text)
        output, finish_reason = client.chat(prompt, SOLVER_TEMPERATURE)
        records.append(
            {
                "problem_id": problem["problem_id"],
                "suite": problem.get("suite"),
                "method": method,
                "idx": idx,
                "n": n,
                "model": client.model,
                "approach": approach.text if approach else None,
                "probability": approach.probability if approach else None,
                "path": approach.path if approach else None,
                "output": output,
                "code": prompts.extract_code(output),
                "finish_reason": finish_reason,
                # Responses that name the approach they were given are
                # discarded when building training data. Flagged rather than
                # dropped here, so the rate stays measurable.
                "leaked": bool(approach and prompts.mentions_approach(output)),
                "planner": planner_record,
            }
        )
    return records


def _planner_record(method: str, drawn: Plan) -> dict | None:
    """Summarises a planning call for the per-sample records."""
    if method == "iid":
        return None
    return {
        "attempts": drawn.attempts,
        "approaches_parsed": len(drawn.approaches),
        "tree": prompts.extract_tree(drawn.raw) if method == "groot" else None,
        "raw": drawn.raw,
    }


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yields the records of a JSON Lines file, skipping blank lines."""
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def done_pairs(path: Path) -> set[tuple[str, str]]:
    """Returns the (problem_id, method) pairs already in an output file."""
    if not path.exists():
        return set()
    return {(row["problem_id"], row["method"]) for row in read_jsonl(path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--problems",
        required=True,
        type=Path,
        help="problems jsonl from load.py",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="samples jsonl, appended to and resumable",
    )
    parser.add_argument(
        "--method",
        default="groot",
        help=f"one or more of {', '.join(METHODS)}, comma-separated",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help="sampling budget per problem",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DIVERSITY_MODEL", ""),
        help="model name the endpoint serves, or $DIVERSITY_MODEL",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="problems in flight at once",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stop after this many problems",
    )
    parser.add_argument(
        "--planner-retries",
        type=int,
        default=PLANNER_RETRIES,
        help="maximum planning calls per problem",
    )
    args = parser.parse_args(argv)

    args.methods = [
        part.strip() for part in args.method.split(",") if part.strip()
    ]
    unknown = [name for name in args.methods if name not in METHODS]
    if unknown:
        parser.error(
            f"unknown method(s): {', '.join(unknown)}; "
            f"choose from {', '.join(METHODS)}"
        )
    if not args.model:
        parser.error("--model is required, or set $DIVERSITY_MODEL")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Samples every problem under every requested method."""
    args = parse_args(argv)

    problems = list(read_jsonl(args.problems))
    if args.limit:
        problems = problems[: args.limit]

    already = done_pairs(args.out)
    units = [
        (problem, method)
        for problem in problems
        for method in args.methods
        if (problem["problem_id"], method) not in already
    ]
    if already:
        print(
            f"resuming: {len(already)} (problem, method) pairs already done",
            file=sys.stderr,
        )
    print(
        f"{len(problems)} problems x {len(args.methods)} method(s) "
        f"-> {len(units)} to run",
        file=sys.stderr,
    )

    client = Client(model=args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    counts = {"done": 0, "failed": 0}

    with open(args.out, "a") as handle:

        def run(unit: tuple[dict, str]) -> None:
            problem, method = unit
            try:
                records = sample_problem(
                    client, problem, method, args.n, args.planner_retries
                )
            except Exception as exc:  # noqa: BLE001
                # Isolation point: one unreachable endpoint or malformed
                # problem should not take a multi-hour run down with it.
                with lock:
                    counts["failed"] += 1
                print(
                    f"  WARN {problem['problem_id']}/{method}: {exc}",
                    file=sys.stderr,
                )
                return
            with lock:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
                handle.flush()
                counts["done"] += 1
                if counts["done"] % 10 == 0:
                    print(f"  {counts['done']}/{len(units)}", file=sys.stderr)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run, units))

    print(
        f"done: {counts['done']} ok, {counts['failed']} failed -> {args.out}",
        file=sys.stderr,
    )
    return 1 if counts["failed"] and not counts["done"] else 0


if __name__ == "__main__":
    sys.exit(main())
