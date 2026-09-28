"""Samples solutions to code problems with IID, VS or Groot.

IID solves each problem n times. VS and Groot first ask the model for n approaches (VS as a list
with probabilities, Groot as paths through a decision tree), then solve once per approach.
Solutions that mention the approach they were given are dropped.

Start a vLLM server, then sample from it:

    vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768
    python sample.py --problems problems.jsonl --method iid vs groot --out samples.jsonl

All problems and methods run at once, with up to --workers requests in flight, and a problem's
solves start as soon as its plan arrives. Each problem's samples are appended to --out when they
finish, and re-running with the same --out skips the problems already there.
"""

import argparse
import asyncio
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import prompts

PLANNER_TEMPERATURE = 0.45
SOLVER_TEMPERATURE = 0.85


def chat(url: str, model: str, message: str, temperature: float) -> tuple[str, str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 8192,
    }
    request = urllib.request.Request(
        f"{url}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        choice = json.load(response)["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]


async def sample_problem(
    args: argparse.Namespace, limit: asyncio.Semaphore, problem: dict, method: str
) -> tuple[list[dict], int]:
    """Returns the problem's samples and how many were dropped for mentioning their approach."""

    async def ask(message: str, temperature: float) -> tuple[str, str]:
        async with limit:
            return await asyncio.to_thread(chat, args.url, args.model, message, temperature)

    statement = problem["prompt_full"]
    plan, approaches = None, [None] * args.n
    if method != "iid":
        plan, _ = await ask(prompts.planner_prompt(method, statement, args.n), PLANNER_TEMPERATURE)
        approaches = prompts.parse_approaches(plan)[: args.n]
    solutions = await asyncio.gather(
        *(
            ask(
                statement if a is None else prompts.solver_prompt(statement, a.text),
                SOLVER_TEMPERATURE,
            )
            for a in approaches
        )
    )
    records = [
        {
            "problem_id": problem["problem_id"],
            "method": method,
            "approach": approach.text if approach else None,
            "probability": approach.probability if approach else None,
            "plan": plan,
            "output": output,
            "finish_reason": finish_reason,
        }
        for approach, (output, finish_reason) in zip(approaches, solutions, strict=True)
        if not (approach and prompts.mentions_approach(output))
    ]
    return records, len(approaches) - len(records)


async def run(args: argparse.Namespace) -> None:
    problems = []
    with args.problems.open() as lines:
        for line in lines:
            row = json.loads(line)
            problems.append({"problem_id": row["problem_id"], "prompt_full": row["prompt_full"]})
    done = set()
    if args.out.exists():
        with args.out.open() as lines:
            done = {(row["problem_id"], row["method"]) for row in map(json.loads, lines)}
    units = [(p, m) for m in args.method for p in problems if (p["problem_id"], m) not in done]
    print(f"{len(units)} problem-method pairs to sample, {len(done)} already in {args.out}")

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(args.workers))
    limit = asyncio.Semaphore(args.workers)
    tasks = [asyncio.create_task(sample_problem(args, limit, p, m)) for p, m in units]
    written = dropped = 0
    with args.out.open("a") as out:
        for finished in asyncio.as_completed(tasks):
            records, leaked = await finished
            out.write("".join(json.dumps(record) + "\n" for record in records))
            out.flush()
            written, dropped = written + len(records), dropped + leaked
    print(f"wrote {written} samples to {args.out}, dropped {dropped} that mentioned their approach")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--method", nargs="+", choices=["iid", "vs", "groot"], default=["groot"])
    parser.add_argument("--n", type=int, default=4, help="samples per problem")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="the vLLM server")
    parser.add_argument("--workers", type=int, default=1024, help="requests in flight")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
