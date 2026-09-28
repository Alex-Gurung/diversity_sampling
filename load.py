"""Converts LiveCodeBench, Cobalt or OJBench into the problem format that sample.py reads.

    python load.py --suite cobalt --path cobalt_train.jsonl --out problems.jsonl

Each problem has a problem_id, the prompt the model sees (prompt_full), and a ground_truth. For
LiveCodeBench and Cobalt the ground truth holds the tests:

    {"eval_type": "stdio" or "call", "fn_name": ...,
     "input_output": {"inputs": [...], "outputs": [...]}}

For OJBench it names the problem for OJBench's judge.
"""

import argparse
import base64
import json
import pickle
import zlib
from pathlib import Path

LCB_PROMPT = (
    "You will be given a question (problem specification) and will generate a correct "
    "Python program that matches the specification and passes all tests.\n\nQuestion:\n"
)
STARTER_INSTRUCTION = (
    "You will use the following starter code to write the solution and enclose "
    "your code within delimiters.\n```python\n{}\n```"
)
STDIO_INSTRUCTION = (
    "Read input from stdin and write output to stdout. Return your final program "
    "inside ```python``` fences.\n\n```python\n# YOUR CODE HERE\n```"
)


def lcb(row: dict) -> dict:
    starter = (row["starter_code"] or "").strip()
    instruction = STARTER_INSTRUCTION.format(starter) if starter else STDIO_INSTRUCTION
    private = row["private_test_cases"]
    if not private.startswith("["):  # a base64 zlib pickle of the JSON; only load files you trust
        private = pickle.loads(zlib.decompress(base64.b64decode(private)))
    tests = json.loads(row["public_test_cases"]) + json.loads(private)
    return {
        "problem_id": f"lcb:{row['question_id']}",
        "prompt_full": f"{LCB_PROMPT}{row['question_content']}\n\n{instruction}".rstrip(),
        "ground_truth": {
            "eval_type": "call" if tests[0]["testtype"] == "functional" else "stdio",
            "fn_name": json.loads(row["metadata"]).get("func_name"),
            "input_output": {
                "inputs": [test["input"] for test in tests],
                "outputs": [test["output"] for test in tests],
            },
        },
    }


def cobalt(row: dict) -> dict:
    prompt_full = row["prompt"]
    if row.get("instruction"):
        prompt_full = f"{prompt_full}\n\n{row['instruction']}".rstrip()
    return {
        "problem_id": row["problem_id"],
        "prompt_full": prompt_full,
        "ground_truth": row["ground_truth"],
    }


def ojbench(row: dict) -> dict:
    dataset = row["dataset"].upper()
    return {
        "problem_id": f"ojbench_{row['difficulty']}:{row['id']}",
        "prompt_full": row["prompt"],
        # OJBench names NOI problem directories loj-<id> and ICPC ones by the bare id.
        "ground_truth": {
            "judge_backend": "ojbench",
            "dataset": dataset,
            "problem_id": f"loj-{row['id']}" if dataset == "NOI" else str(row["id"]),
        },
    }


LOADERS = {"lcb": lcb, "cobalt": cobalt, "ojbench": ojbench}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", required=True, choices=LOADERS)
    parser.add_argument("--path", required=True, type=Path, help="the suite's jsonl file")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.path.read_text().splitlines() if line.strip()]
    if args.suite == "ojbench":
        rows = [row for row in rows if row["language"].lower() == "python"]
    problems = [LOADERS[args.suite](row) for row in rows]
    args.out.write_text("".join(json.dumps(problem) + "\n" for problem in problems))
    print(f"{len(problems)} problems -> {args.out}")


if __name__ == "__main__":
    main()
