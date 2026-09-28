import argparse
import asyncio
import json
import threading
import time

import prompts
import sample
import score


def test_sampling_runs_requests_together_and_resumes(tmp_path, monkeypatch):
    in_flight, peak, lock = [0], [0], threading.Lock()

    def fake_chat(url, model, message, temperature):
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.05)
        with lock:
            in_flight[0] -= 1
        if temperature == sample.PLANNER_TEMPERATURE:
            plan = (
                "<approach>Probability: 0.5\nSort.</approach>"
                "<approach>Probability: 0.5\nUse the hint.</approach>"
            )
            return plan, "stop"
        return (
            "Following the hint: print(1)" if "Use the hint." in message else "print(1)"
        ), "stop"

    monkeypatch.setattr(sample, "chat", fake_chat)
    problems = tmp_path / "problems.jsonl"
    problems.write_text(
        "".join(
            json.dumps({"problem_id": f"p{i}", "prompt_full": f"Problem {i}"}) + "\n"
            for i in range(3)
        )
    )
    args = argparse.Namespace(
        problems=problems,
        out=tmp_path / "samples.jsonl",
        method=["iid", "vs"],
        n=2,
        model="m",
        url="u",
        workers=8,
    )
    asyncio.run(sample.run(args))
    rows = [json.loads(line) for line in args.out.read_text().splitlines()]
    assert sum(row["method"] == "iid" for row in rows) == 6
    assert [row["approach"] for row in rows if row["method"] == "vs"] == ["Sort."] * 3
    assert peak[0] == args.workers

    asyncio.run(sample.run(args))
    assert len(args.out.read_text().splitlines()) == len(rows)


def test_prompts_leave_braces_in_the_problem_alone():
    problem = r"Compute \sum_{i=1}^{N} a_i."
    for method in ("vs", "groot"):
        prompt = prompts.planner_prompt(method, problem, 4)
        assert problem in prompt
        assert prompt.endswith("\n")
    assert "four entirely different subtrees" in prompts.planner_prompt("groot", problem, 4)
    assert prompts.solver_prompt(problem, "Use a prefix sum.").count(problem) == 1


def test_parse_approaches():
    vs = (
        "<approach>\nProbability: 0.6\nSort, then scan.\n</approach>\n"
        "<approach>Probability: 0.4\nUse a heap.</approach>"
    )
    assert [(a.text, a.probability) for a in prompts.parse_approaches(vs)] == [
        ("Sort, then scan.", 0.6),
        ("Use a heap.", 0.4),
    ]
    groot = (
        "<tree>A. Greedy</tree>\nPath: A -> A1\n"
        "<approach>\nGreedy by deadline.\n</approach>\n<approach>unclosed"
    )
    assert [a.text for a in prompts.parse_approaches(groot)] == ["Greedy by deadline."]


def test_extract_code_takes_the_last_closed_block():
    response = "```python\nprint(1)\n```\nWait, that is wrong.\n```python\nprint(2)\n```"
    assert prompts.extract_code(response) == "print(2)"
    assert prompts.extract_code("Here:\n```python\nprint(3)") == "print(3)"


def test_mentions_approach():
    assert prompts.mentions_approach("Following the hint, we sort.")
    assert not prompts.mentions_approach("We sort the array first.")


def test_same_output_compares_numbers_by_value():
    assert score.same_output("1.0 2\n", "1 2.00")
    assert not score.same_output("1 2", "1 3")
    assert not score.same_output("1\n2", "1")


def test_stdio_program():
    tests = {
        "eval_type": "stdio",
        "input_output": {"inputs": ["2 3\n", "5 7\n"], "outputs": ["5\n", "12\n"]},
    }
    assert score.passes("a, b = map(int, input().split())\nprint(a + b)", tests)
    assert not score.passes("a, b = map(int, input().split())\nprint(a * b)", tests)
    assert not score.passes("", tests)


def test_call_program_may_use_typing_names_without_importing_them():
    tests = {
        "eval_type": "call",
        "fn_name": "total",
        "input_output": {"inputs": ["[1, 2, 3]"], "outputs": ["6"]},
    }
    code = "class Solution:\n    def total(self, nums: List[int]) -> int:\n        return sum(nums)"
    assert score.passes(code, tests)


def test_timeout_fails(monkeypatch):
    monkeypatch.setattr(score, "TIMEOUT_SECONDS", 1)
    tests = {"eval_type": "stdio", "input_output": {"inputs": [""], "outputs": [""]}}
    assert not score.passes("while True:\n    pass", tests)
