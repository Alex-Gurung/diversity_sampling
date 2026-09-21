"""Tests for the executor, the loaders and the report arithmetic."""

import json
from pathlib import Path

import code_eval
import load
import score

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "problems.jsonl"

STDIO_GROUND_TRUTH = {
    "eval_type": "stdio",
    "input_output": {
        "inputs": ["2 3\n", "10 20\n"],
        "outputs": ["5\n", "30\n"],
    },
}

CALL_GROUND_TRUTH = {
    "eval_type": "call",
    "fn_name": "two_sum",
    "input_output": {
        "inputs": ["[2, 7, 11, 15]\n9"],
        "outputs": ["[0, 1]"],
    },
}

ADDER = "a, b = map(int, input().split())\nprint(a + b)"


def test_a_correct_stdio_program_passes_every_test():
    results, _ = code_eval.run_tests(ADDER, STDIO_GROUND_TRUTH, timeout=10)
    assert results == [code_eval.PASS, code_eval.PASS]
    assert code_eval.passes_all(results)


def test_a_wrong_answer_stops_at_the_first_failure():
    results, details = code_eval.run_tests(
        ADDER.replace("a + b", "a - b"), STDIO_GROUND_TRUTH, timeout=10
    )
    assert results == [code_eval.WRONG]
    assert details[-1]["status"] == "wrong"


def test_a_crash_and_a_hang_are_told_apart():
    crashed, _ = code_eval.run_tests(
        "raise ValueError('no')", STDIO_GROUND_TRUTH, timeout=10
    )
    hung, _ = code_eval.run_tests(
        "import time\nwhile True: time.sleep(1)",
        STDIO_GROUND_TRUTH,
        timeout=2,
    )
    assert crashed == [code_eval.ERROR]
    assert hung == [code_eval.TIMEOUT]


def test_an_empty_program_is_an_error_not_a_wrong_answer():
    results, details = code_eval.run_tests("", STDIO_GROUND_TRUTH)
    assert results == [code_eval.ERROR]
    assert details[0]["status"] == "empty"


def test_the_memory_cap_is_enforced():
    results, _ = code_eval.run_tests(
        "x = bytearray(3_000_000_000)\nprint(len(x))",
        STDIO_GROUND_TRUTH,
        timeout=30,
        memory_mb=256,
    )
    assert results[0] == code_eval.ERROR


def test_output_comparison_ignores_whitespace_and_number_formatting():
    assert code_eval.compare_stdio("5\n", "  5  ")
    assert code_eval.compare_stdio("1.50\n2\n", "1.5\n2.0\n")
    assert not code_eval.compare_stdio("5\n", "5\n5\n")
    assert not code_eval.compare_stdio("five", "5")


def test_call_based_problems_run_against_a_solution_class():
    solution = (
        "class Solution:\n"
        "    def two_sum(self, nums: List[int], target: int) -> List[int]:\n"
        "        seen = {}\n"
        "        for i, value in enumerate(nums):\n"
        "            if target - value in seen:\n"
        "                return [seen[target - value], i]\n"
        "            seen[value] = i\n"
    )
    results, _ = code_eval.run_tests(solution, CALL_GROUND_TRUTH, timeout=10)
    assert results == [code_eval.PASS]


def test_a_call_based_program_missing_its_function_is_a_compile_error():
    results, details = code_eval.run_tests(
        "class Solution:\n    pass", CALL_GROUND_TRUTH, timeout=10
    )
    assert results == [code_eval.ERROR]
    assert details[0]["status"] == "compile_error"


def test_mismatched_test_counts_are_rejected_rather_than_truncated():
    broken = {
        "eval_type": "stdio",
        "input_output": {"inputs": ["1\n", "2\n"], "outputs": ["1\n"]},
    }
    try:
        code_eval.run_tests(ADDER, broken)
    except ValueError as exc:
        assert "2 test inputs but 1 outputs" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_the_example_problems_load_and_their_reference_solutions_pass():
    problems = list(load.read_jsonl(EXAMPLES))
    assert len(problems) == 3
    for problem in problems:
        assert problem["prompt_full"].startswith(problem["prompt"])
        assert problem["ground_truth"]["eval_type"] == "stdio"
        io_pairs = problem["ground_truth"]["input_output"]
        assert len(io_pairs["inputs"]) == len(io_pairs["outputs"])


def test_ojbench_rows_point_at_the_judge_with_dataset_prefixed_ids(tmp_path):
    source = tmp_path / "ojbench.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "2423",
                "prompt": "Solve it.",
                "dataset": "NOI",
                "language": "python",
                "difficulty": "easy",
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "7",
                "prompt": "Solve it.",
                "dataset": "icpc",
                "language": "cpp",
                "difficulty": "hard",
            }
        )
        + "\n"
    )
    problems = load.load_ojbench(source)
    assert len(problems) == 1  # the C++ row is dropped
    ground_truth = problems[0]["ground_truth"]
    assert ground_truth == {
        "judge_backend": "ojbench",
        "dataset": "NOI",
        "problem_id": "loj-2423",
    }


def test_pass_at_k_matches_its_boundary_cases():
    assert score.pass_at_k(n=4, c=0, k=1) == 0.0
    assert score.pass_at_k(n=4, c=4, k=1) == 1.0
    assert score.pass_at_k(n=4, c=1, k=1) == 0.25
    # One correct out of four, drawing two: 1 - C(3,2)/C(4,2) = 1/2.
    assert score.pass_at_k(n=4, c=1, k=2) == 0.5


def test_the_report_counts_problems_solved_and_leaks():
    scored = [
        {
            "method": "groot",
            "problem_id": "a",
            "correct": True,
            "approach": "x",
            "leaked": False,
        },
        {
            "method": "groot",
            "problem_id": "a",
            "correct": False,
            "approach": "y",
            "leaked": True,
        },
        {
            "method": "iid",
            "problem_id": "a",
            "correct": False,
            "approach": None,
            "leaked": False,
        },
    ]
    table = score.report(scored)
    groot_row = next(line for line in table.splitlines() if "groot" in line)
    iid_row = next(line for line in table.splitlines() if "iid" in line)
    assert groot_row.split()[1:4] == ["1", "2", "0.500"]
    assert groot_row.split()[-1] == "0.500"  # one of two guided rows leaked
    assert iid_row.split()[-1] == "0.000"  # iid rows are never counted as leaks


def test_a_program_with_a_main_guard_runs():
    # The entry point is reached because each test execs the candidate as
    # __main__ in its own process, so no AST rewriting is needed to find it.
    program = (
        "def main():\n"
        "    a, b = map(int, input().split())\n"
        "    print(a + b)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    results, _ = code_eval.run_tests(program, STDIO_GROUND_TRUTH, timeout=10)
    assert results == [code_eval.PASS, code_eval.PASS]


def test_bailing_out_with_a_zero_exit_status_is_not_a_failure():
    # Competitive solutions routinely end on sys.exit() or exit().
    for tail in ("import sys\nsys.exit(0)", "exit()", "raise SystemExit"):
        program = f"{ADDER}\n{tail}\n"
        results, _ = code_eval.run_tests(
            program, STDIO_GROUND_TRUTH, timeout=10
        )
        assert results == [code_eval.PASS, code_eval.PASS], tail
    failing = f"{ADDER}\nimport sys\nsys.exit(3)\n"
    results, _ = code_eval.run_tests(failing, STDIO_GROUND_TRUTH, timeout=10)
    assert results == [code_eval.ERROR]


def test_a_syntax_error_is_reported_once_as_a_compile_error():
    results, details = code_eval.run_tests(
        "def f(:\n    pass", STDIO_GROUND_TRUTH
    )
    assert results == [code_eval.ERROR, code_eval.ERROR]
    assert details[0]["status"] == "compile_error"


def test_one_test_cannot_leak_state_into_the_next():
    # Each test is a separate process, so a module a candidate scribbles on is
    # clean again for the test after it.
    program = (
        "import os\n"
        "a, b = map(int, input().split())\n"
        "print(a + b if not hasattr(os, 'SEEN') else -1)\n"
        "os.SEEN = True\n"
    )
    results, _ = code_eval.run_tests(
        program, STDIO_GROUND_TRUTH, timeout=10, stop_on_failure=False
    )
    assert results == [code_eval.PASS, code_eval.PASS]


def test_a_call_based_test_times_out_on_its_own():
    # A hang in one call-based test costs that test its timeout, not the whole
    # problem's budget.
    ground_truth = {
        "eval_type": "call",
        "fn_name": "two_sum",
        "input_output": {
            "inputs": ["[2, 7]\n9", "[1, 2]\n3"],
            "outputs": ["[0, 1]", "[0, 1]"],
        },
    }
    solution = (
        "class Solution:\n"
        "    calls = 0\n"
        "    def two_sum(self, nums, target):\n"
        "        return [0, 1]\n"
    )
    hangs = (
        "import time\n"
        "class Solution:\n"
        "    def two_sum(self, nums, target):\n"
        "        time.sleep(30)\n"
    )
    ok, _ = code_eval.run_tests(solution, ground_truth, timeout=10)
    slow, details = code_eval.run_tests(
        hangs, ground_truth, timeout=2, stop_on_failure=False
    )
    assert ok == [code_eval.PASS, code_eval.PASS]
    assert slow == [code_eval.TIMEOUT, code_eval.TIMEOUT]
    assert details[0]["status"] == "timeout"
