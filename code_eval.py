#!/usr/bin/env python3
"""Runs a candidate program against a problem's tests.

SAFETY: this executes model-written code. Every test runs in its own process
with a wall-clock timeout and an address-space limit, but the filesystem and
the network are NOT sandboxed. Run it inside a container or a VM you are
willing to lose.

Two test shapes, matching what ``load.py`` emits:

    stdio   feed the input on stdin, compare stdout line by line, with numeric
            lines compared as decimals so "1.0" and "1" agree
    call    import the program, call ``fn_name`` -- on a ``Solution`` instance
            when the program defines one -- with JSON-decoded arguments, and
            compare the return value

Per-test result codes follow the LiveCodeBench convention: 1 pass, 0 wrong
answer, -1 timeout, -2 error, including a program that fails to import.

This module is also its own worker. ``run_tests`` launches
``python code_eval.py``, which reads a job on stdin and writes graded results
on stdout; that worker imports the heavyweight prelude once, compiles the
candidate once, and then forks per test. Forking is what keeps tests isolated
from each other -- a real process each, with its own stdin, stdout and memory
limit -- without emulating any of it in-process. POSIX only.
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

PASS = 1
WRONG = 0
TIMEOUT = -1
ERROR = -2

DEFAULT_TIMEOUT = 10.0
DEFAULT_MEMORY_MB = 4096
MAX_DETAIL_CHARS = 400

# Slack over the sum of the per-test timeouts, to catch a worker that wedges
# somewhere other than inside a test.
_WORKER_SLACK_SECONDS = 30.0

# Both benchmarks' own harnesses run candidates with these names already in
# scope, and solutions routinely rely on it: a bare List[int] annotation, gcd
# without an import. Providing it keeps us from scoring an import error the
# benchmark itself would never have seen. The worker pays for it once.
PRELUDE = """import sys
sys.set_int_max_str_digits(50000)
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from json import *
from builtins import *
from typing import *
import string, re, datetime, collections, heapq, bisect, copy, math, random
import statistics, itertools, functools, operator, io, json
"""


# ------------------------------------------------------------ comparison ----


def _decimals(line: str) -> tuple[bool, list[Decimal]]:
    """Parses a line as whitespace-separated decimals, if it is one."""
    try:
        return True, [Decimal(token) for token in line.split()]
    except (ArithmeticError, TypeError, ValueError):
        return False, []


def compare_stdio(prediction: str, expected: str) -> bool:
    """Compares program output against expected output, line by line.

    Surrounding whitespace is ignored, and lines that parse as numbers are
    compared as decimals, so "1.0" matches "1".

    Args:
        prediction: What the program printed.
        expected: What the test expects.

    Returns:
        True if the two agree.
    """
    predicted_lines = [
        line.strip() for line in (prediction or "").strip().split("\n")
    ]
    expected_lines = [
        line.strip() for line in (expected or "").strip().split("\n")
    ]
    if len(predicted_lines) != len(expected_lines):
        return False
    for predicted, expect in zip(predicted_lines, expected_lines, strict=True):
        if predicted == expect:
            continue
        predicted_ok, predicted_decimal = _decimals(predicted)
        expect_ok, expect_decimal = _decimals(expect)
        if predicted_ok and expect_ok and predicted_decimal == expect_decimal:
            continue
        return False
    return True


def _short(value: Any) -> str:
    """Renders a value for a failure detail, truncated to keep files small."""
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= MAX_DETAIL_CHARS:
        return text
    return text[:MAX_DETAIL_CHARS] + "..."


def passes_all(results: Sequence[int]) -> bool:
    """Reports whether every graded test passed."""
    return bool(results) and all(result == PASS for result in results)


# ---------------------------------------------------------------- worker ----
#
# Everything from here to main() runs in the worker process, one per
# candidate. It is single-threaded by construction, which is what makes
# os.fork() safe to use here.


def _apply_memory_limit(memory_mb: int) -> None:
    """Caps the address space of the calling process."""
    limit = memory_mb * 1024 * 1024
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        # Not every platform honours every limit; the ones it does still bind.
        with contextlib.suppress(AttributeError, OSError, ValueError):
            resource.setrlimit(getattr(resource, name), (limit, limit))


def _resolve_entry_point(namespace: dict, fn_name: str):
    """Finds a call-mode problem's entry point, bound if it is a method."""
    if "Solution" in namespace:
        return getattr(namespace["Solution"](), fn_name)
    return namespace[fn_name]


def _run_in_child(
    compiled: Any,
    namespace: dict,
    job: dict,
    args: Sequence | None,
    paths: dict[str, Path],
) -> NoReturn:
    """Runs one test and exits. Only ever reached inside a forked child.

    The child holds a private copy of the worker's memory, so the namespace it
    scribbles on and the streams it replaces cannot reach the next test.

    Args:
        compiled: The candidate's compiled code object.
        namespace: Globals to execute it in, already holding the prelude.
        job: The job being run, for its mode and fn_name.
        args: Decoded arguments for a call-mode test.
        paths: Where to read stdin from, and write stdout, stderr and a
            call-mode return value to.
    """
    exit_code = 0
    try:
        _apply_memory_limit(job["memory_mb"])
        with (
            open(paths["stdin"]) as stream_in,
            open(paths["stdout"], "w") as stream_out,
            open(paths["stderr"], "w") as stream_err,
        ):
            os.dup2(stream_in.fileno(), 0)
            os.dup2(stream_out.fileno(), 1)
            os.dup2(stream_err.fileno(), 2)
            # __main__, so that an `if __name__ == "__main__":` block runs.
            # That is why no AST rewriting is needed to reach the program's
            # entry point the way an in-process runner would have to.
            namespace["__name__"] = "__main__"
            exec(compiled, namespace)
            if job["mode"] in ("call", "probe"):
                # The probe stops here: resolving the entry point is the whole
                # question, and calling it with no arguments would not be.
                function = _resolve_entry_point(namespace, job["fn_name"])
                if job["mode"] == "call":
                    value = function(*(args or []))
                    if isinstance(value, tuple):
                        value = list(value)
                    paths["result"].write_text(json.dumps(value))
    except SystemExit as exc:
        # Solutions routinely bail out early with sys.exit() or exit(). A zero
        # status is an ordinary end, not a failure.
        exit_code = 0 if exc.code in (None, 0) else 1
    except BaseException:  # noqa: BLE001 - a candidate may raise anything
        traceback.print_exc()
        exit_code = 1
    finally:
        # fd 1 and 2 still point at the files after the streams above close,
        # so this reaches them. os._exit skips the interpreter's own flush.
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(ValueError, OSError):
                stream.flush()
    os._exit(exit_code)


class _TestTimeoutError(Exception):
    """Raised in the worker when a forked test outstays its timeout."""


def _raise_timeout(_signum: int, _frame: object) -> NoReturn:
    """SIGALRM handler, installed only while waiting on a forked test."""
    raise _TestTimeoutError


def _await_child(pid: int, timeout: float) -> str:
    """Waits for a forked test, killing it if it outstays its timeout.

    The wait blocks and a real timer interrupts it, rather than polling: that
    costs nothing per test and keeps the timeout exact. Both are safe here
    because the worker is single-threaded, so the handler runs where it was
    installed.

    Args:
        pid: The child to wait for.
        timeout: Seconds to allow it.

    Returns:
        One of "ok", "error" or "timeout".
    """
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        _finished, status = os.waitpid(pid, 0)
        clean = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        return "ok" if clean else "error"
    except _TestTimeoutError:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        return "timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _fork_test(
    compiled: Any,
    prelude: dict,
    job: dict,
    index: int,
    args: Sequence | None,
    stdin_text: str,
    workdir: Path,
) -> tuple[str, str, str]:
    """Runs one test in its own process and collects what it produced.

    Streams go through files rather than pipes, so a chatty program cannot
    deadlock against a parent that is still writing its input.

    Returns:
        An (outcome, output, stderr) triple. ``output`` is stdout for a stdio
        test and the JSON-encoded return value for a call test.
    """
    paths = {
        "stdin": workdir / f"{index}.in",
        "stdout": workdir / f"{index}.out",
        "stderr": workdir / f"{index}.err",
        "result": workdir / f"{index}.result",
    }
    paths["stdin"].write_text(stdin_text)
    for key in ("stdout", "stderr", "result"):
        paths[key].write_text("")

    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        _run_in_child(compiled, dict(prelude), job, args, paths)

    outcome = _await_child(pid, job["timeout"])
    source = "result" if job["mode"] == "call" else "stdout"
    return (
        outcome,
        paths[source].read_text(errors="replace"),
        paths["stderr"].read_text(errors="replace"),
    )


def _grade(
    outcome: str, output: str, stderr: str, expected: str, mode: str
) -> tuple[int, dict]:
    """Turns one test's raw outcome into a result code and a detail."""
    if outcome == "timeout":
        return TIMEOUT, {"status": "timeout"}
    if outcome == "error":
        return ERROR, {"status": "error", "message": _short(stderr.strip())}
    if mode == "call":
        try:
            correct = json.loads(output) == json.loads(expected)
        except json.JSONDecodeError:
            return ERROR, {
                "status": "error",
                "message": "unreadable return value",
            }
    else:
        correct = compare_stdio(output, expected)
    if correct:
        return PASS, {"status": "pass"}
    return WRONG, {
        "status": "wrong",
        "expected": _short(expected),
        "output": _short(output),
    }


def _probe_entry_point(
    compiled: Any, prelude: dict, job: dict, workdir: Path
) -> str | None:
    """Checks a call-mode program defines its entry point, in a child.

    A missing function is a property of the program rather than of any one
    test, so it is reported once as a compile error instead of n times as a
    failure. The check runs in a fork because resolving it means executing the
    program, which the worker must not do in its own process.

    Returns:
        An error message, or None if the entry point resolved.
    """
    outcome, _output, stderr = _fork_test(
        compiled,
        prelude,
        dict(job, mode="probe"),
        index=-1,
        args=None,
        stdin_text="",
        workdir=workdir,
    )
    if outcome == "ok":
        return None
    lines = stderr.strip().splitlines()
    return lines[-1] if lines else outcome


def _all_failed(size: int, status: str, message: str) -> dict:
    """Builds the result of a candidate that never got as far as a test."""
    return {
        "results": [ERROR] * size,
        "details": [{"status": status, "message": _short(message)}] * size,
    }


def _work(job: dict) -> dict:
    """Grades every test of one candidate. Runs in the worker process."""
    size = len(job["tests"])
    prelude: dict = {}
    exec(compile(PRELUDE, "<prelude>", "exec"), prelude)
    try:
        compiled = compile(job["code"], "<candidate>", "exec")
    except (SyntaxError, ValueError) as exc:
        return _all_failed(size, "compile_error", repr(exc))

    results: list[int] = []
    details: list[dict] = []
    with tempfile.TemporaryDirectory() as raw_workdir:
        workdir = Path(raw_workdir)
        os.chdir(workdir)  # a candidate that writes files writes them here
        if job["mode"] == "call":
            failure = _probe_entry_point(compiled, prelude, job, workdir)
            if failure is not None:
                return _all_failed(size, "compile_error", failure)

        for index, test in enumerate(job["tests"]):
            outcome, output, stderr = _fork_test(
                compiled,
                prelude,
                job,
                index,
                test.get("args"),
                test.get("stdin", ""),
                workdir,
            )
            result, detail = _grade(
                outcome, output, stderr, test["expected"], job["mode"]
            )
            results.append(result)
            details.append(detail)
            if job["stop_on_failure"] and result != PASS:
                break
    return {"results": results, "details": details}


def main() -> int:
    """Worker entry point: reads a job on stdin, writes results on stdout."""
    json.dump(_work(json.loads(sys.stdin.read())), sys.stdout)
    return 0


# ------------------------------------------------------------------- API ----


def _build_job(
    code: str,
    ground_truth: dict,
    inputs: list,
    outputs: list,
    timeout: float,
    memory_mb: int,
    stop_on_failure: bool,
) -> dict:
    """Assembles the job the worker reads, decoding call-mode arguments."""
    mode = "call" if ground_truth.get("eval_type") == "call" else "stdio"
    tests = []
    for test_input, expected in zip(inputs, outputs, strict=True):
        if isinstance(test_input, list):
            test_input = "\n".join(test_input)
        if isinstance(expected, list):
            expected = "\n".join(expected)
        if mode == "call":
            # Each input holds one JSON value per line: the arguments.
            args = [json.loads(line) for line in test_input.split("\n")]
            tests.append({"args": args, "expected": expected})
        else:
            tests.append({"stdin": test_input, "expected": expected})
    return {
        "code": code,
        "mode": mode,
        "fn_name": ground_truth.get("fn_name"),
        "tests": tests,
        "timeout": timeout,
        "memory_mb": memory_mb,
        "stop_on_failure": stop_on_failure,
    }


def _unpack(reported: dict) -> tuple[list[int], list[dict]]:
    """Splits a worker result into the pair ``run_tests`` returns."""
    return reported["results"], reported["details"]


def run_tests(
    code: str,
    ground_truth: dict,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_tests: int | None = None,
    stop_on_failure: bool = True,
) -> tuple[list[int], list[dict]]:
    """Grades one program against one problem's tests.

    Args:
        code: The extracted program.
        ground_truth: The problem's ``ground_truth`` field.
        timeout: Seconds allowed per test.
        memory_mb: Address-space cap for each test.
        max_tests: Grade only the first N tests. None grades all of them.
        stop_on_failure: Stop at the first non-pass, which is all pass@k needs
            and much cheaper.

    Returns:
        A (per-test codes, per-test details) pair.

    Raises:
        ValueError: If the problem carries a different number of inputs and
            outputs, which would silently grade against the wrong test.
    """
    if not (code or "").strip():
        return [ERROR], [{"status": "empty"}]

    input_output = ground_truth.get("input_output") or {}
    inputs = list(input_output.get("inputs") or [])
    outputs = list(input_output.get("outputs") or [])
    if not inputs:
        return [ERROR], [{"status": "no_tests"}]
    if len(inputs) != len(outputs):
        raise ValueError(
            f"{len(inputs)} test inputs but {len(outputs)} outputs"
        )
    if max_tests:
        inputs = inputs[:max_tests]
        outputs = outputs[:max_tests]
    call_mode = ground_truth.get("eval_type") == "call"
    if call_mode and not ground_truth.get("fn_name"):
        return _unpack(
            _all_failed(len(inputs), "no_fn_name", "the problem names none")
        )

    job = _build_job(
        code, ground_truth, inputs, outputs, timeout, memory_mb, stop_on_failure
    )
    budget = timeout * len(inputs) + _WORKER_SLACK_SECONDS
    try:
        completed = subprocess.run(
            [sys.executable, __file__],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=budget,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _unpack(
            _all_failed(len(inputs), "global_timeout", "the worker hung")
        )

    try:
        return _unpack(json.loads(completed.stdout))
    except json.JSONDecodeError:
        message = completed.stderr.strip() or "the worker produced no result"
        return _unpack(_all_failed(len(inputs), "error", message))


if __name__ == "__main__":
    sys.exit(main())
