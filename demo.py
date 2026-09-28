"""Starts a vLLM server, samples one example problem with VS and Groot, and prints the results.

python demo.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LOG = Path("vllm.log")
PROBLEM = Path("demo_problem.jsonl")
SAMPLES = Path("demo_samples.jsonl")
SCORED = Path("demo_scored.jsonl")


def run(*args: object) -> None:
    subprocess.run([sys.executable, *args], check=True)


PROBLEM.write_text(Path("examples/problems.jsonl").read_text().splitlines()[0] + "\n")
SAMPLES.unlink(missing_ok=True)

server = subprocess.Popen(
    ["vllm", "serve", MODEL, "--host", "127.0.0.1", "--port", "8000", "--max-model-len", "32768"],
    stdout=LOG.open("w"),
    stderr=subprocess.STDOUT,
)
try:
    print(f"starting vLLM (log: {LOG})")
    while "Application startup complete" not in LOG.read_text():
        if server.poll() is not None:
            sys.exit(f"vllm serve exited; see {LOG}")
        time.sleep(5)
    run("sample.py", "--problems", PROBLEM, "--method", "vs", "groot", "--out", SAMPLES)
finally:
    server.terminate()
    server.wait()

run("score.py", "--samples", SAMPLES, "--problems", PROBLEM, "--out", SCORED)
print(f"\n{'method':7} {'passed':7} approach")
for line in SCORED.read_text().splitlines():
    sample = json.loads(line)
    passed = "yes" if sample["correct"] else "no"
    print(f"{sample['method']:7} {passed:7} {sample['approach'][:100]}")
