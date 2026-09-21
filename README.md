# diversity_sampling

Minimal implementations of two training-free ways to sample a *strategically
diverse* set of solutions to a code problem, plus the loaders and judge needed
to measure them.

Ordinary IID sampling draws every solution from the same conditional
distribution, so the samples concentrate on whichever strategy the model
already favours; raising the temperature perturbs token choices without
changing the strategy those tokens express. Both methods here instead make the
model commit to a different approach *before* it starts writing a solution.

**VS (Verbalized Sampling)** asks for a flat list of `n` approaches, each with
the model's own estimate of how likely it is. Separation is whatever the
model's enumeration supplies. (Zhang et al., [Verbalized Sampling: How to
Mitigate Mode Collapse and Unlock LLM Diversity](https://arxiv.org/abs/2510.01171))

**GrOoT** asks the model to build a decision tree over the ways the problem
could be approached, then to take `n` root-to-leaf paths through it. Two paths
diverge somewhere, so the approaches differ at a decision the model itself
marked as important.

Both elicit approaches in one planning call and then execute each approach in
its own call, so the resulting traces keep the ordinary reasoning-then-answer
shape. The approach is passed as a hidden instruction and the model is told to
present the reasoning as its own; responses that give the game away are
flagged, because they are not usable training examples.

Sampling and scoring only — no training, no dataset construction, and no math
or next-chapter-prediction domains.

## Setup

The one dependency is the OpenAI *client* library, not the OpenAI API:
generation goes to any OpenAI-compatible endpoint, which is what vLLM serves.

```bash
uv sync                                  # or: pip install openai
```

vLLM belongs to whatever environment serves the model, not to this project:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 --max-model-len 32768

export DIVERSITY_API_BASE=http://127.0.0.1:8000/v1
export DIVERSITY_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

`DIVERSITY_API_KEY` defaults to `EMPTY`, which is what a local vLLM expects.

## Quickstart

Three example problems ship with the repo, so this needs no downloads:

```bash
uv run python sample.py --problems examples/problems.jsonl \
    --method iid,vs,groot --n 4 --out samples.jsonl
uv run python score.py --samples samples.jsonl --problems examples/problems.jsonl \
    --out scored.jsonl
```

`score.py` prints one row per method — the shape of the table, with made-up
numbers:

```
method    probs  samples  sample pass  solved   pass@k  leaked
groot         3       12        0.333       2    0.667   0.083
iid           3       12        0.250       1    0.333   0.000
vs            3       12        0.333       2    0.667   0.000
```

`solved` is problems with at least one correct sample, which is what these
methods are for: at a fixed budget, do the samples between them cover more
problems. `--k` switches `pass@k` to the unbiased estimator. Both scripts are
resumable — re-run with the same `--out` and they pick up where they stopped.

## Benchmarks

`load.py` converts a suite into the one problem format everything else reads.
The source files run to hundreds of megabytes, so they are not vendored here;
pass `--path`, or set `DIVERSITY_DATA_DIR`.

```bash
uv run python load.py --suite lcb --path lcb_test6.jsonl --out problems.jsonl
uv run python sample.py --problems problems.jsonl --method groot --out samples.jsonl
uv run python score.py --samples samples.jsonl --problems problems.jsonl \
    --out scored.jsonl --workers 8
```

| `--suite` | Source | Graded by |
|---|---|---|
| `lcb` | LiveCodeBench `code_generation_lite`, v6 test split | `code_eval.py` |
| `cobalt` | a filtered split of [osunlp/TACO-Cobalt](https://huggingface.co/datasets/osunlp/TACO-Cobalt) | `code_eval.py` |
| `ojbench` | [OJBench](https://github.com/He-Ren/OJBench) prompts, Python only | OJBench's DMOJ judge |

There are two judges because the suites leave no choice: LiveCodeBench and
Cobalt ship tests but no judge, OJBench ships a judge but no tests. Which one
grades a sample is the problem's own declaration, not a flag — point
`--ojbench-problem-dirs` at OJBench's test data and its problems go to its
judge, which is the only way its verdicts match published numbers:

```bash
uv run python score.py --samples samples.jsonl --problems problems.jsonl \
    --out scored.jsonl \
    --ojbench-problem-dirs /data/OJBench_testdata/NOI /data/OJBench_testdata/ICPC
```

Leave it off and those samples are written out unscored rather than counted as
wrong. The built-in judge is a compact reimplementation rather than the
benchmarks' own harnesses: fine for comparing methods against each other, not
a substitute for the official harness when you want leaderboard-comparable
numbers.

`score.py` executes model-written code. Every test gets its own process, a
timeout and a memory cap, but the filesystem and the network are not
sandboxed, so run it in a container. `load.py --suite lcb` unpickles the
benchmark's private test cases, so point it only at a file you trust.

## Prompts

The prompts live in [`prompts/`](prompts/) as plain text, so they can be read
and diffed without going through Python:

| File | Stage |
|---|---|
| `groot_planner.txt` | GrOoT stage 1: the tree, then `n` paths through it |
| `vs_planner.txt` | VS stage 1: `n` approaches with probabilities |
| `solver.txt` | stage 2, shared: solve from a hidden approach |

IID needs no template — the baseline is the problem's own prompt, unchanged.

They carry `{PROBLEM}`, `{APPROACH}`, `{N}`, `{N_WORD}` and `{N_MINUS_1_WORD}`
placeholders. At `n=4` they render byte-identical to the prompts published in
the paper appendix; `tests/golden/` holds those renderings, taken from the
original sources, and a test compares against them so the templates cannot
drift silently.

## Settings

Defaults match the paper's code-sampling configuration:

| | |
|---|---|
| budget per problem | 4 |
| planner temperature / top-p / max tokens | 0.45 / 0.95 / 8192 |
| solver and IID temperature / top-p / max tokens | 0.85 / 0.95 / 8192 |
| planner retries | 3 |

A budget of `n` costs `n` solver calls plus one planning call, for both VS and
GrOoT. Compare against IID at the budget you care about, and decide separately
whether to charge the planning call to it.

If a planner returns fewer than `n` parseable approaches it is retried; if no
attempt reaches `n`, the fullest one is used and `planner.approaches_parsed`
records the shortfall. If nothing parses, the problem falls back to a single
unguided solve with `approach: null`.

## Output format

`sample.py` writes one record per sample:

```json
{"problem_id": "lcb:2035", "suite": "lcb", "method": "groot", "idx": 0, "n": 4,
 "model": "...", "approach": "Sort the intervals by end point, then ...",
 "probability": null, "path": "B -> B1a", "output": "...", "code": "...",
 "finish_reason": "stop", "leaked": false,
 "planner": {"attempts": 1, "approaches_parsed": 4, "tree": "...", "raw": "..."}}
```

`probability` is set by VS, `path` and `tree` by GrOoT. `score.py` adds
`correct`, `tests_passed`, `tests_run` and `status`.

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # Google style, 80 cols
uv run pytest tests/                                 # no endpoint needed
```
