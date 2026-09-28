# diversity_sampling

Samples solutions to programming problems with Groot and VS, the two methods from the paper, and
with IID sampling as the baseline.

- IID solves each problem n times.
- VS (verbalized sampling) asks the model for n approaches, each with a probability, then solves
  the problem once per approach.
- Groot asks the model for a decision tree of approaches and n root-to-leaf paths through it, then
  solves the problem once per path.

The solver gets the approach as a hidden hint. Solutions that mention the hint are dropped.

## Setup

```bash
pip install vllm
```

The scripts themselves use only the Python standard library.

## Usage

Start a vLLM server. `--data-parallel-size` is the number of GPUs to use; leave it out for one GPU.

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768 --data-parallel-size 8
```

Then, in another shell:

```bash
python load.py --suite cobalt --path cobalt_train.jsonl --out problems.jsonl
python sample.py --problems problems.jsonl --method iid vs groot --out samples.jsonl
python score.py --samples samples.jsonl --problems problems.jsonl --out scored.jsonl
```

`sample.py` talks to `http://localhost:8000/v1` (change with `--url`). It runs all problems and
methods at once with up to 1024 requests in flight (change with `--workers`), appends each
problem's samples to `--out` as soon as they finish, and skips problems already in `--out` when
re-run. `examples/problems.jsonl` has three small problems for a quick test.

`score.py` prints the number of samples, correct samples and solved problems for each method. A
problem is solved if at least one of its samples passes every test.

`score.py` runs model-written programs, so run it in a container.

The sampling defaults are the paper's: n = 4, planner temperature 0.45, solver and IID temperature
0.85, top-p 0.95, top-k 20, and at most 8192 new tokens per call. `score.py` gives each test 10
seconds and 4 GB of memory.

## Files

| File | Contents |
|---|---|
| `prompts/` | the VS and Groot planner prompts and the solver prompt |
| `prompts.py` | fills in the prompts, parses approaches, extracts code |
| `sample.py` | sampling |
| `score.py` | grading |
| `call_harness.py` | runs function-call tests (LiveCodeBench) |
| `load.py` | converts LiveCodeBench, Cobalt and OJBench files to the problem format |

## Data

`load.py` reads:

- LiveCodeBench: `test6.jsonl` from `livecodebench/code_generation_lite` (release v6).
- Cobalt: a jsonl with `problem_id`, `prompt`, `instruction` and `ground_truth` on each row, as in
  our filtered split of [osunlp/TACO-Cobalt](https://huggingface.co/datasets/osunlp/TACO-Cobalt).
- OJBench: its prompts file, `full.jsonl`.

## Formats

Problems, one JSON object per line:

```json
{"problem_id": "cobalt:1001", "prompt_full": "...",
 "ground_truth": {"eval_type": "stdio", "input_output": {"inputs": ["..."], "outputs": ["..."]}}}
```

`eval_type` is `stdio` (the program reads stdin and prints its answer) or `call` (LiveCodeBench
functional problems, where `fn_name` names the function). OJBench problems have
`{"judge_backend": "ojbench", ...}` instead. `score.py` skips them; grade them with
[OJBench's judge](https://github.com/He-Ren/OJBench).

Samples:

```json
{"problem_id": "cobalt:1001", "method": "groot", "approach": "...", "probability": null,
 "plan": "...", "output": "...", "finish_reason": "stop"}
```

`approach` and `plan` are null for IID, and `probability` is set only for VS. `score.py` adds
`code` and `correct`.

## Tests

```bash
pytest tests/
```
