"""The code prompts for VS and GrOoT, and the parsers for what they return.

The prompts themselves live as plain text in ``prompts/*.txt`` so they can be
read and diffed without going through Python. They carry four placeholders:

    {PROBLEM}          the problem statement, output-format instruction included
    {APPROACH}         one approach, given to the solver as a hidden instruction
    {N}                the sampling budget, as a digit
    {N_WORD}           the same number spelled out ("four")
    {N_MINUS_1_WORD}   n - 1 spelled out, for "three more <approach> blocks"

At n=4 the rendered prompts are byte-identical to the ones published in the
paper appendix; the files under ``tests/golden/`` pin that down. Substitution
is a plain string replace rather than ``str.format`` because the templates
contain literal braces and backticks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
]


def _template(name: str) -> str:
    """Reads ``prompts/<name>.txt``, minus the trailing newline it ends with."""
    return (PROMPT_DIR / f"{name}.txt").read_text().rstrip("\n")


def _spell(number: int) -> str:
    """Spells a small number, falling back to digits outside the table."""
    if 0 <= number < len(_WORDS):
        return _WORDS[number]
    return str(number)


def _render(template: str, **fields: str) -> str:
    """Substitutes ``{FIELD}`` placeholders by literal replacement."""
    rendered = template
    for key, value in fields.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def groot_planner(problem: str, n: int = 4) -> str:
    """Builds the GrOoT planning prompt.

    Asks for a decision tree over ways to solve the problem, then for the n
    root-to-leaf paths through it that best trade coverage against plausibility.

    Args:
        problem: The problem statement, including its output-format instruction.
        n: How many approaches to ask for.

    Returns:
        The planning prompt, as a single user message.
    """
    return _render(
        _template("groot_planner"),
        PROBLEM=problem,
        N=str(n),
        N_WORD=_spell(n),
        N_MINUS_1_WORD=_spell(n - 1),
    )


def vs_planner(problem: str, n: int = 4) -> str:
    """Builds the VS planning prompt.

    Asks for a flat list of n approaches, each carrying the model's own
    estimate of how likely it is.

    Args:
        problem: The problem statement, including its output-format instruction.
        n: How many approaches to ask for.

    Returns:
        The planning prompt, as a single user message.
    """
    return _render(_template("vs_planner"), PROBLEM=problem, N=str(n))


def solver(problem: str, approach: str) -> str:
    """Builds the stage-2 prompt that both methods share.

    The approach is framed as a hidden instruction: the model is told to
    present the reasoning as its own, so the trace stays usable as an ordinary
    reasoning-then-answer training example.

    Args:
        problem: The problem statement, including its output-format instruction.
        approach: One approach from a planning call.

    Returns:
        The solving prompt, as a single user message.
    """
    return _render(_template("solver"), PROBLEM=problem, APPROACH=approach)


def iid_prompt(problem: str) -> str:
    """Returns the IID baseline prompt, which is the problem unchanged.

    The dataset's own output-format instruction is already part of ``problem``
    (see ``load.py``), so there is nothing to render. Adding a "reason through
    this carefully" preamble would make the baseline less in-distribution
    rather than more.

    Args:
        problem: The problem statement, including its output-format instruction.

    Returns:
        The problem, unchanged.
    """
    return problem


# --------------------------------------------------------------- parsing ----

APPROACH_RE = re.compile(
    r"<approach>(.*?)</approach>", re.DOTALL | re.IGNORECASE
)
TREE_RE = re.compile(r"<tree>(.*?)</tree>", re.DOTALL | re.IGNORECASE)
PATH_RE = re.compile(r"^[ \t]*Path:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
PROBABILITY_RE = re.compile(
    r"^\s*Probability:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)

# The "Path:" line belongs outside the tags, and a planner writing a numbered
# list sometimes repeats the index inside them. Both are headers rather than
# plan, so leading ones are dropped. A line only counts as a header when it
# holds nothing else, so an approach opening "Approach: use a segment tree" or
# "1. Sort the array" keeps its first line.
_STRAY_HEADER_RE = re.compile(
    r"^\s*(Path:.*|\d+\s*[.)]|Approach\s*\d*\s*[:.]?)\s*$", re.IGNORECASE
)


@dataclass
class Approach:
    """One approach from a planning call, ready to hand to the solver.

    Attributes:
        text: The approach itself, with any header lines stripped.
        probability: The model's own likelihood estimate. VS only; GrOoT is
            not asked for one.
        path: The route through the decision tree, such as "B -> B1a". GrOoT
            only.
    """

    text: str
    probability: float | None = None
    path: str | None = None


def extract_tree(planner_output: str) -> str | None:
    """Returns the <tree> block of a GrOoT response, or None if it has none.

    The tree is kept for inspection only; nothing downstream depends on it.

    Args:
        planner_output: A raw planning response.

    Returns:
        The tree contents, or None.
    """
    match = TREE_RE.search(planner_output or "")
    return match.group(1).strip() if match else None


def parse_approaches(
    planner_output: str, n: int | None = None
) -> list[Approach]:
    """Pulls the <approach> blocks out of a planning response.

    VS opens each block with a ``Probability: <p>`` line, and GrOoT writes a
    ``Path: ...`` line just outside each block so the tag contents stay pure
    plan. Both are read here and stripped from the approach text, so the
    solver only ever sees the plan itself.

    Args:
        planner_output: A raw planning response.
        n: Keep at most this many approaches. None keeps all of them.

    Returns:
        The approaches, in the order the planner wrote them.
    """
    text = planner_output or ""
    approaches = []
    for match in APPROACH_RE.finditer(text):
        body, probability = _split_probability(match.group(1))
        approaches.append(
            Approach(
                text=body,
                probability=probability,
                path=_preceding_path(text, match.start()),
            )
        )
    return approaches[:n] if n is not None else approaches


def _split_probability(body: str) -> tuple[str, float | None]:
    """Splits a leading ``Probability:`` line off an approach body."""
    lines = body.strip().splitlines()
    probability = None
    if lines:
        match = PROBABILITY_RE.match(lines[0])
        if match:
            probability = float(match.group(1))
            lines = lines[1:]
    while lines and (not lines[0].strip() or _STRAY_HEADER_RE.match(lines[0])):
        lines = lines[1:]
    return "\n".join(lines).strip(), probability


def _preceding_path(text: str, block_start: int) -> str | None:
    """Returns the last ``Path:`` line before an <approach> block, if any."""
    paths = [match.group(1) for match in PATH_RE.finditer(text, 0, block_start)]
    return paths[-1] if paths else None


# --------------------------------------------------------------- outputs ----

PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
GENERIC_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
OPEN_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+#.-]*[ \t]*\n")

# A response that names the approach it was given has failed the "present it
# as your own work" instruction, so it is not a usable training example.
# Discarding these is part of the method.
LEAK_TERMS = [
    "private note",
    "the suggestion",
    "the memo",
    "prior attempt",
    "hidden instruction",
    "previous attempt",
    "the anchor",
    "sampled attempt",
    "as instructed",
    "the guidance",
    "approaches already tried",
    "the strategy note",
    "the given approach",
    "given approach",
    "the hint",
    "hint structure",
    "copying the hint",
    "the provided approach",
    "was supplied",
]


def extract_code(text: str) -> str:
    """Returns the program from a solver response: the last closed fence.

    Last rather than first, because models routinely write a draft, notice it
    is wrong, and follow it with a corrected block -- and OJBench's own parser
    takes the last one too. The fallbacks handle a fence the model opened but
    never closed, which truncation makes common enough that missing it shows
    up as spurious compile errors.

    Args:
        text: A raw solver response.

    Returns:
        The extracted program, or the whole response if it has no fences.
    """
    text = text or ""
    for pattern in (PYTHON_FENCE_RE, GENERIC_FENCE_RE):
        blocks = pattern.findall(text)
        if blocks:
            return blocks[-1].strip()
    open_fence = OPEN_FENCE_RE.search(text)
    if open_fence:
        return text[open_fence.end() :].strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def mentions_approach(text: str) -> bool:
    """Reports whether a response gives away that it was handed an approach.

    Args:
        text: A raw solver response.

    Returns:
        True if the response names the approach it was given.
    """
    lowered = (text or "").lower()
    return any(term in lowered for term in LEAK_TERMS)
