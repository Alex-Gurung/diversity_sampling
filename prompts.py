"""Prompts for VS and Groot, and parsing of model responses."""

import re
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty"
).split()
PLACEHOLDER = re.compile(r"\{(PROBLEM|APPROACH|N|N_WORD|N_MINUS_1_WORD)\}")

APPROACH_TAG = re.compile(r"<approach>(.*?)</approach>", re.DOTALL | re.IGNORECASE)
PROBABILITY_LINE = re.compile(
    r"^\s*Probability\s*:\s*([01](?:\.\d+)?)\s*$", re.IGNORECASE | re.MULTILINE
)
PYTHON_BLOCK = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ANY_BLOCK = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
OPEN_FENCE = re.compile(r"```[a-zA-Z0-9_+#.-]*[ \t]*\n")

# A solution containing any of these phrases reveals that it was given an approach.
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
    "given approach",
    "the hint",
    "hint structure",
    "the provided approach",
    "was supplied",
]


@dataclass
class Approach:
    text: str
    probability: float | None  # VS only


def render(template: str, **fields: str) -> str:
    text = (PROMPT_DIR / f"{template}.txt").read_text()
    return PLACEHOLDER.sub(lambda match: fields[match.group(1)], text)


def planner_prompt(method: str, problem: str, n: int) -> str:
    return render(
        f"{method}_planner",
        PROBLEM=problem,
        N=str(n),
        N_WORD=NUMBER_WORDS[n],
        N_MINUS_1_WORD=NUMBER_WORDS[n - 1],
    )


def solver_prompt(problem: str, approach: str) -> str:
    return render("solver", PROBLEM=problem, APPROACH=approach)


def parse_approaches(planner_output: str) -> list[Approach]:
    approaches = []
    for block in APPROACH_TAG.findall(planner_output):
        probability = PROBABILITY_LINE.search(block.strip())
        approaches.append(
            Approach(
                text=PROBABILITY_LINE.sub("", block.strip()).strip(),
                probability=float(probability.group(1)) if probability else None,
            )
        )
    return approaches


def extract_code(response: str) -> str:
    """Returns the last closed code block, or the text after an unclosed fence."""
    for pattern in (PYTHON_BLOCK, ANY_BLOCK):
        blocks = pattern.findall(response)
        if blocks:
            return blocks[-1].strip()
    fence = OPEN_FENCE.search(response)
    return response[fence.end() :].strip() if fence else response.strip()


def mentions_approach(response: str) -> bool:
    return any(term in response.lower() for term in LEAK_TERMS)
