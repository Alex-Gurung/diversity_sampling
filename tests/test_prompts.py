"""Tests for prompt rendering and for parsing what the planners return."""

from pathlib import Path

import prompts

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def golden(name: str) -> str:
    return (GOLDEN_DIR / name).read_text()


def test_rendered_prompts_match_the_published_ones():
    # The goldens were taken from the prompts the paper's runs used. If this
    # fails, prompts/*.txt has drifted from what was published.
    assert prompts.groot_planner("{PROBLEM}", 4) == golden(
        "groot_planner_n4.txt"
    )
    assert prompts.vs_planner("{PROBLEM}", 4) == golden("vs_planner_n4.txt")
    assert prompts.solver("{PROBLEM}", "{APPROACH}") == golden("solver.txt")


def test_no_placeholders_survive_rendering():
    rendered = prompts.groot_planner("a problem", 4)
    for placeholder in ("{PROBLEM}", "{N}", "{N_WORD}", "{N_MINUS_1_WORD}"):
        assert placeholder not in rendered
    assert "{N}" not in prompts.vs_planner("a problem", 4)
    assert "{APPROACH}" not in prompts.solver("a problem", "an approach")


def test_budget_is_spelled_consistently():
    rendered = prompts.groot_planner("a problem", 8)
    assert "Then pick the 8 root-to-leaf paths" in rendered
    assert "eight approaches from eight entirely different subtrees" in rendered
    assert "(seven more <approach> blocks, 8 in total" in rendered
    assert "Generate 8 high-level approaches" in prompts.vs_planner("p", 8)


def test_budget_outside_the_spelling_table_falls_back_to_digits():
    assert "20 approaches from 20 entirely" in prompts.groot_planner("p", 20)


def test_vs_approaches_carry_probabilities():
    output = (
        "<approach>\nProbability: 0.45\nSort, then index.\n</approach>\n"
        "<approach>\nProbability: 0.1\nBinary search the answer.\n</approach>"
    )
    parsed = prompts.parse_approaches(output)
    assert [a.probability for a in parsed] == [0.45, 0.1]
    assert parsed[0].text == "Sort, then index."
    assert parsed[0].path is None


def test_groot_approaches_carry_the_path_written_outside_the_tags():
    output = (
        "<tree>\nA. Sort\nB. Heap\n</tree>\n\n"
        "Path: A -> A1a\n<approach>\nSort ascending.\n</approach>\n"
        "Path: B -> B1b\n<approach>\nMin-heap of everything.\n</approach>"
    )
    parsed = prompts.parse_approaches(output)
    assert [a.path for a in parsed] == ["A -> A1a", "B -> B1b"]
    assert [a.text for a in parsed] == [
        "Sort ascending.",
        "Min-heap of everything.",
    ]
    assert prompts.extract_tree(output) == "A. Sort\nB. Heap"


def test_a_path_header_leaked_inside_the_tags_is_dropped():
    output = "<approach>\nPath: C -> C1\nQuickselect.\n</approach>"
    assert prompts.parse_approaches(output)[0].text == "Quickselect."


def test_parsing_caps_at_the_budget_and_survives_junk():
    output = "<approach>one</approach>" * 6
    assert len(prompts.parse_approaches(output, 4)) == 4
    assert prompts.parse_approaches("no tags at all") == []
    assert prompts.parse_approaches("") == []
    assert prompts.extract_tree("no tree here") is None


def test_extract_code_takes_the_last_closed_fence():
    text = (
        "First attempt:\n```python\nprint(1)\n```\n"
        "That is wrong, here is the fix:\n```python\nprint(2)\n```\n"
    )
    assert prompts.extract_code(text) == "print(2)"


def test_extract_code_recovers_a_fence_truncation_left_open():
    assert prompts.extract_code("Reasoning.\n```python\nprint(1)") == "print(1)"


def test_extract_code_falls_back_to_the_whole_response():
    assert prompts.extract_code("print(1)") == "print(1)"
    assert prompts.extract_code("") == ""


def test_leaks_are_detected_case_insensitively():
    assert prompts.mentions_approach("Following The Given Approach, we sort.")
    assert prompts.mentions_approach("based on the hint, use a heap")
    assert not prompts.mentions_approach("Sort the array, then scan it once.")


def test_only_lines_that_are_purely_a_header_are_stripped():
    # A leaked "Path:" line and a bare list index are headers. An approach
    # that merely opens with one of those words is prose and must survive.
    dropped = [
        (
            "<approach>\nPath: C -> C1\nQuickselect.\n</approach>",
            "Quickselect.",
        ),
        ("<approach>\n1.\nQuickselect.\n</approach>", "Quickselect."),
        ("<approach>\nApproach 3:\nQuickselect.\n</approach>", "Quickselect."),
    ]
    kept = [
        "Approach: use a segment tree.",
        "Path finding is the core of this problem.",
        "2024 is the year in the statement, so parse it.",
        "1. Sort the array.\n2. Scan it once.",
    ]
    for output, expected in dropped:
        assert prompts.parse_approaches(output)[0].text == expected
    for body in kept:
        output = f"<approach>\n{body}\n</approach>"
        assert prompts.parse_approaches(output)[0].text == body
