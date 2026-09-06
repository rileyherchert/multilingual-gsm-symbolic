"""Check every copy of template 0062, including inactive and reference copies."""

import tomllib
from pathlib import Path
from random import Random

import pytest

from multilingual_gsm_symbolic._helpers import build_eval_context, eval_node, parse_expr

_ROOT = Path(__file__).resolve().parents[1] / "src/multilingual_gsm_symbolic/data/templates"
_FILES = sorted(_ROOT.rglob("0062.toml"))


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.relative_to(_ROOT).as_posix())
def test_card_total_formula_matches_solution(path):
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if "question_annotated" not in data:
        assert data == {"ignore": True}, "Only an empty ignored placeholder may omit its formula"
        return

    formula = parse_expr(data["question_annotated"].split("#answer:", 1)[1].strip())
    final = data["answer_annotated"].split("####", 1)[1].strip()
    assert final.startswith("{") and final.endswith("}")
    solution = parse_expr(final[1:-1])

    # All numeric assignments allowed by this template's ranges and integrality constraint.
    # Independent integer arithmetic avoids using either template expression as the oracle.
    for red in range(15, 81):
        for percent in range(20, 90):
            if red * percent % 100:
                continue
            green = red + red * percent // 100
            yellow = red + green
            expected = red + green + yellow
            env = build_eval_context(Random(0), {"n1": red, "p": percent})
            assert eval_node(formula, env) == expected, (path, red, percent)
            assert eval_node(solution, env) == expected, (path, red, percent)
