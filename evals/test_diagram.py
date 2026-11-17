"""DeepEval tests for the chat->board diagram feature.

Eval 1 (arrow binding): do arrows stay attached when a node is dragged?
Eval 2 (node overlap): do step nodes overlap each other?
Eval 3 (text visibility): do drawn texts render immediately, no refresh?
Eval 4 (scene isolation): does each conversation keep its own board scene?

The elements under test come from the REAL builder in diagram.html
(see diagram_gen.py) — the evals measure the code's output, they do not
call an LLM and need no API key.

Run:  python3 -m venv .venv && .venv/bin/pip install -r requirements-evals.txt
       .venv/bin/pytest test_diagram.py -v
"""

import json

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from diagram_gen import generate_elements, scene_isolation_report
from metrics import (
    ArrowBindingMetric,
    NodeOverlapMetric,
    SceneIsolationMetric,
    TextVisibilityMetric,
)

TWO_STEPS = {
    "title": "شروع تا پایان",
    "steps": [
        {"title": "شروع", "desc": "نقطه‌ی آغاز فرایند"},
        {"title": "پایان", "desc": "نقطه‌ی پایان فرایند"},
    ],
}

DEMO_3 = {
    "title": "ثبت تا تحویل",
    "steps": [
        {"title": "ثبت درخواست", "desc": "مشتری درخواست را ثبت می‌کند"},
        {"title": "بررسی و تایید", "desc": "کارشناس درخواست را بررسی می‌کند"},
        {"title": "اجرا و تحویل", "desc": "کار انجام و تحویل داده می‌شود"},
    ],
}

SEVEN_STEPS = {
    "title": "فرایند هفت مرحله‌ای",
    "steps": [{"title": f"مرحله‌ی {i}", "desc": f"توضیح مرحله‌ی {i}"} for i in range(1, 8)],
}

LONG_DESC = {
    "title": "",
    "steps": [
        {
            "title": "قدم خیلی طولانی با توضیح مفصل",
            "desc": "این یک توضیح بسیار طولانی است که حتماً باید به چند خط بشکند تا در باکس جا شود و باکس هم باید بلندتر شود",
        },
        {"title": "پایان", "desc": ""},
    ],
}

CASES = [
    (TWO_STEPS, "2-steps"),
    (DEMO_3, "3-steps-demo"),
    (SEVEN_STEPS, "7-steps"),
    (LONG_DESC, "long-desc"),
]


def _case(diagram, name):
    elements = generate_elements(diagram)
    return LLMTestCase(
        input=(
            f"Draw a {len(diagram['steps'])}-step process diagram"
            + (f": {diagram['title']}" if diagram.get("title") else "")
        ),
        actual_output=json.dumps(elements, ensure_ascii=False),
        expected_output="Arrows bound to their nodes; no overlapping nodes.",
        context=[f"case: {name}"],
    )


@pytest.mark.parametrize("diagram,name", CASES, ids=[c[1] for c in CASES])
def test_arrows_bound(diagram, name):
    assert_test(_case(diagram, name), [ArrowBindingMetric()])


@pytest.mark.parametrize("diagram,name", CASES, ids=[c[1] for c in CASES])
def test_nodes_do_not_overlap(diagram, name):
    assert_test(_case(diagram, name), [NodeOverlapMetric()])


@pytest.mark.parametrize("diagram,name", CASES, ids=[c[1] for c in CASES])
def test_text_visible_immediately(diagram, name):
    assert_test(_case(diagram, name), [TextVisibilityMetric()])


SCENE_CASES = [(True, "with-legacy-scene"), (False, "fresh-start")]


@pytest.mark.parametrize("with_legacy,name", SCENE_CASES, ids=[c[1] for c in SCENE_CASES])
def test_scene_isolation(with_legacy, name):
    report = scene_isolation_report(with_legacy)
    test_case = LLMTestCase(
        input=f"Scene isolation ({name}): each conversation keeps its own board scene",
        actual_output=json.dumps(report, ensure_ascii=False),
        expected_output="All isolation checks true.",
        context=[f"case: {name}"],
    )
    assert_test(test_case, [SceneIsolationMetric()])
