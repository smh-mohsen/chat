"""DeepEval tests for the chat->board diagram feature.

Eval 1 (arrow binding): do arrows stay attached when a node is dragged?
Eval 2 (node overlap): do step nodes overlap each other?
Eval 3 (faithfulness): does the drawn diagram say exactly what was asked?
Eval 4 (arrow geometry): do arrows visually touch the rectangles?
Eval 5 (layout order): are steps stacked top-to-bottom, aligned, evenly spaced?
Eval 6 (text fit): does every text fit inside its own box?

The elements under test come from the REAL parser + builder in diagram.html
(see diagram_gen.py) — the evals measure the code's output, they do not
call an LLM and need no API key.

Run:  python3 -m venv .venv && .venv/bin/pip install -r requirements-evals.txt
       .venv/bin/pytest test_diagram.py -v
"""

import json

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from diagram_gen import build_from_reply, generate_elements
from metrics import (
    ArrowBindingMetric,
    ArrowGeometryMetric,
    FaithfulnessMetric,
    LayoutOrderMetric,
    NodeOverlapMetric,
    TextFitMetric,
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

# (name, user request, model reply, expected steps)
FAITH_CASES = [
    (
        "correct-2-steps",
        "نمودار شروع و پایان بکش",
        "```diagram-steps\n{\"title\": \"شروع تا پایان\", \"steps\": [{\"title\": \"شروع\", \"desc\": \"نقطه‌ی آغاز\"}, {\"title\": \"پایان\", \"desc\": \"نقطه‌ی پایان\"}]}\n```",
        [
            {"title": "شروع", "desc": "نقطه‌ی آغاز"},
            {"title": "پایان", "desc": "نقطه‌ی پایان"},
        ],
    ),
    (
        "invented-step",
        "نمودار شروع و پایان بکش",
        "```diagram-steps\n{\"title\": \"شروع تا پایان\", \"steps\": [{\"title\": \"شروع\", \"desc\": \"نقطه‌ی آغاز\"}, {\"title\": \"بازبینی\", \"desc\": \"بررسی میانی\"}, {\"title\": \"پایان\", \"desc\": \"نقطه‌ی پایان\"}]}\n```",
        [
            {"title": "شروع", "desc": "نقطه‌ی آغاز"},
            {"title": "پایان", "desc": "نقطه‌ی پایان"},
        ],
    ),
    (
        "missing-step",
        "نمودار شروع و پایان بکش",
        "```diagram-steps\n{\"title\": \"شروع تا پایان\", \"steps\": [{\"title\": \"شروع\", \"desc\": \"نقطه‌ی آغاز\"}]}\n```",
        [
            {"title": "شروع", "desc": "نقطه‌ی آغاز"},
            {"title": "پایان", "desc": "نقطه‌ی پایان"},
        ],
    ),
    (
        "wrong-title",
        "نمودار شروع و پایان بکش",
        "```diagram-steps\n{\"title\": \"شروع تا پایان\", \"steps\": [{\"title\": \"شروع\", \"desc\": \"نقطه‌ی آغاز\"}, {\"title\": \"اتمام\", \"desc\": \"نقطه‌ی پایان\"}]}\n```",
        [
            {"title": "شروع", "desc": "نقطه‌ی آغاز"},
            {"title": "پایان", "desc": "نقطه‌ی پایان"},
        ],
    ),
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


@pytest.mark.parametrize(
    "name,user_request,reply,expected", FAITH_CASES, ids=[c[0] for c in FAITH_CASES]
)
def test_faithful_to_request(name, user_request, reply, expected):
    elements = build_from_reply(reply)
    test_case = LLMTestCase(
        input=user_request,
        actual_output=json.dumps(elements, ensure_ascii=False),
        expected_output=json.dumps(expected, ensure_ascii=False),
        context=[f"case: {name}"],
    )
    assert_test(test_case, [FaithfulnessMetric()])


@pytest.mark.parametrize("diagram,name", CASES, ids=[c[1] for c in CASES])
def test_arrows_touch_rectangles(diagram, name):
    assert_test(_case(diagram, name), [ArrowGeometryMetric()])


@pytest.mark.parametrize("diagram,name", CASES, ids=[c[1] for c in CASES])
def test_layout_order(diagram, name):
    assert_test(_case(diagram, name), [LayoutOrderMetric()])


@pytest.mark.parametrize("diagram,name", CASES, ids=[c[1] for c in CASES])
def test_text_fits_in_box(diagram, name):
    assert_test(_case(diagram, name), [TextFitMetric()])
