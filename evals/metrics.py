"""Custom DeepEval metrics for the chat->board diagram feature.

No LLM is used: both metrics measure the generated Excalidraw elements
deterministically. The test case's actual_output must be the JSON list of
elements produced by buildProcessElements (see diagram_gen.py).
"""

import json

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def _load_elements(test_case):
    try:
        elements = json.loads(test_case.actual_output)
    except (TypeError, ValueError):
        return None
    return elements if isinstance(elements, list) else None


def _rects(elements):
    return [e for e in elements if e.get("type") == "rectangle"]


def _arrows(elements):
    return [e for e in elements if e.get("type") == "arrow"]


def _overlap_area(a, b):
    ox = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
    oy = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
    return ox * oy if ox > 0 and oy > 0 else 0


class ArrowBindingMetric(BaseMetric):
    """Eval 1: is every arrow bound to the two rectangles it connects?

    In Excalidraw an arrow follows a node when dragged only if it carries
    startBinding/endBinding pointing at that node (elementId). The arrow
    between step i and step i+1 must bind rect[i] (start) and rect[i+1] (end).
    Score = fraction of arrows fully bound on both ends.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase):
        elements = _load_elements(test_case)
        if elements is None:
            self.score, self.reason = (
                0.0,
                "actual_output is not a JSON list of elements",
            )
            return self.score
        rects = _rects(elements)
        arrows = _arrows(elements)
        if not arrows:
            self.score, self.reason = (
                0.0,
                "no arrows found in the generated elements",
            )
            return self.score
        problems, bound = [], 0
        for idx, arrow in enumerate(arrows):
            start = arrow.get("startBinding") or {}
            end = arrow.get("endBinding") or {}
            want_start = rects[idx]["id"] if idx < len(rects) else None
            want_end = rects[idx + 1]["id"] if idx + 1 < len(rects) else None
            ok_start = start.get("elementId") == want_start
            ok_end = end.get("elementId") == want_end
            if ok_start and ok_end:
                bound += 1
            else:
                problems.append(
                    f"arrow {idx} (y={arrow.get('y')}): "
                    f"startBinding={start.get('elementId')!r} (want {want_start!r}), "
                    f"endBinding={end.get('elementId')!r} (want {want_end!r})"
                )
        self.score = bound / len(arrows)
        if bound == len(arrows):
            self.reason = (
                f"{bound}/{len(arrows)} arrows bound on both ends. "
                "All arrows follow their nodes when dragged."
            )
        else:
            self.reason = (
                f"{bound}/{len(arrows)} arrows bound on both ends. "
                "Unbound: " + "; ".join(problems)
            )
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Arrow Binding"


class NodeOverlapMetric(BaseMetric):
    """Eval 2: do any step nodes overlap each other?

    Checks every pair of rectangles for a positive-area AABB intersection.
    Score is 1.0 when no pair overlaps, else 0.0.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase):
        elements = _load_elements(test_case)
        if elements is None:
            self.score, self.reason = (
                0.0,
                "actual_output is not a JSON list of elements",
            )
            return self.score
        rects = _rects(elements)
        conflicts = []
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                area = _overlap_area(rects[i], rects[j])
                if area > 0:
                    conflicts.append(
                        f"rect {i} (y={rects[i]['y']}..{rects[i]['y'] + rects[i]['height']}) "
                        f"overlaps rect {j} (y={rects[j]['y']}..{rects[j]['y'] + rects[j]['height']}) "
                        f"by {area:.0f}px²"
                    )
        if conflicts:
            self.score = 0.0
            self.reason = (
                f"{len(conflicts)} overlapping node pair(s): " + "; ".join(conflicts)
            )
        else:
            self.score = 1.0
            self.reason = f"No overlap among {len(rects)} nodes."
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Node Overlap"


class TextVisibilityMetric(BaseMetric):
    """Eval 3: do drawn texts render immediately, without a page refresh?

    Excalidraw 0.18 needs lineHeight (default 1.25) and autoResize on text
    elements and measures them with a canvas; without lineHeight the height
    is NaN and the text stays invisible until a reload rebuilds it. A text
    is valid when lineHeight == 1.25, autoResize is true, width/height are
    finite and > 0, and no stale 'baseline' field is present (not a 0.18
    text field). Score = fraction of valid texts.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase):
        import math

        elements = _load_elements(test_case)
        if elements is None:
            self.score, self.reason = (
                0.0,
                "actual_output is not a JSON list of elements",
            )
            return self.score
        texts = [e for e in elements if e.get("type") == "text"]
        if not texts:
            self.score, self.reason = 0.0, "no text elements found"
            return self.score
        bad = []
        for idx, text in enumerate(texts):
            problems = []
            if text.get("lineHeight") != 1.25:
                problems.append(f"lineHeight={text.get('lineHeight')!r} (want 1.25)")
            if text.get("autoResize") is not True:
                problems.append(f"autoResize={text.get('autoResize')!r} (want True)")
            if "baseline" in text:
                problems.append("has stale 'baseline' field")
            for field in ("width", "height"):
                value = text.get(field)
                if not (
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0
                ):
                    problems.append(f"{field}={value!r} (want finite > 0)")
            if problems:
                bad.append(f"text {idx}: " + ", ".join(problems))
        ok = len(texts) - len(bad)
        self.score = ok / len(texts)
        self.reason = (
            f"{ok}/{len(texts)} texts render immediately. "
            + ("All texts visible without refresh." if ok == len(texts) else "Invalid: " + "; ".join(bad))
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Text Visibility"


class SceneIsolationMetric(BaseMetric):
    """Eval 4: does each conversation keep its own board scene?

    The test case's actual_output is the JSON report from
    diagram_gen.scene_isolation_report: boolean checks for legacy migration,
    empty new chats, per-chat keys, and scenes surviving each other's writes.
    Score = fraction of checks that are true.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase):
        try:
            report = json.loads(test_case.actual_output)
        except (TypeError, ValueError):
            report = None
        if not isinstance(report, dict) or not report:
            self.score, self.reason = (
                0.0,
                "actual_output is not a JSON object report",
            )
            return self.score
        failed = [key for key, value in report.items() if value is not True]
        ok = len(report) - len(failed)
        self.score = ok / len(report)
        self.reason = (
            f"{ok}/{len(report)} isolation checks true. "
            + (
                "Each conversation keeps its own scene."
                if not failed
                else "Failed: " + ", ".join(failed)
            )
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Scene Isolation"
