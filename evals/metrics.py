"""Custom DeepEval metrics for the chat->board diagram feature.

No LLM is used: all metrics measure the generated Excalidraw elements
deterministically. The test case's actual_output must be the JSON list of
elements produced by buildProcessElements (see diagram_gen.py); the
faithfulness metric also reads expected_output as the JSON list of
expected steps [{title, desc}].
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
    return sorted(
        [e for e in elements if e.get("type") == "rectangle"],
        key=lambda e: e["y"],
    )


def _arrows(elements):
    return sorted(
        [e for e in elements if e.get("type") == "arrow"],
        key=lambda e: e["y"],
    )


def _overlap_area(a, b):
    ox = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
    oy = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
    return ox * oy if ox > 0 and oy > 0 else 0


def _norm(text):
    return " ".join(str(text or "").split())


def _drawn_steps(elements):
    """Steps as drawn: one (title, desc) per rectangle, from its bound text.

    The builder stores title + desc in one text element whose originalText
    is "title\\ndesc" and whose containerId is the rectangle's id.
    """
    steps = []
    for rect in _rects(elements):
        bound = [
            e
            for e in elements
            if e.get("type") == "text" and e.get("containerId") == rect["id"]
        ]
        if not bound:
            steps.append({"title": "", "desc": ""})
            continue
        raw = bound[0].get("originalText", bound[0].get("text", ""))
        title, _, desc = raw.partition("\n")
        steps.append({"title": _norm(title), "desc": _norm(desc)})
    return steps


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


class FaithfulnessMetric(BaseMetric):
    """Eval 3: does the drawn diagram say exactly what was asked?

    The expected steps come from expected_output (JSON list of
    {title, desc}); the drawn steps are read back from the elements.
    A step matches when its normalized title is equal and its desc is
    equal (when the expected desc is non-empty). Score is the F1 of
    matched steps over drawn vs expected steps, so an invented extra step
    hurts precision and a missing step hurts recall.
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
        try:
            expected = json.loads(test_case.expected_output or "[]")
        except (TypeError, ValueError):
            expected = []
        expected = [
            {"title": _norm(s.get("title")), "desc": _norm(s.get("desc"))}
            for s in expected
            if isinstance(s, dict)
        ]
        drawn = _drawn_steps(elements)
        if not expected or not drawn:
            self.score = 0.0
            self.reason = (
                f"expected {len(expected)} step(s), drew {len(drawn)} — "
                "nothing usable was drawn."
                if not drawn
                else "no expected steps given."
            )
            return self.score
        unused = list(drawn)
        matched, missing = 0, []
        for want in expected:
            hit = None
            for got in unused:
                if got["title"] == want["title"] and (
                    not want["desc"] or got["desc"] == want["desc"]
                ):
                    hit = got
                    break
            if hit is None:
                missing.append(want["title"] or "(untitled)")
            else:
                matched += 1
                unused.remove(hit)
        extra = [g["title"] for g in unused]
        precision = matched / len(drawn)
        recall = matched / len(expected)
        self.score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        self.reason = (
            f"F1={self.score:.2f} ({matched}/{len(expected)} expected steps drawn). "
            + (
                "Diagram says exactly what was asked."
                if self.score == 1.0
                else "Missing: " + ", ".join(missing) + "; " + "Extra: " + ", ".join(extra)
                if missing or extra
                else ""
            )
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Faithfulness"


class ArrowGeometryMetric(BaseMetric):
    """Eval 4: do the arrows visually touch the rectangles they connect?

    Even unbound, an arrow must start at the bottom edge of the upper
    rectangle and end at the top edge of the lower one (the builder leaves
    a 6px gap on each side), run vertically through the rectangles'
    center, and carry a downward arrowhead. Score = fraction of arrows
    passing all checks.
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
        problems, ok_count = [], 0
        for idx, arrow in enumerate(arrows):
            if idx + 1 >= len(rects):
                problems.append(f"arrow {idx}: no next rectangle")
                continue
            upper, lower = rects[idx], rects[idx + 1]
            center_x = upper["x"] + upper["width"] / 2
            points = arrow.get("points") or []
            failed = []
            if abs(arrow["x"] - center_x) > 0.5:
                failed.append(f"x={arrow['x']} (want center {center_x})")
            if abs(arrow["y"] - (upper["y"] + upper["height"] + 6)) > 0.5:
                failed.append(
                    f"y={arrow['y']} (want bottom of upper rect + 6 = "
                    f"{upper['y'] + upper['height'] + 6})"
                )
            if abs((arrow["y"] + arrow["height"]) - (lower["y"] - 6)) > 0.5:
                failed.append(
                    f"end y={arrow['y'] + arrow['height']} (want top of lower "
                    f"rect - 6 = {lower['y'] - 6})"
                )
            if not (
                arrow.get("width") == 0
                and len(points) == 2
                and points[0] == [0, 0]
                and points[1][0] == 0
                and points[1][1] == arrow["height"]
            ):
                failed.append(f"not a vertical segment (points={points})")
            if arrow.get("endArrowhead") != "arrow":
                failed.append(f"endArrowhead={arrow.get('endArrowhead')!r}")
            if failed:
                problems.append(f"arrow {idx}: " + "; ".join(failed))
            else:
                ok_count += 1
        self.score = ok_count / len(arrows)
        self.reason = (
            f"{ok_count}/{len(arrows)} arrows touch both rectangles. "
            + (
                "All arrows visually connect their nodes."
                if ok_count == len(arrows)
                else "Bad: " + "; ".join(problems)
            )
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Arrow Geometry"


class LayoutOrderMetric(BaseMetric):
    """Eval 5: are the steps stacked top-to-bottom, aligned, evenly spaced?

    Checks: rectangles in step order (y strictly increasing), all share
    the same x and width, the gaps between consecutive rectangles are all
    equal, and a title text (if any) is centered above the first rectangle.
    Score = fraction of checks passed.
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
        if len(rects) < 2:
            self.score, self.reason = 0.0, "fewer than 2 rectangles to order"
            return self.score
        checks, failed = [], []
        order_ok = all(rects[i]["y"] < rects[i + 1]["y"] for i in range(len(rects) - 1))
        checks.append(("step order top-to-bottom", order_ok))
        same_x = len({r["x"] for r in rects}) == 1
        checks.append(("same x (aligned)", same_x))
        same_w = len({r["width"] for r in rects}) == 1
        checks.append(("same width", same_w))
        gaps = [
            rects[i + 1]["y"] - (rects[i]["y"] + rects[i]["height"])
            for i in range(len(rects) - 1)
        ]
        equal_gaps = all(abs(g - gaps[0]) <= 0.5 for g in gaps)
        checks.append((f"equal gaps ({gaps[0]:.0f}px)", equal_gaps))
        titles = [
            e
            for e in elements
            if e.get("type") == "text" and e.get("containerId") is None
        ]
        if titles:
            t = titles[0]
            centered = abs((t["x"] + t["width"] / 2) - (rects[0]["x"] + rects[0]["width"] / 2)) <= 0.5
            above = t["y"] + t["height"] <= rects[0]["y"]
            checks.append(("title centered above first rect", centered and above))
        failed = [name for name, ok in checks if not ok]
        self.score = (len(checks) - len(failed)) / len(checks)
        self.reason = (
            f"{len(checks) - len(failed)}/{len(checks)} layout checks passed. "
            + (
                "Steps are ordered, aligned and evenly spaced."
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
        return "Layout Order"


class TextFitMetric(BaseMetric):
    """Eval 6: does every text fit inside its own box?

    Each step's bound text must lie fully inside its rectangle, and a
    standalone title text must not be wider than the 360px column.
    Score = fraction of texts that fit.
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
        texts = [e for e in elements if e.get("type") == "text"]
        if not texts:
            self.score, self.reason = 0.0, "no text elements found"
            return self.score
        bad, ok_count = [], 0
        for idx, text in enumerate(texts):
            if text.get("containerId"):
                rect = next(
                    (r for r in rects if r["id"] == text["containerId"]), None
                )
                if rect is None:
                    bad.append(f"text {idx}: container rect missing")
                    continue
                fits = (
                    text["x"] >= rect["x"] - 0.5
                    and text["x"] + text["width"] <= rect["x"] + rect["width"] + 0.5
                    and text["y"] >= rect["y"] - 0.5
                    and text["y"] + text["height"] <= rect["y"] + rect["height"] + 0.5
                )
                if not fits:
                    bad.append(
                        f"text {idx} overflows its rect "
                        f"(text y={text['y']}..{text['y'] + text['height']}, "
                        f"rect y={rect['y']}..{rect['y'] + rect['height']})"
                    )
                else:
                    ok_count += 1
            else:
                if text["width"] <= 360 + 0.5:
                    ok_count += 1
                else:
                    bad.append(f"title text {idx} wider than the 360px column")
        self.score = ok_count / len(texts)
        self.reason = (
            f"{ok_count}/{len(texts)} texts fit inside their boxes. "
            + ("No text overflows its box." if ok_count == len(texts) else "Overflowing: " + "; ".join(bad))
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Text Fit"
