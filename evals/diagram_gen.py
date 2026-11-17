"""Generate Excalidraw elements with the REAL code from diagram.html.

Reads <repo>/diagram.html (read-only, never edited) and runs its
parseDiagramSteps / buildProcessElements in Node with a stubbed canvas
(measureText ~= 9px per char — enough to exercise wrapping and positions;
the real browser measures with the system font).
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path


def _find_diagram_html() -> Path:
    # The evals live in <repo>/evals/, so diagram.html is one level up.
    # Fall back to this agent's work copy when run outside a checkout.
    here = Path(__file__).resolve().parent.parent / "diagram.html"
    if here.exists():
        return here
    return Path.home() / "workspace" / "chat-diagram" / "chat" / "diagram.html"


DIAGRAM_HTML = _find_diagram_html()

_NODE_PRELUDE = """
var document = { getElementById: function() { return null; }, createElement: function() { return { getContext: function() { return { measureText: function(t) { return { width: t.length * 9 }; }, set font(v) {} }; } }; } };
var localStorage = { _d: {}, getItem: function(k) { return this._d[k] || null; }, setItem: function(k, v) { this._d[k] = String(v); }, removeItem: function(k) { delete this._d[k]; } };
var window = {};
"""


def _extract_builder(html: str) -> str:
    m = re.search(
        r"/\* ---------- ساختن عناصر فرایند.*?^function buildProcessElements\(diagram, topY\) \{.*?\n\}\n",
        html,
        re.S | re.M,
    )
    if not m:
        raise RuntimeError("buildProcessElements not found in diagram.html")
    return m.group(0)


def _extract_parser(html: str) -> str:
    m = re.search(
        r"function parseDiagramSteps\(text\) \{.*?\n  \}\n",
        html,
        re.S,
    )
    if not m:
        raise RuntimeError("parseDiagramSteps not found in diagram.html")
    return m.group(0)


def _run_node(harness: str) -> str:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False, encoding="utf-8"
    ) as f:
        f.write(harness)
        path = f.name
    res = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        raise RuntimeError(f"node failed: {res.stderr.strip()}")
    return res.stdout


def generate_elements(diagram: dict) -> list:
    """Run the real builder for one diagram and return its elements."""
    html = DIAGRAM_HTML.read_text(encoding="utf-8")
    builder = _extract_builder(html)
    harness = (
        _NODE_PRELUDE
        + builder
        + "\nconsole.log(JSON.stringify(buildProcessElements("
        + json.dumps(diagram, ensure_ascii=False)
        + ", 0)));\n"
    )
    return json.loads(_run_node(harness))


def build_from_reply(reply: str) -> list:
    """Run the real parser + builder on a model reply and return elements.

    Empty list when the reply holds no usable diagram-steps block.
    """
    html = DIAGRAM_HTML.read_text(encoding="utf-8")
    parser = _extract_parser(html)
    builder = _extract_builder(html)
    harness = (
        _NODE_PRELUDE
        + parser
        + "\n"
        + builder
        + "\nvar r = parseDiagramSteps("
        + json.dumps(reply, ensure_ascii=False)
        + ");\nvar d = r.diagrams.length ? r.diagrams[0] : null;\n"
        + "console.log(JSON.stringify(d ? buildProcessElements(d, 0) : []));\n"
    )
    return json.loads(_run_node(harness))
