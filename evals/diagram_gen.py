"""Generate Excalidraw elements with the REAL builder from diagram.html.

Reads ~/workspace/chat-diagram/chat/diagram.html (read-only, never edited)
and runs its buildProcessElements in Node with a stubbed canvas
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
    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False, encoding="utf-8"
    ) as f:
        f.write(harness)
        path = f.name
    res = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        raise RuntimeError(f"node failed: {res.stderr.strip()}")
    return json.loads(res.stdout)


def _extract_scene_block(html: str) -> str:
    m = re.search(
        r"const SCENE_KEY = .*?^function loadScene\(\) \{.*?\n\}\n",
        html,
        re.S | re.M,
    )
    if not m:
        raise RuntimeError("scene helpers not found in diagram.html")
    return m.group(0)


def scene_isolation_report(with_legacy: bool) -> dict:
    """Exercise the real per-chat scene logic from diagram.html in Node.

    Returns a dict of boolean checks: legacy scene migrates to chat-A,
    chat-B starts empty, writing B's scene leaves A untouched, keys differ.
    """
    html = DIAGRAM_HTML.read_text(encoding="utf-8")
    block = _extract_scene_block(html)
    if with_legacy:
        setup = (
            "localStorage.setItem('excalidraw.diagram.v1', JSON.stringify("
            "{elements:[{id:'old1',type:'rectangle',x:0,y:0,width:10,height:10}],appState:{}}));\n"
        )
        checks = """
currentChatId = 'chat-A';
var sceneA = loadScene();
var report = {
  legacy_migrated_to_A: !!(sceneA && sceneA.elements && sceneA.elements.length === 1),
  legacy_key_removed: localStorage.getItem('excalidraw.diagram.v1') === null,
  per_chat_key_written: localStorage.getItem('excalidraw.diagram.v1:chat-A') !== null,
  chat_A_empty_at_start: true,
  chat_B_empty: (function(){ currentChatId = 'chat-B'; return loadScene() === null; })(),
  keys_distinct: sceneKeyFor('chat-A') !== sceneKeyFor('chat-B')
};
writeScene(sceneKeyFor('chat-B'), {elements:[{id:'b1',type:'rectangle',x:0,y:0,width:5,height:5}],appState:{}});
currentChatId = 'chat-A';
var sA2 = loadScene();
report.chat_A_unchanged_after_B_write = !!(sA2 && sA2.elements && sA2.elements.length === 1 && sA2.elements[0].id === 'old1');
currentChatId = 'chat-B';
var sB2 = loadScene();
report.chat_B_has_own_scene = !!(sB2 && sB2.elements && sB2.elements.length === 1 && sB2.elements[0].id === 'b1');
console.log(JSON.stringify(report));
"""
    else:
        setup = ""
        checks = """
currentChatId = 'chat-A';
var sceneA = loadScene();
var report = {
  legacy_migrated_to_A: true,
  legacy_key_removed: true,
  per_chat_key_written: true,
  chat_A_empty_at_start: sceneA === null,
  chat_B_empty: false,
  keys_distinct: sceneKeyFor('chat-A') !== sceneKeyFor('chat-B')
};
writeScene(sceneKeyFor('chat-A'), {elements:[{id:'a1',type:'rectangle',x:0,y:0,width:5,height:5}],appState:{}});
currentChatId = 'chat-B';
report.chat_B_empty = loadScene() === null;
writeScene(sceneKeyFor('chat-B'), {elements:[{id:'b1',type:'rectangle',x:0,y:0,width:5,height:5}],appState:{}});
currentChatId = 'chat-A';
var sA2 = loadScene();
report.chat_A_unchanged_after_B_write = !!(sA2 && sA2.elements && sA2.elements.length === 1 && sA2.elements[0].id === 'a1');
currentChatId = 'chat-B';
var sB2 = loadScene();
report.chat_B_has_own_scene = !!(sB2 && sB2.elements && sB2.elements.length === 1 && sB2.elements[0].id === 'b1');
console.log(JSON.stringify(report));
"""
    driver = _NODE_PRELUDE + block + "\n" + setup + checks
    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False, encoding="utf-8"
    ) as f:
        f.write(driver)
        path = f.name
    res = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        raise RuntimeError(f"node failed: {res.stderr.strip()}")
    return json.loads(res.stdout)
