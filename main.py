import json
import os

from dotenv import load_dotenv

load_dotenv()  # باید قبل از ساختن OpenAI() و TavilyClient() اجرا شود

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient

MODEL = "gpt-4o-mini"  # هر مدلی که الان استفاده می‌کنید
SYSTEM = (
    "You are a helpful assistant with access to the user's local computer. "
    "Use the web_search tool when you need current or factual information. "
    "Use list_dir to explore folders, read_file to read a file before "
    "changing it, edit_file for small changes to an existing file, and "
    "write_file to create a new file or rewrite one completely. Always use "
    "absolute paths, prefer edit_file over write_file for existing files, "
    "and after each change tell the user exactly which path you touched. "
    "Answer in the user's language and mention sources when you used search."
)

client = OpenAI()
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
app = FastAPI()


# ---------- tools ----------
def web_search(query: str, max_results: int = 5) -> str:
    res = tavily.search(query=query, max_results=max_results)
    items = [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in res["results"]
    ]
    return json.dumps(items, ensure_ascii=False)


# ---------- filesystem tools ----------
# ابزارهای فایل فقط داخل این ریشهها کار میکنند (با کاما جدا شوند).
# پیشفرض پوشهی خانه است؛ برای دسترسی به کل لپتاپ در .env بگذار: FILES_ROOTS=/
FILES_ROOTS = [
    os.path.realpath(os.path.expanduser(p.strip()))
    for p in os.environ.get("FILES_ROOTS", "~").split(",")
    if p.strip()
]
MAX_READ_LINES = 2000
MAX_READ_CHARS = 47000


def _resolve(path: str) -> str:
    """مسیر را مطلق میکند و مطمئن میشود داخل یکی از ریشههای مجاز است."""
    if not path or not str(path).strip():
        raise ValueError("path لازم است.")
    full = os.path.realpath(os.path.expanduser(str(path).strip()))
    for root in FILES_ROOTS:
        if full == root or full.startswith(root.rstrip(os.sep) + os.sep):
            return full
    raise ValueError(
        f"دسترسی به این مسیر مجاز نیست: {full} "
        f"(ریشههای مجاز: {', '.join(FILES_ROOTS)})"
    )


def list_dir(path: str = "~") -> str:
    """نام پوشهها و فایلهای یک مسیر را برمیگرداند."""
    full = _resolve(path)
    if not os.path.isdir(full):
        raise ValueError(f"پوشه پیدا نشد: {full}")
    lines = []
    for name in sorted(os.listdir(full)):
        child = os.path.join(full, name)
        if os.path.isdir(child):
            lines.append(f"{name}/")
        else:
            try:
                lines.append(f"{name} ({os.path.getsize(child)} bytes)")
            except OSError:
                lines.append(name)
    body = "\n".join(lines) if lines else "(پوشه خالی است)"
    return f"{full}\n{body}"


def read_file(
    path: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    """محتوای یک فایل متنی را با شمارهی خط برمیگرداند."""
    full = _resolve(path)
    if os.path.isdir(full):
        raise ValueError(f"این مسیر یک پوشه است؛ از list_dir استفاده کن: {full}")
    if not os.path.isfile(full):
        raise ValueError(f"فایل پیدا نشد: {full}")
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    total = len(lines)
    if total == 0:
        return f"{full} (فایل خالی است)"
    start = max(1, start_line or 1)
    end = total if not end_line else min(total, end_line)
    if start > end:
        raise ValueError(f"بازهی نامعتبر: {start}-{end} (فایل {total} خط دارد).")

    out: list[str] = []
    used = 0
    truncated = False
    for i in range(start, end + 1):
        row = f"{i:>5}\t{lines[i - 1]}"
        if len(out) >= MAX_READ_LINES or used + len(row) > MAX_READ_CHARS:
            truncated = True
            break
        out.append(row)
        used += len(row) + 1

    last = start + len(out) - 1
    header = f"{full} (خطوط {start}-{last} از {total})"
    if truncated:
        out.append(f"... [ادامه دارد: از start_line={last + 1} ادامه بده]")
    return header + "\n" + "\n".join(out)


def write_file(path: str, content: str) -> str:
    """فایل جدید میسازد یا فایل موجود را کاملاً بازنویسی میکند."""
    if content is None:
        raise ValueError("content لازم است.")
    full = _resolve(path)
    if os.path.isdir(full):
        raise ValueError(f"این مسیر یک پوشه است: {full}")
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    existed = os.path.exists(full)
    with open(full, "w", encoding="utf-8") as f:
        f.write(str(content))
    verb = "بازنویسی شد" if existed else "ساخته شد"
    return f"{full} {verb} ({len(str(content))} کاراکتر)"


def edit_file(
    path: str, old_text: str, new_text: str, replace_all: bool = False
) -> str:
    """یک تکهی متن را در فایل موجود جایگزین میکند."""
    if not old_text:
        raise ValueError("old_text نباید خالی باشد.")
    full = _resolve(path)
    if not os.path.isfile(full):
        raise ValueError(f"فایل پیدا نشد: {full}")
    with open(full, "r", encoding="utf-8") as f:
        data = f.read()
    count = data.count(old_text)
    if count == 0:
        raise ValueError(
            "متن old_text در فایل پیدا نشد؛ اول read_file بزن و عیناً همان متن را بفرست."
        )
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_text {count} بار در فایل تکرار شده؛ آن را دقیقتر/طولانیتر کن "
            "یا replace_all=true بده."
        )
    updated = (
        data.replace(old_text, new_text)
        if replace_all
        else data.replace(old_text, new_text, 1)
    )
    with open(full, "w", encoding="utf-8") as f:
        f.write(updated)
    return f"{full} ویرایش شد ({count if replace_all else 1} جایگذاری)."


_STATUS_ICONS = {
    "list_dir": "📂 مرور پوشه",
    "read_file": "📖 خواندن فایل",
    "write_file": "📝 نوشتن فایل",
    "edit_file": "✏️ ویرایش فایل",
}


def _file_status(name: str, path: str | None) -> str:
    """یک خط وضعیت خوانا برای ابزارهای فایل میسازد."""
    label = _STATUS_ICONS.get(name, name)
    return f"{label}: {path}" if path else label


TOOL_FUNCS = {
    "web_search": web_search,
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List the files and folders inside a directory on the user's computer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "absolute directory path (default: home folder)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file from the user's computer. Returns the content "
                "with line numbers; use start_line/end_line for large files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute file path"},
                    "start_line": {
                        "type": "integer",
                        "description": "first line to read (1-based)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "last line to read, inclusive",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new file or overwrite an existing file with the given "
                "content. Missing parent folders are created automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute file path"},
                    "content": {"type": "string", "description": "full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact piece of text inside an existing file. old_text "
                "must match the current content exactly and appear only once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute file path"},
                    "old_text": {
                        "type": "string",
                        "description": "exact existing text to replace",
                    },
                    "new_text": {"type": "string", "description": "replacement text"},
                    "replace_all": {
                        "type": "boolean",
                        "default": False,
                        "description": "replace every occurrence instead of the first",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]


# ---------- agent loop ----------
def run_agent(messages: list[dict], max_steps: int = 5) -> str:
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append(msg.model_dump(exclude_none=True))

        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments)
                result = TOOL_FUNCS[call.function.name](**args)
            except Exception as e:
                result = f"tool error: {e}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    return "تعداد مراحل مجاز تمام شد."


# ---------- streaming agent ----------
def sse(payload: dict) -> str:
    """یک رویداد SSE با بدنه‌ی JSON می‌سازد."""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def stream_agent(messages: list[dict], max_steps: int = 5):
    """مانند run_agent ولی توکن‌ها را به‌صورت زنده (SSE) بیرون می‌دهد.

    رویدادها:
      {"type": "status", "text": "..."}     وضعیت فعلی (جست‌وجو، نوشتن پاسخ)
      {"type": "delta",  "text": "..."}     تکه‌ی تازه‌ی پاسخ
      {"type": "restart"}                   متن قبلی پاسخ نهایی نبود؛ از نو بنویس
      {"type": "error",  "message": "..."}  خطا
      {"type": "done"}                      پایان موفق
    """
    try:
        for _ in range(max_steps):
            stream = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, stream=True
            )
            calls: dict[int, dict] = {}
            text = ""
            try:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue
                    if delta.content:
                        text += delta.content
                        yield sse({"type": "delta", "text": delta.content})
                    for call in delta.tool_calls or []:
                        slot = calls.setdefault(
                            call.index or 0, {"id": "", "name": "", "arguments": ""}
                        )
                        if call.id:
                            slot["id"] = call.id
                        if call.function is not None:
                            if call.function.name:
                                slot["name"] += call.function.name
                            if call.function.arguments:
                                slot["arguments"] += call.function.arguments
            finally:
                stream.close()  # اگر کلاینت قطع شد، اتصال OpenAI هم بسته شود

            if not calls:
                yield sse({"type": "done"})
                return

            # مدل ابزار خواست؛ متنی که تا اینجا فرستادیم پاسخ نهایی نبود
            if text:
                yield sse({"type": "restart"})

            ordered = [calls[i] for i in sorted(calls)]
            for i, call in enumerate(ordered):  # بعضی سرورها id نفرستند؛ tool_call_id لازم است
                if not call["id"]:
                    call["id"] = f"call_{i}"
            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": c["arguments"] or "{}",
                            },
                        }
                        for c in ordered
                    ],
                }
            )

            for call in ordered:
                try:
                    args = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                query = args.get("query")
                name = call["name"]
                target = args.get("path")
                if name == "web_search" and query:
                    yield sse({"type": "status", "text": f"🔎 جست‌وجوی وب: {query}"})
                elif name in _STATUS_ICONS:
                    yield sse({"type": "status", "text": _file_status(name, target)})
                else:
                    yield sse({"type": "status", "text": f"🔧 اجرای ابزار {call['name']}"})
                try:
                    result = TOOL_FUNCS[call["name"]](**args)
                except Exception as e:
                    result = f"tool error: {e}"
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result}
                )
            yield sse({"type": "status", "text": "✍️ در حال نوشتن پاسخ…"})

        yield sse({"type": "error", "message": "تعداد مراحل مجاز تمام شد."})
        yield sse({"type": "done"})
    except Exception as e:  # به‌جای استریم خراب، خطا را به کاربر نشان بده
        yield sse({"type": "error", "message": f"خطای سرور: {e}"})
        yield sse({"type": "done"})


# ---------- API ----------
class ChatReq(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]


@app.post("/chat")
def chat(req: ChatReq):
    msgs = [{"role": "system", "content": SYSTEM}] + req.messages
    return {"reply": run_agent(msgs)}


@app.post("/chat/stream")
def chat_stream(req: ChatReq):
    """همان /chat ولی پاسخ را تکه‌تکه (Server-Sent Events) می‌فرستد."""
    msgs = [{"role": "system", "content": SYSTEM}] + req.messages
    return StreamingResponse(
        stream_agent(msgs),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # پروکسی‌ها بافر نکنند
        },
    )


@app.get("/")
def index():
    return FileResponse("index.html")