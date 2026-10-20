import json
import os
import time

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


# ---------- token usage & cost ----------
# قیمتها بر اساس هر یک میلیون توکن و به دلار آمریکا هستند.
# برای افزودن مدل جدید فقط یک سطر به این جدول اضافه کن.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60, "context": 128_000},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00, "context": 128_000},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00, "context": 1_047_576},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60, "context": 1_047_576},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40, "context": 1_047_576},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40, "context": 200_000},
    "gpt-4-turbo": {"input": 10.00, "cached_input": 10.00, "output": 30.00, "context": 128_000},
    "gpt-3.5-turbo": {"input": 0.50, "cached_input": 0.50, "output": 1.50, "context": 16_385},
}
DEFAULT_PRICING = PRICING["gpt-4o-mini"]
PER_TOKENS = 1_000_000

# اگر سرور سازگار با OpenAI پارامتر stream_options را نپذیرد، خودکار یکبار
# دیگر بدون آن تلاش میکنیم. برای خاموشکردن کامل: STREAM_USAGE=0 در .env
STREAM_USAGE = os.environ.get("STREAM_USAGE", "1") != "0"


def _env_float(name: str, default: float) -> float:
    """عدد اعشاری را از محیط میخواند و در صورت مقدار نامعتبر، پیشفرض میدهد."""
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


# تأخیر کوتاه (به ثانیه) بین تکههای متن تا پاسخ نرم و یکنواخت نمایش داده شود؛
# مدل گاهی چند تکه را با هم میفرستد و همین باعث پرش متن میشود.
# برای خاموشکردن: STREAM_DELAY=0 در .env
STREAM_DELAY = _env_float("STREAM_DELAY", 0.01)


def _price_for(model: str) -> dict:
    """قیمت مدل را پیدا میکند (اول تطابق دقیق، بعد تطابق پیشوندی)."""
    name = (model or "").strip().lower()
    if name in PRICING:
        return PRICING[name]
    for key, value in PRICING.items():
        if name.startswith(key):
            return value
    return DEFAULT_PRICING


def _approx_tokens(value) -> int:
    """تخمین تقریبی توکن (≈۴ کاراکتر برای هر توکن) وقتی سرور usage نمیفرستد."""
    if value is None:
        return 0
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    text = text.strip()
    return max(1, len(text) // 4) if text else 0


class UsageTracker:
    """مصرف توکن و هزینهی یک پاسخ را در چند مرحله (tool-calling) جمع میکند."""

    def __init__(self, model: str = MODEL):
        self.model = model
        self.pricing = _price_for(model)
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.total_tokens = 0
        self.context_peak = 0
        self.steps = 0
        self.estimated_steps = 0

    def add(self, usage) -> None:
        """آمار واقعی یک فراخوانی مدل را اضافه میکند."""
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        self.input_tokens += prompt
        self.output_tokens += completion
        self.cached_tokens += min(cached, prompt)  # cached داخل prompt_tokens است
        self.total_tokens += total
        self.context_peak = max(self.context_peak, total)
        self.steps += 1

    def add_estimate(self, messages: list[dict], output_text: str) -> None:
        """وقتی سرور usage برنمیگرداند، مصرف را تقریبی حدس میزند."""
        parts = [m.get("content") for m in messages]
        parts += [m.get("tool_calls") for m in messages if m.get("tool_calls")]
        prompt = sum(_approx_tokens(p) for p in parts if p)
        prompt += _approx_tokens(TOOLS)  # تعریف ابزارها هم توکن مصرف میکند
        completion = _approx_tokens(output_text)
        total = prompt + completion
        self.input_tokens += prompt
        self.output_tokens += completion
        self.total_tokens += total
        self.context_peak = max(self.context_peak, total)
        self.steps += 1
        self.estimated_steps += 1

    def to_dict(self) -> dict:
        """خروجی JSON-friendly برای فرستادن به رابط کاربری."""
        price = self.pricing
        fresh_input = max(0, self.input_tokens - self.cached_tokens)
        cost_input = fresh_input / PER_TOKENS * price["input"]
        cost_cached = self.cached_tokens / PER_TOKENS * price["cached_input"]
        cost_output = self.output_tokens / PER_TOKENS * price["output"]
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "context_used": self.context_peak,
            "context_limit": price["context"],
            "steps": self.steps,
            "estimated": self.estimated_steps > 0,
            "estimated_steps": self.estimated_steps,
            "cost_input_usd": round(cost_input, 10),
            "cost_cached_usd": round(cost_cached, 10),
            "cost_output_usd": round(cost_output, 10),
            "cost_usd": round(cost_input + cost_cached + cost_output, 10),
            "pricing": {
                "input": price["input"],
                "cached_input": price["cached_input"],
                "output": price["output"],
                "unit": "USD per 1M tokens",
            },
        }


# ---------- agent loop ----------
def run_agent(messages: list[dict], max_steps: int = 5) -> dict:
    """پاسخ کامل را میسازد و مصرف توکن/هزینهی آن را هم برمیگرداند."""
    tracker = UsageTracker()
    for _ in range(max_steps):
        steps_before = tracker.steps
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        tracker.add(getattr(resp, "usage", None))
        msg = resp.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""
            if tracker.steps == steps_before:  # سرور usage نداد
                tracker.add_estimate(messages, content)
            return {"reply": content, "usage": tracker.to_dict()}

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

    return {"reply": "تعداد مراحل مجاز تمام شد.", "usage": tracker.to_dict()}


# ---------- streaming agent ----------
def sse(payload: dict) -> str:
    """یک رویداد SSE با بدنه‌ی JSON می‌سازد."""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _stream_chunks(messages: list[dict]):
    """تکههای استریم مدل را تحویل میدهد (و در پایان chunk مربوط به usage را).

    اگر سرور پارامتر stream_options را نپذیرد، یکبار دیگر بدون آن تلاش
    میکند؛ بهشرط آنکه هنوز چیزی به کاربر فرستاده نشده باشد.
    """
    attempts: list[dict] = []
    if STREAM_USAGE:
        attempts.append({"stream_options": {"include_usage": True}})
    attempts.append({})

    last_error: Exception | None = None
    for extra in attempts:
        try:
            stream = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, stream=True, **extra
            )
        except Exception as e:  # خود درخواست رد شد
            last_error = e
            continue
        sent_any = False
        try:
            for chunk in stream:
                sent_any = True
                yield chunk
            return
        except Exception as e:
            if sent_any:  # وسط استریم خراب شد؛ دیگر دوباره تلاش نکن
                raise
            last_error = e
        finally:
            stream.close()  # اگر کلاینت قطع شد، اتصال OpenAI هم بسته شود
    if last_error is not None:
        raise last_error
    raise RuntimeError("استریم باز نشد.")


def stream_agent(messages: list[dict], max_steps: int = 5):
    """مانند run_agent ولی توکن‌ها را به‌صورت زنده (SSE) بیرون می‌دهد.

    رویدادها:
      {"type": "status", "text": "..."}     وضعیت فعلی (جست‌وجو، نوشتن پاسخ)
      {"type": "delta",  "text": "..."}     تکه‌ی تازه‌ی پاسخ
      {"type": "restart"}                   متن قبلی پاسخ نهایی نبود؛ از نو بنویس
      {"type": "usage",  "usage": {...}}     مصرف توکن و هزینه این پاسخ
      {"type": "error",  "message": "..."}  خطا
      {"type": "done"}                      پایان موفق

    بین تکههای متن یک تأخیر کوتاه (STREAM_DELAY) میگذاریم تا نمایش نرم باشد.
    """
    tracker = UsageTracker()
    try:
        for _ in range(max_steps):
            steps_before = tracker.steps
            calls: dict[int, dict] = {}
            text = ""
            for chunk in _stream_chunks(messages):
                if getattr(chunk, "usage", None):
                    tracker.add(chunk.usage)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    text += delta.content
                    yield sse({"type": "delta", "text": delta.content})
                    if STREAM_DELAY:
                        # چند میلیثانیه صبر تا مرورگر تکهها را یکییکی و نرم بنویسد
                        time.sleep(STREAM_DELAY)
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

            if tracker.steps == steps_before:
                tracker.add_estimate(messages, text)

            if not calls:
                yield sse({"type": "usage", "usage": tracker.to_dict()})
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

        yield sse({"type": "usage", "usage": tracker.to_dict()})
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
    """پاسخ یکجا: {"reply": "...", "usage": {توکن/هزینه}}"""
    msgs = [{"role": "system", "content": SYSTEM}] + req.messages
    return run_agent(msgs)


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