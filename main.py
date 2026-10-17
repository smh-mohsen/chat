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
    "You are a helpful assistant. Use the web_search tool when you need "
    "current or factual information. Answer in the user's language and "
    "mention sources when you used search."
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


TOOL_FUNCS = {"web_search": web_search}

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
    }
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
                if call["name"] == "web_search" and query:
                    yield sse({"type": "status", "text": f"🔎 جست‌وجوی وب: {query}"})
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