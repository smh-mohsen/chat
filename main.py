import json
import os

from dotenv import load_dotenv

load_dotenv()  # باید قبل از ساختن OpenAI() و TavilyClient() اجرا شود

from fastapi import FastAPI
from fastapi.responses import FileResponse
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


# ---------- API ----------
class ChatReq(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]


@app.post("/chat")
def chat(req: ChatReq):
    msgs = [{"role": "system", "content": SYSTEM}] + req.messages
    return {"reply": run_agent(msgs)}


@app.get("/")
def index():
    return FileResponse("index.html")