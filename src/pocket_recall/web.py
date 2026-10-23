"""Local web UI for trying a memory store: browse sessions, run recall, add conversations.

Run: pocket-recall web [--dir ~/.pocket-recall] [--port 8765]. Every *.db file in --dir is selectable,
so demo stores (e.g. imported WildChat users) stay apart from your own memory.db.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from .store import MemoryStore, normalize_time

STATIC = Path(__file__).parent / "web" / "index.html"


class Stores:
    def __init__(self, directory: Path):
        self.dir = directory.expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.open: dict[str, MemoryStore] = {}

    def names(self) -> list[str]:
        return sorted(p.name for p in self.dir.glob("*.db"))

    def get(self, name: str) -> MemoryStore:
        if name not in self.names():
            raise KeyError(name)
        if name not in self.open:
            self.open[name] = MemoryStore(self.dir / name)
        return self.open[name]


def _session_json(s) -> dict:
    return {"id": s.session_id, "started_at": s.started_at, "rank": getattr(s, "rank", 0),
            "messages": [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in s.messages]}


def create_app(directory: Path) -> Starlette:
    stores = Stores(directory)

    def store_of(request: Request, body: dict | None = None) -> MemoryStore:
        name = (body or {}).get("db") or request.query_params.get("db", "")
        return stores.get(name)

    async def index(request: Request):
        return FileResponse(STATIC, headers={"Cache-Control": "no-store"})  # always serve the current page

    async def list_stores(request: Request):
        out = []
        for name in stores.names():
            s = stores.get(name)
            sidecar = stores.dir / (name.removesuffix(".db") + ".questions.json")  # optional example questions
            out.append({"name": name, **{k: v for k, v in s.stats().items() if k != "path"},
                        "range": s.time_range(),
                        "questions": json.loads(sidecar.read_text()) if sidecar.exists() else []})
        return JSONResponse(out)

    async def sessions(request: Request):
        q = request.query_params
        total, page = store_of(request).list_sessions(
            search=q.get("q", ""), limit=int(q.get("limit", 50)), offset=int(q.get("offset", 0)))
        return JSONResponse({"total": total, "sessions": page})

    async def session(request: Request):
        st = store_of(request)
        s = st.get_session(request.query_params.get("id", ""))
        if not s:
            return JSONResponse(None, status_code=404)
        return JSONResponse({**_session_json(s), "facts": [f.text for f in st.facts.get(s.session_id)]})

    async def recall(request: Request):
        body = await request.json()
        query = (body.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "Type a question"}, status_code=400)
        budget = max(200, min(int(body.get("budget", 8000)), 64000))
        now = datetime.fromisoformat(normalize_time(body["now"])) if body.get("now") else None
        t = time.perf_counter()
        r = store_of(request, body).recall(query, budget, now=now)
        return JSONResponse({
            "ms": round((time.perf_counter() - t) * 1000),
            "window": [str(d) for d in r.window] if r.window else None,
            "time_only": r.time_only,
            "window_records": r.window_records,
            "tokens_est": r.tokens_est,
            "budget": budget,
            "rounds_considered": r.rounds_considered,
            "sessions": [_session_json(s) for s in r.sessions],
            "text": r.text,
        })

    async def save(request: Request):
        body = await request.json()
        msgs = [m for m in body.get("messages", []) if m.get("role") in ("user", "assistant") and m.get("content")]
        if not msgs:
            return JSONResponse({"error": "No messages to save: start each line with user: or assistant:"}, status_code=400)
        try:
            sid = store_of(request, body).add_session(msgs, started_at=body.get("started_at") or None)
        except ValueError as e:
            return JSONResponse({"error": f"Bad date: {e}"}, status_code=400)
        return JSONResponse({"id": sid})

    async def delete(request: Request):
        body = await request.json()
        return JSONResponse({"deleted": store_of(request, body).forget_session(body.get("id", ""))})

    async def create_store(request: Request):
        body = await request.json()
        name = "".join(c for c in body.get("name", "") if c.isalnum() or c in "-_")
        if not name:
            return JSONResponse({"error": "Names may only contain letters, digits, - and _"}, status_code=400)
        MemoryStore(stores.dir / f"{name}.db").close()
        return JSONResponse({"name": f"{name}.db"})

    async def unknown_store(request: Request, exc: KeyError):
        return JSONResponse({"error": f"No such store: {exc}"}, status_code=404)

    app = Starlette(exception_handlers={KeyError: unknown_store}, routes=[
        Route("/", index),
        Route("/api/stores", list_stores),
        Route("/api/stores", create_store, methods=["POST"]),
        Route("/api/sessions", sessions),
        Route("/api/session", session),
        Route("/api/recall", recall, methods=["POST"]),
        Route("/api/save", save, methods=["POST"]),
        Route("/api/delete", delete, methods=["POST"]),
    ])

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(prog="pocket-recall web")
    p.add_argument("--dir", default="~/.pocket-recall", help="folder of memory stores (*.db)")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    uvicorn.run(create_app(Path(args.dir)), host="127.0.0.1", port=args.port)
