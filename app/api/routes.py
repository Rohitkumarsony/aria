"""
WebSocket API  ws://{host}/ws/{session_id}

Client → Server:
  { type: "ping" }
  { type: "chat",          message: "..." }
  { type: "upload",        filename: "x.pdf", data: "<base64>" }
  { type: "sessions_list" }
  { type: "session_load",   session_id: "..." }
  { type: "session_delete", session_id: "..." }
  { type: "status" }

Server → Client events:
  pong
  status              { total_chunks }
  searching_web       { query }           ← fires when web_search tool actually executes
  chat_token          { token }
  chat_done           { confidence, verdict }
  chat_error          { detail }
  upload_start        { filename, total_pages, has_ocr, message }
  page_done           { page, total_pages, pct, used_ocr }
  upload_complete     { filename, total_chunks, doc_id }
  upload_error        { detail }
  sessions_list       { sessions: [...] }
  session_history     { session_id, messages: [...] }
  session_deleted     { session_id }
"""

import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.dependencies import get_agent, get_retriever, get_validator, get_vector_store
from app.core.session_store import (
    append_message, delete_session, get_messages, list_sessions,
)
from app.core.ws_manager import ws_manager
from app.models.schemas import AllowedFileType
from app.rag.document_processor import DocumentProcessor

router = APIRouter()
_pool  = ThreadPoolExecutor(max_workers=4)


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket handler
# ══════════════════════════════════════════════════════════════════════════════

@router.websocket("/ws/main")
async def ws_handler(ws: WebSocket):
    """
    Single persistent connection per browser tab.
    session_id is read from each message body — no reconnect needed on session switch.
    """
    # Use a stable connection ID for ws_manager routing
    conn_id = str(__import__('uuid').uuid4())
    await ws_manager.connect(conn_id, ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws_manager.send(conn_id, "error", {"detail": "Invalid JSON"})
                continue

            t = msg.get("type")
            # session_id comes from the message; fall back to conn_id for backward compat
            session_id = msg.get("session_id", conn_id)

            if t == "ping":
                await ws_manager.send(conn_id, "pong", {})

            elif t == "status":
                vs = get_vector_store()
                await ws_manager.send(conn_id, "status", {"total_chunks": vs.total_chunks})

            elif t == "sessions_list":
                await ws_manager.send(conn_id, "sessions_list", {"sessions": list_sessions()})

            elif t == "session_load":
                sid  = msg.get("session_id", "")
                msgs = get_messages(sid)
                await ws_manager.send(conn_id, "session_history", {
                    "session_id": sid, "messages": msgs
                })

            elif t == "session_delete":
                sid = msg.get("session_id", "")
                delete_session(sid)
                await ws_manager.send(conn_id, "session_deleted", {"session_id": sid})

            elif t == "upload":
                asyncio.create_task(
                    _handle_upload(conn_id, session_id, msg.get("filename", ""), msg.get("data", ""))
                )

            elif t == "chat":
                asyncio.create_task(
                    _handle_chat(conn_id, session_id, msg.get("message", ""))
                )

            else:
                await ws_manager.send(conn_id, "error", {"detail": f"Unknown type: {t}"})

    except WebSocketDisconnect:
        ws_manager.disconnect(conn_id, ws)
    except Exception:
        ws_manager.disconnect(conn_id, ws)


# ══════════════════════════════════════════════════════════════════════════════
# Upload handler (background task)
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_upload(conn_id: str, session_id: str, filename: str, b64_data: str) -> None:
    send = lambda ev, d: ws_manager.send(conn_id, ev, d)

    # Validate inputs
    if not filename:
        await send("upload_error", {"detail": "No filename provided."}); return

    try:
        file_type = DocumentProcessor.validate_extension(filename)
    except ValueError as e:
        await send("upload_error", {"detail": str(e)}); return

    try:
        file_bytes = base64.b64decode(b64_data)
    except Exception:
        await send("upload_error", {"detail": "Invalid file data."}); return

    if len(file_bytes) > 20 * 1024 * 1024:
        await send("upload_error", {"detail": "File exceeds 20 MB limit."}); return

    vs        = get_vector_store()
    retriever = get_retriever()
    processor = DocumentProcessor()
    all_chunks = []
    doc_id     = None

    try:
        # ── TXT ──────────────────────────────────────────────────────────────
        if file_type == AllowedFileType.TXT:
            await send("upload_start", {
                "filename": filename, "total_pages": 1,
                "has_ocr": False, "message": "Parsing text file…"
            })
            doc_id, chunks = await asyncio.get_event_loop().run_in_executor(
                _pool, processor.process_txt, file_bytes, filename
            )
            if chunks:
                vs.add_chunks(chunks)
                all_chunks = chunks
            await send("page_done", {
                "page": 1, "total_pages": 1, "chunks": len(chunks),
                "used_ocr": False, "pct": 100,
            })

        # ── PDF ───────────────────────────────────────────────────────────────
        else:
            import pypdf
            import io as _io
            reader      = pypdf.PdfReader(_io.BytesIO(file_bytes))
            total_pages = len(reader.pages)
            first_text  = processor._native_extract(reader.pages[0]) if total_pages else ""
            has_ocr     = not bool(first_text)

            await send("upload_start", {
                "filename": filename,
                "total_pages": total_pages,
                "has_ocr": has_ocr,
                "message": (
                    "⚠ Scanned PDF detected — OCR will extract text. This may take a moment."
                    if has_ocr else f"Processing {total_pages} page(s)…"
                ),
            })

            def _process_pages():
                return list(processor.iter_pdf_pages(file_bytes, filename))

            page_results = await asyncio.get_event_loop().run_in_executor(_pool, _process_pages)

            for pr in page_results:
                doc_id = pr["doc_id"]
                if pr["chunks"]:
                    vs.add_chunks(pr["chunks"])
                    all_chunks.extend(pr["chunks"])

                pct = int(pr["page_num"] / pr["total_pages"] * 100)
                await send("page_done", {
                    "page":       pr["page_num"],
                    "total_pages": pr["total_pages"],
                    "chunks":     len(pr["chunks"]),
                    "used_ocr":   pr["used_ocr"],
                    "text_found": pr["text_found"],
                    "pct":        pct,
                })
                await asyncio.sleep(0)  # yield to event loop

            if not all_chunks:
                await send("upload_error", {
                    "detail": (
                        "No text could be extracted from this PDF.\n"
                        "For scanned PDFs, install: sudo apt install tesseract-ocr poppler-utils"
                    )
                })
                return

        # ── Rebuild BM25 index ────────────────────────────────────────────────
        retriever.rebuild()

        await send("upload_complete", {
            "filename":     filename,
            "total_chunks": len(all_chunks),
            "doc_id":       doc_id,
        })

    except Exception as e:
        await send("upload_error", {"detail": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# Chat handler (background task)
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_chat(conn_id: str, session_id: str, message: str) -> None:
    if not message.strip():
        return

    agent     = get_agent()
    validator = get_validator()

    # Persist user message immediately
    append_message(session_id, "user", message)

    collected: list[str] = []

    # Callback: fires when web_search tool actually executes (accurate timing)
    async def on_searching(query: str):
        await ws_manager.send(conn_id, "searching_web", {"query": query})

    try:
        async for token in agent.stream(
            session_id=session_id,
            user_message=message,
            on_searching=on_searching,
        ):
            await ws_manager.send(conn_id, "chat_token", {"token": token})
            collected.append(token)

        full_response = "".join(collected)

        # Persist assistant response
        append_message(session_id, "assistant", full_response)

        # Validate response quality (non-blocking)
        try:
            validation = await validator.validate(message, full_response)
        except Exception:
            validation = {"confidence": "medium", "verdict": ""}

        await ws_manager.send(conn_id, "chat_done", {
            "confidence": validation.get("confidence", "medium"),
            "verdict":    validation.get("verdict", ""),
        })

        # Refresh session list in sidebar
        await ws_manager.send(conn_id, "sessions_list", {"sessions": list_sessions()})

    except Exception as e:
        await ws_manager.send(conn_id, "chat_error", {"detail": str(e)})