"""
RAGAgent — production streaming agent.
Intent is classified by the LLM itself — no hardcoded keyword lists.
The model decides when to use tools based on the system prompt.
"""

from __future__ import annotations
import json
import uuid
from typing import AsyncIterator, Callable, Awaitable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.prompts.templates import RAG_CONTEXT_TEMPLATE, SYSTEM_PROMPT
from app.rag.hybrid_retriever import HybridRetriever
from app.tools.tools import CalculatorTool, WebSearchTool

settings = get_settings()


class RAGAgent:
    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever
        self._tools     = [WebSearchTool(), CalculatorTool()]
        self._tool_map  = {t.name: t for t in self._tools}

        _base = dict(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.4,
            streaming=True,
        )
        # Single LLM — the model decides when to call tools based on intent
        self._llm = ChatOpenAI(**_base).bind_tools(self._tools)

        self._sessions: dict[str, list] = {}

    # ── Public ───────────────────────────────────────────────────────────────

    async def stream(
        self,
        session_id: str,
        user_message: str,
        on_searching: Callable[[str], Awaitable[None]] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens.
        The LLM decides whether to call tools — no keyword matching.
        on_searching: async callback fired when web_search tool executes.
        """
        if user_message.strip() == "__clear__":
            self._sessions.pop(session_id, None)
            return

        history  = self._sessions.setdefault(session_id, [])
        human    = self._build_human_message(user_message)
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [human]

        # ── Pass 1: stream ────────────────────────────────────────────────
        text_chunks: list[str] = []
        tool_acc: dict[int, dict] = {}

        async for chunk in self._llm.astream(messages):
            if chunk.content:
                text_chunks.append(chunk.content)
                yield chunk.content

            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    idx = tc.get("index", 0)
                    if idx not in tool_acc:
                        tool_acc[idx] = {
                            "id":   tc.get("id", str(uuid.uuid4())),
                            "name": tc.get("name", ""),
                            "args": "",
                        }
                    if tc.get("name"): tool_acc[idx]["name"] = tc["name"]
                    if tc.get("args"): tool_acc[idx]["args"] += tc["args"]

        # No tool calls → done
        if not tool_acc:
            self._save(session_id, history, user_message, "".join(text_chunks))
            return

        # ── Resolve tool arguments ────────────────────────────────────────
        resolved = []
        for tc in tool_acc.values():
            try:
                args = json.loads(tc["args"]) if tc["args"].strip() else {}
            except json.JSONDecodeError:
                args = {"query": tc["args"]}
            resolved.append({
                "id":   tc["id"],
                "name": tc["name"],
                "args": args,
                "type": "tool_call",
            })

        # Notify frontend that web search is running
        if on_searching:
            search_call = next((r for r in resolved if r["name"] == "web_search"), None)
            if search_call:
                query = search_call["args"].get("query", "")
                await on_searching(query)

        # ── Execute tools ─────────────────────────────────────────────────
        ai_msg    = AIMessage(content="", tool_calls=resolved)
        tool_msgs = await self._run_tools(resolved)
        messages2 = messages + [ai_msg] + tool_msgs

        # ── Pass 2: stream final answer ───────────────────────────────────
        final: list[str] = []
        async for chunk in self._llm.astream(messages2):
            if chunk.content:
                final.append(chunk.content)
                yield chunk.content

        self._save(session_id, history, user_message, "".join(final))

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ── Private ───────────────────────────────────────────────────────────

    def _build_human_message(self, query: str) -> HumanMessage:
        """Inject RAG context if documents exist and retrieval finds hits."""
        if not self._retriever.has_documents:
            return HumanMessage(content=query)
        hits = self._retriever.retrieve(query)
        if not hits:
            return HumanMessage(content=query)
        context = "\n\n---\n\n".join(
            f"[{h['metadata'].get('filename', 'doc')}, "
            f"page {h['metadata'].get('page_number', '?')}]\n{h['content']}"
            for h in hits
        )
        return HumanMessage(
            content=RAG_CONTEXT_TEMPLATE.format(context=context, question=query)
        )

    async def _run_tools(self, calls: list[dict]) -> list[ToolMessage]:
        import asyncio

        async def _one(tc: dict) -> ToolMessage:
            tool = self._tool_map.get(tc["name"])
            try:
                result = await tool.arun(tc["args"]) if tool else f"Unknown tool: {tc['name']}"
            except Exception as e:
                result = f"Tool error: {e}"
            return ToolMessage(content=str(result), tool_call_id=tc["id"])

        return list(await asyncio.gather(*[_one(tc) for tc in calls]))

    def _save(self, session_id: str, history: list, user_msg: str, ai_content: str) -> None:
        self._sessions[session_id] = (
            history
            + [HumanMessage(content=user_msg)]
            + [AIMessage(content=ai_content)]
        )