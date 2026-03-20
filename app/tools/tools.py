"""
Tools: WebSearchTool (Serper.dev) + CalculatorTool

Search strategy:
  1. Serper API → Google search results (title + snippet + URL)
  2. Fetch actual page content for top 2 results via httpx
  3. Return rich content to LLM
"""

import ast
import math
import operator
import re
from typing import Any

import httpx
from langchain_core.tools import BaseTool

from app.core.config import get_settings
from app.prompts.templates import CALCULATOR_DESCRIPTION, WEB_SEARCH_DESCRIPTION

settings = get_settings()

# ── Safe math ─────────────────────────────────────────────────────────────────

_BINOPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log,
    "log10": math.log10, "ceil": math.ceil, "floor": math.floor,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):  return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return float(node.value)
    if isinstance(node, ast.Name) and node.id in _FUNCS: return _FUNCS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        fn = _safe_eval(node.func)
        if callable(fn): return fn(*[_safe_eval(a) for a in node.args])
    raise ValueError(f"Unsupported: {type(node).__name__}")


# ── Page content fetcher ──────────────────────────────────────────────────────

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _fetch_page_text(url: str, max_chars: int = 2500) -> str:
    """Fetch a URL and return clean extracted text."""
    try:
        resp = httpx.get(url, headers=_BROWSER_HEADERS, timeout=8, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        html = resp.text
        # Remove noise
        for tag in ["script", "style", "nav", "footer", "header", "aside", "noscript"]:
            html = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Strip tags
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text[:max_chars]
    except Exception:
        return ""


# ── Serper API ────────────────────────────────────────────────────────────────

def _serper_search(query: str, api_key: str) -> list[dict]:
    """
    Serper.dev Google Search API.
    Docs: https://serper.dev
    Returns list of {title, snippet, url}
    """
    try:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={
                "q": query,
                "num": 5,
                "hl": "en",
                "gl": "in",   # India locale for better local results
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []

        # Answer box — direct answer (e.g. "Who is PM of India")
        answer_box = data.get("answerBox", {})
        if answer_box:
            answer = (
                answer_box.get("answer")
                or answer_box.get("snippet")
                or answer_box.get("snippetHighlighted", [""])[0]
            )
            if answer:
                results.append({
                    "title": answer_box.get("title", "Direct Answer"),
                    "snippet": answer,
                    "url": answer_box.get("link", ""),
                    "is_answer_box": True,
                })

        # Knowledge graph (e.g. celebrities, places)
        kg = data.get("knowledgeGraph", {})
        if kg.get("description"):
            results.append({
                "title": kg.get("title", ""),
                "snippet": kg.get("description", ""),
                "url": kg.get("website", kg.get("descriptionLink", "")),
                "is_kg": True,
            })

        # Organic results
        for r in data.get("organic", [])[:5]:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("link", ""),
            })

        return results

    except Exception as e:
        return []


# ── Main search function ──────────────────────────────────────────────────────

def _search_sync(query: str) -> str:
    api_key = settings.serper_api_key
    if not api_key or api_key == "your_serper_key_here":
        return "Serper API key not configured. Add SERPER_API_KEY to .env"

    results = _serper_search(query, api_key)
    if not results:
        return "No search results found."

    output_parts = []

    for i, r in enumerate(results[:5]):
        title   = r.get("title", "")
        snippet = r.get("snippet", "")
        url     = r.get("url", "")

        # For answer box / knowledge graph — these already have the answer
        if r.get("is_answer_box") or r.get("is_kg"):
            output_parts.append(f"**{title}**\n{snippet}" + (f"\nSource: {url}" if url else ""))
            continue

        # For organic results — fetch actual page content for top 2
        page_text = ""
        if i < 2 and url:
            page_text = _fetch_page_text(url, max_chars=2000)

        if page_text:
            output_parts.append(f"**{title}**\nSource: {url}\n\n{page_text}")
        else:
            output_parts.append(f"**{title}**\n{snippet}\nSource: {url}")

    return "\n\n---\n\n".join(output_parts)


# ── Tool classes ──────────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = WEB_SEARCH_DESCRIPTION

    def _run(self, query: str) -> str:
        return _search_sync(query)

    async def _arun(self, query: str) -> str:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, _search_sync, query)


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = CALCULATOR_DESCRIPTION

    def _run(self, expression: str) -> str:
        try:
            result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
            return f"Result: {result:,.10g}"
        except ZeroDivisionError:
            return "Error: Division by zero."
        except Exception as e:
            return f"Error: {e}"

    async def _arun(self, expression: str) -> str:
        return self._run(expression)