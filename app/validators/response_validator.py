"""
ResponseValidator — checks AI responses for accuracy, hallucination, and completeness.
Runs asynchronously after each response. Results are sent to the frontend via WebSocket.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.core.config import get_settings
from app.models.schemas import ConfidenceLevel
from app.prompts.templates import VALIDATOR_PROMPT

settings = get_settings()

_WEB_TRIGGERS = [
    "news", "today", "weather", "president", "prime minister", "ceo", "who is",
    "current", "latest", "now", "price", "stock", "score", "election",
    "2024", "2025", "2026", "this week", "yesterday", "just",
]


class ResponseValidator:
    """Validates AI responses for confidence and potential issues."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o-mini",   # cheaper model for validation
            openai_api_key=settings.openai_api_key,
            temperature=0,
        )

    async def validate(self, question: str, response: str) -> dict:
        """
        Returns:
          {
            confidence: "high"|"medium"|"low"|"unverified",
            issues: [...],
            should_search_web: bool,
            verdict: "..."
          }
        """
        # Quick heuristic: if answer contains stale phrases, flag immediately
        stale_phrases = [
            "as of my last update", "as of my knowledge", "i don't have access",
            "i recommend checking", "i cannot browse", "my training data",
        ]
        response_lower = response.lower()
        has_stale = any(p in response_lower for p in stale_phrases)
        needs_web = any(kw in question.lower() for kw in _WEB_TRIGGERS)

        if has_stale and needs_web:
            return {
                "confidence": ConfidenceLevel.UNVERIFIED,
                "issues": ["Response uses training data for a real-time question"],
                "should_search_web": True,
                "verdict": "This answer may be outdated. Web search was not used.",
            }

        # LLM-based validation (async)
        try:
            prompt = VALIDATOR_PROMPT.format(question=question, response=response)
            result = await self._llm.ainvoke([HumanMessage(content=prompt)])
            text = result.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception:
            return {
                "confidence": ConfidenceLevel.MEDIUM,
                "issues": [],
                "should_search_web": False,
                "verdict": "Validation unavailable.",
            }
