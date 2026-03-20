# Aria — AI Research Assistant

A production-grade conversational AI chatbot with document Q&A (RAG), real-time web search, LangGraph agent, persistent session history, and a Chatbot-style single-page frontend.

![Alt text](./images/landing_page.png)

Realtime notify when document uploading
![alt text](images/realtimenotify_processing_page.png)
---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Technology Stack & Why](#technology-stack--why)
- [Hybrid Search — Deep Dive](#hybrid-search--deep-dive)
- [Decision Logic — When Does What Run](#decision-logic--when-does-what-run)
- [WebSocket API](#websocket-api)
- [Advanced Features](#advanced-features)
- [Environment Variables](#environment-variables)
- [Design Decisions & Trade-offs](#design-decisions--trade-offs)

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/Rohitkumarsony/aria.git
cd aria
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. OCR support for scanned PDFs
sudo apt install tesseract-ocr poppler-utils

# 4. Configure environment
cp .env.example .env
# Edit .env — add your API keys

# 5. Run
uvicorn main:app --reload

# Open http://localhost:8000
```

---

## Project Structure

```
aria/
│
├── main.py                          # FastAPI app factory + lifespan startup
├── requirements.txt
├── .env.example
│
├── frontend/
│   └── index.html                   # Single-file GPT-style UI (HTML + CSS + JS)
│
├── sessions/                        # Persisted chat history (one JSON per session)
│   └── {session_id}.json
│
├── chroma_db/                       # ChromaDB vector store (auto-created on first run)
│
└── app/
    ├── api/
    │   └── routes.py                # WebSocket handler — all client communication
    │
    ├── agents/
    │   └── rag_agent.py             # LangGraph agent — orchestrates LLM + RAG + tools
    │
    ├── core/
    │   ├── config.py                # Pydantic Settings — typed env var loading
    │   ├── dependencies.py          # Singleton instances via lru_cache
    │   ├── session_store.py         # Disk-based JSON chat history
    │   └── ws_manager.py            # WebSocket connection registry (conn_id → sockets)
    │
    ├── models/
    │   └── schemas.py               # Pydantic models + AllowedFileType enum (pdf/txt only)
    │
    ├── prompts/
    │   └── templates.py             # All prompt strings — system, RAG, validator, tools
    │
    ├── rag/
    │   ├── document_processor.py    # PDF/TXT parsing, OCR fallback, chunking
    │   ├── vector_store.py          # ChromaDB wrapper — embed + store + dense search
    │   └── hybrid_retriever.py      # BM25 sparse + dense embedding + RRF fusion
    │
    ├── tools/
    │   └── tools.py                 # WebSearchTool (Serper + page fetch) + CalculatorTool
    │
    └── validators/
        └── response_validator.py    # GPT-4o-mini confidence scorer for each response
```

---

## Architecture Overview

```
Browser (WebSocket ws://host/ws/main)
        │
        │  { type: "chat",   session_id: "...", message: "..." }
        │  { type: "upload", session_id: "...", filename: "...", data: "<b64>" }
        ▼
┌─────────────────────────────────────────────────────────┐
│  routes.py — WebSocket Handler                          │
│                                                         │
│  • Single /ws/main endpoint (no reconnect on switch)    │
│  • session_id read from message body                    │
│  • asyncio.create_task() — never blocks receive loop    │
│  • Fan-out responses via ws_manager (conn_id routing)   │
└──────────┬──────────────────────────┬───────────────────┘
           │ chat                     │ upload
    ┌──────▼──────┐            ┌──────▼────────────────┐
    │  RAGAgent   │            │  DocumentProcessor    │
    │             │            │                       │
    │ 1. Retrieve │            │ 1. Validate ext (enum)│
    │    from RAG │            │ 2. pypdf extraction   │
    │    (hybrid) │            │ 3. OCR fallback       │
    │ 2. Filter   │            │ 4. Chunk text         │
    │    by score │            │ 5. OpenAI embed       │
    │ 3. Inject   │            │ 6. Store ChromaDB     │
    │    context  │            │ 7. Rebuild BM25 index │
    │ 4. Stream   │            │ 8. Broadcast progress │
    │    GPT-4o   │            └───────────────────────┘
    │ 5. Tools    │
    │ 6. Validate │
    └─────────────┘
```

---

## Technology Stack & Why

### FastAPI
**Why:** Async-native Python framework with built-in WebSocket support. Pydantic integration gives automatic request validation. Handles concurrent LLM streaming, embedding calls, and web fetches without blocking.
![Alt text](./images/input_field.png)

### WebSocket (pure — no HTTP for chat/upload)
**Why:** HTTP request-response is wrong for streaming AI. WebSocket gives:
- Bidirectional — server pushes tokens as they generate
- Real-time page-by-page upload progress
- One persistent connection handles chat, upload, sessions simultaneously
- Session switch = zero reconnect (session_id travels in the message, not the URL)

### OpenAI GPT-4o
**Why:** Best tool-calling accuracy. The structured function-calling API lets the model decide when to call `web_search` vs answer from knowledge vs use document context — no hardcoded keyword matching needed.

### LangGraph
**Why:** Explicit, inspectable agent state machine. The loop `retrieve → llm → tools → llm` is a named graph — debuggable and extensible. Plain LangChain chains hide the loop internals. Adding a new node (e.g. a routing node) is a clean graph modification.

### ChromaDB
**Why:** Embedded vector database — zero external service to deploy. Cosine similarity search with persistent storage. Abstracts to a single class so swapping to Pinecone for production is a one-file change.

### OpenAI text-embedding-3-small
**Why:** Best cost/quality ratio for embeddings. Creates 1536-dimensional semantic vectors. Used for dense retrieval — finds conceptually similar chunks regardless of exact wording.

### BM25 (rank-bm25)
**Why:** Keyword-based sparse retrieval runs in-memory with zero infrastructure. Catches exact matches (names, codes, acronyms, IDs) that semantic search misses.

### Serper API
**Why:** Returns real Google results including answer boxes and knowledge graphs. Only called for live/time-sensitive queries. Also fetches actual page content (not just snippets) for richer LLM context.

### pypdf + pytesseract + pdf2image
**Why:** Two-stage PDF extraction — fast native text first, OCR fallback for scanned documents. Works with both digital PDFs and image-only scanned files.

---

## Hybrid Search — Deep Dive

### Why not just embeddings?

| Query | Dense (embeddings) | BM25 (keyword) |
|---|---|---|
| "what is machine learning" | ✅ finds semantic matches | ❌ misses paraphrases |
| "RFC-2616 section 14" | ❌ number not semantic | ✅ exact keyword match |
| "John Smith phone number" | ⚠️ partial | ⚠️ partial |
| "contact info" (chunk says "reach me at") | ✅ | ❌ |

Neither method alone is reliable. Hybrid search uses both.

### How Hybrid Retrieval Works

```
Query: "what is Rohit's email address"
              │
     ┌────────┴────────┐
     │                 │
Dense Search        BM25 Search
(ChromaDB cosine)   (keyword match)
     │                 │
Rank 1: contact     Rank 1: email chunk    ← appears in both
Rank 2: about me    Rank 2: contact info
Rank 3: skills      Rank 3: reach out
     │                 │
     └────────┬────────┘
              │
        RRF Fusion
        score(chunk) = 1/(60+rank_dense) + 1/(60+rank_sparse)
              │
        Chunks appearing in BOTH lists get highest combined score
              │
        Relevance filter: drop chunks with score < 0.30
              │
        Top 4 chunks → injected as LLM context
```
![Alt text](./images/metadata.png)

### RRF Formula

```
RRF_score = Σ  1 / (k + rank)    where k = 60
```

`k=60` is the standard constant that prevents a rank-1 result from dominating. A chunk at rank 1 in both lists scores `0.0328`. A chunk at rank 1 in only one list scores `0.0164`. The fusion naturally promotes consensus.

### Relevance Threshold (score ≥ 0.30)

Without this, every query injects whatever ChromaDB returns — even a 0.05 similarity score. Threshold = 0.30 means: if retrieved chunks aren't genuinely related to the query, the query gets sent to the LLM without document context. The LLM then uses its own knowledge or calls `web_search` instead of hallucinating from an irrelevant document section.

![Alt text](./images/calling_web_tool.png)

---

## Decision Logic — When Does What Run

```
User message received
        │
        ▼
Documents uploaded + relevant chunks found (score ≥ 0.30)?
        │
   YES ─┤→ Inject document context into prompt
        │       │
        │       ▼
        │  LLM answers from document, cites (filename, page N)
        │  If doc doesn't fully answer → uses own knowledge or web_search
        │
   NO ──┤→ No document context injected
        │
        ▼
LLM evaluates with system prompt — decides independently:
        │
        ├── General knowledge question?
        │   ("what is Python", "explain REST", "history of WW2")
        │   → LLM answers from its own built-in knowledge base
        │   → No tool call, fastest response
        │
        ├── Live / time-sensitive data?
        │   ("today's news", "current weather", "who is PM of India now")
        │   → Calls web_search tool
        │   → Serper fetches Google results
        │   → httpx fetches actual page content from top 2 URLs
        │   → LLM streams answer with source URLs
        │
        ├── Math / computation?
        │   ("calculate 15% of 2400", "compound interest formula")
        │   → Calls calculator tool
        │   → Safe AST evaluator returns exact result
        │
        └── Casual chitchat?
            ("hi", "thanks", "nice", "lol")
            → LLM responds naturally, no tools
```
![Alt text](./images/web_result.png)

### Why the LLM decides (not keyword rules)

Hardcoded rules (`if "news" in query: search`) break constantly. "What's new in Python 3.12" has "new" but needs knowledge-base, not live search. "Who is the president" needs live search but has no trigger keyword. GPT-4o reads the full context and makes the right call reliably.

---

## WebSocket API

**Endpoint:** `ws://localhost:8000/ws/main`

Single persistent connection. Session ID travels inside each message — no URL changes, no reconnects.
![Alt text](./images/follow-up.png)

### Client → Server

```json
{ "type": "ping" }
{ "type": "status" }
{ "type": "sessions_list" }
{ "type": "session_load",   "session_id": "uuid" }
{ "type": "session_delete", "session_id": "uuid" }
{ "type": "chat",   "session_id": "uuid", "message": "your question" }
{ "type": "upload", "session_id": "uuid", "filename": "doc.pdf", "data": "<base64>" }
```

### Server → Client Events

```json
{ "event": "pong" }
{ "event": "status",          "total_chunks": 42 }
{ "event": "sessions_list",   "sessions": [{"session_id":"...","title":"...","updated_at":"..."}] }
{ "event": "session_history", "session_id": "...", "messages": [{"role":"user","content":"..."}] }
{ "event": "session_deleted", "session_id": "..." }

// Upload lifecycle
{ "event": "upload_start",    "filename": "doc.pdf", "total_pages": 10, "has_ocr": false }
{ "event": "page_done",       "page": 3, "total_pages": 10, "pct": 30, "used_ocr": false }
{ "event": "upload_complete", "filename": "doc.pdf", "total_chunks": 42, "doc_id": "uuid" }
{ "event": "upload_error",    "detail": "No extractable text found." }

// Chat lifecycle
{ "event": "searching_web",   "query": "India news today" }
{ "event": "chat_token",      "token": "The " }
{ "event": "chat_done",       "confidence": "high", "verdict": "Accurate web search result." }
{ "event": "chat_error",      "detail": "Tool error: ..." }
```
![Alt text](./images/tool_using.png)

---

## Advanced Features

| Feature | File | What it does |
|---|---|---|
| Hybrid RAG (BM25 + dense + RRF) | `rag/hybrid_retriever.py` | Best-of-both retrieval with fusion scoring |
| Relevance threshold filter | `agents/rag_agent.py` | Prevents hallucination from irrelevant doc chunks |
| OCR fallback | `rag/document_processor.py` | Handles scanned/image PDFs via tesseract |
| Page-by-page ingestion | `rag/document_processor.py` | Large PDFs never hit token limits |
| Response confidence validator | `validators/response_validator.py` | GPT-4o-mini scores each answer (high/medium/low) |
| Session persistence | `core/session_store.py` | JSON files survive server restarts |
| Single WebSocket connection | `app/api/routes.py` | No reconnect on session switch |
| Background task execution | `app/api/routes.py` | Upload + chat never block WS receive loop |
| Real page content fetch | `tools/tools.py` | httpx fetches actual text from URLs (not just snippets) |
| Safe AST calculator | `tools/tools.py` | Math eval without eval() — no code injection possible |
| Intent-based tool routing | `agents/rag_agent.py` | LLM decides tools, not keyword lists |
| Streaming tokens | `agents/rag_agent.py` | First token in < 1 second, GPT-like feel |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI key for GPT-4o and embeddings |
| `SERPER_API_KEY` | ✅ | — | Serper.dev Google Search key |
| `CHROMA_PERSIST_DIR` | | `./chroma_db` | ChromaDB storage path |
| `SESSIONS_DIR` | | `./sessions` | Chat history folder |
| `OPENAI_MODEL` | | `gpt-4o` | LLM model name |
| `OPENAI_EMBEDDING_MODEL` | | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | | `600` | Characters per chunk |
| `CHUNK_OVERLAP` | | `100` | Overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | | `6` | Dense retrieval candidates |
| `BM25_TOP_K` | | `6` | BM25 retrieval candidates |
| `HYBRID_FINAL_K` | | `4` | Final chunks after RRF fusion |

---

## Design Decisions & Trade-offs

**Pure WebSocket vs REST + SSE**
SSE only flows server → client. WebSocket is bidirectional, so one connection handles streaming tokens, upload progress, session events, and status. Also enabled eliminating reconnects on session switch.

**LangGraph vs plain LangChain**
LangGraph makes the agent loop an explicit, named state machine. When adding a new capability (routing node, memory summarizer), it's a clean graph modification rather than digging through chain internals.

**session_id in message body vs URL**
Original design used `/ws/{session_id}` — switching sessions required closing and reopening the socket, creating a reconnect loop. Moving session_id to the message body gives one stable connection forever.

**JSON files vs Redis for sessions**
JSON files require zero infrastructure. The `session_store.py` interface is fully abstracted — switching to Redis for horizontal scaling is a single-file change.

**ChromaDB vs Pinecone**
ChromaDB is embedded — no external service needed for development or assessment. The `VectorStore` class abstracts all calls; swap implementation without changing agent code.

**Relevance threshold 0.30**
Without it, every question injects whatever ChromaDB returns regardless of relevance. The threshold ensures only genuinely matching chunks reach the LLM, preventing it from hallucinating answers from unrelated document sections.

**Safe AST calculator vs eval()**
Python eval() would execute arbitrary code. The AST walker only permits whitelisted node types. `__import__('os').system('rm -rf /')` is rejected at parse time.

---
## Note ##
# Intent-Based Routing Model for Assistant Responses

This document describes how the assistant decides when to use uploaded document context, web search, calculator tools, or built-in LLM knowledge.

## Chatbot Summary

Our assistant follows an intent-first response model. It first checks whether the user's query can be answered from uploaded documents; if relevant document context exists, it answers from that context with citations. If the document context is missing, incomplete, or not sufficient for a reliable answer, it decides whether external freshness is required. For queries involving current or changing information such as news, weather, sports, prices, public figures, schedules, or recent events, it invokes web search. For numerical computation, formulas, percentages, interest, conversions, or explicit math expressions, it invokes the calculator tool. For conversational queries, explanations, rewriting, summarization, brainstorming, or stable general knowledge that does not require live verification, it answers directly using the language model’s built-in knowledge. This design avoids unnecessary tool calls, keeps responses natural, and uses tools only when they improve accuracy, freshness, or precision.

## Source Priority

The assistant should use sources in this order:

1. **Uploaded document context** when the question is about user-provided files or internal content.
2. **Web search** when the answer depends on live, recent, or changing information.
3. **Calculator** when the task requires exact computation.
4. **Built-in LLM knowledge** for stable general knowledge, reasoning, rewriting, summarization, and conversational help.

## When to Use Each Source

### 1. Document Context

Use document context when:

* the user asks about uploaded files
* the answer should come from PDFs, docs, manuals, or company documents
* citations from the document are needed
* the query is clearly grounded in provided material

### 2. Web Search

Use web search when:

* the answer needs current information
* the topic may have changed recently
* the user asks about news, weather, sports, prices, public figures, schedules, regulations, releases, or recent events

### 3. Calculator

Use calculator when:

* the user asks for arithmetic
* the task includes percentages, formulas, ratios, interest, conversions, or exact numeric evaluation
* precision matters more than natural language estimation

### 4. Built-in LLM Knowledge

Use built-in knowledge when:

* the question is general and stable
* the user wants explanation, rewriting, summarization, comparison, brainstorming, or drafting help
* no document grounding is required
* no live verification is required
* no exact computation is required

## Why the LLM Should Not Be Fully Restricted

The assistant should not be completely restricted from using built-in knowledge.

### Reasons

* Many user requests do not require tools, such as rewriting, summarizing, explaining concepts, drafting messages, or brainstorming.
* Full restriction makes the assistant feel robotic and overly dependent on tools.
* It can trigger unnecessary web searches for simple, stable questions.
* If document retrieval misses useful information, the assistant may wrongly fail instead of still helping with general knowledge.
* Conversational quality becomes worse when every answer must come from a tool.

### Recommended Approach

Use a grounded-first architecture:

* **Documents first** when relevant
* **Web search** for freshness
* **Calculator** for exact math
* **LLM knowledge** for reasoning, writing, and stable general knowledge

## Recommended Routing Principle

**Ground first, reason second.**

That means:

* prefer documents when the question is about provided material
* prefer web search when freshness matters
* prefer calculator when exact math is needed
* otherwise let the LLM answer naturally

## Prompt Changes for Stronger Restriction

If stricter control is needed, do not remove LLM knowledge entirely. Instead, make the routing more explicit in the prompts.

### 1. Update `SYSTEM_PROMPT`

```python
SYSTEM_PROMPT = """You are Aria, an intelligent AI assistant.

Your goal is to answer using the most reliable source available in this order:
1. Uploaded document context, when relevant
2. Web search, when the question needs current or real-world information
3. Calculator, when exact computation is needed
4. Built-in model knowledge, only when the question is general, stable, and does not require documents, live data, or exact calculation

Rules:
- If the user’s question is about uploaded files, prefer document context and cite it.
- If the answer may depend on recent or changing information, use web_search.
- If the query involves arithmetic or formulas, use calculator.
- Use built-in knowledge only for stable general knowledge, explanations, rewriting, summarization, brainstorming, or conversational help.
- Do not present built-in knowledge as document-backed or live-verified information.
- When documents are incomplete, say so briefly, then supplement with web search or built-in knowledge as appropriate.
"""
```

### 2. Update `RAG_CONTEXT_TEMPLATE`

```python
RAG_CONTEXT_TEMPLATE = """Here is relevant content from the user's uploaded documents:

{context}

User's question: {question}

Answer using the document content first and cite sources as (filename, page N).
If the document only partially answers the question, clearly separate:
1. what is supported by the document
2. what is supplemented from web search or general model knowledge

Use web search if freshness or real-world verification is needed.
Use general model knowledge only for stable background explanation when documents do not fully cover the topic.
"""
```

### 3. Update `VALIDATOR_PROMPT`

```python
VALIDATOR_PROMPT = """Evaluate this AI assistant response.

Question: {question}
Response: {response}

Return JSON only:
{{
  "confidence": "high|medium|low|unverified",
  "issues": [],
  "should_search_web": false,
  "used_unsupported_llm_knowledge": false,
  "verdict": "one sentence assessment"
}}

Mark used_unsupported_llm_knowledge as true if:
- the response claims document support without citation
- the response answers a current-events question without web search
- the response uses model knowledge where exact tools should have been used
- the response presents uncertain knowledge as verified fact
"""
```

## Final Recommendation

Do not fully block the LLM from using its own knowledge. Instead, make the routing policy explicit and prioritize the best source for each task. This gives better accuracy, better user experience, and better tool efficiency.
