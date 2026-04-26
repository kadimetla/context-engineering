# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Training materials and MCP server implementations for "Context Engineering with MCP" — a course teaching production MCP (Model Context Protocol) deployment. The flagship teaching application is WARNERCO Schematica, a FastAPI + FastMCP + LangGraph application that exercises **all four CoALA memory tiers** (Working / Episodic / Semantic / Procedural) in one coherent codebase. See `research_synthesis/` for the four independent deep-research reports that drove the design.

## Repository Structure

```
context-engineering/
├── src/warnerco/backend/          # WARNERCO Schematica - Primary Teaching App
│   ├── app/                       # FastAPI + FastMCP + LangGraph
│   │   ├── adapters/              # Memory backends (JSON, Chroma, Azure, Graph, Scratchpad, Episodic, CoALA overview)
│   │   ├── langgraph/             # 9-node hybrid RAG pipeline + consolidation cycle
│   │   ├── models/                # Pydantic models (schematic, graph, scratchpad, episodic)
│   │   ├── main.py                # FastAPI application
│   │   └── mcp_tools.py           # FastMCP tool definitions
│   ├── data/                      # JSON schematics + vector stores
│   ├── scripts/                   # Indexing utilities
│   ├── static/                    # SPA dashboards
│   └── tests/                     # Test suite
├── labs/lab-01-hello-mcp/         # Hands-on exercises (starter + solution)
├── docs/                          # Student materials and tutorials
│   ├── diagrams/                  # Architecture diagrams (SVG + Mermaid)
│   ├── tutorials/                 # Step-by-step tutorials
│   └── api/                       # API reference docs
├── config/                        # Sample MCP client configs
├── diagrams/                      # High-level architecture (Mermaid)
├── instructor/                    # Instructor materials
├── research_synthesis/            # Deep-research reports (Claude/ChatGPT/Gemini/Perplexity) on agent memory
├── .claude/agents/                # Claude Code agents
└── .claude/skills/                # Claude Code skills
```

## Development Commands

### WARNERCO Schematica (FastAPI + FastMCP + LangGraph)

**Python 3.13 required** — `.python-version` is pinned to 3.13. `onnxruntime` (chromadb dependency) does not ship 3.14 wheels, so `uv sync` fails on 3.14.

```bash
cd src/warnerco/backend
uv sync                                    # Install dependencies
uv run uvicorn app.main:app --reload       # Start server (http://localhost:8000)
uv run warnerco-serve                      # Same as above, via console script
uv run warnerco-mcp                        # MCP stdio server (for Claude Desktop)
uv run warnerco-restart                    # Force-kill anything on port 8000, then restart
uv run warnerco-restart --kill-only        # Just free the port (no restart)
uv run warnerco-restart --port 9000        # Use a different port
```

**Restart helper:** `scripts/restart_server.py` — Windows uses `netstat -ano` + `taskkill /F /T /PID` (also kills uvicorn reload children); POSIX uses `lsof -t` + SIGKILL. Refuses to kill its own PID. Exit 0 if port freed, exit 1 if not.

**Memory Backends** (set `MEMORY_BACKEND` in `.env`):
- `json` - Fastest startup, keyword search (default)
- `chroma` - Local semantic search (recommended for dev)
- `azure_search` - Enterprise deployment

**Index Schematics**:
```bash
# Chroma (local vectors)
uv run python -c "from app.adapters.chroma_store import ChromaMemoryStore; import asyncio; asyncio.run(ChromaMemoryStore().index_all())"

# Azure AI Search (enterprise vectors)
uv run python scripts/index_azure_search.py

# Graph Memory (knowledge graph)
uv run python scripts/index_graph.py
```

### Lab 01 - Hello MCP (Beginner Entry Point)

```bash
cd labs/lab-01-hello-mcp/starter
npm install
npm start
# Test with: npx @modelcontextprotocol/inspector node src/index.js
```

## MCP Server Architecture Patterns

### Tool Response Format
All tools must return content array:
```javascript
return {
  content: [{
    type: 'text',
    text: JSON.stringify(result)
  }]
};
```

### Logging Convention
Use `console.error()` for all logging - stdout reserved for MCP protocol:
```javascript
console.error('Tool called:', toolName);  // Goes to stderr
```

### Resource URIs
Resources use URI scheme: `memory://overview`, `memory://context-stream`

## Key Files

### WARNERCO Schematica
- `src/warnerco/backend/app/main.py` - FastAPI application
- `src/warnerco/backend/app/mcp_tools.py` - FastMCP tool/resource/prompt definitions (28 tools, 11 resources, 5 prompts)
- `src/warnerco/backend/app/langgraph/flow.py` - 9-node hybrid RAG orchestration
- `src/warnerco/backend/app/langgraph/consolidate.py` - "Sleep cycle" — promotes working/episodic memory to semantic via MCP Sampling
- `src/warnerco/backend/app/adapters/` - Memory backends (JSON, Chroma, Azure, Graph, Scratchpad, Episodic, CoALA overview)
- `src/warnerco/backend/app/adapters/episodic_store.py` - SQLite event log with Park et al. recency × importance × relevance recall
- `src/warnerco/backend/app/adapters/coala_overview.py` - Live four-tier snapshot helper for `memory://coala-overview`
- `src/warnerco/backend/app/models/graph.py` - Entity and Relationship models
- `src/warnerco/backend/app/models/scratchpad.py` - ScratchpadEntry, ScratchpadStats, predicate vocabulary
- `src/warnerco/backend/app/models/episodic.py` - EventKind, EpisodicEvent, EpisodicRecallResult, ConsolidationResult
- `src/warnerco/backend/app/adapters/graph_store.py` - SQLite + NetworkX graph store
- `src/warnerco/backend/app/adapters/scratchpad_store.py` - SQLite store with LLM minimization/enrichment
- `src/warnerco/backend/data/schematics/schematics.json` - Source of truth (25 robot schematics)
- `src/warnerco/backend/scripts/index_graph.py` - Graph indexing script
- `docs/tutorials/coala-memory-walkthrough.md` - Classroom demo script for the four-tier path

### Lab 01
- `labs/lab-01-hello-mcp/starter/src/index.js` - Starting point for students
- `labs/lab-01-hello-mcp/solution/src/index.js` - Completed solution

### Configuration
- `config/claude_desktop_config.json` - Sample Claude Desktop configuration

## MCP Client Configuration

### Claude Desktop
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "warnerco": {
      "command": "uv",
      "args": ["run", "warnerco-mcp"],
      "cwd": "C:/github/context-engineering/src/warnerco/backend"
    }
  }
}
```

### VS Code
Create `.vscode/mcp.json` in workspace:
```json
{
  "mcpServers": {
    "warnerco": {
      "command": "uv",
      "args": ["run", "warnerco-mcp"],
      "cwd": "${workspaceFolder}/src/warnerco/backend"
    }
  }
}
```

## Testing with MCP Inspector

Primary debugging tool - opens web UI to call tools and view resources:
```bash
npx @modelcontextprotocol/inspector uv run warnerco-mcp
# Opens http://localhost:5173
```

## Environment Variables

Set in `src/warnerco/backend/.env`:

```bash
# Memory backend selection
MEMORY_BACKEND=json  # json, chroma, or azure_search

# Scratchpad Memory (CoALA Tier 1 — working memory, persistent SQLite)
SCRATCHPAD_DB_PATH=data/scratchpad/notes.db
SCRATCHPAD_INJECT_BUDGET=1500           # Tokens for LangGraph injection

# Episodic Memory (CoALA Tier 2 — timestamped events, persistent SQLite)
EPISODIC_DB_PATH=data/episodic/events.db
EPISODIC_MAX_RETRIEVAL_K=5               # Top-k for recall
EPISODIC_RECENCY_HALF_LIFE_HOURS=24.0    # Park et al. half-life
EPISODIC_WEIGHT_RECENCY=0.4              # α_recency
EPISODIC_WEIGHT_IMPORTANCE=0.3           # α_importance
EPISODIC_WEIGHT_RELEVANCE=0.3            # α_relevance

# Azure AI Search (production)
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=<from-portal>
AZURE_SEARCH_INDEX=warnerco-schematics

# Azure OpenAI (for embeddings and reasoning)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=<from-portal>
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

## WARNERCO Schematica Architecture

```
+--------------------------------------------------------------------------+
|                          FastAPI + FastMCP                               |
+--------------------------------------------------------------------------+
|  LangGraph Flow (9-node CoALA-tiered RAG)                                |
|  parse_intent -> query_graph -> inject_scratchpad -> recall_episodes ->  |
|  retrieve -> compress_context -> reason -> respond -> log_episode        |
+--------------------------------------------------------------------------+
|  Four CoALA Memory Tiers (Sumers et al. 2024)                            |
|  +------------+  +-----------+  +----------+  +------------------------+ |
|  | Working    |  | Episodic  |  | Semantic |  | Procedural             | |
|  | Scratchpad |  | events.db |  | Vector   |  | MCP Prompts (versioned)| |
|  | (SQLite)   |  | (SQLite)  |  | store    |  | catalog://procedural   | |
|  +------------+  +-----------+  +----------+  +------------------------+ |
+--------------------------------------------------------------------------+
|  Consolidation ("sleep cycle"): scratchpad+episodic --(ctx.sample)--> semantic |
+--------------------------------------------------------------------------+
```

**MCP Tools** (28 total registered):
- Vector/Schema: `warn_list_robots`, `warn_get_robot`, `warn_semantic_search`, `warn_memory_stats`, `warn_index_schematic`, `warn_compare_schematics`, `warn_create_schematic`, `warn_update_schematic`, `warn_delete_schematic`, `warn_explain_schematic`
- Interactive (Elicitation/Sampling): `warn_guided_search`, `warn_feedback_loop`, `warn_replacement_advisor`
- Graph: `warn_add_relationship`, `warn_graph_neighbors`, `warn_graph_path`, `warn_graph_stats`
- Scratchpad (CoALA working): `warn_scratchpad_write`, `warn_scratchpad_read`, `warn_scratchpad_clear`, `warn_scratchpad_stats`
- Episodic (CoALA Tier 2): `warn_episodic_log`, `warn_episodic_recall`, `warn_episodic_recent`, `warn_episodic_stats`
- Consolidation (CoALA "sleep cycle"): `warn_consolidate_memory`
- Progressive tool loading (meta): `warn_search_tools`, `warn_describe_tool`

### Progressive Tool Loading

Per Anthropic's "code execution with MCP" guidance — clients can discover tools cheaply instead of pre-loading every full schema. The `warn_search_tools` and `warn_describe_tool` meta tools self-exclude from search results (so `count` ≤ 26 when `total` = 28). See `docs/tutorials/progressive-tool-loading.md`.

```python
warn_search_tools(query="", detail="name")            # cheapest discovery
warn_search_tools(query="graph", detail="summary")    # narrow it down
warn_describe_tool(name="warn_graph_neighbors")       # full schema for one
```

**API Endpoints**:
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/robots` | List schematics |
| GET | `/api/robots/{id}` | Get by ID |
| POST | `/api/search` | Semantic search |
| GET | `/api/memory/stats` | Backend stats |
| GET | `/api/graph/stats` | Graph statistics |
| GET | `/api/graph/neighbors/{id}` | Entity neighbors |
| GET | `/api/scratchpad/stats` | Scratchpad statistics |
| GET | `/api/scratchpad/entries` | Scratchpad entries |
| GET | `/docs` | OpenAPI docs |
| GET | `/dash/scratchpad/` | Scratchpad dashboard |

### Graph Memory (Knowledge Graph Layer)

WARNERCO Schematica includes a Graph Memory layer demonstrating hybrid RAG architectures. It runs alongside the vector store to enable relationship-based queries.

**Why Graph Memory?** Vector search finds *similar* things; graph queries find *connected* things. Use both for comprehensive retrieval.

**Components**:

| File | Purpose |
|------|---------|
| `app/models/graph.py` | Entity and Relationship Pydantic models |
| `app/adapters/graph_store.py` | SQLite persistence + NetworkX traversal |
| `scripts/index_graph.py` | Populate graph from schematics.json |
| `data/graph/knowledge.db` | SQLite database (117 entities, 221 relationships) |

**MCP Graph Tools**:

| Tool | Description |
|------|-------------|
| `warn_add_relationship` | Create triplet (subject, predicate, object) |
| `warn_graph_neighbors` | Get connected entities (in/out/both) |
| `warn_graph_path` | Find shortest path between entities |
| `warn_graph_stats` | Node count, edge count, density |

**Indexed Predicates** (from `scripts/index_graph.py`): `has_tag` (75), `compatible_with` (50), `belongs_to_model` (25), `has_category` (25), `has_status` (25), `contains` (21). The `warn_add_relationship` tool also accepts the legacy vocabulary (`depends_on`, `manufactured_by`, `related_to`).

**Index the Graph**:

```bash
cd src/warnerco/backend
uv run python scripts/index_graph.py
```

**LangGraph Integration**: The `query_graph` node (Node 2 in the pipeline) enriches retrieval context with graph relationships before vector search. It activates for DIAGNOSTIC and ANALYTICS intents, or when queries mention explicit relationships.

### CoALA Four-Tier Memory (Sumers et al. 2024)

The 9-node LangGraph pipeline maps each node to a CoALA tier so a class can see all four tiers exercised in one turn:

| CoALA Tier | What it stores | Backed by | LangGraph node | Read tools | Write tools |
|------------|----------------|-----------|----------------|------------|-------------|
| Working | This-session observations & inferences | `data/scratchpad/notes.db` (SQLite) | `inject_scratchpad` | `warn_scratchpad_read` | `warn_scratchpad_write` |
| Episodic | Timestamped past events with importance | `data/episodic/events.db` (SQLite) | `recall_episodes` (gated) + `log_episode` (always) | `warn_episodic_recall`, `warn_episodic_recent`, `warn_episodic_stats` | `warn_episodic_log` (auto from `log_episode`) |
| Semantic | Durable, generalizable facts (incl. consolidated `FACT-*` records) | Vector store (Chroma/Azure/JSON) | `retrieve` | `warn_semantic_search`, `warn_get_robot`, `warn_list_robots` | `warn_index_schematic`, `warn_consolidate_memory` |
| Procedural | Versioned skills/workflows | MCP `@mcp.prompt()` registrations | (user-invoked, not in pipeline) | `memory://procedural-catalog` | source control + CI |

**Episodic recall scoring** (`app/adapters/episodic_store.py`) follows Park et al.'s formula:

```
total = α_recency · 0.5^(hours_since / half_life)
      + α_importance · stored_importance
      + α_relevance · bag_of_words_cosine(query, summary+content)
```

Per-event score breakdown is exposed via `warn_episodic_recall` so students can see why each memory surfaced. **Pedagogical simplification:** relevance is bag-of-words cosine, not embeddings — swap-in is at `_relevance()`. Recall is gated to ANALYTICS/DIAGNOSTIC intents only (LOOKUP/SEARCH skip — narrate during demo).

**Consolidation cycle** (`app/langgraph/consolidate.py`) is the "sleep cycle" — uses `ctx.sample()` to read recent scratchpad+episodic memory, extract durable facts, and write them to the vector store as synthetic `Schematic` records (`id=FACT-*`, `category=consolidated_fact`, `model=MEMORY`). Logs an OBSERVATION back to episodic memory so consolidation itself becomes a memory. **ADD-only** — no Mem0 AUDN dedup.

**Two new resources:**
- `memory://coala-overview` — live JSON snapshot of all four tiers with current counts
- `memory://procedural-catalog` — registered MCP Prompts with version metadata

**Run the four-tier demo:** see `docs/tutorials/coala-memory-walkthrough.md` for the ~4-minute classroom path.

### Scratchpad Memory (Session Working Memory)

WARNERCO Schematica includes a Scratchpad Memory layer for session-scoped observations and inferences. It provides working memory that persists during a conversation but resets between sessions.

**Why Scratchpad Memory?** Vector stores find similar things. Graph stores find connected things. Scratchpad stores remember things from the current session. Use all three for comprehensive context.

**Components**:

| File | Purpose |
|------|---------|
| `app/models/scratchpad.py` | ScratchpadEntry, ScratchpadStats, predicate vocabulary |
| `app/adapters/scratchpad_store.py` | In-memory store with LLM minimization/enrichment |
| `static/dash/scratchpad/` | Scratchpad dashboard |

**MCP Scratchpad Tools**:

| Tool | Description |
|------|-------------|
| `warn_scratchpad_write` | Store an observation with optional LLM minimization |
| `warn_scratchpad_read` | Retrieve entries with optional filtering and enrichment |
| `warn_scratchpad_clear` | Clear entries by subject or age |
| `warn_scratchpad_stats` | Token usage, entry counts, savings metrics |

**Supported Predicates**: `observed`, `inferred`, `relevant_to`, `summarized_as`, `contradicts`, `supersedes`, `depends_on`

**Configuration** (in `.env` or `app/config.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| `scratchpad_db_path` | `data/scratchpad/notes.db` | SQLite database path (relative to backend/) |
| `scratchpad_inject_budget` | 1500 | Tokens for LangGraph injection |

**LangGraph Integration**: The `inject_scratchpad` node (Node 3 in the pipeline) adds session context between graph query and vector retrieval. Entries are formatted as `[predicate] subject -> object: content` and injected into the compressed context under "Session Memory (Scratchpad)".

## Claude Code Agents and Skills

This repo includes Claude Code agents (`.claude/agents/`) and skills (`.claude/skills/`):

**Agents**:
- `python-mcp-server-expert` - FastMCP development guidance
- `azure-principal-architect` - Azure WAF assessments

**Skills**:
- `mcp-server-builder` - Build MCP servers in Python/JS/TS
- `warnerco-schematica` - WARNERCO Robotics Schematica development

## MCP Resources

- **MCP Specification**: https://spec.modelcontextprotocol.io/
- **TypeScript SDK**: `@modelcontextprotocol/sdk` (npm)
- **Python SDK**: `mcp` (pip) or `fastmcp` (pip)
- **MCP Inspector**: `@modelcontextprotocol/inspector`
