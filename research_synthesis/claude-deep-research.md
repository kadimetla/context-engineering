# Memory architectures for MCP-native agent systems

## Section 1 — Executive summary

Enterprise teams building agent systems in 2026 are converging on a **layered memory model** that has stopped pretending a single vector store will suffice. The dominant production pattern stratifies memory into four CoALA-aligned tiers (working, episodic, semantic, procedural), exposes each tier through distinct **Model Context Protocol (MCP) primitives**, and isolates tenants with namespace- or partition-level controls rather than runtime metadata filters. The shift from RAG-as-memory to **fact-extraction-then-embed** pipelines (Mem0, Zep) is now the default for any agent expected to live longer than a single session, and bi-temporal knowledge graphs (Graphiti) are the ascendant primitive for any workload that needs auditable "what was true on date X" semantics.

**Five highest-confidence enterprise patterns:**

First, **vector stores alone fail past a few thousand turns** — every published long-conversation benchmark (LongMemEval, LoCoMo) shows 30–60 % accuracy collapse for naive RAG over multi-session histories ([LongMemEval, ICLR 2025](https://arxiv.org/abs/2410.10813)). Production teams now combine vector retrieval with graph-structured semantic memory and explicit summarization. Second, **multi-tenancy is decided in the data plane**, not the query: Pinecone namespaces, Weaviate native multi-tenancy, Qdrant tiered partitioning, and Azure AI Search index-per-tenant are vendor-recommended over filter-by-tenant approaches that silently leak under filter-validation bugs ([OWASP LLM08:2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)). Third, **MCP Tools express procedural and write-side memory; MCP Resources express semantic memory; MCP Prompts encode procedural skills** — this primitive-to-memory mapping is now stable enough to design against. Fourth, **fact-extraction at write time** (Mem0's ADD/UPDATE/DELETE/NOOP and Zep's bi-temporal invalidation) beats summarization at read time on token economics and contradiction handling. Fifth, **observability remains the weakest link**: no major platform ships a first-class memory dashboard, and teams instrument recall@k, faithfulness, and drift through Phoenix or Langfuse on top of OTel GenAI semantic conventions.

**Three contested questions where the field has not converged:**

The first is **Mem0 v2 vs v3**: Mem0's April 2026 release replaced the AUDN write-time conflict-resolution pipeline with single-pass extraction plus retrieval-time conflict scoring ([release notes](https://newreleases.io/project/github/mem0ai/mem0/release/v2.0.0)). Whether write-time or query-time conflict handling is correct at scale is unsettled, and the academic paper still describes the v1/v2 algorithm. The second is the **temporal-graph vs vector-with-decay debate**: Zep's [2501.13956](https://arxiv.org/abs/2501.13956) and Mem0's [2504.19413](https://arxiv.org/abs/2504.19413) papers report mutually contradictory benchmark wins on overlapping evals — both are vendor-authored, neither is independently reproduced, and engineering teams should treat both as marketing-adjacent. The third is whether **GDPR Article 17 is satisfied by deleting the vector**: legal opinion is unsettled because embeddings are partially invertible and bi-temporal graphs (Graphiti) preserve invalidated edges by design — making them GDPR-hostile without an explicit hard-delete pipeline.

The recommended posture for a senior engineering team is to **design for memory typology first, framework second**: pick the CoALA tiers your workload actually needs, decide which MCP primitive carries each, then choose a framework that does not fight you. The remainder of this synthesis maps that decision onto current tooling and enterprise constraints.

## Section 2 — Memory type reference card

The taxonomy below uses the [CoALA paper](https://arxiv.org/abs/2309.02427) (Sumers, Yao, Narasimhan, Griffiths, TMLR Feb 2024, v3 March 2024) as the canonical reference. Each memory type is defined by its retention semantics, not its storage substrate.

### Working memory

**Definition.** The active scratchpad of the current decision cycle: perceptual inputs, in-flight tool outputs, the rolling conversation turn, and any active goals — *"a data structure that persists across LLM calls"* (CoALA §4.1).

**Canonical implementation.** A typed state object passed across an agent's reasoning loop. In LangGraph this is the **checkpointer state** (`MemorySaver` for dev, `PostgresSaver` for production). In MemGPT/Letta it is the **core memory blocks**. In Google ADK it is **Session State** (`session.state` dict with `user:`, `app:`, `temp:` prefix conventions). In FastMCP it is `ctx.set_state` / `ctx.get_state` keyed off the MCP session.

**MCP primitive.** **Elicitation** writes into working memory from the user mid-task; **Sampling** transforms working memory contents via host-side LLM calls. Working memory itself is not exposed as an MCP primitive — it is the host's responsibility.

**WARNERCO Schematica.** Implemented as the in-memory scratchpad node `inject_scratchpad` in the 7-node LangGraph pipeline (verified in repo README architecture diagram). No GAP.

**Common confusion.** Working memory is *not* the context window — the context window is its serialization. A 200K-token Claude run with no scratchpad object has a large context window and effectively zero working memory.

**Production examples.** Letta's editable `human` and `persona` blocks; LangGraph's checkpointed `messages` and structured fields; ADK Session State.

### Episodic memory

**Definition.** Specific past events with temporal indexing — what happened on Tuesday, what tools were called in run 4f2c, what the user said three sessions ago.

**Canonical implementation.** Append-only event log indexed by time and (usually) by content embedding. In Letta this is **recall memory** (the perpetual searchable message history kept out of context). In Zep it is the **episode subgraph** (raw messages preserved non-lossily) plus the bi-temporal edges in Graphiti. In Mem0 it is the **history table** logging every ADD/UPDATE/DELETE event. In OpenAI's stack it is **Conversations API** items.

**MCP primitive.** **Resources** with stable URIs are the natural carrier — episodic events are addressable, idempotent reads. **Tools** wrap the write path (`add_episode`, `archival_memory_insert`). **Roots** namespace the episodic scope (per-project, per-workspace).

**WARNERCO Schematica.** **GAP.** The repo's graph store is described as "SQLite + NetworkX (Knowledge Graph)" with no temporal/bi-temporal dimension surfaced. Adding Graphiti or a homegrown bi-temporal edge model is the single largest extension opportunity.

**Common confusion.** Episodic memory is *not* conversation history. Conversation history is a chronological dump; episodic memory is *retrievable* by relevance, recency, and importance — Park et al.'s [Generative Agents](https://arxiv.org/abs/2304.03442) `α_recency·recency + α_importance·importance + α_relevance·relevance` formula is the canonical retrieval signature.

**Production examples.** Letta recall memory; Zep Graphiti episode subgraph; OpenAI Conversations items; Generative Agents memory stream.

### Semantic memory

**Definition.** Generalized facts and knowledge extracted from experience — *"the agent's knowledge about the world and itself"* (CoALA §4.1). Crystallized truth, not raw transcript.

**Canonical implementation.** Vector store of extracted fact-statements, optionally augmented with a knowledge graph. Mem0 extracts facts via two-phase LLM pipeline and stores them in Qdrant/Chroma/PGVector with optional Neo4j/Memgraph (Mem0g). Zep stores them as edges in Graphiti's semantic entity subgraph. Reference MCP memory server stores entity-relation-observation triples in a JSON knowledge graph.

**MCP primitive.** **Resources** for stable knowledge documents; **Tools** for `search_memory`/`add_memory` write/read paths. The 2025-06-18 spec's **resource_link** in tool results enables the citation pattern: a `search_memory` tool returns `resource_link` URIs, the host fetches passages on demand.

**WARNERCO Schematica.** Implemented via the JSON → ChromaDB → Azure AI Search progression and the SQLite+NetworkX knowledge graph. No GAP for the basic semantic tier; teams will outgrow JSON quickly and want Azure AI Search wired earlier than the course's progression suggests.

**Common confusion.** Semantic memory is *not* RAG. RAG is a retrieval pattern that can serve any memory type; semantic memory is the *content category* of extracted facts. A RAG over yesterday's chat log is episodic-memory retrieval, not semantic.

**Production examples.** Mem0 fact store; Zep semantic entity subgraph; OpenAI vector_stores; Anthropic Claude Skills package contents.

### Procedural memory

**Definition.** Skills, workflows, and learned action sequences — *"procedures that implement actions … and procedures that implement decision-making itself"* (CoALA §4.1). System prompts, tool definitions, code, and slash-command playbooks all qualify.

**Canonical implementation.** Three forms in production: **(a)** prompts and prompt templates (system prompts, MCP Prompts, OpenAI Prompts dashboard objects), **(b)** tool/skill registries (Anthropic Skills, Letta Code, FastMCP `SkillsProvider`), **(c)** learned prompt-rewrite outputs (LangMem prompt optimization, Letta sleep-time agents).

**MCP primitive.** **Prompts** are the canonical carrier — user-invoked, parameterized, completion-aware playbooks. **Tools** carry executable skills. CoALA explicitly warns *"learning new actions by writing to procedural memory … is significantly riskier than writing to episodic or semantic memory, as it can easily introduce bugs or allow an agent to subvert its designers' intentions"* — which justifies the spec's choice to make Prompts user-controlled rather than model-selectable.

**WARNERCO Schematica.** **GAP.** The repo exposes Tools but the README does not document MCP Prompts. Adding a `/code_review`, `/incident_runbook`, or `/schematica_query` Prompt suite is a clear procedural-memory extension and maps onto Segment 2 of the course directly.

**Common confusion.** Procedural memory is *not* the system prompt. A system prompt is one *instance* of procedural memory baked into a session; procedural memory is the broader category that includes versioned playbooks, tool definitions, and skill packages that survive across sessions.

**Production examples.** Anthropic Skills with `skills-2025-10-02` beta header; LangMem prompt optimizer; Letta Code MemFS; OpenAI Prompts dashboard objects.

### Reconciliation matrix — CoALA across industry framings

| CoALA | Letta / MemGPT | Mem0 | Zep / Graphiti | LangMem / LangGraph | OpenAI Responses | Anthropic Claude |
|---|---|---|---|---|---|---|
| Working | Core memory blocks | (in prompt context) | Assembled context block | Checkpointer state | `instructions` + Conversation items in flight | Context window + memory tool view |
| Episodic | Recall memory | History DB rows | Episode subgraph (bi-temporal) | LangGraph Store (namespaced) | Conversations API items | Memory tool `/memories` files (append) |
| Semantic | Archival memory | Fact store (vector + optional graph) | Semantic entity + community subgraphs | LangGraph Store + LangMem fact extraction | Vector stores (`vector_store_ids`) | Skills resources; memory tool persistent files |
| Procedural | Tools + Letta Code skills | Procedural-memory parameter (sparse) | None first-class | LangMem prompt optimization | Prompts dashboard objects | Skills (`skills-2025-10-02`) |

The matrix exposes two truths the marketing literature obscures: **no framework cleanly implements all four CoALA types**, and **procedural memory is the most under-served tier** — only Letta, LangMem, and Anthropic Skills make a serious attempt at it.

## Section 3 — MCP memory layering architecture (keystone)

The MCP specification version **2025-06-18** is the reference for everything below ([spec index](https://modelcontextprotocol.io/specification/2025-06-18)). Material differences from earlier versions are noted inline. Where 2025-11-25 introduced further changes (CIMD client registration, JSON Schema 2020-12, ElicitResult enhancements), these are flagged but out of scope.

### Reference architecture

```mermaid
flowchart TB
  subgraph Host[ MCP Host - Claude Desktop / VS Code / LangGraph runtime ]
    H[Host orchestrator]
    WM[Working memory: scratchpad / checkpointer state]
  end

  subgraph EpiMCP[Episodic memory MCP server - stateful HTTP]
    ET[Tools: add_episode, search_episodes, recall_at]
    ER[Resources: episode://session/{id}]
    EB[(Bi-temporal graph: Graphiti / Neo4j)]
  end

  subgraph SemMCP[Semantic memory MCP server - stateless HTTP]
    ST[Tools: add_fact, search_facts, delete_fact]
    SR[Resources: fact://entity/{id}]
    SB[(Vector store + KG: Azure AI Search + NetworkX)]
  end

  subgraph ProcMCP[Procedural memory MCP server - stdio or HTTP]
    PT[Tools: invoke_skill]
    PP[Prompts: /code_review, /incident_runbook]
    PRes[Resources: skill://name/{version}]
  end

  subgraph IdP[OAuth 2.1 Authorization Server - Entra / Auth0]
    AS[Token issuance, RFC 8707 audience binding]
  end

  H -- Sampling, Elicitation --> H
  H <-- JSON-RPC over Streamable HTTP --> EpiMCP
  H <-- JSON-RPC over Streamable HTTP --> SemMCP
  H <-- JSON-RPC over stdio --> ProcMCP
  EpiMCP -.token validation.-> AS
  SemMCP -.token validation.-> AS
  WM -. compress via Sampling .-> SemMCP
```

The host owns working memory and orchestrates Sampling and Elicitation. Each long-term tier lives behind its own MCP server, isolating failure domains and per-tier scaling characteristics. The host is the only component that sees all three; servers do not coordinate horizontally — composition is host-side by spec ([architecture overview](https://modelcontextprotocol.io/specification/2025-06-18/architecture)).

### Primitive-to-memory mapping (concrete)

**Tools** are the canonical write/read interface for any memory tier. Memory mutations are model-controlled and have side effects — exactly the Tool semantic. The 2025-06-18 spec adds `structuredContent` plus optional `outputSchema`, which means a `search_memory` tool's result is now a typed object the host can validate, not a JSON-stringified blob. Pair with `resource_link` (also new in 2025-06-18) for citation-based retrieval: `search_memory` returns links, host fetches passages lazily.

**Resources** are the carrier for **semantic-memory documents and episodic-memory snapshots** with stable URIs. The `subscribe`/`notifications/resources/updated` channel makes them suitable for time-aware memory views — a fact whose value just changed pushes an update to subscribed hosts. Resources are *application-driven, not model-driven*: the host decides when to inject them, which keeps them out of the chain-of-thought trace and saves tokens.

**Prompts** carry **procedural memory** that requires human invocation: codified playbooks, review checklists, multi-turn workflow seeds. They are exposed as slash commands in clients like VS Code's [June 2025 full-MCP rollout](https://code.visualstudio.com/blogs/2025/06/12/full-mcp-spec-support). The CoALA injunction against autonomous procedural learning is operationally enforced by Prompts being user-controlled, not model-selectable.

**Sampling** is not memory — it is the substrate for **memory transformations**. Servers use it to summarize episodic memory into semantic memory, extract facts, re-rank retrieved episodes, and rewrite Letta-style core memory blocks. Production caveat: as of 2025-06-18, only VS Code reliably implements sampling among major hosts; Claude Desktop did not at the spec date. Servers depending on sampling have constrained client coverage.

**Elicitation** (new in 2025-06-18, [PR #382](https://mcpcn.com/en/specification/2025-06-18/changelog/)) is the channel for **acquiring working-memory contents from the human** mid-task. Schemas are deliberately constrained to flat objects with primitive properties — string, number, boolean, enum — to keep client UIs simple. The spec rule *"Servers MUST NOT use elicitation to request sensitive information"* is binding. Elicitation is the right channel when the answer is authoritative human knowledge; Sampling is the right channel when the answer is model reasoning.

**Roots** delimit the **scope of episodic and semantic memory** available to the server — not memory themselves but the namespace boundary. The 2025-06-18 restriction to `file://` URIs is a real gap for multi-tenant memory servers; abstract namespaces (org IDs, customer IDs) must be passed as tool arguments instead. The 2025-11-25 revision adds richer URI support, partially addressing this.

### Stateful vs stateless server decision tree

```
Does the server need to:
├── make sampling/elicitation/roots requests back to the client? → STATEFUL (mandatory)
├── persist scratch state across calls in the same session? → STATEFUL
├── support resumability via Last-Event-ID? → STATEFUL
├── run behind a serverless / edge runtime without sticky sessions? → STATELESS
└── horizontal scale across regions without coordination? → STATELESS

Default for memory servers:
├── Episodic memory MCP → STATEFUL HTTP (sampling for summarization, sticky session)
├── Semantic memory MCP → STATELESS HTTP (every call self-contained, scales horizontally)
└── Procedural memory MCP → STATEFUL stdio (local, single-user, low latency)
```

The C# SDK's stateless-mode caveat is operationally important: *"Client sampling, elicitation, and roots capabilities are also disabled in stateless mode, because the server cannot make requests"* ([HttpServerTransportOptions](https://csharp.sdk.modelcontextprotocol.io/api/ModelContextProtocol.AspNetCore.HttpServerTransportOptions.html)). A horizontally scaled stateless memory MCP cannot use elicitation to clarify ambiguous queries — that is the price of stateless deployment.

The `Mcp-Session-Id` header is the unit of state association. The spec requires cryptographically secure UUIDs, ASCII-printable, returned on `InitializeResult`, sent on every subsequent request, and explicitly invalidatable via `HTTP DELETE`. Session ID is **not a stable user identifier** — a new client connection produces a new session and therefore new working memory. User identity must be carried separately via OAuth bearer tokens with RFC 8707 resource indicators.

### Multi-server composition patterns

Composition is host-side by spec; the host maintains 1:1 client connections to each server. Three patterns dominate:

The **dedicated-server-per-tier pattern** (illustrated above) puts episodic, semantic, and procedural memory behind separate MCP servers. Tool naming uses dotted namespacing — `episodic.search`, `semantic.search` — and the 2025-06-18 split between `name` (ID) and `title` (display) lets you keep IDs namespaced while showing clean labels. This is the strongest isolation; failure of the procedural server does not degrade semantic recall.

The **umbrella-proxy pattern** uses FastMCP's [Proxy + Transforms](https://deepwiki.com/jlowin/fastmcp/4.4-client-side-authentication) to front a single host-facing MCP server that internally proxies to per-tier backends. Transforms rename tools, hide internals, and apply visibility-by-session rules. This trades isolation for a single connection from the host's perspective; useful when host configuration is constrained (Claude Desktop's per-server setup overhead).

The **embedded-server pattern** runs each tier's logic in-process under a shared server. Lowest latency, highest blast radius. Appropriate for development and single-tenant deployments; not recommended for multi-tenant production.

### Schematica's pipeline alignment and extension

The 7-node LangGraph pipeline in WARNERCO Schematica (`parse_intent → query_graph → inject_scratchpad → retrieve → compress → reason → respond`) maps onto the architecture cleanly:

- `parse_intent` and `inject_scratchpad` operate on **working memory** (scratchpad, current turn).
- `query_graph` and `retrieve` are **semantic-memory reads** through MCP Tools.
- `compress` is a **summarization-as-compression** step that should call **Sampling** when running under a sampling-capable host.
- `reason` and `respond` are pure LLM steps that consume working memory.

**Extensions Schematica should add to fully exercise the course material:**

1. An **episodic-memory MCP server** with bi-temporal edges (Graphiti pattern), exposing `add_episode` and `recall_at(timestamp)` Tools plus `episode://` Resources. This addresses the verified GAP.
2. A **procedural-memory Prompt suite** (`/schematica_query`, `/code_review`) registered as MCP Prompts. This addresses the verified GAP.
3. A **Sampling-driven reflection node** between `compress` and `reason` that periodically distills episodic memories into semantic facts via host-side LLM calls. This addresses the third verified GAP.

## Section 4 — Framework comparison matrix

All claims below are from primary docs; tagged **PROD** (≥6 months in market), **EMERGING** (<6 months), or **RESEARCH** where applicable. Production readiness is on a 1–5 scale calibrated against multi-tenant, audited enterprise deployment.

| Framework | Memory types (CoALA) | Persistence | Multi-tenancy | MCP-native | Observability | License | Prod readiness | Best fit | Notable limitation | Last meaningful activity |
|---|---|---|---|---|---|---|---|---|---|---|
| [LangMem](https://github.com/langchain-ai/langmem) | Sem + Ep + **Proc** (prompt opt) | LangGraph BaseStore | Namespace tuples | Adapter only | LangSmith | MIT | 3 | Background memory consolidation in LangGraph stacks | ~60s p95 latency; not for hot path | Active main, Apr 2026 |
| [LangGraph built-in](https://docs.langchain.com/oss/python/langgraph/add-memory) | Work + Ep + Sem | Memory/SQLite/Postgres+pgvector | Namespace tuples | Adapter only | LangSmith | MIT | 5 | Default storage primitive for LangGraph | No built-in conflict resolution or temporal | 1.x train, Apr 2026 |
| [Mem0](https://github.com/mem0ai/mem0) | Sem + Ep | Hybrid: vector + KV + opt graph | `user_id`/`agent_id`/`run_id` filters | Native server (`mcp.mem0.ai`) | PostHog telemetry | Apache 2.0 | 4 | Cross-framework personalization | v2→v3 algorithm fork | v3 Apr 2026 |
| [Letta](https://github.com/letta-ai/letta) | Work + Ep + Sem + Proc | Postgres + vector | Per-agent + MemFS git | Native (host + server) | Agent Dev Environment | Apache 2.0 | 4 | Long-running autonomous agents | Adopt the whole runtime | Letta Code Mar 2026 |
| [Zep / Graphiti](https://github.com/getzep/graphiti) | Ep + Sem (bi-temporal) | Neo4j / FalkorDB / Kuzu | Per-user graph | Graphiti MCP | Zep Cloud dash | Apache 2.0 (Graphiti); Cloud proprietary | 4 (Cloud) / 2 (self-host) | Audit / "as-of" temporal queries | CE deprecated Apr 2025 | Graphiti continuous |
| [Cognee](https://github.com/topoteretes/cognee) | Ep + Sem + Proc (Tasks) | Vector + Graph | Per-user graphs + permissions | Native | Logging | Apache 2.0 | 3 | Privacy-critical local-first | Smaller ecosystem | Continuous through 2026 |
| [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/) | Sem (+Ep via KM) | 15+ vector connectors | Per-collection or filter | Via M.E.AI | OTel + Azure Monitor | MIT | 4 | Azure-native enterprise .NET | Mid-migration to Vector Stores; M.E.AI convergence | Convergence in flight |
| [Google ADK](https://google.github.io/adk-docs/) | Work (state) + Ep + Sem | InMemory / DB / Vertex Memory Bank | `(app, user, session)` keys | Tool-side only | Cloud Trace | Apache 2.0 (SDK); Memory Bank proprietary | 4 (3 for Memory Bank Preview) | GCP / Vertex AI shops | Memory Bank still Preview | v1.22 Apr 2026 |
| [AutoGen](https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/memory.html) | Sem (protocol-agnostic) | Plugin: Chroma / Mem0 / Redis | Backend-defined | Tool-side | OTel | MIT | 3.5 | Multi-agent orchestration | Converging with SK | Convergence in flight |
| [OpenAI Responses + Conversations](https://developers.openai.com/api/docs/assistants/migration) | Ep + Sem | Managed | Conv + vector_store IDs | Native | OpenAI dash | Proprietary | 4 | OpenAI-only greenfield | Assistants sunset 26 Aug 2026 | Active |
| [Anthropic Memory Tool + Skills](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool) | Sem + Proc (file-shaped) | Client-side (your storage) | Your design | MCP author | Anthropic dash | Proprietary | 3.5 | Claude-native, regulated, ZDR | Beta header; ~2.5K-token system prompt overhead | Memory Tool Sep 2025; Skills Oct 2025 |

### Decision guide — three deployment scenarios

**Enterprise greenfield.** Choose **Mem0 or Letta** as the long-term memory layer based on autonomy needs. Mem0 if your agents are LLM-frontends to existing systems and you want personalization that travels across frameworks (CrewAI, LangChain, AutoGen, AWS Strands — Mem0 is the [exclusive memory provider for AWS Agent SDK](https://mem0.ai/series-a)). Letta if you are building a long-running autonomous agent and willing to commit to its runtime. Wrap both behind your own MCP umbrella server using FastMCP transforms so you can swap providers without re-wiring host configurations. Add Zep Cloud only if your domain demands "as-of" temporal queries (legal, healthcare, financial advisory) — the bi-temporal graph cost is not justified otherwise.

**Retrofit onto existing LangChain stack.** **LangGraph built-in is the foundation, not the question.** Use `PostgresSaver` for short-term state and `PostgresStore` with pgvector for long-term store. Add **LangMem** as a *background* consolidation worker — it is too slow (~60s p95 on LOCOMO) for hot path. If you need cross-framework portability or richer fact extraction than LangMem provides, layer **Mem0** on top via its memory tools. Avoid migrating off LangGraph just to adopt Letta or Zep — the integration cost rarely pays off, and LangGraph 1.x is the most enterprise-credible OSS primitive in the lineup at production-readiness 5/5.

**Microsoft / Azure shop.** **Semantic Kernel Vector Stores plus Azure AI Search**, with the explicit understanding that you are adopting a stack mid-migration. The legacy `IMemoryStore` API is deprecated in favor of `Microsoft.Extensions.VectorData.VectorStore`; SK and AutoGen are converging into the [Microsoft Agent Framework](https://github.com/microsoft/semantic-kernel/discussions/12299). Use the index-per-tenant pattern up to 200 indexes/service on the S3 High-Density tier; switch to Cosmos DB DiskANN with hierarchical partition keys above that. Wire OpenTelemetry GenAI semantic conventions through M.E.AI to Azure Monitor. MCP integration ships via M.E.AI and is production-supported. The trade-off is preview labels on the new APIs and a real migration project off legacy Memory Stores.

## Section 5 — Enterprise patterns playbook

### Pattern 1 — Hot/warm/cold memory tiering

**Problem.** Every token in the prompt costs money and increases latency, but evicting too aggressively causes the model to "forget" relevant context mid-task. Naive single-tier memory either bloats the context window or starves the agent of necessary state.

**Forces.** Token cost grows linearly with context length; LLM attention quality degrades non-linearly with context length; retrieval latency is bounded by network and vector-store query time; user-perceived latency must stay under ~2 seconds for interactive flows.

**Solution sketch.** Three tiers, mapped to MCP primitives. **Hot** tier is the in-context working memory injected via the prompt (carried by the host's checkpointer state). **Warm** tier is fact-extracted semantic memory in a vector store, retrieved via MCP Tools (`search_memory`) with sub-100ms p95. **Cold** tier is full episodic transcripts in object storage or a deactivated vector partition (Weaviate's OFFLOADED state, Mem0 archived buckets), addressable via MCP Resources but not pre-fetched. Promotion from cold to warm is on-demand via Sampling-driven reflection; demotion from warm to cold is by importance score below a threshold combined with last-access decay.

**Known uses.** [Weaviate ACTIVE/INACTIVE/OFFLOADED](https://weaviate.io/blog/launching-into-production); Letta core/recall/archival mapping directly to hot/warm/cold; Mem0's history-table archival pattern; Anthropic's [memory tool + context editing](https://www.anthropic.com/news/context-management) which reports 84 % token reduction on 100-turn web-search tasks (vendor self-reported, validate per workload).

**Consequences.** Operational complexity goes up; you now run three storage substrates. Misconfigured promotion policies cause "warming storms" where many cold items rehydrate at once. Cost savings are real (often 40–70 % token spend reduction) but require dashboard discipline.

### Pattern 2 — Reflective consolidation ("sleep cycle")

**Problem.** Raw episodic memory accumulates faster than the agent can usefully retrieve. Every conversation adds dozens of turns; vector recall over thousands of raw turns degrades into noise. Without consolidation, semantic memory stays empty and episodic memory becomes a haystack.

**Forces.** Real-time response budget excludes synchronous LLM-driven extraction; users return after gaps so consolidation can happen offline; contradicting facts must be reconciled; reflection itself can introduce hallucinations into long-term memory.

**Solution sketch.** A scheduled or trigger-based **reflection job** runs out-of-band against accumulated episodic memory. The job uses **MCP Sampling** to call the host's LLM, prompting it to extract salient facts, summarize sessions, and produce higher-order insights — directly modeled on [Park et al.'s Generative Agents](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763) reflection mechanism (trigger when accumulated importance scores cross a threshold; ask "what 3 high-level questions can we answer" → retrieve memories → emit 5 insights with citations). Outputs write into semantic memory via `add_fact` Tools; reflections themselves are stored as episodic memories enabling reflection-of-reflections (the reflection tree).

**Known uses.** Generative Agents; [Letta sleep-time agents](https://www.letta.com/blog/letta-v1-agent); LangMem background memory manager; Anthropic's pairing of memory tool with `clear_tool_uses_20250919` context editing.

**Consequences.** Reflection cost is non-trivial (each cycle is multiple LLM calls); poor prompting produces hallucinated "facts" that pollute semantic memory; without **eviction policies** reflection compounds storage indefinitely. The eviction discipline is what separates production reflection from research reflection.

### Pattern 3 — Per-tenant memory isolation in shared MCP server

**Problem.** A multi-tenant SaaS serving thousands of customers cannot deploy one MCP server per tenant. Yet running shared infrastructure with metadata-filter-only isolation is the OWASP LLM08 textbook vulnerability — one bad filter and tenant A's memory leaks into tenant B's response.

**Forces.** Operational cost of N MCP processes is prohibitive above ~100 tenants; data isolation is contractually mandated for most B2B; compliance frameworks (SOC 2, HIPAA, GDPR) require demonstrable isolation; MCP Roots cannot represent abstract tenant IDs (file:// only in 2025-06-18).

**Solution sketch.** Enforce isolation in the **data plane**, not in application-supplied filter strings. Recommended layering: **(a)** OAuth 2.1 token with `tenant_id` claim, validated by MCP server using RFC 9728 metadata discovery and RFC 8707 audience binding. **(b)** Server extracts `tenant_id` from validated token and applies it as a Pinecone namespace, Weaviate tenant, Qdrant `group_id` payload, or Azure AI Search index name — never as a user-supplied tool argument. **(c)** Tool input schemas explicitly forbid tenant identifiers in arguments to prevent tool-call-induced leaks. **(d)** Integration tests with cross-tenant token swap to verify isolation at every release. **(e)** Audit logging includes tenant ID on every memory operation.

**Known uses.** [Pinecone namespace-per-tenant](https://docs.pinecone.io/guides/index-data/implement-multitenancy) (vendor-recommended default); [Weaviate native multi-tenancy](https://docs.weaviate.io/weaviate/manage-collections/multi-tenancy) with HOT/WARM/COLD lifecycle; [Qdrant tiered multitenancy](https://qdrant.tech/blog/qdrant-1.16.x/) with whales-vs-minnows shard promotion; [Azure AI Search S3 High-Density](https://learn.microsoft.com/en-us/azure/search/search-modeling-multitenant-saas-applications) up to 200 indexes/service.

**Consequences.** Per-tenant indexes scale to ~10K tenants comfortably; above that, payload partitioning forces shared indexes with stricter filter discipline. The OAuth claim-to-namespace mapping must be cryptographically bound; do not pass tenant ID as a path parameter or HTTP header.

### Pattern 4 — Episodic-to-semantic distillation pipeline

**Problem.** Episodic memory is verbose and contains contradictions; semantic memory must be concise and consistent. Direct retrieval over episodic memory is noisy; using it as the only memory tier means every retrieval competes with every prior turn.

**Forces.** Contradictions arise as users change preferences, facts update, agents make mistakes; users want recency-weighted truth, not transcript archaeology; the distillation must be auditable for compliance.

**Solution sketch.** A two-phase pipeline, exactly the [Mem0 architecture](https://arxiv.org/abs/2504.19413). **Phase 1 — Extraction:** LLM reads recent messages plus rolling summary and emits candidate facts as structured statements. **Phase 2 — Update:** for each candidate, vector-search the top-*s* (default 10) similar existing facts and let the LLM tool-call decide **ADD** (new), **UPDATE** (augment, retain ID), **DELETE** (contradicted), or **NOOP** (already present). Bi-temporal variants (Zep / Graphiti) instead mark contradicted facts with `t_invalid` and create new edges with current `t_valid` — preserving auditability at the cost of GDPR-deletion ergonomics. Distillation runs both inline (write-time, AUDN model) or asynchronously (sleep-cycle reflection).

**Known uses.** Mem0 v1/v2 AUDN pipeline; [Zep Graphiti bi-temporal invalidation](https://arxiv.org/abs/2501.13956); LangMem `create_memory_manager`; Trend Micro's "Trend's Companion" on [Bedrock + Neptune + Mem0](https://aws.amazon.com/blogs/machine-learning/company-wise-memory-in-amazon-bedrock-with-amazon-neptune-and-mem0/) (the cleanest named enterprise case study in this category).

**Consequences.** Inline extraction adds 200–500ms to every turn; asynchronous extraction means recently added facts aren't retrievable until consolidation runs. Mem0 v3's April 2026 redesign collapsed AUDN into single-pass extraction with retrieval-time conflict scoring — **pin your version explicitly**; v2 and v3 produce different memory contents from identical inputs.

### Pattern 5 — Procedural memory as MCP Prompts

**Problem.** Skills, playbooks, and "how we do X here" knowledge frequently lives in private wikis and shadow-IT runbooks invisible to agents. Embedding them as system prompts couples them to specific agent versions and prevents user discovery; embedding them as tools makes them autonomously selectable, which CoALA explicitly warns against.

**Forces.** Procedural knowledge changes more slowly than semantic facts but more often than agent code; humans must be able to discover and trigger it; provenance and version are auditable requirements; CoALA: *"learning new actions by writing to procedural memory … is significantly riskier than writing to episodic or semantic memory."*

**Solution sketch.** Encode procedural knowledge as **MCP Prompts** with parameter schemas, version metadata, and embedded resources. Surface in clients as slash commands (VS Code, Claude Code, Cursor all support this in their full-spec rollouts). Store prompt templates in a versioned source repo; CI publishes new versions to a procedural-memory MCP server that exposes them via `prompts/list`. Use the 2025-06-18 `title` field for human-readable display while keeping `name` as a stable namespaced ID (e.g. `name: "ops.incident.runbook"`, `title: "Run incident response"`). Pair with MCP Tools for execution side-effects.

**Known uses.** [VS Code MCP slash commands](https://code.visualstudio.com/blogs/2025/06/12/full-mcp-spec-support); [Anthropic Skills](https://www.anthropic.com/news/skills) with `skills-2025-10-02` beta header; Letta Code's MemFS approach; FastMCP `SkillsProvider` in v3.

**Consequences.** Discovery surface depends on host UX — Claude Desktop's slash menu is less discoverable than VS Code's command palette. Version drift across hosts is a real problem; pin protocol version on `MCP-Protocol-Version` header (mandatory in 2025-06-18). User-controlled invocation means the agent cannot autonomously chain procedures — that's a feature, not a bug.

### Pattern 6 — Memory poisoning containment

**Problem.** Stored prompt injection — malicious instructions embedded in retrieved memories that survive across sessions — is a confirmed production threat (Slack AI exfiltration, Microsoft 365 Copilot's [EchoLeak CVE-2025-32711](https://msrc.microsoft.com/) at CVSS 9.3). OpenAI's December 2025 statement that AI-browser prompt injection "may never be fully solved" applies equally to agent memory.

**Forces.** Indirect prompt injection is unsolved (Greshake et al., [arXiv:2302.12173](https://arxiv.org/abs/2302.12173)); summarization-as-compression can launder injected content into semantic memory; cross-session persistence means one poisoned turn affects future users sharing memory; tool autonomy amplifies the blast radius (OWASP LLM06 Excessive Agency).

**Solution sketch.** Layer defenses. **Provenance tagging:** every memory record carries `source_user_id`, `source_trust_level`, and `created_by_role`. **Quarantine read paths:** retrieved memories from untrusted sources are wrapped in delimited blocks the host's system prompt explicitly marks as untrusted content. **Dual-LLM patterns:** a Quarantined LLM reads untrusted retrieved content and produces structured facts; a Privileged LLM (with tool access) only sees the structured outputs, never raw retrieved text. **Output validation:** every action-taking tool call is gated by allow-list checks; high-risk operations (email send, file delete) require explicit user confirmation via Elicitation. **Drift detection:** statistical monitoring on cosine-similarity distributions in the vector store flags unusual additions (OWASP LLM08 mitigation). **Sandboxed execution:** retrieved memories that produce tool calls execute in restricted contexts with no network egress.

**Known uses.** OWASP [LLM Top 10 v2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/) categories LLM01, LLM04, LLM06, LLM08; OWASP [Agentic Top 10 (Dec 2025)](https://genai.owasp.org/) ASI categories explicitly including memory poisoning; PromptArmor's Slack AI [exfiltration disclosure](https://www.promptarmor.com/resources/data-exfiltration-from-slack-ai-via-indirect-prompt-injection); Simon Willison's "lethal trifecta" risk model.

**Consequences.** Quarantined-LLM patterns add latency and cost (two LLM calls per turn). Provenance tagging requires a write-side discipline that legacy systems often lack. False positives in drift detection cause operational toil. **No defense is complete** — the field consensus in 2026 is that defense-in-depth is mandatory and that high-agency agents handling untrusted content require human-in-the-loop confirmation for irreversible actions.

### Pattern 7 — Embedding model migration with dual-read

**Problem.** Embedding models retire faster than memory stores. OpenAI deprecated `text-embedding-ada-002` on 4 January 2025 with retirement no earlier than 14 June 2025; same-dimensionality successors are *not* drop-in compatible — vector spaces are not interoperable. Naive migration causes silent retrieval-quality collapse.

**Forces.** Re-embedding terabytes of memory is expensive and time-bounded; vector-store storage costs double during parallel-index migrations; users expect continuous service; embedding model identity must be auditable per record.

**Solution sketch.** Tag every record with `embedding_model_id`, `embedding_dim`, `chunking_version`, and `pii_redaction_version`. Run **dual-write** during migration window: every new memory writes to both old and new indexes. Run **dual-read** with model-tag-aware query routing: queries hit the matching index per record's tag, results are merged and re-ranked. After re-embedding completes, retire the old index. For hot/warm tiers re-embed eagerly; for cold tier accept lazy re-embed-on-access. Microsoft's [explicit guidance](https://learn.microsoft.com/en-us/answers/questions/2071109/how-to-migrate-to-the-latest-model) is unambiguous: "upgrading between embedding models is not automatic." [Qdrant 1.16's conditional update API](https://qdrant.tech/blog/qdrant-1.16.x/) was motivated specifically for this scenario.

**Known uses.** Standard practice across regulated enterprises migrating off ada-002; Azure AI Search index-alias swap pattern; Mem0's versioned schema migrations.

**Consequences.** Storage doubles during the migration window; query latency increases ~10–30 % under dual-read; migration windows of 4–12 weeks are realistic for multi-TB stores. **Drift-adapter** approaches (training a learned transformation between embedding spaces) report 95–99 % retrieval recovery in research settings (RESEARCH; not productized); treat any vendor claiming this is "solved" with skepticism.

## Section 6 — Anti-patterns

The following failure modes are observed repeatedly in production post-mortems, vendor case-study disclosures, and OWASP guidance.

**Vector store as the only memory.** Treating Pinecone or Chroma as the sole long-term memory tier conflates retrieval mechanism with memory typology. RAG over a single embedding pool fails on temporal questions, multi-session reasoning, and contradiction handling — the LongMemEval and LoCoMo benchmarks systematically demonstrate this. Add at minimum a fact-extraction layer (Mem0-style) and either a graph or bi-temporal store for episodic facts.

**Unbounded conversation history reaching the context-window cliff.** Carrying every prior turn in the prompt eventually exceeds context limits or degrades attention to the point of incoherence. Even with 1M-token contexts, ICLR 2025's LongMemEval shows 30–60 % accuracy drop on multi-session reasoning. Compress on a cadence; never assume "more context is always better."

**Memory writes without idempotency keys.** Tool retries, network blips, and host crashes all cause duplicate write attempts. Without idempotency keys (deterministic content hash plus user_id and timestamp), the same fact appears multiple times in semantic memory, polluting retrieval. Mem0's hash-based deduplication and Zep's edge-uniqueness checks address this; bespoke memory layers frequently miss it.

**Embedding model swap without re-indexing strategy.** Changing from ada-002 to embedding-3-small without dual-write/dual-read silently corrupts retrieval quality because the two vector spaces are incompatible. Same-dimensionality is a particular trap — the system "works" but recall drops 20–40 % invisibly.

**Conflating session state with long-term memory.** Storing user preferences in the LangGraph checkpointer rather than the long-term Store means preferences vanish when the thread ends. Conversely, putting per-turn scratch state into the long-term store pollutes semantic memory with ephemera. The distinction is operational, not academic: short-term checkpointer = thread-scoped; long-term Store = cross-thread.

**Reflection without eviction.** Running periodic reflection that writes summaries back into memory without an eviction policy compounds storage indefinitely. Generative Agents' approach — reflections become memories that can themselves trigger reflection — without bounded importance-decay produces a write amplifier.

**Multi-tenant data in shared collection without filter validation.** Relying on application-supplied tenant filters in queries is OWASP LLM08's textbook leak vector. Filter strings are user input and must be cryptographically bound to authenticated identity, not trusted from tool arguments. Enforce in the data plane (namespace, partition, index-per-tenant), not the query.

**MCP Tool sprawl — too many fine-grained memory ops.** Exposing 30+ tools (`get_memory_by_id`, `list_memories_by_user`, `update_memory_field`, `add_memory_tag`, ...) consumes context budget and confuses the model. Production MCP memory servers converge on 5–8 tools (`add`, `search`, `update`, `delete`, `list`, optionally `get`, `delete_all`, `delete_entity`). Hierarchical tool-search patterns (a meta-tool that progressively discloses tools by category) are emerging for cases where 30+ tools are unavoidable.

**Trusting retrieved memory as instruction.** The most operationally damaging anti-pattern: agents that read a retrieved memory and follow its content as if it were a system prompt. This is the indirect-prompt-injection root cause. Memories are *data*; only the system prompt and user message are *instruction*. Wrap retrieved content in delimited untrusted-content blocks.

**Bi-temporal graph as GDPR-deletion mechanism.** Graphiti's `t_invalid` marks an edge as logically superseded but preserves it for auditability. This is *not* deletion; an Article 17 erasure request requires hard removal of the underlying edge plus any backups. Treat bi-temporal models as compliance-friendly for *audit trails* and compliance-hostile for *forgetting* until you build an explicit hard-delete path.

**Embedding raw turns instead of extracted facts.** Embedding "User: I prefer espresso. Assistant: Got it!" produces a high-dimensional vector contaminated by conversational filler. Extract the fact ("user prefers espresso") and embed *that*. Mem0 and Zep both demonstrate this; teams building bespoke RAG-as-memory frequently miss it and wonder why retrieval is noisy.

**Putting MCP Tools where MCP Resources belong.** Exposing every memory document as a Tool means every retrieval enters the chain-of-thought trace and consumes tokens twice (once in the tool call, once in the result). Static or slow-changing knowledge belongs behind Resources with stable URIs; the host fetches when needed without polluting the trace.

## Section 7 — Observability and eval

The state of agent-memory observability in April 2026 is genuinely poor — not because the tools are bad but because **no major platform ships a first-class memory dashboard**. Production teams compose memory-specific instrumentation on top of general LLM observability platforms.

### Tool landscape

[**Langfuse**](https://langfuse.com/) is the strongest open-source baseline (Apache 2.0, fully self-hostable on Postgres + ClickHouse + Redis + S3). Best-in-class general LLM tracing, prompt management with versioning, and LLM-as-judge evals. Free unlimited self-hosting with no per-seat cost. PII masking is supported. Acquired by ClickHouse in January 2026; OSS code remains but long-term direction is uncertain.

[**Arize Phoenix**](https://arize.com/docs/phoenix) ships under Elastic License 2.0 and is OpenTelemetry-native via OpenInference. Single-Docker self-host. RAG-specific metrics (context precision, recall, faithfulness) and clustering / anomaly detection on traces. Phoenix's auto-instrument captures LangChain, LlamaIndex, and OpenAI hierarchical spans without configuration. Strongest agent-eval support among true OSS options.

[**LangSmith**](https://docs.smith.langchain.com/) is closed-source with the tightest LangChain / LangGraph fit and the best agent IDE (LangGraph Studio with breakpoints, state mutation, checkpoint-resume). Self-host is Enterprise-only; per-seat pricing scales painfully. OTel ingest was added March 2026.

**OpenLLMetry** ([traceloop/openllmetry](https://github.com/traceloop/openllmetry)) is the reference implementation that fed conventions into [OpenTelemetry's official `semconv/gen-ai`](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Standard attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens` / `output_tokens`, plus operation-name spans for `chat`, `embeddings`, `tool_call`. Legacy `gen_ai.prompt` / `gen_ai.completion` attributes were deprecated in semconv v1.38.0 in favor of structured message events under the Logs API.

### Benchmarks for memory specifically

**LongMemEval** ([arXiv:2410.10813](https://arxiv.org/abs/2410.10813), ICLR 2025, [code](https://github.com/xiaowu0162/LongMemEval)) is the most comprehensive multi-session memory benchmark. 500 manually curated questions across 7 question types, 5 core abilities — Information Extraction, Multi-Session Reasoning, Knowledge Updates, Temporal Reasoning, Abstention. Two scaled settings: `LongMemEval_S` ≈115 K tokens / 30–40 sessions and `LongMemEval_M` ≈1.5 M tokens / 500 sessions. The headline finding: long-context LLMs show a 30–60 % accuracy drop on `LongMemEval_S` versus oracle conditions. This is the benchmark that should be run against any production memory layer.

**LoCoMo** ([arXiv:2402.17753](https://arxiv.org/abs/2402.17753), ACL 2024 Findings, [site](https://snap-research.github.io/locomo/)) tests very long-term conversational memory with multimodal turns. The paper itself reports 304.9 turns / 19.3 sessions / 9,209 tokens per conversation on average — *not* the "600 dialogues / 26K tokens" figure that appears in vendor blogs (that comes from Mem0's restatement and counts differently). Categories: single-hop, multi-hop, temporal reasoning, open-domain, adversarial.

**DialSim**, **PerLTQA** ([arXiv:2402.16288](https://arxiv.org/abs/2402.16288)), **MemoryBank** ([arXiv:2305.10250](https://arxiv.org/abs/2305.10250)), **MemBench**, and **MemoryBench** (2025) round out the benchmark suite. RAGAS, TruLens, and Phoenix RAG metrics evaluate single-turn RAG quality and should be paired with — not substituted for — multi-session benchmarks.

### Metrics that matter

**Recall@k** is the fraction of ground-truth relevant memories appearing in the top-k retrieved set. **Precision@k** is the inverse. **NDCG@k** is the rank-aware variant. **Faithfulness** (RAGAS sense) is the proportion of atomic claims in the response entailable from retrieved context, computed via LLM-as-judge claim decomposition. **Memory-induced hallucination rate** has no single canonical definition — operationalize as 1 − faithfulness restricted to memory-grounded responses, or as the fraction of LongMemEval ABS failures. **Temporal accuracy** is correctness on questions whose answer depends on recency or time-ordering of facts (LongMemEval TR category). **Multi-session reasoning accuracy** is correctness on questions requiring synthesis across distinct sessions (LongMemEval MR, LoCoMo multi-hop).

### Production memory dashboard

A useful memory dashboard surfaces the following on a per-tenant basis. Memory operation counts (ADD / UPDATE / DELETE / NOOP — Mem0's AUDN distribution) tell you whether extraction is producing useful signal. Recall@k against a held-out eval set, evaluated nightly, catches retrieval-quality drift. Embedding-similarity distribution histograms catch poisoning attempts and embedding model drift. Token-budget utilization per session tells you whether compression is firing correctly. Cost per memory operation, sliced by tenant, identifies abusive usage. P50/P95/P99 latency on `search_memory` and `add_memory` — separate the read and write paths because they fail differently. Compression-firing rate (how often `summarize_messages` fired vs how often it should have). Per-tenant memory size growth and eviction rate — unbounded growth flags missing eviction policies. None of Langfuse, Phoenix, or LangSmith ship this dashboard out of the box; teams compose it from custom OTel metrics plus eval-set runners.

## Section 8 — Demo translations

These five demos run against the verified WARNERCO Schematica architecture. Each is designed for under 10 minutes, Claude Desktop plus local Python only, and surfaces a deliberate failure mode for teaching contrast.

### Demo 1 — Working memory via scratchpad injection

**Learning objective.** Show that the LangGraph `inject_scratchpad` node carries working memory across the 7-node pipeline, and that absence of scratchpad injection produces context loss within a single conversation.

**Memory types exercised.** Working.

**MCP primitives exercised.** Tools (the host's tool call into Schematica's MCP server).

**Modules involved.** `src/warnerco/backend/` LangGraph pipeline (the `inject_scratchpad` node, verified in README architecture diagram); MCP server entry point invoked via `uv run warnerco-mcp`.

**Runbook.**
1. Configure Claude Desktop's `claude_desktop_config.json` to launch `uv run warnerco-mcp` via stdio.
2. Start a conversation: "What is a transformer block?"
3. Follow-up: "What are its three core components?" — verify Schematica's response references the prior turn coherently.
4. Open a fresh Claude Desktop session, paste the second question only.
5. Observe degraded answer quality — the scratchpad was reset.
6. Re-run with the original session; observe the working-memory trace in LangGraph state if LangSmith is wired.

**Failure mode to surface.** Confusion between thread state and long-term memory: the second-session degradation is *not* a long-term-memory failure, it is a working-memory reset. Use this to motivate Demo 3.

### Demo 2 — Semantic memory via ChromaDB retrieval

**Learning objective.** Demonstrate semantic-memory retrieval through the `retrieve` and `query_graph` nodes, and show the JSON → Chroma progression in action.

**Memory types exercised.** Semantic.

**MCP primitives exercised.** Tools (`search_memory` / `retrieve` style); optionally Resources if Schematica exposes documents via `resource_link`.

**Modules involved.** `src/warnerco/backend/` retrieval node; ChromaDB-backed vector store; SQLite + NetworkX knowledge graph.

**Runbook.**
1. Ingest a small corpus (the course's sample data) into Schematica with `uv run` against the ingestion script.
2. Ask Claude Desktop a fact-grounded question via the Schematica MCP tool.
3. Inspect the LangGraph trace (LangSmith or stdout) to see the `retrieve` node output.
4. Repeat with the JSON backend (toggle in config).
5. Repeat with ChromaDB.
6. Compare retrieval latency and recall qualitatively.

**Failure mode to surface.** Same-dimension embedding swap. Mid-demo, change the embedding model in config without re-ingesting; observe degraded retrieval. Use this to motivate Pattern 7 (dual-read migration).

### Demo 3 — Episodic memory gap → temporal knowledge graph

**Learning objective.** Demonstrate the **verified GAP** in Schematica's current episodic-memory support, and motivate the bi-temporal extension.

**Memory types exercised.** Episodic (gap demonstration).

**MCP primitives exercised.** Tools (the gap is what we're showing).

**Modules involved.** SQLite + NetworkX knowledge graph (currently non-temporal, per repo verification).

**Runbook.**
1. Ingest two contradictory facts at different timestamps: "User X works at Acme" (Jan 2025), "User X works at BetaCorp" (Apr 2026).
2. Query: "Where does User X work now?"
3. Query: "Where did User X work in February 2026?"
4. Observe Schematica returns both facts or one arbitrarily — no temporal disambiguation.
5. Pull a [Graphiti MCP server](https://github.com/getzep/graphiti) into the Claude Desktop config alongside Schematica.
6. Re-ingest with timestamps, re-query, observe correct bi-temporal answers.

**Failure mode to surface.** This is the failure mode by design — it makes the procedural-vs-episodic distinction concrete and shows why a flat knowledge graph is insufficient.

### Demo 4 — Procedural memory via MCP Prompts (gap)

**Learning objective.** Add MCP Prompts to Schematica to demonstrate procedural memory; another **verified GAP** extension.

**Memory types exercised.** Procedural.

**MCP primitives exercised.** Prompts (newly added), Tools (existing).

**Modules involved.** A new `prompts.py` (or equivalent) registering FastMCP `@mcp.prompt` decorations.

**Runbook.**
1. Add a `/schematica_query` prompt to the MCP server with a single `{topic}` parameter.
2. Add a `/code_review` prompt that takes `{language}` and `{focus_area}` and returns a 5-message playbook.
3. Restart the MCP server; reload Claude Desktop.
4. Type `/` in Claude Desktop; observe the new slash commands appear.
5. Invoke `/schematica_query topic="vector indexing"` and observe the parameterized prompt template execute.
6. Compare with calling the equivalent as a Tool (autonomously selectable) — discuss CoALA's procedural-memory caveat about model-controlled rewrites.

**Failure mode to surface.** Slash-command discoverability differs across hosts. Run the same demo in VS Code's Copilot Chat to show the command-palette UX vs Claude Desktop's slash menu.

### Demo 5 — Reflective consolidation via Sampling (gap)

**Learning objective.** Add a Sampling-based reflection node between `compress` and `reason`; the third **verified GAP**.

**Memory types exercised.** Episodic → Semantic distillation.

**MCP primitives exercised.** Sampling.

**Modules involved.** A new reflection step in the LangGraph pipeline; FastMCP server's Sampling client integration.

**Runbook.**
1. Run Schematica from VS Code with Copilot Chat (sampling-capable host as of [June 2025 full-spec rollout](https://code.visualstudio.com/blogs/2025/06/12/full-mcp-spec-support)).
2. Have a 10-turn conversation about a project's history.
3. Trigger the reflection node manually (or wait for the threshold trigger).
4. Observe the `sampling/createMessage` request in the trace; the server asks the host's LLM to extract 3 high-level questions and 5 insights.
5. Observe new entries written to semantic memory.
6. Restart, ask a question that requires the reflected insight rather than raw transcript; verify retrieval works.

**Failure mode to surface.** Run the same demo in Claude Desktop. Sampling is not implemented; the server falls back to "no LLM available" or reflection skipped. Use this to discuss the Sampling-coverage gap across hosts and the tradeoff with stateless deployment.

## Section 9 — Glossary

**Bi-temporal model.** A graph or table where every fact carries two timestamps: when it held in the world (`t_valid` / `t_invalid`) and when the system observed it (`t'_created` / `t'_expired`). Enables "as-of" queries.

**CoALA.** Cognitive Architectures for Language Agents (Sumers et al., TMLR 2024) — the canonical four-tier memory taxonomy: working, episodic, semantic, procedural.

**Context window.** The maximum number of tokens the LLM accepts as input on a single inference call. The serialization of working memory plus retrieved content plus instructions.

**CoALA decision cycle.** Proposal → Evaluation → Selection → Execution loop, executing internal (reasoning, retrieval, learning) and external (grounding) actions.

**Drift adapter.** A learned transformation between two embedding spaces, allowing one model's queries to retrieve over another model's stored embeddings. RESEARCH; not productized.

**Ebbinghaus forgetting curve.** R = e^(−t/S); the exponential decay of memory retention over time, modulated by rehearsal-driven strength S. Used as inspiration in MemoryBank and Generative Agents.

**Elicitation.** MCP primitive (added in 2025-06-18) by which a server requests structured input from the user mid-task via `elicitation/create`. Schemas restricted to flat objects with primitive properties.

**Episodic memory.** CoALA tier storing specific past events with temporal indexing. NOT the same as raw conversation history — episodic memory is *retrievable* by relevance, recency, and importance.

**Faithfulness.** RAGAS metric: proportion of atomic claims in a response entailable from retrieved context. Computed via LLM-as-judge claim decomposition.

**FastMCP.** Python framework for building MCP servers (`PrefectHQ/fastmcp`, formerly `jlowin/fastmcp`); v3 GA February 2026. Three pillars: Components, Providers, Transforms.

**Graphiti.** Open-source bi-temporal knowledge graph engine underpinning Zep; preserves invalidated edges with `t_invalid` rather than deleting.

**Hot/warm/cold tiering.** Memory architecture pattern: hot = in-context, warm = vector store, cold = object storage / deactivated partitions.

**Importance × Recency × Relevance.** Generative Agents retrieval formula: `α_recency·recency + α_importance·importance + α_relevance·relevance`, with αs typically 1.

**LongMemEval.** ICLR 2025 benchmark for multi-session memory; 500 questions across 5 abilities (IE, MR, KU, TR, ABS); LongMemEval_S ≈115K tokens.

**MCP host.** The application that orchestrates one or more MCP clients and presents the user interface (Claude Desktop, VS Code, LangGraph runtime).

**MCP Resource.** Server primitive exposing application-driven data via stable URIs; idempotent reads; subscribable for change notifications.

**MCP Tool.** Server primitive exposing model-controlled actions with side effects; declared via JSON Schema input and (since 2025-06-18) optional output schema.

**Memory poisoning.** Attack class in which malicious instructions are written into stored memory (directly or via summarized injected content) and execute on later retrieval. OWASP Agentic ASI category.

**Mem0 AUDN.** ADD / UPDATE / DELETE / NOOP write-time conflict-resolution pipeline; v1/v2 only — replaced in v3 (April 2026) by single-pass extraction with retrieval-time conflict scoring.

**Procedural memory.** CoALA tier storing skills, workflows, and decision-making procedures. NOT the same as the system prompt — system prompts are instances; procedural memory is the broader category including versioned playbooks, MCP Prompts, Anthropic Skills.

**RAG.** Retrieval-Augmented Generation — a *retrieval pattern*, not a memory type. RAG can serve any memory tier; semantic memory is content category, not a retrieval mechanism.

**Resource link.** A `{type: "resource_link", uri: ...}` content item that may appear in tool results (added in 2025-06-18); enables citation-based retrieval where the tool returns links and the host fetches passages on demand.

**Sampling.** MCP primitive by which a server requests an LLM completion through the host's client. Substrate for memory transformations (summarization, extraction, re-ranking).

**Semantic memory.** CoALA tier storing generalized facts and knowledge extracted from experience. NOT the same as RAG — RAG is retrieval; semantic memory is content category.

**Streamable HTTP.** MCP transport introduced in 2025-03-26 spec; replaced HTTP+SSE; single endpoint accepting POST and GET; per-request SSE upgrade for streaming.

**Stateful vs stateless server.** Stateful servers maintain per-session context across requests via `Mcp-Session-Id`; stateless servers do not and cannot use Sampling, Elicitation, or Roots.

**Temporal accuracy.** Correctness on questions whose answer depends on recency or time-ordering of facts; LongMemEval TR category.

**Working memory.** CoALA tier holding active state for the current decision cycle. NOT the same as the context window — the context window is its serialization.

## Section 10 — Verified bibliography

All URLs verified resolving as of 26 April 2026.

**Primary specifications**
- [SPEC] [Model Context Protocol specification 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18) — Tools, Resources, Prompts, Sampling, Elicitation, Roots, Transports, Authorization sub-pages.
- [SPEC] [MCP 2025-06-18 changelog (mirror)](https://mcpcn.com/en/specification/2025-06-18/changelog/) — material differences from 2024-11-05 and 2025-03-26.
- [SPEC] [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — standard attributes for LLM observability.
- [SPEC] [GDPR Article 17](https://gdpr-info.eu/art-17-gdpr/) — right to erasure.
- [SPEC] [EU AI Act regulatory framework](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — phased implementation through 2026–2028.

**Framework documentation**
- [VENDOR-DOC] [FastMCP](https://gofastmcp.com/) and [State Management DeepWiki](https://deepwiki.com/jlowin/fastmcp/13.3-state-management-and-caching-strategies).
- [VENDOR-DOC] [LangGraph add-memory docs](https://docs.langchain.com/oss/python/langgraph/add-memory).
- [VENDOR-DOC] [LangMem](https://github.com/langchain-ai/langmem) and [LangMem summarization guide](https://langchain-ai.github.io/langmem/guides/summarization/).
- [VENDOR-DOC] [Mem0 GitHub](https://github.com/mem0ai/mem0) and [Mem0 MCP integration](https://docs.mem0.ai/platform/features/mcp-integration).
- [VENDOR-DOC] [Letta docs](https://docs.letta.com/) and [MemGPT-and-Letta blog](https://www.letta.com/blog/memgpt-and-letta).
- [VENDOR-DOC] [Zep / Graphiti](https://www.getzep.com/) with [open-source Graphiti repo](https://github.com/getzep/graphiti).
- [VENDOR-DOC] [Cognee](https://github.com/topoteretes/cognee).
- [VENDOR-DOC] [Microsoft Semantic Kernel Vector Stores](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/) and [Memory Store migration guide](https://learn.microsoft.com/en-us/semantic-kernel/support/migration/memory-store-migration).
- [VENDOR-DOC] [Google ADK memory docs](https://google.github.io/adk-docs/sessions/memory/).
- [VENDOR-DOC] [AutoGen memory](https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/memory.html).
- [VENDOR-DOC] [OpenAI Assistants migration guide](https://developers.openai.com/api/docs/assistants/migration) and [deprecations](https://developers.openai.com/api/docs/deprecations).
- [VENDOR-DOC] [Anthropic Memory Tool docs](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool) and [context management announcement](https://www.anthropic.com/news/context-management).

**Academic papers**
- [PAPER] Sumers, Yao, Narasimhan, Griffiths. "Cognitive Architectures for Language Agents." TMLR 2024. [arXiv:2309.02427](https://arxiv.org/abs/2309.02427).
- [PAPER] Packer et al. "MemGPT: Towards LLMs as Operating Systems." 2023. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560).
- [PAPER] Park, O'Brien, Cai et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442); [ACM full text](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763).
- [PAPER] Chhikara, Khant, Aryan, Singh, Yadav. "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory." ECAI 2025. [arXiv:2504.19413](https://arxiv.org/abs/2504.19413).
- [PAPER] Rasmussen et al. "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." 2025. [arXiv:2501.13956](https://arxiv.org/abs/2501.13956).
- [PAPER] Zhang et al. "A Survey on the Memory Mechanism of Large Language Model based Agents." TOIS 2025. [arXiv:2404.13501](https://arxiv.org/abs/2404.13501).
- [PAPER] Wu et al. "From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs." 2025. [arXiv:2504.15965](https://arxiv.org/abs/2504.15965).
- [PAPER] Zhong et al. "MemoryBank: Enhancing Large Language Models with Long-Term Memory." AAAI 2024. [arXiv:2305.10250](https://arxiv.org/abs/2305.10250).
- [PAPER] Du et al. "PerLTQA: A Personal Long-Term Memory Dataset for Memory Classification, Retrieval, and Synthesis." NAACL 2024. [arXiv:2402.16288](https://arxiv.org/abs/2402.16288).
- [PAPER] Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." EACL 2024. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217).
- [PAPER] Greshake et al. "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." AISec 2023. [arXiv:2302.12173](https://arxiv.org/abs/2302.12173).

**Benchmarks**
- [BENCHMARK] Wu et al. "LongMemEval." ICLR 2025. [arXiv:2410.10813](https://arxiv.org/abs/2410.10813); [code](https://github.com/xiaowu0162/LongMemEval).
- [BENCHMARK] Maharana et al. "LoCoMo." ACL 2024 Findings. [arXiv:2402.17753](https://arxiv.org/abs/2402.17753); [site](https://snap-research.github.io/locomo/).

**Vendor blog posts and case studies**
- [VENDOR-BLOG] [Trend Micro on Mem0 + Bedrock + Neptune](https://aws.amazon.com/blogs/machine-learning/company-wise-memory-in-amazon-bedrock-with-amazon-neptune-and-mem0/) — the cleanest named enterprise case study in the category.
- [VENDOR-BLOG] [Mem0 Series A announcement](https://mem0.ai/series-a) — AWS Agent SDK exclusive memory provider claim (Oct 2025).
- [VENDOR-BLOG] [Zep Community Edition deprecation announcement](https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/) — April 2025.
- [VENDOR-BLOG] [Cloudflare on Streamable HTTP MCP servers](https://blog.cloudflare.com/streamable-http-mcp-servers-python/).
- [VENDOR-BLOG] [VS Code full-MCP spec support](https://code.visualstudio.com/blogs/2025/06/12/full-mcp-spec-support) — June 2025.
- [VENDOR-BLOG] [Anthropic Skills announcement](https://www.anthropic.com/news/skills).
- [VENDOR-BLOG] [Pinecone multi-tenancy guide](https://docs.pinecone.io/guides/index-data/implement-multitenancy).
- [VENDOR-BLOG] [Weaviate multi-tenancy](https://weaviate.io/blog/multi-tenancy-vector-search) and [docs](https://docs.weaviate.io/weaviate/manage-collections/multi-tenancy).
- [VENDOR-BLOG] [Qdrant 1.16 tiered multitenancy](https://qdrant.tech/blog/qdrant-1.16.x/) and [multitenancy guide](https://qdrant.tech/documentation/manage-data/multitenancy/).
- [VENDOR-BLOG] [Azure AI Search multi-tenant SaaS guidance](https://learn.microsoft.com/en-us/azure/search/search-modeling-multitenant-saas-applications) and [Azure Cosmos DB DiskANN multi-tenancy](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/multi-tenancy-vector-search).
- [VENDOR-BLOG] [Microsoft embedding model migration guidance](https://learn.microsoft.com/en-us/answers/questions/2071109/how-to-migrate-to-the-latest-model).
- [VENDOR-BLOG] [Microsoft Presidio docs](https://microsoft.github.io/presidio/).
- [VENDOR-BLOG] [Pinecone HIPAA / DPA](https://www.pinecone.io/dpa/) and [security center](https://security.pinecone.io/).
- [VENDOR-BLOG] [Auth0 on MCP authorization](https://auth0.com/blog/mcp-specs-update-all-about-auth/); [Descope on the auth spec](https://www.descope.com/blog/post/mcp-auth-spec).
- [VENDOR-BLOG] [Cisco on MCP elicitation, structured content, OAuth](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements).
- [VENDOR-BLOG] [PromptArmor Slack AI exfiltration](https://www.promptarmor.com/resources/data-exfiltration-from-slack-ai-via-indirect-prompt-injection).

**Independent / standards bodies**
- [INDEPENDENT] [OWASP Top 10 for LLM Applications v2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — categories LLM01, LLM04, LLM06, LLM08 directly relevant to agent memory.
- [INDEPENDENT] [OWASP Agentic Top 10](https://genai.owasp.org/) — December 2025; ASI categories including memory poisoning.
- [INDEPENDENT] [Stack Overflow: authentication and authorization in MCP](https://stackoverflow.blog/2026/01/21/is-that-allowed-authentication-and-authorization-in-model-context-protocol/).

**WARNERCO Schematica reference**
- [VENDOR-DOC] [timothywarner-org/context-engineering](https://github.com/timothywarner-org/context-engineering) — repo verified public, MIT, with WARNERCO Schematica under `src/warnerco/backend/`.
- [VENDOR-DOC] [O'Reilly Live: Context Engineering with MCP](https://www.oreilly.com/live-events/context-engineering-with-mcp/0642572241292/) — corroborates 4-segment course shape.

**Verified gaps in WARNERCO Schematica (per repo verification)**
- MCP Prompts for procedural memory — not present in README architecture or topic tags.
- Temporal knowledge graph for episodic memory — README describes flat SQLite + NetworkX; no bi-temporal dimension.
- MCP Sampling for reflective memory — not mentioned in README or topic tags.
- Specific Python file names under `src/warnerco/backend/` (e.g., `graph.py`) — not directly enumerable; CLAUDE.md described as source of truth but inaccessible to fetch path used.

**Uncertainty flags carried into this report**
- Vendor-reported benchmark numbers (Mem0 LOCOMO claims, Zep DMR/LongMemEval claims) are authored by competing vendors and not independently reproduced — treat as marketing-adjacent.
- LoCoMo statistics differ between paper-as-written (~300 turns / 9K tokens) and Mem0's restatement (~600 dialogues / 26K tokens); citations distinguish.
- Mem0 v2 → v3 algorithm change replaces AUDN with single-pass extraction; the Mem0 paper documents v1/v2 only — pin versions in production.
- "Drift adapter" embedding migration recovery numbers are RESEARCH; not productized as of April 2026.
- GDPR Article 17 against partially invertible embeddings is legally unsettled; consult counsel.
- 2025-11-25 MCP spec revision exists with further changes (CIMD, JSON Schema 2020-12, ElicitResult enhancements); this synthesis is anchored to 2025-06-18 as requested.
- Memory MCP enterprise case studies are dominated by vendor blogs; only the Trend Micro / Mem0 / Bedrock case is fully named with architecture detail.