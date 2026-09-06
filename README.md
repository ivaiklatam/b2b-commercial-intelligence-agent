# B2B Commercial Intelligence Agent

**CMU Capstone Project — Agentic AI Program**
*Jorge Javier Lozano Díaz · Data & AI Architecture Lead, Accenture Colombia*

---

## Overview

An autonomous multi-agent system that monitors the Colombian B2B market daily, detects commercial signals, and delivers actionable recommendations to sales advisors — before the competition does.

The system combines a **ReAct reasoning loop**, **Tree of Thought hypothesis generation**, **semantic RAG retrieval**, and a **multi-agent LangGraph workflow** to transform raw market signals into grounded, scored, and advisor-ready commercial recommendations.

> *"An agent solves a problem. A Living Intelligence System ensures the ecosystem no longer faces it the same way."*
> — Living Intelligence White Paper, Jorge Lozano Díaz, 2026

---

## The Problem

B2B commercial teams in Telco operate with fragmented, outdated market information. There is no system to continuously monitor company news, track market shifts, or prepare context for client visits. Real business opportunities go unnoticed because no one was watching at the right moment.

**Intended users:** B2B commercial manager (Milena Rodríguez, Gerencia Nacional B2B) and regional sales advisors across Colombia.

---

## System Architecture

```
                    ┌─────────────────────────────────────┐
                    │         MCP SERVER                   │
                    │  knowledge_graph_search              │
                    │  custom_search                       │
                    │  news_rss                            │
                    └──────────────┬──────────────────────┘
                                   │ MCP Protocol
                    ┌──────────────▼──────────────────────┐
                    │         LANGGRAPH WORKFLOW           │
                    │                                      │
                    │  ┌─────────┐                         │
                    │  │  SCOUT  │ Detects & classifies    │
                    │  │  AGENT  │ market signals          │
                    │  └────┬────┘                         │
                    │       │                              │
                    │  ┌────▼──────────┐                   │
                    │  │  STRATEGIST   │ ToT: 3 hypotheses │
                    │  │  AGENT        │ Beam search top 2 │
                    │  │               │ RAG refinement    │
                    │  └────┬──────────┘                   │
                    │       │                              │
                    │  ┌────▼────┐  score < 0.6            │
                    │  │  CRITIC │ ──────────────► REVISE  │
                    │  │  AGENT  │  (max 2 iter)           │
                    │  └────┬────┘                         │
                    │       │ score ≥ 0.6                  │
                    │  ┌────▼────┐                         │
                    │  │ MEMORY  │ Long-term learning      │
                    │  │  AGENT  │ Pattern tracking        │
                    │  └─────────┘                         │
                    └─────────────────────────────────────┘
```

### Agent Roles

| Agent | Role | Objective |
|---|---|---|
| **Scout** | Signal collection | Recall — surface everything relevant |
| **Strategist** | ToT recommendation generation | Precision — select strongest hypothesis |
| **Critic** | Quality evaluation | Reliability — prevent weak recommendations |
| **Memory** | Knowledge consolidation | Improvement — learn from every cycle |

### Key Design Decisions

- **ReAct loop** in Scout: Reason → Act (MCP tool call) → Observe → Repeat
- **Tree of Thought** in Strategist: 3 hypotheses, beam search selects top 2, RAG refines
- **Revision gate**: Critic returns rejected recommendations to Strategist (max 2 iterations)
- **MCP Server**: Google tools exposed via Model Context Protocol
- **ChromaDB RAG**: Section-level chunking with timestamp metadata and expiry warnings
- **Two-level human intervention**: Regional directors (tactical) + National manager (strategic)
- **A2A Architecture**: Each agent is designed to operate as an independent A2A server. See [A2A Design](#a2a-architecture-design) below.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini (via Gemini API) |
| Embeddings | Google Gemini Embedding |
| Orchestration | LangGraph StateGraph |
| RAG / Vectorstore | ChromaDB |
| Tool Protocol | MCP (Model Context Protocol) |
| Observability | LangSmith |
| External Tools | Google Knowledge Graph API, Custom Search API, News RSS |
| Internal Tools | Pipeline analysis, Cross-sell insights |
| Tests | pytest |

---

## Project Structure

```
b2b-commercial-intelligence-agent/
├── README.md
├── requirements.txt
├── .env.example
├── cli.py                          # Entry point + metrics reporting
├── memory.json                     # Long-term memory (auto-generated)
├── sample_output.md                # Example system output
├── data/
│   ├── companies.json              # 20 Colombian B2B companies
│   ├── portfolio.json              # 4 document types with timestamps
│   ├── pipeline.json               # Salesforce-equivalent exports
│   └── config.json                 # System parameters and thresholds
├── mcp_server/
│   └── google_tools_server.py      # MCP Server: 3 Google tools
├── agent/
│   ├── state.py                    # LangGraph AgentState definition
│   ├── scout.py                    # Scout Agent
│   ├── strategist.py               # Strategist Agent + ToT
│   ├── critic.py                   # Critic Agent + rubric
│   ├── memory.py                   # Memory Agent
│   └── graph.py                    # LangGraph workflow + revision gate
├── tools/
│   ├── mcp_client.py               # MCP Client → LangChain tools
│   ├── rag.py                      # ChromaDB retrieval + timestamps
│   ├── pipeline.py                 # Pipeline analysis tool
│   └── crosssell.py                # Cross-sell insights tool
└── tests/
    └── test_agent.py               # 8 tests
```

---

## Setup

### Prerequisites

- Python 3.11+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/b2b-commercial-intelligence-agent.git
cd b2b-commercial-intelligence-agent
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Google APIs
GOOGLE_API_KEY=your_google_api_key
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# LangSmith Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=b2b-commercial-intelligence

# Models
GEMINI_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001

# System Config
MAX_COMPANIES=5
SCORE_THRESHOLD=0.6
MAX_ITERATIONS=2
RAG_TOP_K=5
```

### API Keys Required

| Key | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Credentials |
| `GOOGLE_GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GOOGLE_CSE_ID` | [Programmable Search Engine](https://programmablesearchengine.google.com) |
| `LANGCHAIN_API_KEY` | [LangSmith](https://smith.langchain.com) → Settings → API Keys |

---

## Usage

### Basic run

```bash
python cli.py --sector manufacturing --region Bogota --max-companies 3
```

### All sectors, all regions

```bash
python cli.py --max-companies 5 --output full_report.md
```

### Available parameters

| Parameter | Description | Default |
|---|---|---|
| `--sector` | Target sector (manufacturing, retail, logistics...) | All |
| `--region` | Target region (Bogota, Medellin, Cali...) | All |
| `--max-companies` | Maximum companies to analyze | 3 |
| `--output` | Output markdown file | sample_output.md |

### Run tests

```bash
pytest tests/test_agent.py -v
```

---

## Sample Output

See [sample_output.md](sample_output.md) for a complete example of system output including:
- Detected signals with urgency classification
- Approved recommendations with score breakdown
- RAG evidence citations
- Cycle metrics (groundedness rate, pruning rate, latency)
- Long-term memory statistics

---

## Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| **Groundedness rate** | Recommendations with RAG evidence | > 85% |
| **Pruning rate** | Hypotheses rejected by Critic | < 30% |
| **Escalation rate** | Recommendations without precedent | Monitored |
| **Cycle latency** | End-to-end processing time | < 180s |
| **Adoption rate** | Recommendations acted on by advisors | Primary metric |

---

## Safety and Guardrails

1. **Source verification** — Only pre-approved external sources
2. **Output grounding** — Every recommendation requires RAG or pipeline evidence
3. **Timestamp expiration** — Pricing content flagged after 90 days
4. **Strategic alignment filter** — Excluded sectors blocked before Strategist
5. **Critic score threshold** — 0.6 minimum score for delivery
6. **Two-level human intervention** — Regional (tactical) + National (strategic)

---

## A2A Architecture Design

While the current implementation uses LangGraph for inter-agent communication,
each agent is designed to operate as an independent
**Agent-to-Agent (A2A) server** in a production deployment:

```
Scout A2A Server          Strategist A2A Server
┌─────────────────┐       ┌──────────────────────┐
│ Agent Card:     │       │ Agent Card:           │
│ name: scout     │  A2A  │ name: strategist      │
│ capabilities:   │──────►│ capabilities:         │
│ - signal_detect │       │ - tot_reasoning       │
│ - classify      │       │ - rag_retrieval       │
└─────────────────┘       └──────────────────────┘
```

Each agent would expose:
- `GET /.well-known/agent.json` — Agent Card with capabilities
- `POST /run` — Execute task and return structured response
- `POST /run/stream` — Streaming execution for long tasks

---

## Limitations and Next Steps

**Current limitations:**
- Dataset is synthetic — real Salesforce integration pending
- Memory Agent improves pattern weights but does not retrain the LLM
- A2A protocol implemented as design, not running servers
- Adoption rate metric requires real advisor feedback loop

**Next steps:**
- Real Salesforce API integration replacing Excel exports
- Implement full A2A servers for each agent
- Add Tree of Thought depth 3 for complex multi-product recommendations
- Implement Animus component for strategic direction evaluation
- Deploy Memory Agent as async background service

---

## Connection to Living Intelligence

This capstone implements the foundational architecture of a
**Living Intelligence System (LIS)** as defined in the
[Living Intelligence White Paper](https://www.linkedin.com/in/jorge-lozano-diaz)
(Lozano Díaz, 2026):

| LIS Component | Implementation |
|---|---|
| **Agents** | Scout, Strategist, Critic |
| **Cognitors** | Memory Agent (knowledge consolidation) |
| **Sentinels** | Critic Agent (signal quality monitoring) |
| **Living Memory** | ChromaDB RAG + memory.json |
| **Ethos** | config.json (immutable strategic constraints) |
| **Animus** | Designed — next implementation phase |

---

## Author

**Jorge Javier Lozano Díaz**
Data & AI Architecture Lead · Accenture Colombia
[linkedin.com/in/jorge-lozano-diaz](https://linkedin.com/in/jorge-lozano-diaz)

*CMU Agentic AI Program · School of Computer Science · Executive Education · 2026*
```