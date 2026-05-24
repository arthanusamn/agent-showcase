# 🤖 Agent Showcase — Rocky & Art's Agent Collection

A portfolio of production-grade AI agents built to demonstrate agentic AI
capabilities for business consulting and custom development.

## Stack

- **LLM**: DeepSeek Chat (reasoning + function calling)
- **Orchestration**: OpenClaw sub-agents + Python control flows
- **Tools**: Composio (Gmail, Slack, GitHub), web scraping, file I/O
- **Integration**: Email delivery, Telegram alerts, scheduled cron jobs

## Agents

| # | Agent | What It Does | Tech |
|---|-------|-------------|------|
| 1 | **Research Synth** | Multi-agent research → structured report | Sub-agent orchestration, web search, markdown synthesis |
| 2 | **BizFlow** | End-to-end business workflow automation | Composio tool-use, email, API pipelines |
| 3 | **Data Whisperer** | Upload CSV → auto EDA → insights report | pandas, matplotlib, DeepSeek reasoning |
| 4 | **Pipeline Pilot** | Scheduled data pipeline with alerts | Cron + conditional branching |
| 5 | **Lead Scraper** | Extract + enrich leads from website | Playwright, trafilatura, enrichment |

## How to Run Each

```bash
# Agent 1: Research Synth
cd ~/.openclaw/workspace/agent-showcase
python3 agent_1_research_synth.py --topic "Agentic AI for small businesses"
python3 agent_1_research_synth.py --topic "RAG security best practices 2026" --depth deep

# Agent 2: BizFlow
python3 agent_2_bizflow.py --demo  # Runs a simulated business workflow
python3 agent_2_bizflow.py --live # Runs with real Gmail (if connected)

# Agent 3: Data Whisperer
python3 agent_3_data_whisperer.py --file data.csv

# Agent 4: Pipeline Pilot
python3 agent_4_pipeline_pilot.py --config config.json

# Agent 5: Lead Scraper
python3 agent_5_lead_scraper.py --url https://example.com
```

## Notes for Client Conversations

Each agent is designed to showcase a different capability clients care about:

1. **Research Synth** → "Show me you can synthesize complex information"
2. **BizFlow** → "Show me you can automate my actual business workflows"
3. **Data Whisperer** → "Show me you understand data science"
4. **Pipeline Pilot** → "Show me you can build reliable production systems"
5. **Lead Scraper** → "Show me you can generate leads/cost savings"

These are reference implementations. Real client agents get custom-built per their
specific workflow, data sources, and compliance requirements.
