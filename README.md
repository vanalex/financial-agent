# Financial Intelligent Agent

Financial Intelligent Agent is an async Python research assistant built with `deepagents.create_deep_agent`. It combines market quotes, iterative financial-news research, reflection, and structured LLM synthesis in a checkpointed LangGraph workflow.

The current entry point analyzes one broad market proxy to stay within strict Alpha Vantage rate limits:

- `SPY`

## Purpose

The project helps collect and interpret market context without manually switching between market data, news search, and an LLM prompt. It does not make trading decisions or provide personalized investment advice. Instead, it summarizes evidence, explains likely market drivers, highlights contradictions, and calls out short-term risks based only on the supplied data.

## Features

- Fetches latest global quote data from Alpha Vantage with async HTTP calls. The default run uses one symbol, so it makes one Alpha Vantage request.
- Uses `deepagents.create_deep_agent` for planning, tool use, reflection, subagent-capable execution, and structured final output.
- Collects recent financial market news with Tavily search using async LangChain calls.
- Exposes Alpha Vantage quote lookup, Tavily news search, and Markdown report saving as agent tools.
- Uses OpenAI through the Deep Agents model interface for final structured synthesis.
- Persists agent state with LangGraph's Postgres checkpointer so interrupted runs can resume.
- Validates Alpha Vantage responses with Pydantic models.
- Applies a small request delay before Alpha Vantage calls to reduce rate-limit pressure.

## How It Works

The Deep Agent is defined in `app/deep_agent.py` and compiled in `app/graph.py`.
It receives a market-brief task, then uses these domain tools:

1. `get_market_quote`
   Fetches the latest Alpha Vantage global quote for one ticker symbol.

2. `search_market_news`
   Runs a targeted Tavily financial-news search.

3. `save_market_report`
   Saves the final Markdown report under `output/`.

The agent is instructed to gather quote data, run targeted news searches, reflect
on evidence gaps, save the report, and return a structured `DeepMarketBrief`.

The graph uses LangGraph's Postgres checkpointer. Checkpoints are stored in the
database configured by `DATABASE_URL`, keyed by the `thread_id` passed in the
graph config.

## Project Structure

```text
.
├── app
│   ├── clients
│   │   ├── alphavintage_collector.py
│   │   └── tavily.py
│   ├── nodes
│   │   ├── analysis.py
│   │   ├── market.py
│   │   ├── news.py
│   │   └── report.py
│   ├── deep_agent.py
│   ├── graph.py
│   └── state.py
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python 3.14 or newer
- `uv` for dependency management
- API keys for:
  - Alpha Vantage
  - OpenAI
  - Tavily

## Setup

Install dependencies:

```bash
uv sync
```

Start the local Postgres database:

```bash
docker compose up -d postgres
```

Create a local environment file:

```bash
cp .env.example .env
```

Fill in the required values:

```bash
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
DEEP_AGENT_MODEL=openai:gpt-5
DATABASE_URL=postgresql://financial_agent:financial_agent_password@localhost:5432/financial_agent
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING="true"
LANGSMITH_TRACING_V2="true"
LANGSMITH_PROJECT="financial agent"
```

The LangSmith variables are optional for local execution, but required if you want traces sent to LangSmith.

## Usage

Run the default analysis:

```bash
uv run python main.py
```

The script invokes the LangGraph workflow and prints the final LLM-generated market analysis.
It also prints the generated `thread_id`; keep that value if you want to inspect
or resume the run later.

To choose symbols from the command line:

```bash
uv run python main.py run --symbols SPY QQQ
```

Each symbol maps to one Alpha Vantage `GLOBAL_QUOTE` API request.

To allow more or fewer research/reflection passes:

```bash
uv run python main.py run --symbols SPY QQQ --max-research-iterations 3
```

To make a run easy to resume, provide your own stable `thread_id`:

```bash
uv run python main.py run --thread-id market-2026-08-15 --symbols SPY QQQ
```

The `run` command refuses to start if that `thread_id` already has checkpointed
state. Use `resume` for an existing interrupted run, or choose a new `thread_id`
for a fresh run.

If a run fails after one or more checkpoints have been saved, inspect the saved
state:

```bash
uv run python main.py checkpoint --thread-id market-2026-08-15
```

Then resume from the next saved graph step:

```bash
uv run python main.py resume --thread-id market-2026-08-15
```

To confirm checkpoint persistence is active after a run, inspect the checkpoint
tables:

```bash
docker compose exec postgres psql -U financial_agent -d financial_agent -c "\dt checkpoint*"
docker compose exec postgres psql -U financial_agent -d financial_agent -c "select thread_id, checkpoint_id from checkpoints order by checkpoint_id desc limit 5;"
```

## Tracing

LangSmith tracing is added at the boundaries that are most useful for debugging and observability:

- `Financial Agent Workflow` wraps the full graph invocation.
- Deep Agents traces the model reasoning, tool calls, and structured response.
- `Deep Agent Market Quote Tool` traces quote lookups.
- `Alpha Vantage Global Quote` traces each symbol quote request.
- `Deep Agent Financial News Tool` traces news-search tool calls.
- `Tavily Financial News Query` traces each news provider call.
- `Deep Agent Report Save Tool` traces report persistence.

The Deep Agent is invoked through LangGraph's async `ainvoke`, so LangSmith can show model calls, tool calls, and checkpointed agent state under the workflow trace.

## Evaluations

The project includes a LangSmith evaluation suite in `evals/`. The evals use
controlled synthetic market/news examples so results do not depend on live
Alpha Vantage or Tavily data.

Create the dataset:

```bash
uv run python evals/create_dataset.py
```

Run the default model comparison:

```bash
uv run python evals/run_eval.py
```

Run one model:

```bash
uv run python evals/run_eval.py --single-model gpt-5-mini
```

The runner evaluates `app.nodes.analysis.analyze_market` directly. It does not
use Postgres, the full LangGraph workflow, Alpha Vantage, or Tavily.

Evaluators check:

- required market brief sections
- expected market themes and price/news relationships
- absence of unsupported causal claims
- evidence citation discipline
- insufficient-evidence handling and disclaimer language
- overall analytical quality using an LLM judge

## Output

The final analysis asks the model to identify:

- the three most important developments
- likely explanations for market movements
- connections between news and price movements
- cases where market behavior contradicts the news
- important risks for the next 24 to 48 hours

The prompt also instructs the model to distinguish facts from interpretation, avoid inventing market data, treat collection errors as missing evidence, and avoid personalized investment advice.

## Notes

- Alpha Vantage may return API limit messages, missing data, or invalid-symbol errors. These are captured in the graph state's `errors` list when possible.
- Tavily news results depend on the current search index and your Tavily account access.
- The analysis quality depends on the completeness and freshness of the collected quote and news data.
- This project is a research assistant, not a production trading system.
