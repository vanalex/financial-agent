# Financial Intelligent Agent

Financial Intelligent Agent is an async Python research assistant that combines market quotes, financial news, and an LLM analysis step into a single LangGraph workflow. It is designed to produce a short market research brief from recent quote data and news context.

The current entry point analyzes one broad market proxy to stay within strict Alpha Vantage rate limits:

- `SPY`

## Purpose

The project helps collect and interpret market context without manually switching between market data, news search, and an LLM prompt. It does not make trading decisions or provide personalized investment advice. Instead, it summarizes evidence, explains likely market drivers, highlights contradictions, and calls out short-term risks based only on the supplied data.

## Features

- Fetches latest global quote data from Alpha Vantage with async HTTP calls. The default run uses one symbol, so it makes one Alpha Vantage request.
- Collects recent financial market news with Tavily search using async LangChain calls.
- Runs a LangGraph pipeline with separate nodes for market data, news, and analysis.
- Uses OpenAI through async `langchain-openai` calls for the final market research analysis.
- Tracks data collection errors in graph state so missing or failed data is visible to the analysis step.
- Validates Alpha Vantage responses with Pydantic models.
- Applies a small request delay before Alpha Vantage calls to reduce rate-limit pressure.

## How It Works

The workflow is defined in `app/graph.py`:

1. `prepare_market_data`
   Initializes the pending-symbol queue.

2. `fetch_market_data`
   Retrieves quote data for one pending symbol using Alpha Vantage. The graph
   loops through this node until all symbols are complete, creating a checkpoint
   after each symbol.

3. `fetch_news`
   Searches for recent financial market news focused on macroeconomic data, interest rates, technology stocks, oil, inflation, and market-moving events.

4. `analyze_market`
   Sends the collected market data, news, and any collection errors to an OpenAI chat model and asks for a structured market analysis.

The graph state is represented by `MarketState` in `app/state.py`.

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
│   │   └── news.py
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
- `Fetch Market Data Node` shows requested symbols, returned quote count, and collection errors.
- `Alpha Vantage Global Quote` traces each symbol quote request.
- `Fetch News Node` shows news result counts and URLs.
- `Tavily Financial News Search` traces the news provider call.
- `Analyze Market Node` shows evidence counts and final analysis size.

The OpenAI call is made through LangChain's async `ainvoke`, so when LangSmith tracing is enabled it can appear as a child model run under the analysis node.

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
