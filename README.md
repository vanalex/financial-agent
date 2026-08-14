# Financial Intelligent Agent

Financial Intelligent Agent is an async Python research assistant that combines market quotes, financial news, and an LLM analysis step into a single LangGraph workflow. It is designed to produce a short market research brief from recent quote data and news context.

The current entry point analyzes a fixed watchlist:

- `SPY`
- `QQQ`
- `NVDA`
- `AAPL`
- `MSFT`

## Purpose

The project helps collect and interpret market context without manually switching between market data, news search, and an LLM prompt. It does not make trading decisions or provide personalized investment advice. Instead, it summarizes evidence, explains likely market drivers, highlights contradictions, and calls out short-term risks based only on the supplied data.

## Features

- Fetches latest global quote data from Alpha Vantage with async HTTP calls.
- Collects recent financial market news with Tavily search using async LangChain calls.
- Runs a LangGraph pipeline with separate nodes for market data, news, and analysis.
- Uses OpenAI through async `langchain-openai` calls for the final market research analysis.
- Tracks data collection errors in graph state so missing or failed data is visible to the analysis step.
- Validates Alpha Vantage responses with Pydantic models.
- Applies a small request delay before Alpha Vantage calls to reduce rate-limit pressure.

## How It Works

The workflow is defined in `app/graph.py`:

1. `fetch_market_data`
   Retrieves quote data for each symbol using Alpha Vantage.

2. `fetch_news`
   Searches for recent financial market news focused on macroeconomic data, interest rates, technology stocks, oil, inflation, and market-moving events.

3. `analyze_market`
   Sends the collected market data, news, and any collection errors to an OpenAI chat model and asks for a structured market analysis.

The graph state is represented by `MarketState` in `app/state.py`.

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

Create a local environment file:

```bash
cp .env.example .env
```

Fill in the required values:

```bash
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

## Usage

Run the default analysis:

```bash
uv run python main.py
```

The script invokes the LangGraph workflow and prints the final LLM-generated market analysis.

To analyze a different set of symbols, edit the `symbols` list in `main.py`:

```python
result = await graph.ainvoke(
    {
        "symbols": ["SPY", "QQQ", "AAPL"],
        "errors": [],
    }
)
```

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
