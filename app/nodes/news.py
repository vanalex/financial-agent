from langsmith import traceable

from app.clients.tavily import get_financial_news


def _summarize_news_state_inputs(inputs: dict) -> dict:
    state = inputs.get("state", {})
    return {
        "symbol_count": len(state.get("symbols", [])),
        "market_data_count": len(state.get("market_data", [])),
        "error_count": len(state.get("errors", [])),
    }


def _summarize_news_state_outputs(output: dict) -> dict:
    return {
        "news_count": len(output.get("news", [])),
        "urls": [
            article.get("url")
            for article in output.get("news", [])
        ],
    }


@traceable(
    name="Fetch News Node",
    run_type="chain",
    tags=["graph-node", "news"],
    process_inputs=_summarize_news_state_inputs,
    process_outputs=_summarize_news_state_outputs,
)
async def fetch_news(state):

    news = await get_financial_news()

    return {
        "news": news
    }
