from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clients.tavily import get_financial_news


class NewsArticle(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    content: str | None = None
    published_date: str | None = None
    score: float | None = None
    query: str | None = None


def _validate_news_articles(news: list[dict]) -> tuple[list[dict], list[str]]:
    articles = []
    errors = []

    for index, article in enumerate(news, 1):
        try:
            articles.append(NewsArticle.model_validate(article).model_dump())
        except ValidationError as exc:
            errors.append(f"news article {index}: {exc}")

    return articles, errors


def _summarize_news_state_inputs(inputs: dict) -> dict:
    state = inputs.get("state", {})
    return {
        "symbol_count": len(state.get("symbols", [])),
        "market_data_count": len(state.get("market_data", [])),
        "error_count": len(state.get("errors", [])),
    }


def _summarize_news_state_outputs(output: dict | None) -> dict:
    if output is None:
        return {
            "status": "failed",
        }

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
    errors = list(state.get("errors", []))

    raw_news = await get_financial_news()
    news, validation_errors = _validate_news_articles(raw_news)
    errors.extend(validation_errors)

    return {
        "news": news,
        "errors": errors,
    }
