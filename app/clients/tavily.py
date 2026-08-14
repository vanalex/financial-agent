from langchain_tavily import TavilySearch
from langsmith import traceable

from dotenv import load_dotenv
load_dotenv()

search = TavilySearch(
    max_results=5,
    topic="news",
)


def _summarize_news_output(news: list[dict]) -> dict:
    return {
        "article_count": len(news),
        "articles": [
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "score": article.get("score"),
            }
            for article in news
        ],
    }


@traceable(
    name="Tavily Financial News Search",
    run_type="tool",
    tags=["news", "tavily"],
    metadata={"provider": "tavily", "topic": "news"},
    process_outputs=_summarize_news_output,
)
async def get_financial_news() -> list[dict]:

    query = """
    Most important EU financial market news today.
    Focus on:
    Federal Reserve,
    inflation,
    interest rates,
    stock market,
    technology stocks,
    oil,
    macroeconomic data.
    """

    result = await search.ainvoke({
        "query": query
    })

    return result["results"]

if __name__ == '__main__':
    import asyncio

    news = asyncio.run(get_financial_news())

    for article in news:
        print(article["title"])
        print(article["url"])
