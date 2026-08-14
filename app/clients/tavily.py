from langchain_tavily import TavilySearch

from dotenv import load_dotenv
load_dotenv()

search = TavilySearch(
    max_results=5,
    topic="news",
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
