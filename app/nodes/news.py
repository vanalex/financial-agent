from app.clients.tavily import get_financial_news


async def fetch_news(state):

    news = await get_financial_news()

    return {
        "news": news
    }
