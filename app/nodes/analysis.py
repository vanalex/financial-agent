from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
)


async def analyze_market(state):

    market_data = state["market_data"]
    news = state["news"]
    errors = state.get("errors", [])

    prompt = f"""
You are a financial market research analyst.

You receive structured market data and recent news.

MARKET DATA

{market_data}

NEWS

{news}

DATA COLLECTION ERRORS

{errors}

Analyze this information.

Identify:

1. The three most important developments.
2. Likely explanations for the market movements.
3. Connections between news and price movements.
4. Cases where market behavior contradicts the news.
5. Important risks for the next 24-48 hours.

Rules:

- Never invent market data.
- Clearly distinguish facts from interpretation.
- Only make claims supported by the supplied evidence.
- Treat data collection errors as missing evidence, not as market signals.
- If evidence is insufficient, explicitly say so.
- Do not provide personalized investment advice.
"""

    response = await llm.ainvoke(prompt)

    return {
        "analysis": response.content
    }
