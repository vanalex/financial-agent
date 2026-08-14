from langchain_openai import ChatOpenAI
from langsmith import traceable


llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
)


def _summarize_analysis_inputs(inputs: dict) -> dict:
    state = inputs.get("state", {})
    return {
        "symbols": state.get("symbols", []),
        "market_data_count": len(state.get("market_data", [])),
        "news_count": len(state.get("news", [])),
        "error_count": len(state.get("errors", [])),
    }


def _summarize_analysis_outputs(output: dict) -> dict:
    analysis = output.get("analysis", "")
    return {
        "analysis_char_count": len(analysis),
        "has_analysis": bool(analysis),
    }


@traceable(
    name="Analyze Market Node",
    run_type="chain",
    tags=["graph-node", "analysis"],
    metadata={"model": "gpt-5"},
    process_inputs=_summarize_analysis_inputs,
    process_outputs=_summarize_analysis_outputs,
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
