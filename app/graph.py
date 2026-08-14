from langgraph.graph import StateGraph, START, END

from app.state import MarketState
from app.nodes.market import fetch_market_data
from app.nodes.news import fetch_news
from app.nodes.analysis import analyze_market


builder = StateGraph(MarketState)

builder.add_node(
    "fetch_market_data",
    fetch_market_data,
)

builder.add_node(
    "fetch_news",
    fetch_news,
)

builder.add_node(
    "analyze_market",
    analyze_market,
)


builder.add_edge(
    START,
    "fetch_market_data",
)

builder.add_edge(
    "fetch_market_data",
    "fetch_news",
)

builder.add_edge(
    "fetch_news",
    "analyze_market",
)

builder.add_edge(
    "analyze_market",
    END,
)


graph = builder.compile()