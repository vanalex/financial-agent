import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END

from app.state import MarketState
from app.nodes.market import fetch_market_data, prepare_market_data
from app.nodes.news import fetch_news
from app.nodes.analysis import analyze_market

load_dotenv()

builder = StateGraph(MarketState)

builder.add_node(
    "prepare_market_data",
    prepare_market_data,
)

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
    "prepare_market_data",
)


def route_market_data(state: MarketState) -> str:
    if state.get("pending_symbols"):
        return "fetch_market_data"
    return "fetch_news"


builder.add_conditional_edges(
    "prepare_market_data",
    route_market_data,
    {
        "fetch_market_data": "fetch_market_data",
        "fetch_news": "fetch_news",
    },
)

builder.add_conditional_edges(
    "fetch_market_data",
    route_market_data,
    {
        "fetch_market_data": "fetch_market_data",
        "fetch_news": "fetch_news",
    },
)

builder.add_edge(
    "fetch_news",
    "analyze_market",
)

builder.add_edge(
    "analyze_market",
    END,
)



def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is required for the Postgres LangGraph checkpointer."
        )
    return database_url


@asynccontextmanager
async def create_graph() -> AsyncIterator:
    async with AsyncPostgresSaver.from_conn_string(get_database_url()) as checkpointer:
        await checkpointer.setup()
        yield builder.compile(checkpointer=checkpointer)
