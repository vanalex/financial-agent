import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.deep_agent import create_financial_deep_agent

load_dotenv()


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
        yield create_financial_deep_agent(checkpointer=checkpointer)
