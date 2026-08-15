import asyncio

import aiohttp
from langsmith import traceable
from pydantic import ValidationError

from app.clients.alphavintage_collector import AlphaVantageApiError, get_daily


def _summarize_market_state_inputs(inputs: dict) -> dict:
    state = inputs.get("state", {})
    return {
        "symbols": state.get("symbols", []),
        "pending_symbols": state.get("pending_symbols", []),
        "completed_symbols": state.get("completed_symbols", []),
        "incoming_error_count": len(state.get("errors", [])),
    }


def _summarize_market_state_outputs(output: dict | None) -> dict:
    if output is None:
        return {
            "status": "failed",
        }

    return {
        "symbols_returned": [
            item.get("symbol")
            for item in output.get("market_data", [])
        ],
        "market_data_count": len(output.get("market_data", [])),
        "pending_count": len(output.get("pending_symbols", [])),
        "completed_count": len(output.get("completed_symbols", [])),
        "error_count": len(output.get("errors", [])),
    }


def prepare_market_data(state):
    return {
        "pending_symbols": list(state["symbols"]),
        "completed_symbols": [],
        "market_data": [],
        "errors": list(state.get("errors", [])),
    }


@traceable(
    name="Fetch Market Data Node",
    run_type="chain",
    tags=["graph-node", "market-data"],
    process_inputs=_summarize_market_state_inputs,
    process_outputs=_summarize_market_state_outputs,
)
async def fetch_market_data(state):
    pending_symbols = list(state.get("pending_symbols", []))
    completed_symbols = list(state.get("completed_symbols", []))
    market_data = list(state.get("market_data", []))
    errors = list(state.get("errors", []))

    if not pending_symbols:
        return {}

    symbol = pending_symbols.pop(0)
    try:
        market_data.append(await get_daily(symbol))
    except (
        AlphaVantageApiError,
        aiohttp.ClientError,
        asyncio.TimeoutError,
        ValidationError,
    ) as exc:
        errors.append(f"{symbol}: {exc}")

    completed_symbols.append(symbol)

    return {
        "pending_symbols": pending_symbols,
        "completed_symbols": completed_symbols,
        "market_data": market_data,
        "errors": errors,
    }
