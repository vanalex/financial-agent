import asyncio

import aiohttp
from langsmith import traceable

from app.clients.alphavintage_collector import AlphaVantageApiError, get_daily


def _summarize_market_state_inputs(inputs: dict) -> dict:
    state = inputs.get("state", {})
    return {
        "symbols": state.get("symbols", []),
        "incoming_error_count": len(state.get("errors", [])),
    }


def _summarize_market_state_outputs(output: dict) -> dict:
    return {
        "symbols_returned": [
            item.get("symbol")
            for item in output.get("market_data", [])
        ],
        "market_data_count": len(output.get("market_data", [])),
        "error_count": len(output.get("errors", [])),
    }


@traceable(
    name="Fetch Market Data Node",
    run_type="chain",
    tags=["graph-node", "market-data"],
    process_inputs=_summarize_market_state_inputs,
    process_outputs=_summarize_market_state_outputs,
)
async def fetch_market_data(state):

    results = []
    errors = list(state.get("errors", []))

    async def fetch_symbol(symbol: str):
        try:
            return await get_daily(symbol), None
        except (AlphaVantageApiError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return None, f"{symbol}: {exc}"

    symbol_results = await asyncio.gather(
        *(fetch_symbol(symbol) for symbol in state["symbols"])
    )

    for market_data, error in symbol_results:
        if market_data is not None:
            results.append(market_data)
        if error is not None:
            errors.append(error)

    return {
        "market_data": results,
        "errors": errors,
    }
