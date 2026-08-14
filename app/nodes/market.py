import asyncio

import aiohttp

from app.clients.alphavintage_collector import AlphaVantageApiError, get_daily


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
