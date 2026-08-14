import asyncio
import os
import time
from datetime import date

import aiohttp
from dotenv import load_dotenv
from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

BASE_URL = "https://www.alphavantage.co/query"
MIN_REQUEST_INTERVAL_SECONDS = 1.1
_last_request_at = 0.0
_request_lock = asyncio.Lock()


class AlphaVantageApiError(RuntimeError):
    """Raised when Alpha Vantage returns a valid HTTP response with an API error."""


class GlobalQuote(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(alias="01. symbol")
    close: float = Field(alias="05. price")
    previous_close: float = Field(alias="08. previous close")
    change_pct: str = Field(alias="10. change percent")
    latest_trading_day: date = Field(alias="07. latest trading day")


class AlphaVantageQuoteResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    global_quote: GlobalQuote = Field(alias="Global Quote")


def _get_api_key() -> str:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise AlphaVantageApiError(
            "ALPHA_VANTAGE_API_KEY is not set. Add it to your environment or .env file."
        )

    return api_key


def _parse_change_pct(value: str) -> float:
    return float(value.rstrip("%"))


def _validate_alpha_vantage_payload(symbol: str, payload: dict) -> None:
    for key in ("Information", "Error Message", "Note"):
        message = payload.get(key)
        if message:
            raise AlphaVantageApiError(f"Alpha Vantage response for {symbol}: {message}")

    if payload.get("Global Quote") == {}:
        raise AlphaVantageApiError(
            f"Alpha Vantage returned no quote data for {symbol}. "
            "Check that the ticker symbol is valid."
        )


async def _wait_for_rate_limit_slot() -> None:
    global _last_request_at

    async with _request_lock:
        elapsed = time.monotonic() - _last_request_at
        wait_seconds = MIN_REQUEST_INTERVAL_SECONDS - elapsed
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        _last_request_at = time.monotonic()


@traceable(
    name="Alpha Vantage Global Quote",
    run_type="tool",
    tags=["market-data", "alpha-vantage"],
    metadata={"provider": "alpha_vantage", "function": "GLOBAL_QUOTE"},
)
async def get_daily(symbol: str) -> dict:
    await _wait_for_rate_limit_slot()

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
            BASE_URL,
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": _get_api_key(),
            },
        ) as response:
            response.raise_for_status()
            payload = await response.json()

    _validate_alpha_vantage_payload(symbol, payload)

    data = AlphaVantageQuoteResponse.model_validate(payload)
    quote = data.global_quote

    return {
        "symbol": quote.symbol,
        "date": quote.latest_trading_day,
        "close": quote.close,
        "previous_close": quote.previous_close,
        "change_pct": round(_parse_change_pct(quote.change_pct), 2),
    }

if __name__ == '__main__':
    print(asyncio.run(get_daily("AAPL")))
