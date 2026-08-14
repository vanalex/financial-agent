from typing import TypedDict


class MarketState(TypedDict, total=False):

    symbols: list[str]

    market_data: list[dict]

    news: list[dict]

    analysis: str

    report: str

    errors: list[str]