from typing import TypedDict


class MarketState(TypedDict, total=False):

    symbols: list[str]

    pending_symbols: list[str]

    completed_symbols: list[str]

    market_data: list[dict]

    news: list[dict]

    analysis: str

    report: str

    report_path: str

    errors: list[str]
