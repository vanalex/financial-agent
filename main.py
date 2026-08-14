import asyncio

from app.graph import graph


async def main():
    result = await graph.ainvoke(
        {
            "symbols": [
                "SPY",
                "QQQ",
                "NVDA",
                "AAPL",
                "MSFT",
            ],
            "errors": [],
        }
    )


    print(
        result["analysis"]
    )

if __name__ == '__main__':
    asyncio.run(main())
