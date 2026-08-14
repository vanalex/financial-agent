import asyncio

from langsmith import traceable

from app.graph import graph


def _summarize_workflow_outputs(result: dict) -> dict:
    return {
        "symbols": result.get("symbols", []),
        "market_data_count": len(result.get("market_data", [])),
        "news_count": len(result.get("news", [])),
        "error_count": len(result.get("errors", [])),
        "analysis_char_count": len(result.get("analysis", "")),
    }


@traceable(
    name="Financial Agent Workflow",
    run_type="chain",
    tags=["workflow", "financial-agent"],
    process_outputs=_summarize_workflow_outputs,
)
async def run_market_analysis(symbols: list[str]) -> dict:
    return await graph.ainvoke(
        {
            "symbols": symbols,
            "errors": [],
        }
    )


async def main():
    result = await run_market_analysis(
        [
            "SPY",
        ]
    )

    print(
        result["analysis"]
    )

if __name__ == '__main__':
    asyncio.run(main())
