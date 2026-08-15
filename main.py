import argparse
import asyncio
from datetime import UTC, datetime

from langsmith import traceable

from app.graph import create_graph

DEFAULT_SYMBOLS = ["SPY"]


def build_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def default_thread_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"market-{timestamp}"


def _summarize_workflow_outputs(result: dict | None) -> dict:
    if result is None:
        return {
            "status": "failed",
        }

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
async def run_market_analysis(symbols: list[str], thread_id: str) -> dict:
    async with create_graph() as graph:
        existing_state = await graph.aget_state(build_config(thread_id))
        if existing_state.values:
            raise RuntimeError(
                f"Checkpoint already exists for thread_id={thread_id!r}. "
                "Use the resume command or choose a new thread_id."
            )

        return await graph.ainvoke(
            {
                "symbols": symbols,
                "errors": [],
            },
            config=build_config(thread_id),
        )


async def resume_market_analysis(thread_id: str) -> dict:
    async with create_graph() as graph:
        config = build_config(thread_id)
        state = await graph.aget_state(config)
        if not state.values:
            raise RuntimeError(f"No checkpoint found for thread_id={thread_id!r}.")
        if not state.next:
            return state.values

        return await graph.ainvoke(None, config=config)


async def show_checkpoint(thread_id: str) -> None:
    async with create_graph() as graph:
        state = await graph.aget_state(build_config(thread_id))
        if not state.values:
            print(f"No checkpoint found for thread_id={thread_id!r}.")
            return

        print(f"thread_id: {thread_id}")
        print(f"next: {state.next}")
        print(f"symbols: {state.values.get('symbols', [])}")
        print(f"pending_symbols: {state.values.get('pending_symbols', [])}")
        print(f"completed_symbols: {state.values.get('completed_symbols', [])}")
        print(f"market_data_count: {len(state.values.get('market_data', []))}")
        print(f"news_count: {len(state.values.get('news', []))}")
        print(f"error_count: {len(state.values.get('errors', []))}")
        print(f"has_analysis: {bool(state.values.get('analysis'))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--thread-id")
    run_parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--thread-id", required=True)

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--thread-id", required=True)

    parser.set_defaults(command="run", thread_id=None, symbols=DEFAULT_SYMBOLS)
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.command == "checkpoint":
        await show_checkpoint(args.thread_id)
        return

    if args.command == "resume":
        result = await resume_market_analysis(args.thread_id)
    else:
        args.thread_id = args.thread_id or default_thread_id()
        print(f"thread_id: {args.thread_id}")
        result = await run_market_analysis(args.symbols, args.thread_id)

    print(
        result.get("analysis", "No analysis is available yet.")
    )

if __name__ == '__main__':
    asyncio.run(main())
