import argparse
import asyncio
import json
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, ToolMessage
from langsmith import traceable

from app.deep_agent import (
    DeepMarketBrief,
    build_market_research_prompt,
    render_deep_market_brief,
)
from app.graph import create_graph
from app.nodes.report import save_report_text

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
        "error_count": len(result.get("errors", [])),
        "analysis_char_count": len(result.get("analysis", "")),
        "has_report": bool(result.get("report")),
        "report_path": result.get("report_path"),
    }


def _message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def _last_ai_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) or getattr(message, "type", None) == "ai":
            text = _message_text(message).strip()
            if text:
                return text
    return ""


def _coerce_deep_brief(value) -> DeepMarketBrief | None:
    if isinstance(value, DeepMarketBrief):
        return value
    if isinstance(value, dict):
        return DeepMarketBrief.model_validate(value)
    return None


def _extract_report_path(messages: list) -> str | None:
    for message in reversed(messages):
        if not (
            isinstance(message, ToolMessage)
            or getattr(message, "type", None) == "tool"
        ):
            continue

        content = _message_text(message)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and parsed.get("report_path"):
            return str(parsed["report_path"])

    return None


def _normalize_deep_agent_result(result: dict, symbols: list[str]) -> dict:
    messages = result.get("messages", [])
    errors = []

    brief = _coerce_deep_brief(result.get("structured_response"))
    if brief:
        report = render_deep_market_brief(brief)
        analysis = report
    else:
        analysis = _last_ai_text(messages) or "No analysis is available yet."
        report = analysis
        errors.append("Deep agent did not return the expected structured response.")

    report_path = _extract_report_path(messages)
    if not report_path:
        report_path = save_report_text(report, symbols)

    return {
        "symbols": symbols,
        "analysis": analysis,
        "report": report,
        "report_path": report_path,
        "errors": errors,
        "messages": messages,
        "structured_response": result.get("structured_response"),
    }


@traceable(
    name="Financial Agent Workflow",
    run_type="chain",
    tags=["workflow", "financial-agent"],
    process_outputs=_summarize_workflow_outputs,
)
async def run_market_analysis(
    symbols: list[str],
    thread_id: str,
    max_research_iterations: int,
) -> dict:
    async with create_graph() as graph:
        existing_state = await graph.aget_state(build_config(thread_id))
        if existing_state.values:
            raise RuntimeError(
                f"Checkpoint already exists for thread_id={thread_id!r}. "
                "Use the resume command or choose a new thread_id."
            )

        result = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": build_market_research_prompt(
                            symbols,
                            max_research_iterations,
                        ),
                    }
                ],
            },
            config=build_config(thread_id),
        )
        return _normalize_deep_agent_result(result, symbols)


async def resume_market_analysis(thread_id: str) -> dict:
    async with create_graph() as graph:
        config = build_config(thread_id)
        state = await graph.aget_state(config)
        if not state.values:
            raise RuntimeError(f"No checkpoint found for thread_id={thread_id!r}.")
        if not state.next:
            return _normalize_deep_agent_result(state.values, [])

        result = await graph.ainvoke(None, config=config)
        return _normalize_deep_agent_result(result, [])


async def show_checkpoint(thread_id: str) -> None:
    async with create_graph() as graph:
        state = await graph.aget_state(build_config(thread_id))
        if not state.values:
            print(f"No checkpoint found for thread_id={thread_id!r}.")
            return

        print(f"thread_id: {thread_id}")
        print(f"next: {state.next}")
        print(f"message_count: {len(state.values.get('messages', []))}")
        print(
            "has_structured_response: "
            f"{bool(state.values.get('structured_response'))}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--thread-id")
    run_parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    run_parser.add_argument("--max-research-iterations", type=int, default=2)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--thread-id", required=True)

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--thread-id", required=True)

    parser.set_defaults(
        command="run",
        thread_id=None,
        symbols=DEFAULT_SYMBOLS,
        max_research_iterations=2,
    )
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
        if args.max_research_iterations < 1:
            raise ValueError("--max-research-iterations must be at least 1.")
        print(f"thread_id: {args.thread_id}")
        result = await run_market_analysis(
            args.symbols,
            args.thread_id,
            args.max_research_iterations,
        )

    print(result.get("analysis", "No analysis is available yet."))

    if report_path := result.get("report_path"):
        print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    asyncio.run(main())
