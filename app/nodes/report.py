import json
import re
from datetime import UTC, datetime
from pathlib import Path

from langsmith import traceable


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def _summarize_report_inputs(inputs: dict) -> dict:
    state = inputs.get("state", {})
    return {
        "symbols": state.get("symbols", []),
        "has_analysis": bool(state.get("analysis")),
        "market_data_count": len(state.get("market_data", [])),
        "news_count": len(state.get("news", [])),
        "error_count": len(state.get("errors", [])),
    }


def _summarize_report_outputs(output: dict | None) -> dict:
    if output is None:
        return {
            "has_report": False,
        }

    return {
        "has_report": bool(output.get("report")),
        "report_path": output.get("report_path"),
    }


def _safe_filename_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:80] or "market-brief"


def _json_block(value) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def _build_report(state: dict) -> str:
    generated_at = datetime.now(UTC).isoformat()
    analysis = state.get("analysis") or "No analysis is available."

    return "\n".join(
        [
            analysis.rstrip(),
            "",
            "---",
            "",
            "## Evidence Snapshot",
            "",
            f"- Generated at: {generated_at}",
            f"- Symbols: {', '.join(state.get('symbols', [])) or 'None'}",
            f"- Market data records: {len(state.get('market_data', []))}",
            f"- News records: {len(state.get('news', []))}",
            f"- Collection errors: {len(state.get('errors', []))}",
            "",
            "### Market Data",
            "",
            "```json",
            _json_block(state.get("market_data", [])),
            "```",
            "",
            "### News",
            "",
            "```json",
            _json_block(state.get("news", [])),
            "```",
            "",
            "### Data Collection Errors",
            "",
            "```json",
            _json_block(state.get("errors", [])),
            "```",
            "",
        ]
    )


@traceable(
    name="Write Market Report Node",
    run_type="chain",
    tags=["graph-node", "report"],
    process_inputs=_summarize_report_inputs,
    process_outputs=_summarize_report_outputs,
)
def write_market_report(state):
    OUTPUT_DIR.mkdir(exist_ok=True)

    report = _build_report(state)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    symbols = "-".join(state.get("symbols", [])) or "market"
    filename = f"market_brief_{_safe_filename_part(symbols)}_{timestamp}.md"
    report_path = OUTPUT_DIR / filename
    report_path.write_text(report, encoding="utf-8")

    return {
        "report": report,
        "report_path": str(report_path),
    }
