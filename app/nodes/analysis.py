import json
import os
from functools import cache
from typing import Literal

from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field


DEFAULT_ANALYSIS_MODEL = "gpt-5"
REQUIRED_DISCLAIMER = (
    "This market brief is for informational research only and does not "
    "constitute personalized investment advice."
)


class EvidenceReference(BaseModel):
    source_type: Literal["market_data", "news", "error"] = Field(
        description="Type of supplied evidence supporting the claim."
    )
    source: str = Field(
        description="Specific symbol/date, article title/URL, or collection error."
    )


class MarketDevelopment(BaseModel):
    title: str = Field(description="Short name for the development.")
    facts: list[str] = Field(
        description="Facts supported directly by supplied market data or news."
    )
    interpretation: str = Field(
        description="Analyst interpretation clearly separated from facts."
    )
    evidence: list[EvidenceReference] = Field(
        description="Evidence references for the facts and interpretation."
    )


class MarketBrief(BaseModel):
    executive_summary: str
    market_data_reviewed: list[str]
    top_developments: list[MarketDevelopment]
    price_news_linkage: list[str]
    contradictions: list[str]
    risks_next_24_48h: list[str]
    evidence_gaps: list[str]
    disclaimer: str = Field(default=REQUIRED_DISCLAIMER)


@cache
def _analysis_llm():
    model = os.getenv("ANALYSIS_MODEL", DEFAULT_ANALYSIS_MODEL)
    return ChatOpenAI(
        model=model,
        temperature=0,
    ).with_structured_output(MarketBrief)


def _summarize_analysis_inputs(inputs: dict) -> dict:
    state = inputs.get("state", {})
    return {
        "symbols": state.get("symbols", []),
        "market_data_count": len(state.get("market_data", [])),
        "news_count": len(state.get("news", [])),
        "error_count": len(state.get("errors", [])),
    }


def _summarize_analysis_outputs(output: dict | None) -> dict:
    if output is None:
        return {
            "has_analysis": False,
        }

    analysis = output.get("analysis", "")
    return {
        "analysis_char_count": len(analysis),
        "has_analysis": bool(analysis),
        "analysis_model": os.getenv("ANALYSIS_MODEL", DEFAULT_ANALYSIS_MODEL),
    }


def _json_block(value) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def _format_market_data(market_data: list[dict]) -> list[dict]:
    return [
        {
            "symbol": item.get("symbol"),
            "latest_trading_day": item.get("date"),
            "close": item.get("close"),
            "previous_close": item.get("previous_close"),
            "change_pct": item.get("change_pct"),
        }
        for item in market_data
    ]


def _format_news(news: list[dict]) -> list[dict]:
    formatted = []
    for article in news:
        formatted.append(
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "published_date": article.get("published_date"),
                "score": article.get("score"),
                "content": article.get("content"),
            }
        )
    return formatted


def render_market_brief(brief: MarketBrief) -> str:
    sections = [
        "# Market Brief",
        "",
        "## Executive Summary",
        brief.executive_summary,
        "",
        "## Market Data Reviewed",
    ]

    sections.extend(f"- {item}" for item in brief.market_data_reviewed)
    sections.extend(["", "## Top Developments"])

    for index, development in enumerate(brief.top_developments, 1):
        sections.append(f"{index}. {development.title}")
        if development.facts:
            sections.append("   Facts:")
            sections.extend(f"   - {fact}" for fact in development.facts)
        sections.append(f"   Interpretation: {development.interpretation}")
        if development.evidence:
            evidence = "; ".join(
                f"{item.source_type}: {item.source}" for item in development.evidence
            )
            sections.append(f"   Evidence: {evidence}")

    sections.extend(["", "## Price and News Linkage"])
    sections.extend(f"- {item}" for item in brief.price_news_linkage)

    sections.extend(["", "## Contradictions"])
    sections.extend(f"- {item}" for item in brief.contradictions)

    sections.extend(["", "## Risks: Next 24-48 Hours"])
    sections.extend(f"- {item}" for item in brief.risks_next_24_48h)

    sections.extend(["", "## Evidence Gaps"])
    sections.extend(f"- {item}" for item in brief.evidence_gaps)

    sections.extend(["", "## Disclaimer", brief.disclaimer])

    return "\n".join(sections).strip() + "\n"


def _insufficient_evidence_analysis(errors: list[str]) -> str:
    gaps = [
        "No market data was available.",
        "No news articles were available.",
    ]
    if errors:
        gaps.append("Collection errors were reported: " + "; ".join(errors))

    brief = MarketBrief(
        executive_summary=(
            "There is insufficient supplied evidence to produce a market brief. "
            "No factual market movement or news linkage claims can be made from "
            "empty market data and news inputs."
        ),
        market_data_reviewed=[],
        top_developments=[],
        price_news_linkage=[
            "No price/news linkage can be assessed without market data or news."
        ],
        contradictions=[
            "No contradictions can be assessed without market data or news."
        ],
        risks_next_24_48h=[
            "Short-term risks cannot be identified from the supplied evidence."
        ],
        evidence_gaps=gaps,
        disclaimer=REQUIRED_DISCLAIMER,
    )
    return render_market_brief(brief)


@traceable(
    name="Analyze Market Node",
    run_type="chain",
    tags=["graph-node", "analysis"],
    metadata={"model": "gpt-5"},
    process_inputs=_summarize_analysis_inputs,
    process_outputs=_summarize_analysis_outputs,
)
async def analyze_market(state):

    market_data = state.get("market_data", [])
    news = state.get("news", [])
    errors = state.get("errors", [])

    if not market_data and not news:
        return {
            "analysis": _insufficient_evidence_analysis(errors),
        }

    prompt = f"""
You are a financial market research analyst.

You receive structured market data, recent news, and data collection errors.

Return a concise structured market brief using the requested schema.

MARKET DATA

{_json_block(_format_market_data(market_data))}

NEWS

{_json_block(_format_news(news))}

DATA COLLECTION ERRORS

{_json_block(errors)}

Analyze this information.

Required sections:

- Executive summary.
- Market data reviewed.
- The three most important developments.
- Likely explanations for market movements.
- Connections between news and price movements.
- Cases where market behavior contradicts the news.
- Important risks for the next 24-48 hours.
- Evidence gaps.

Rules:

- Never invent market data.
- Clearly distinguish facts from interpretation.
- Only make claims supported by the supplied evidence.
- Every factual news claim must cite an article title or URL from NEWS.
- Every market movement claim must cite the relevant symbol and latest_trading_day from MARKET DATA.
- Treat data collection errors as missing evidence, not as market signals.
- If evidence is insufficient, explicitly say so.
- Do not provide personalized investment advice.
- Do not call anything "today" unless the supplied evidence includes today's date.
- Always mention latest_trading_day when discussing price data.
- Include this exact disclaimer: {REQUIRED_DISCLAIMER}
"""

    brief = await _analysis_llm().ainvoke(prompt)

    return {
        "analysis": render_market_brief(brief),
    }
