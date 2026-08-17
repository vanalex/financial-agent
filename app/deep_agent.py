import os

from deepagents import create_deep_agent
from langsmith import traceable
from pydantic import BaseModel, Field

from app.clients.alphavintage_collector import get_daily
from app.clients.tavily import search_financial_news
from app.nodes.analysis import REQUIRED_DISCLAIMER
from app.nodes.report import save_report_text


DEFAULT_DEEP_AGENT_MODEL = "openai:gpt-5"

FINANCIAL_RESEARCH_SYSTEM_PROMPT = f"""
You are a financial market research deep agent.

Your job is to produce evidence-grounded market briefs. Use tools to gather
current quote and news evidence, plan the research, track evidence gaps, and
iterate when another search would materially improve the brief.

Required behavior:

- Fetch quote data for every requested symbol before discussing price movement.
- Search financial news with targeted queries rather than one broad query.
- Reflect on evidence gaps before writing the final brief.
- Never invent market data, dates, article titles, URLs, or sources.
- Clearly distinguish facts from interpretation.
- Cite supplied market data with symbol and latest trading day.
- Cite news claims with article title or URL.
- Treat tool errors as missing evidence, not as market signals.
- Do not call anything "today" unless collected evidence includes today's date.
- Do not provide personalized investment advice.
- Include this exact disclaimer: {REQUIRED_DISCLAIMER}

When the brief is ready, call save_market_report with the final Markdown report.
Then provide the same brief as your structured final response.
"""


class DeepMarketDevelopment(BaseModel):
    title: str = Field(description="Short name for the development.")
    facts: list[str] = Field(
        description="Facts supported directly by collected market data or news."
    )
    interpretation: str = Field(
        description="Analyst interpretation clearly separated from facts."
    )
    evidence: list[str] = Field(
        description="Specific symbol/date, article title/URL, or tool error."
    )


class DeepMarketBrief(BaseModel):
    executive_summary: str
    market_data_reviewed: list[str]
    top_developments: list[DeepMarketDevelopment]
    price_news_linkage: list[str]
    contradictions: list[str]
    risks_next_24_48h: list[str]
    evidence_gaps: list[str]
    report_markdown: str = Field(
        description="Complete final Markdown report, including all required sections."
    )
    disclaimer: str = Field(default=REQUIRED_DISCLAIMER)


@traceable(
    name="Deep Agent Market Quote Tool",
    run_type="tool",
    tags=["deep-agent", "market-data", "alpha-vantage"],
)
async def get_market_quote(symbol: str) -> dict:
    """Fetch the latest Alpha Vantage global quote for one ticker symbol."""
    return await get_daily(symbol)


@traceable(
    name="Deep Agent Financial News Tool",
    run_type="tool",
    tags=["deep-agent", "news", "tavily"],
)
async def search_market_news(query: str) -> list[dict]:
    """Search recent financial news for a targeted market research query."""
    return await search_financial_news(query)


@traceable(
    name="Deep Agent Report Save Tool",
    run_type="tool",
    tags=["deep-agent", "report"],
)
def save_market_report(markdown: str, symbols: list[str]) -> dict:
    """Save the final market brief Markdown report and return its local path."""
    report_path = save_report_text(markdown, symbols)
    return {
        "report_path": report_path,
    }


def create_financial_deep_agent(checkpointer=None):
    model = os.getenv("DEEP_AGENT_MODEL", DEFAULT_DEEP_AGENT_MODEL)
    return create_deep_agent(
        model=model,
        tools=[
            get_market_quote,
            search_market_news,
            save_market_report,
        ],
        system_prompt=FINANCIAL_RESEARCH_SYSTEM_PROMPT,
        response_format=DeepMarketBrief,
        checkpointer=checkpointer,
        name="financial-market-deep-agent",
    )


def build_market_research_prompt(
    symbols: list[str],
    max_research_iterations: int,
) -> str:
    return f"""
Produce a market research brief for these symbols: {", ".join(symbols)}.

Research depth:
- Run no more than {max_research_iterations} research/reflection passes.
- In each pass, use targeted news searches only when they address a concrete
  evidence gap.

Deliverable:
- Gather quote data for each symbol.
- Gather recent financial news relevant to price movement, macro context,
  rates, inflation, technology stocks, oil, and market-moving events as needed.
- Reflect on evidence gaps before final synthesis.
- Save the final Markdown report with save_market_report.
- Return the final brief in the requested structured response format.
""".strip()


def render_deep_market_brief(brief: DeepMarketBrief) -> str:
    if brief.report_markdown.strip():
        return brief.report_markdown.strip() + "\n"

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
            sections.append(f"   Evidence: {'; '.join(development.evidence)}")

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
