"""
Evaluators for the Financial Intelligent Agent.

The suite mixes deterministic code checks with an LLM judge:
- structure: required market brief sections are present
- evidence discipline: market/news claims cite supplied symbols and news titles
- expected reasoning: expected themes and relationships appear
- unsupported claims: forbidden claims are absent
- risk controls: insufficient evidence and advice disclaimers are handled
- analytical quality: LLM judge checks causal reasoning and uncertainty handling
"""

import re
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI

from app.nodes.analysis import REQUIRED_DISCLAIMER


REQUIRED_SECTIONS = [
    "Executive Summary",
    "Market Data Reviewed",
    "Top Developments",
    "Price and News Linkage",
    "Contradictions",
    "Risks: Next 24-48 Hours",
    "Evidence Gaps",
    "Disclaimer",
]


def _get_outputs(obj):
    return obj.outputs if hasattr(obj, "outputs") else obj.get("outputs", {}) or {}


def _get_inputs(obj):
    return obj.inputs if hasattr(obj, "inputs") else obj.get("inputs", {}) or {}


def _output_text(run) -> str:
    outputs = _get_outputs(run)
    output = outputs.get("output", "")
    return output if isinstance(output, str) else str(output)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _meaningful_words(phrase: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", phrase.lower())
    return [
        word
        for word in words
        if len(word) > 3 and word not in {"with", "from", "that", "this", "into"}
    ]


def _matches_expected_text(text: str, phrase: str) -> bool:
    if phrase.lower() in text:
        return True

    meaningful_words = _meaningful_words(phrase)
    if not meaningful_words:
        return phrase.lower() in text

    matched = sum(1 for word in meaningful_words if word in text)
    return matched / len(meaningful_words) >= 0.6


def _matches_forbidden_claim(text: str, phrase: str) -> bool:
    normalized_phrase = _normalize(phrase)
    if normalized_phrase in text:
        return True

    meaningful_words = _meaningful_words(phrase)
    if not meaningful_words:
        return False

    matched = sum(1 for word in meaningful_words if word in text)
    return matched == len(meaningful_words)


def structural_quality(run, example) -> dict:
    output = _output_text(run)
    output_lower = output.lower()

    found = [
        section
        for section in REQUIRED_SECTIONS
        if section.lower() in output_lower
    ]
    missing = [
        section
        for section in REQUIRED_SECTIONS
        if section.lower() not in output_lower
    ]

    score = len(found) / len(REQUIRED_SECTIONS)
    comment = (
        f"Found {len(found)}/{len(REQUIRED_SECTIONS)} required sections."
        if not missing
        else f"Missing sections: {', '.join(missing)}."
    )
    return {"score": score, "comment": comment}


def expected_reasoning(run, example) -> dict:
    output = _normalize(_output_text(run))
    expected = _get_outputs(example)

    checks = []
    for theme in expected.get("expected_themes", []):
        checks.append((f"theme: {theme}", _matches_expected_text(output, theme)))
    for relationship in expected.get("expected_relationships", []):
        checks.append(
            (f"relationship: {relationship}", _matches_expected_text(output, relationship))
        )

    if not checks:
        return {"score": 1.0, "comment": "No expected reasoning checks defined."}

    passed = [label for label, ok in checks if ok]
    failed = [label for label, ok in checks if not ok]
    score = len(passed) / len(checks)
    comment = (
        f"Matched {len(passed)}/{len(checks)} expected reasoning checks."
        if not failed
        else "Missing: " + "; ".join(failed[:5])
    )
    return {"score": score, "comment": comment}


def unsupported_claims(run, example) -> dict:
    output = _normalize(_output_text(run))
    forbidden = _get_outputs(example).get("must_not_claim", [])

    matched = [
        claim
        for claim in forbidden
        if _matches_forbidden_claim(output, claim)
    ]
    score = 0.0 if matched else 1.0
    comment = (
        "No forbidden claims detected."
        if not matched
        else "Forbidden claims detected: " + "; ".join(matched)
    )
    return {"score": score, "comment": comment}


def evidence_discipline(run, example) -> dict:
    output = _output_text(run)
    output_lower = output.lower()
    inputs = _get_inputs(example)

    symbols = inputs.get("symbols", [])
    news_titles = [
        article.get("title", "")
        for article in inputs.get("news", [])
        if article.get("title")
    ]

    checks = []
    for item in inputs.get("market_data", []):
        symbol = item.get("symbol")
        if symbol:
            checks.append((f"market symbol cited: {symbol}", symbol.lower() in output_lower))

    if news_titles:
        title_matched = any(title.lower() in output_lower for title in news_titles)
        checks.append(("at least one supplied news title cited", title_matched))

    if symbols:
        any_symbol = any(symbol.lower() in output_lower for symbol in symbols)
        checks.append(("at least one supplied symbol cited", any_symbol))

    if not checks:
        return {"score": 1.0, "comment": "No evidence references required."}

    passed = [label for label, ok in checks if ok]
    failed = [label for label, ok in checks if not ok]
    score = len(passed) / len(checks)
    comment = (
        f"Passed {len(passed)}/{len(checks)} evidence checks."
        if not failed
        else "Failed: " + "; ".join(failed)
    )
    return {"score": score, "comment": comment}


def risk_controls(run, example) -> dict:
    output = _normalize(_output_text(run))
    expected = _get_outputs(example)

    checks = [
        (
            "required disclaimer present",
            REQUIRED_DISCLAIMER.lower() in output,
        ),
        (
            "no personalized investment advice",
            "personalized investment advice" in output
            or "does not constitute investment advice" in output,
        ),
    ]

    if expected.get("evidence_sufficient") is False:
        checks.append(
            (
                "insufficient evidence acknowledged",
                "insufficient evidence" in output
                or "does not adequately explain" in output
                or "cannot be assessed" in output
                or "missing explanation" in output,
            )
        )

    if expected.get("should_research_more"):
        checks.append(
            (
                "additional evidence or research need identified",
                "additional" in output
                or "more evidence" in output
                or "further research" in output
                or "evidence gaps" in output,
            )
        )

    passed = [label for label, ok in checks if ok]
    failed = [label for label, ok in checks if not ok]
    score = len(passed) / len(checks)
    comment = (
        f"Passed {len(passed)}/{len(checks)} risk-control checks."
        if not failed
        else "Failed: " + "; ".join(failed)
    )
    return {"score": score, "comment": comment}


class AnalyticalQualityGrade(TypedDict):
    reasoning: Annotated[str, ..., "Concise assessment of analytical quality"]
    evidence_use_score: Annotated[int, ..., "1-5 score for use of supplied evidence"]
    causality_score: Annotated[int, ..., "1-5 score for causal discipline"]
    uncertainty_score: Annotated[int, ..., "1-5 score for uncertainty handling"]
    overall_score: Annotated[int, ..., "1-5 overall score"]


_analytical_quality_judge = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
).with_structured_output(
    AnalyticalQualityGrade,
    method="json_schema",
    strict=True,
)


def analytical_quality(run, example) -> dict:
    output = _output_text(run)
    inputs = _get_inputs(example)
    expected = _get_outputs(example)

    prompt = f"""You are evaluating a market research brief.

SUPPLIED MARKET DATA:
{inputs.get("market_data", [])}

SUPPLIED NEWS:
{inputs.get("news", [])}

EXPECTED THEMES:
{expected.get("expected_themes", [])}

EXPECTED RELATIONSHIPS:
{expected.get("expected_relationships", [])}

MUST NOT CLAIM:
{expected.get("must_not_claim", [])}

AGENT OUTPUT:
{output[:8000]}

Evaluate whether the brief uses only supplied evidence, avoids unsupported
causal claims, identifies contradictions, and acknowledges missing evidence."""

    grade = _analytical_quality_judge.invoke([{"role": "user", "content": prompt}])
    score = grade["overall_score"] / 5.0
    return {
        "score": score,
        "comment": (
            f"Evidence: {grade['evidence_use_score']}/5, "
            f"Causality: {grade['causality_score']}/5, "
            f"Uncertainty: {grade['uncertainty_score']}/5, "
            f"Overall: {grade['overall_score']}/5. "
            f"Reasoning: {grade['reasoning']}"
        ),
    }


ALL_EVALUATORS = [
    structural_quality,
    expected_reasoning,
    unsupported_claims,
    evidence_discipline,
    risk_controls,
    analytical_quality,
]
