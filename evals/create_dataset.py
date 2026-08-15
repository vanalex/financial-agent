"""
Create the LangSmith evaluation dataset for the
Financial Intelligence Agent.

Run:

    python evals/create_dataset.py

Environment variables required:

    LANGSMITH_API_KEY=...
    LANGSMITH_TRACING=true

Optionally:

    LANGSMITH_PROJECT=financial-intelligence-agent
"""

from langsmith import Client
from dotenv import load_dotenv
load_dotenv()


DATASET_NAME = "financial-intelligence-agent-v1"

DATASET_DESCRIPTION = """
Evaluation dataset for a LangGraph-based financial intelligence agent.

Each example contains:
- structured market movements
- financial news available to the agent
- reference expectations describing what a good analysis should detect

The dataset evaluates reasoning quality rather than exact text generation.

Important capabilities being tested:
- identifying the market regime
- connecting news with market movements
- detecting contradictions
- avoiding unsupported causal claims
- recognizing insufficient evidence
- deciding when additional research is necessary
""".strip()


def build_examples() -> list[dict]:
    """
    Build controlled synthetic examples.

    We intentionally use synthetic market situations rather than live data.

    This makes expected reasoning deterministic and prevents the evaluation
    dataset from changing every day.
    """

    return [
        # ------------------------------------------------------------
        # CASE 1
        #
        # Higher inflation + higher yields + tech selloff.
        #
        # The agent should recognize a macro/rates-driven risk-off move.
        # It should also notice that NVDA is falling despite positive
        # company-specific news.
        # ------------------------------------------------------------
        {
            "inputs": {
                "symbols": [
                    "SPY",
                    "QQQ",
                    "NVDA",
                ],
                "market_data": [
                    {
                        "symbol": "SPY",
                        "change_pct": -1.2,
                    },
                    {
                        "symbol": "QQQ",
                        "change_pct": -2.1,
                    },
                    {
                        "symbol": "NVDA",
                        "change_pct": -4.5,
                    },
                ],
                "news": [
                    {
                        "title": "Inflation comes in above expectations",
                        "summary": (
                            "Latest inflation data was stronger than expected, "
                            "raising concerns that interest rates may remain "
                            "higher for longer."
                        ),
                    },
                    {
                        "title": "Treasury yields rise after inflation report",
                        "summary": (
                            "US government bond yields moved higher following "
                            "the stronger inflation release."
                        ),
                    },
                    {
                        "title": "Nvidia reports strong revenue growth",
                        "summary": (
                            "Nvidia reported stronger-than-expected revenue "
                            "and maintained a positive business outlook."
                        ),
                    },
                ],
            },
            "outputs": {
                "expected_direction": "risk_off",
                "expected_themes": [
                    "higher inflation",
                    "higher interest rates or treasury yields",
                    "pressure on growth and technology stocks",
                ],
                "expected_relationships": [
                    (
                        "Technology underperformance is consistent with "
                        "rising interest-rate expectations."
                    ),
                    (
                        "Nvidia's decline contradicts the positive "
                        "company-specific news."
                    ),
                ],
                "must_not_claim": [
                    "Nvidia declined because of weak earnings",
                    "Nvidia missed revenue expectations",
                ],
                "evidence_sufficient": True,
                "should_research_more": False,
            },
        },

        # ------------------------------------------------------------
        # CASE 2
        #
        # Oil rises and energy stocks outperform.
        #
        # This should be an easy causal relationship for the agent.
        # ------------------------------------------------------------
        {
            "inputs": {
                "symbols": [
                    "SPY",
                    "XLE",
                ],
                "market_data": [
                    {
                        "symbol": "SPY",
                        "change_pct": -0.4,
                    },
                    {
                        "symbol": "XLE",
                        "change_pct": 2.8,
                    },
                ],
                "news": [
                    {
                        "title": "Oil prices surge on supply concerns",
                        "summary": (
                            "Oil prices rose sharply following renewed "
                            "concerns about global crude supply."
                        ),
                    }
                ],
            },
            "outputs": {
                "expected_direction": "mixed",
                "expected_themes": [
                    "oil price increase",
                    "energy sector outperformance",
                ],
                "expected_relationships": [
                    (
                        "Energy sector outperformance is consistent with "
                        "higher oil prices."
                    )
                ],
                "must_not_claim": [
                    "the broad equity market strongly rallied",
                ],
                "evidence_sufficient": True,
                "should_research_more": False,
            },
        },

        # ------------------------------------------------------------
        # CASE 3
        #
        # Very small market move and neutral Fed communication.
        #
        # The agent should NOT manufacture a dramatic narrative.
        # ------------------------------------------------------------
        {
            "inputs": {
                "symbols": [
                    "SPY",
                    "QQQ",
                ],
                "market_data": [
                    {
                        "symbol": "SPY",
                        "change_pct": 0.1,
                    },
                    {
                        "symbol": "QQQ",
                        "change_pct": 0.2,
                    },
                ],
                "news": [
                    {
                        "title": (
                            "Federal Reserve official gives neutral remarks"
                        ),
                        "summary": (
                            "A Federal Reserve official reiterated that future "
                            "policy decisions will depend on incoming economic "
                            "data."
                        ),
                    }
                ],
            },
            "outputs": {
                "expected_direction": "neutral",
                "expected_themes": [
                    "limited market movement",
                    "neutral monetary policy communication",
                ],
                "expected_relationships": [
                    (
                        "The supplied evidence does not indicate a major "
                        "risk-on or risk-off event."
                    )
                ],
                "must_not_claim": [
                    "the Federal Reserve caused a major market selloff",
                    "the Federal Reserve announced an interest-rate cut",
                    "the Federal Reserve announced an interest-rate increase",
                ],
                "evidence_sufficient": True,
                "should_research_more": False,
            },
        },

        # ------------------------------------------------------------
        # CASE 4
        #
        # Apple drops heavily, but supplied news does not explain it.
        #
        # Important test:
        # the agent must admit that evidence is insufficient.
        # ------------------------------------------------------------
        {
            "inputs": {
                "symbols": [
                    "AAPL",
                ],
                "market_data": [
                    {
                        "symbol": "AAPL",
                        "change_pct": -5.0,
                    }
                ],
                "news": [
                    {
                        "title": "Apple announces new product",
                        "summary": (
                            "Apple announced a new product. No major negative "
                            "company-specific developments were included in "
                            "the supplied information."
                        ),
                    }
                ],
            },
            "outputs": {
                "expected_direction": "company_specific_selloff",
                "expected_themes": [
                    "large Apple price decline",
                    "insufficient evidence explaining the move",
                ],
                "expected_relationships": [
                    (
                        "The available news does not adequately explain "
                        "Apple's large decline."
                    )
                ],
                "must_not_claim": [
                    "Apple fell because investors disliked the new product",
                    "the product announcement caused Apple's decline",
                ],
                "evidence_sufficient": False,
                "should_research_more": True,
            },
        },

        # ------------------------------------------------------------
        # CASE 5
        #
        # Positive earnings but strongly negative price action.
        #
        # Tests contradiction detection.
        # ------------------------------------------------------------
        {
            "inputs": {
                "symbols": [
                    "NVDA",
                ],
                "market_data": [
                    {
                        "symbol": "NVDA",
                        "change_pct": -6.2,
                    }
                ],
                "news": [
                    {
                        "title": "Nvidia beats earnings expectations",
                        "summary": (
                            "Nvidia reported revenue and earnings above "
                            "analyst expectations."
                        ),
                    }
                ],
            },
            "outputs": {
                "expected_direction": "bearish_price_action",
                "expected_themes": [
                    "positive fundamentals",
                    "negative market reaction",
                ],
                "expected_relationships": [
                    (
                        "Nvidia's negative price action contradicts the "
                        "apparently positive earnings news."
                    )
                ],
                "must_not_claim": [
                    "Nvidia fell because it missed earnings expectations",
                    "Nvidia reported weak revenue",
                ],
                "evidence_sufficient": False,
                "should_research_more": True,
            },
        },

        # ------------------------------------------------------------
        # CASE 6
        #
        # Broad risk-on environment.
        #
        # Tests whether the agent can combine macro and equity evidence.
        # ------------------------------------------------------------
        {
            "inputs": {
                "symbols": [
                    "SPY",
                    "QQQ",
                    "IWM",
                ],
                "market_data": [
                    {
                        "symbol": "SPY",
                        "change_pct": 1.3,
                    },
                    {
                        "symbol": "QQQ",
                        "change_pct": 1.7,
                    },
                    {
                        "symbol": "IWM",
                        "change_pct": 2.0,
                    },
                ],
                "news": [
                    {
                        "title": "Inflation data comes in below expectations",
                        "summary": (
                            "Latest inflation figures were softer than "
                            "economists expected."
                        ),
                    },
                    {
                        "title": (
                            "Treasury yields fall following inflation release"
                        ),
                        "summary": (
                            "US Treasury yields moved lower as investors "
                            "reassessed expectations for future interest rates."
                        ),
                    },
                ],
            },
            "outputs": {
                "expected_direction": "risk_on",
                "expected_themes": [
                    "lower inflation",
                    "falling treasury yields",
                    "broad equity strength",
                ],
                "expected_relationships": [
                    (
                        "Softer inflation and lower yields are consistent "
                        "with stronger equity markets."
                    ),
                    (
                        "Strength across SPY, QQQ and IWM suggests the rally "
                        "is relatively broad."
                    ),
                ],
                "must_not_claim": [
                    "markets sold off because inflation accelerated",
                ],
                "evidence_sufficient": True,
                "should_research_more": False,
            },
        },

        # ------------------------------------------------------------
        # CASE 7
        #
        # Conflicting signals.
        #
        # Tests nuanced reasoning instead of forcing bullish/bearish.
        # ------------------------------------------------------------
        {
            "inputs": {
                "symbols": [
                    "SPY",
                    "QQQ",
                    "XLE",
                ],
                "market_data": [
                    {
                        "symbol": "SPY",
                        "change_pct": -0.2,
                    },
                    {
                        "symbol": "QQQ",
                        "change_pct": -1.3,
                    },
                    {
                        "symbol": "XLE",
                        "change_pct": 2.2,
                    },
                ],
                "news": [
                    {
                        "title": "Treasury yields move higher",
                        "summary": (
                            "US Treasury yields increased during the session."
                        ),
                    },
                    {
                        "title": "Oil prices rise sharply",
                        "summary": (
                            "Crude oil prices increased following concerns "
                            "about supply."
                        ),
                    },
                ],
            },
            "outputs": {
                "expected_direction": "mixed",
                "expected_themes": [
                    "technology weakness",
                    "higher treasury yields",
                    "energy strength",
                    "higher oil prices",
                ],
                "expected_relationships": [
                    (
                        "Technology weakness is consistent with higher yields."
                    ),
                    (
                        "Energy strength is consistent with higher oil prices."
                    ),
                ],
                "must_not_claim": [
                    "all sectors moved in the same direction",
                    "the entire market experienced a strong selloff",
                ],
                "evidence_sufficient": True,
                "should_research_more": False,
            },
        },

        # ------------------------------------------------------------
        # CASE 8
        #
        # Strong market move with essentially irrelevant news.
        #
        # Tests whether agent requests more evidence.
        # ------------------------------------------------------------
        {
            "inputs": {
                "symbols": [
                    "SPY",
                    "QQQ",
                ],
                "market_data": [
                    {
                        "symbol": "SPY",
                        "change_pct": -2.4,
                    },
                    {
                        "symbol": "QQQ",
                        "change_pct": -3.1,
                    },
                ],
                "news": [
                    {
                        "title": "Major technology conference begins",
                        "summary": (
                            "Several technology companies are presenting "
                            "new products at an industry conference."
                        ),
                    }
                ],
            },
            "outputs": {
                "expected_direction": "risk_off",
                "expected_themes": [
                    "broad equity weakness",
                    "technology underperformance",
                    "missing explanation for the selloff",
                ],
                "expected_relationships": [
                    (
                        "The supplied news is insufficient to establish "
                        "the cause of the market decline."
                    )
                ],
                "must_not_claim": [
                    "the technology conference caused the market selloff",
                ],
                "evidence_sufficient": False,
                "should_research_more": True,
            },
        },
    ]


def create_dataset() -> None:
    """
    Create and populate the LangSmith dataset.

    The function is intentionally idempotent:
    if the dataset already exists, it does not insert all examples again.
    """

    client = Client()

    print(f"Checking LangSmith dataset: {DATASET_NAME}")

    if client.has_dataset(dataset_name=DATASET_NAME):
        print(
            f"Dataset '{DATASET_NAME}' already exists. "
            "Nothing was created."
        )
        return

    print("Creating dataset...")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )

    examples = build_examples()

    print(
        f"Creating {len(examples)} evaluation examples..."
    )

    client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
    )

    print()
    print("Dataset successfully created.")
    print(f"Name: {DATASET_NAME}")
    print(f"ID:   {dataset.id}")
    print(f"Examples: {len(examples)}")


if __name__ == "__main__":
    create_dataset()