"""
Run LangSmith evaluations for the Financial Intelligent Agent.

The runner evaluates the analysis node directly with synthetic inputs from
evals/create_dataset.py. It does not call Alpha Vantage, Tavily, Postgres, or
the full LangGraph workflow.

Usage:
    uv run python evals/create_dataset.py
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --single-model gpt-5
    uv run python evals/run_eval.py --models gpt-5-mini gpt-5
"""

import argparse
import asyncio
import os
import sys
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langsmith import evaluate

from app.nodes import analysis as analysis_node
from evals.create_dataset import DATASET_NAME
from evals.evaluators import ALL_EVALUATORS

load_dotenv()


DEFAULT_MODELS = ["gpt-5-mini", "gpt-5"]
SYNTHETIC_TRADING_DAY = "2026-08-14"


def _enrich_inputs(inputs: dict) -> dict:
    """Add stable metadata needed by the production prompt's citation rules."""
    enriched = deepcopy(inputs)

    for item in enriched.get("market_data", []):
        item.setdefault("date", SYNTHETIC_TRADING_DAY)
        change_pct = item.get("change_pct", 0)
        item.setdefault("close", 100.0 + float(change_pct))
        item.setdefault("previous_close", 100.0)

    for index, article in enumerate(enriched.get("news", []), 1):
        article.setdefault("published_date", SYNTHETIC_TRADING_DAY)
        article.setdefault(
            "url",
            f"https://example.test/financial-news/{index}",
        )
        if "summary" in article and "content" not in article:
            article["content"] = article["summary"]

    enriched.setdefault("errors", [])
    return enriched


def _run_analysis(inputs: dict) -> str:
    state = _enrich_inputs(inputs)
    result = asyncio.run(analysis_node.analyze_market(state))
    return result.get("analysis", "")


def make_run_function(model: str):
    def run_agent(inputs: dict) -> dict:
        os.environ["ANALYSIS_MODEL"] = model
        analysis_node._analysis_llm.cache_clear()
        return {
            "output": _run_analysis(inputs),
        }

    return run_agent


def build_experiment_metadata(model: str) -> dict:
    return {
        "models": [f"openai:{model}"],
        "prompts": ["market-analysis-structured-brief"],
        "tools": [],
        "notes": (
            "Evaluates app.nodes.analysis.analyze_market directly with synthetic "
            "market/news inputs. Live data clients and Postgres are not used."
        ),
    }


def run_single_model(model: str, experiment_prefix: str):
    print(f"\nModel: {model}")
    print(f"Experiment prefix: {experiment_prefix}")

    return evaluate(
        make_run_function(model),
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=experiment_prefix,
        description=(
            "Financial Intelligent Agent market-analysis evaluation using "
            f"{model} as the analysis model."
        ),
        metadata=build_experiment_metadata(model),
        max_concurrency=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run market-analysis evals.")
    parser.add_argument(
        "--experiment-prefix",
        default="financial-intelligence-agent",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to compare. Default: {' '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--single-model",
        default=None,
        help="Run one model only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = [args.single_model] if args.single_model else args.models or DEFAULT_MODELS

    print(f"Dataset: {DATASET_NAME}")
    print(f"Evaluators: {', '.join(evaluator.__name__ for evaluator in ALL_EVALUATORS)}")
    print(f"Models: {', '.join(models)}")

    for model in models:
        prefix = f"{args.experiment_prefix}-{model}"
        run_single_model(model, prefix)

    print("\nEvaluation run submitted. View results in LangSmith.")


if __name__ == "__main__":
    main()
