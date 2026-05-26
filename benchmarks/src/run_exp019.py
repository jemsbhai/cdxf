"""
EXP-019: LLM-in-the-Loop Config QA — Does Metadata Improve Accuracy?

Tests whether LLMs answer configuration questions more accurately when
YAML comments are preserved (CDXF path) vs stripped (standard path).

Two models:
  - gemma4:31b-cloud (ollama, local)
  - gemini-2.5-flash (Google API)

All API responses are recorded as fixtures for reproducible test replay.

Usage:
    python benchmarks/src/run_exp019.py          # Live calls + record
    python benchmarks/src/run_exp019.py --replay  # Replay from fixtures
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from benchmarks.src.config_corpus import YAML_CONFIGS


# ===========================================================================
# Constants
# ===========================================================================

COMMENT_CONDITIONS = ["with_comments", "without_comments"]

FIXTURES_DIR = Path("benchmarks/results/exp_019/fixtures")

MODELS = [
    {
        "name": "gemma4",
        "backend": "ollama",
        "model_id": "gemma4:31b-cloud",
    },
    {
        "name": "gemini-flash",
        "backend": "gemini",
        "model_id": "gemini-2.5-flash",
    },
]


# ===========================================================================
# QA Dataset
# ===========================================================================

QA_DATASET = [
    # --- comment_only: answers exist ONLY in comments ---
    {
        "id": "q01",
        "question": "Why was this learning rate chosen? What alternatives were considered?",
        "ground_truth": ["grid search", "1e-5", "5e-5", "1e-4"],
        "category": "comment_only",
    },
    {
        "id": "q02",
        "question": "What is the effective batch size and how is it calculated?",
        "ground_truth": ["128", "effective batch", "accumulation"],
        "category": "comment_only",
    },
    {
        "id": "q03",
        "question": "Who owns this configuration?",
        "ground_truth": ["ML Platform Team"],
        "category": "comment_only",
    },
    {
        "id": "q04",
        "question": "When was this configuration created?",
        "ground_truth": ["2026-01-15"],
        "category": "comment_only",
    },
    {
        "id": "q05",
        "question": "What determines the head architecture in this model?",
        "ground_truth": ["task type", "task_type"],
        "category": "comment_only",
    },
    {
        "id": "q06",
        "question": "Why is the seed parameter included?",
        "ground_truth": ["reproducibility"],
        "category": "comment_only",
    },
    # --- value_only: answers are in values, not comments (control) ---
    {
        "id": "q07",
        "question": "What is the learning rate value?",
        "ground_truth": ["2.0e-5", "2e-5", "0.00002"],
        "category": "value_only",
    },
    {
        "id": "q08",
        "question": "How many training epochs are configured?",
        "ground_truth": ["3"],
        "category": "value_only",
    },
    {
        "id": "q09",
        "question": "What dataset is being used for training?",
        "ground_truth": ["tatsu-lab/alpaca", "alpaca"],
        "category": "value_only",
    },
    {
        "id": "q10",
        "question": "What is the minimum acceptable accuracy?",
        "ground_truth": ["0.45"],
        "category": "value_only",
    },
    # --- mixed: benefits from both ---
    {
        "id": "q11",
        "question": "What serving engine is used and what is its configuration purpose?",
        "ground_truth": ["vllm", "serving"],
        "category": "mixed",
    },
    {
        "id": "q12",
        "question": "Describe the evaluation strategy including when evaluations run.",
        "ground_truth": ["epoch", "mmlu", "hellaswag"],
        "category": "mixed",
    },
]


# ===========================================================================
# Config construction
# ===========================================================================


def build_config_with_comments() -> str:
    """Return the large config with all comments (CDXF path)."""
    return YAML_CONFIGS["large"]["text"]


def build_config_without_comments() -> str:
    """Return the same config with comments stripped (standard path)."""
    text = YAML_CONFIGS["large"]["text"]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Remove inline comments
        if " #" in line:
            hash_pos = line.find(" #")
            before = line[:hash_pos]
            if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                line = before.rstrip()
        if line.strip():  # Skip blank lines left by removed comments
            lines.append(line)
    return "\n".join(lines) + "\n"


# ===========================================================================
# Grading
# ===========================================================================


def grade_response(response: str, qa: dict) -> dict:
    """Grade an LLM response against ground truth keywords.

    Returns:
        {"grade": "correct"|"partial"|"wrong"|"hallucinated",
         "matched_keywords": [...]}
    """
    ground_truth = qa["ground_truth"]
    response_lower = response.lower()

    matched = []
    for keyword in ground_truth:
        if keyword.lower() in response_lower:
            matched.append(keyword)

    match_ratio = len(matched) / len(ground_truth) if ground_truth else 0

    if match_ratio >= 0.5:
        grade = "correct"
    elif match_ratio > 0:
        grade = "partial"
    else:
        # Check for hallucination indicators
        # If the response is confident but wrong, it's hallucinated
        confident_phrases = [
            "the answer is", "it is", "this is set to",
            "was chosen", "was selected",
        ]
        is_confident = any(p in response_lower for p in confident_phrases)
        grade = "hallucinated" if is_confident else "wrong"

    return {
        "grade": grade,
        "matched_keywords": matched,
        "match_ratio": match_ratio,
    }


# ===========================================================================
# Model calling with fixture support
# ===========================================================================


def _fixture_key(model_name: str, condition: str, question_id: str) -> str:
    """Generate a fixture filename."""
    return f"{model_name}_{condition}_{question_id}.json"


def _load_fixture(key: str) -> str | None:
    """Load a recorded fixture if it exists."""
    path = FIXTURES_DIR / key
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("response", "")
    return None


def _save_fixture(key: str, model_name: str, condition: str,
                  question_id: str, prompt: str, response: str):
    """Save a response as a fixture."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model_name,
        "condition": condition,
        "question_id": question_id,
        "prompt": prompt,
        "response": response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = FIXTURES_DIR / key
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


async def call_model(
    model: dict,
    prompt: str,
    use_fixtures: bool = False,
    fixture_key: str | None = None,
) -> str:
    """Call an LLM model, with optional fixture recording/replay.

    Args:
        model: Model config dict with name, backend, model_id.
        prompt: The prompt to send.
        use_fixtures: If True, try loading from fixture first.
        fixture_key: Key for fixture storage.

    Returns:
        The model's response text.
    """
    # Try fixture first
    if use_fixtures:
        if fixture_key:
            cached = _load_fixture(fixture_key)
            if cached is not None:
                return cached
        # In fixture mode, never make live calls
        return "[NO FIXTURE RECORDED]"

    # Live call
    if model["backend"] == "ollama":
        response = await _call_ollama(model["model_id"], prompt)
    elif model["backend"] == "gemini":
        response = await _call_gemini(model["model_id"], prompt)
    else:
        raise ValueError(f"Unknown backend: {model['backend']}")

    # Record fixture
    if fixture_key:
        _save_fixture(fixture_key, model["name"], "", "", prompt, response)

    return response


async def _call_ollama(model_id: str, prompt: str) -> str:
    """Call ollama API."""
    import requests
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


async def _call_gemini(model_id: str, prompt: str) -> str:
    """Call Gemini API."""
    try:
        import google.generativeai as genai
        import os

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "ERROR: No GOOGLE_API_KEY or GEMINI_API_KEY in environment"

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0),
        )
        return response.text
    except Exception as e:
        return f"ERROR: {e}"


# ===========================================================================
# QA session
# ===========================================================================


def _build_prompt(config_text: str, question: str) -> str:
    """Build a QA prompt."""
    return (
        "You are an ML engineer reviewing a YAML configuration file. "
        "Answer the question based ONLY on the information present in "
        "the configuration below. Be specific and concise.\n\n"
        "--- CONFIGURATION ---\n"
        f"{config_text}\n"
        "--- END CONFIGURATION ---\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


async def run_qa_session(
    use_fixtures: bool = False,
    models: list[dict] | None = None,
) -> list[dict]:
    """Run QA across all models × conditions × questions.

    Returns a list of result dicts.
    """
    if models is None:
        models = MODELS

    config_with = build_config_with_comments()
    config_without = build_config_without_comments()

    results = []

    for model in models:
        for condition in COMMENT_CONDITIONS:
            config_text = (
                config_with if condition == "with_comments"
                else config_without
            )

            for qa in QA_DATASET:
                prompt = _build_prompt(config_text, qa["question"])
                fkey = _fixture_key(model["name"], condition, qa["id"])

                response = await call_model(
                    model=model,
                    prompt=prompt,
                    use_fixtures=use_fixtures,
                    fixture_key=fkey,
                )

                grading = grade_response(response, qa)

                # Save fixture with full metadata
                if not use_fixtures or not _load_fixture(fkey):
                    _save_fixture(
                        fkey, model["name"], condition,
                        qa["id"], prompt, response,
                    )

                results.append({
                    "model": model["name"],
                    "condition": condition,
                    "question_id": qa["id"],
                    "question": qa["question"],
                    "category": qa["category"],
                    "response": response,
                    "grade": grading["grade"],
                    "matched_keywords": grading["matched_keywords"],
                    "match_ratio": grading["match_ratio"],
                })

    return results


# ===========================================================================
# Full experiment
# ===========================================================================


async def run_experiment(
    output_dir: Path | str | None = None,
    use_fixtures: bool = False,
) -> dict:
    """Run the full EXP-019 experiment."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_019")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-019: LLM-in-the-Loop Config QA")
    print("=" * 70)

    mode_label = "FIXTURE REPLAY" if use_fixtures else "LIVE API CALLS"
    print(f"Mode: {mode_label}\n")

    results = await run_qa_session(use_fixtures=use_fixtures)

    # Compute summary
    summary = {}
    for model in MODELS:
        model_name = model["name"]
        summary[model_name] = {}

        for condition in COMMENT_CONDITIONS:
            by_category = {}
            for cat in ("comment_only", "value_only", "mixed"):
                rows = [
                    r for r in results
                    if r["model"] == model_name
                    and r["condition"] == condition
                    and r["category"] == cat
                ]
                if rows:
                    correct = sum(1 for r in rows if r["grade"] == "correct")
                    total = len(rows)
                    by_category[cat] = {
                        "correct": correct,
                        "total": total,
                        "accuracy": round(correct / total, 4) if total else 0,
                    }

            all_rows = [
                r for r in results
                if r["model"] == model_name
                and r["condition"] == condition
            ]
            overall_correct = sum(1 for r in all_rows if r["grade"] == "correct")
            overall_total = len(all_rows)

            summary[model_name][condition] = {
                "by_category": by_category,
                "overall_accuracy": round(
                    overall_correct / overall_total, 4
                ) if overall_total else 0,
                "overall_correct": overall_correct,
                "overall_total": overall_total,
            }

    # Print results
    for model_name, model_summary in summary.items():
        print(f"\n--- {model_name} ---")
        for condition, data in model_summary.items():
            print(f"  {condition}: {data['overall_accuracy']:.1%} overall "
                  f"({data['overall_correct']}/{data['overall_total']})")
            for cat, cat_data in data["by_category"].items():
                print(f"    {cat:15s}: {cat_data['accuracy']:.1%} "
                      f"({cat_data['correct']}/{cat_data['total']})")

    # Key metric: comment_only accuracy delta
    print("\n--- Key Metric: Comment-Only Accuracy Delta ---")
    for model in MODELS:
        mn = model["name"]
        with_c = summary[mn].get("with_comments", {}).get("by_category", {}).get("comment_only", {})
        without_c = summary[mn].get("without_comments", {}).get("by_category", {}).get("comment_only", {})
        if with_c and without_c:
            delta = with_c["accuracy"] - without_c["accuracy"]
            print(f"  {mn}: with={with_c['accuracy']:.1%}, "
                  f"without={without_c['accuracy']:.1%}, "
                  f"delta={delta:+.1%}")

    # Write outputs
    output = {
        "experiment": "EXP-019",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": [m["name"] for m in MODELS],
        "n_questions": len(QA_DATASET),
        "results": results,
        "summary": summary,
        "use_fixtures": use_fixtures,
    }

    json_path = output_dir / "exp_019_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults: {json_path}")

    # CSV
    csv_path = output_dir / "qa_results.csv"
    if results:
        fieldnames = ["model", "condition", "question_id", "category",
                      "grade", "match_ratio", "question"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    print(f"CSV: {csv_path}")

    print(f"\n{'=' * 70}")
    print("EXP-019 COMPLETE")
    print("=" * 70)

    return output


def _get_version(package: str) -> str:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return "unknown"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true",
                        help="Replay from fixtures instead of live API calls")
    args = parser.parse_args()
    asyncio.run(run_experiment(use_fixtures=args.replay))


if __name__ == "__main__":
    main()
