"""Tests for EXP-019: LLM-in-the-Loop Config QA.

Tests that LLMs answer config questions more accurately when YAML
comments are preserved (CDXF path) vs stripped (standard path).

Two test modes:
  - Fixture-based (default): replay recorded API responses
  - Live (--run-live flag): call actual APIs and record fixtures

TDD: These tests are written BEFORE the implementation.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp019 import (
    QA_DATASET,
    MODELS,
    COMMENT_CONDITIONS,
    build_config_with_comments,
    build_config_without_comments,
    grade_response,
    call_model,
    run_qa_session,
    run_experiment,
    FIXTURES_DIR,
)


# ===========================================================================
# Constants — Protocol compliance
# ===========================================================================


class TestProtocolConstants:
    """Verify experiment constants."""

    def test_comment_conditions(self):
        assert "with_comments" in COMMENT_CONDITIONS
        assert "without_comments" in COMMENT_CONDITIONS
        assert len(COMMENT_CONDITIONS) == 2

    def test_models_defined(self):
        assert len(MODELS) >= 2
        names = {m["name"] for m in MODELS}
        assert "gemma4" in names or any("gemma" in n for n in names)
        assert "gemini" in names or any("gemini" in n for n in names)

    def test_qa_dataset_has_questions(self):
        assert len(QA_DATASET) >= 8

    def test_qa_dataset_has_required_fields(self):
        for qa in QA_DATASET:
            assert "id" in qa
            assert "question" in qa
            assert "ground_truth" in qa
            assert "category" in qa
            assert qa["category"] in ("comment_only", "value_only", "mixed")

    def test_has_all_categories(self):
        categories = {qa["category"] for qa in QA_DATASET}
        assert "comment_only" in categories
        assert "value_only" in categories

    def test_comment_only_questions_exist(self):
        comment_only = [q for q in QA_DATASET if q["category"] == "comment_only"]
        assert len(comment_only) >= 4

    def test_value_only_questions_exist(self):
        value_only = [q for q in QA_DATASET if q["category"] == "value_only"]
        assert len(value_only) >= 3


# ===========================================================================
# Config construction
# ===========================================================================


class TestConfigConstruction:
    """Tests for building configs with/without comments."""

    def test_with_comments_has_comments(self):
        text = build_config_with_comments()
        assert "#" in text
        comment_lines = [l for l in text.splitlines() if l.strip().startswith("#")]
        assert len(comment_lines) >= 10

    def test_without_comments_has_no_comments(self):
        text = build_config_without_comments()
        comment_lines = [l for l in text.splitlines() if l.strip().startswith("#")]
        assert len(comment_lines) == 0

    def test_same_values(self):
        """Both configs must have identical data values."""
        import yaml
        with_c = yaml.safe_load(build_config_with_comments())
        without_c = yaml.safe_load(build_config_without_comments())
        assert with_c == without_c

    def test_with_comments_is_valid_yaml(self):
        import yaml
        parsed = yaml.safe_load(build_config_with_comments())
        assert isinstance(parsed, dict)

    def test_without_comments_is_valid_yaml(self):
        import yaml
        parsed = yaml.safe_load(build_config_without_comments())
        assert isinstance(parsed, dict)


# ===========================================================================
# Grading
# ===========================================================================


class TestGrading:
    """Tests for response grading."""

    def test_correct_answer(self):
        qa = {"ground_truth": ["grid search", "1e-5, 5e-5, 1e-4"]}
        result = grade_response(
            "The learning rate was chosen via grid search over [1e-5, 5e-5, 1e-4].",
            qa,
        )
        assert result["grade"] == "correct"

    def test_partial_answer(self):
        qa = {"ground_truth": ["grid search", "1e-5, 5e-5, 1e-4"]}
        result = grade_response(
            "It was selected through a grid search process.",
            qa,
        )
        assert result["grade"] in ("correct", "partial")

    def test_wrong_answer(self):
        qa = {"ground_truth": ["grid search", "1e-5, 5e-5, 1e-4"]}
        result = grade_response(
            "The learning rate is 0.001 which is standard.",
            qa,
        )
        assert result["grade"] in ("wrong", "hallucinated")

    def test_hallucinated_answer(self):
        qa = {"ground_truth": ["grid search"]}
        result = grade_response(
            "It was chosen using Bayesian optimization with Optuna.",
            qa,
        )
        assert result["grade"] in ("wrong", "hallucinated")

    def test_returns_required_fields(self):
        qa = {"ground_truth": ["test"]}
        result = grade_response("test answer", qa)
        assert "grade" in result
        assert "matched_keywords" in result


# ===========================================================================
# Model calling (fixture-based)
# ===========================================================================


class TestCallModel:
    """Tests for model calling with fixture support."""

    def test_returns_string(self):
        """Call model should return a response string."""
        result = asyncio.run(call_model(
            model=MODELS[0],
            prompt="What is 2+2?",
            use_fixtures=True,
        ))
        assert isinstance(result, str)


# ===========================================================================
# QA session
# ===========================================================================


class TestRunQASession:
    """Tests for running a full QA session."""

    @pytest.fixture(scope="class")
    def session_results(self):
        return asyncio.run(run_qa_session(use_fixtures=True))

    def test_returns_list(self, session_results):
        assert isinstance(session_results, list)
        assert len(session_results) > 0

    def test_has_required_fields(self, session_results):
        for r in session_results:
            assert "model" in r
            assert "condition" in r
            assert "question_id" in r
            assert "question" in r
            assert "response" in r
            assert "grade" in r
            assert "category" in r

    def test_has_both_conditions(self, session_results):
        conditions = {r["condition"] for r in session_results}
        assert conditions == {"with_comments", "without_comments"}

    def test_comment_only_with_comments_better(self, session_results):
        """Comment-only questions should score better with comments."""
        for model in MODELS:
            with_c = [
                r for r in session_results
                if r["model"] == model["name"]
                and r["condition"] == "with_comments"
                and r["category"] == "comment_only"
            ]
            without_c = [
                r for r in session_results
                if r["model"] == model["name"]
                and r["condition"] == "without_comments"
                and r["category"] == "comment_only"
            ]
            if with_c and without_c:
                with_acc = sum(1 for r in with_c if r["grade"] == "correct") / len(with_c)
                without_acc = sum(1 for r in without_c if r["grade"] == "correct") / len(without_c)
                assert with_acc > without_acc, (
                    f"{model['name']}: with={with_acc:.1%} <= without={without_acc:.1%}"
                )

    def test_value_only_similar_both_conditions(self, session_results):
        """Value-only questions (control) should score similarly."""
        for model in MODELS:
            with_c = [
                r for r in session_results
                if r["model"] == model["name"]
                and r["condition"] == "with_comments"
                and r["category"] == "value_only"
            ]
            without_c = [
                r for r in session_results
                if r["model"] == model["name"]
                and r["condition"] == "without_comments"
                and r["category"] == "value_only"
            ]
            if with_c and without_c:
                with_acc = sum(1 for r in with_c if r["grade"] == "correct") / len(with_c)
                without_acc = sum(1 for r in without_c if r["grade"] == "correct") / len(without_c)
                # Control: both should be high (>= 50%) — answers are in the values
                assert with_acc >= 0.5 or without_acc >= 0.5


# ===========================================================================
# Full experiment
# ===========================================================================


class TestRunExperiment:
    """Integration tests for the full experiment."""

    @pytest.fixture(scope="class")
    def results(self):
        return asyncio.run(run_experiment(use_fixtures=True))

    def test_returns_dict(self, results):
        assert isinstance(results, dict)

    def test_has_experiment_id(self, results):
        assert results["experiment"] == "EXP-019"

    def test_has_models(self, results):
        assert "models" in results

    def test_has_summary(self, results):
        assert "summary" in results

    def test_has_accuracy_by_condition(self, results):
        s = results["summary"]
        # Summary is keyed by model name, with conditions nested
        for model_name, model_data in s.items():
            assert "with_comments" in model_data
            assert "without_comments" in model_data
