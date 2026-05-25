"""Tests for EXP-009: End-to-End Fine-Tuning Pipeline State Capture.

Tests pipeline construction, state capture methods, component extraction,
state diffing, and cross-language readability assessment.

TDD: These tests are written BEFORE the implementation.
"""

import json
import pickle
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp009 import (
    build_pipeline_state,
    capture_cdxf,
    capture_tar_gz,
    capture_mega_json,
    capture_pickle,
    extract_component,
    diff_states,
    assess_cross_language,
    measure_pipeline,
    CAPTURE_METHODS,
    PIPELINE_COMPONENTS,
)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

class TestBuildPipelineState:
    def test_returns_dict(self):
        state = build_pipeline_state("full")
        assert isinstance(state, dict)

    def test_full_has_8_components(self):
        state = build_pipeline_state("full")
        assert len(state["components"]) == 8

    def test_minimal_has_3_components(self):
        state = build_pipeline_state("minimal")
        assert len(state["components"]) == 3

    def test_component_required_fields(self):
        state = build_pipeline_state("full")
        for comp in state["components"]:
            assert "name" in comp
            assert "filename" in comp
            assert "format" in comp
            assert "text" in comp
            assert len(comp["text"]) > 0

    def test_full_has_all_pipeline_stages(self):
        """Full pipeline must include all 8 stages from protocol."""
        state = build_pipeline_state("full")
        names = {c["name"] for c in state["components"]}
        required = {
            "dataset_card", "preprocessing", "training_hparams",
            "adapter_config", "quantization_config", "serving_config",
            "eval_results", "deployment_manifest",
        }
        assert required.issubset(names), f"Missing: {required - names}"

    def test_minimal_subset_of_full(self):
        """Minimal components should be a subset of full."""
        minimal = build_pipeline_state("minimal")
        full = build_pipeline_state("full")
        min_names = {c["name"] for c in minimal["components"]}
        full_names = {c["name"] for c in full["components"]}
        assert min_names.issubset(full_names)

    def test_has_yaml_with_comments(self):
        state = build_pipeline_state("full")
        yaml_comps = [c for c in state["components"] if c["format"] == "yaml"]
        assert len(yaml_comps) >= 1
        has_comments = any("#" in c["text"] for c in yaml_comps)
        assert has_comments

    def test_has_toml_component(self):
        state = build_pipeline_state("full")
        toml_comps = [c for c in state["components"] if c["format"] == "toml"]
        assert len(toml_comps) >= 1

    def test_has_multiple_formats(self):
        state = build_pipeline_state("full")
        formats = {c["format"] for c in state["components"]}
        assert len(formats) >= 3

    def test_invalid_complexity_raises(self):
        with pytest.raises(ValueError):
            build_pipeline_state("invalid")

    def test_variant_returns_modified_state(self):
        """build_pipeline_state with variant should change one HP."""
        base = build_pipeline_state("full")
        variant = build_pipeline_state("full", variant="lr_change")
        # Both should have same number of components
        assert len(base["components"]) == len(variant["components"])
        # At least one component should differ
        base_texts = {c["name"]: c["text"] for c in base["components"]}
        var_texts = {c["name"]: c["text"] for c in variant["components"]}
        diffs = [n for n in base_texts if base_texts[n] != var_texts[n]]
        assert len(diffs) >= 1, "Variant should differ from base"


# ---------------------------------------------------------------------------
# State capture methods
# ---------------------------------------------------------------------------

class TestCaptureCdxf:
    def test_returns_bytes(self):
        state = build_pipeline_state("minimal")
        result = capture_cdxf(state["components"])
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_full_pipeline(self):
        state = build_pipeline_state("full")
        result = capture_cdxf(state["components"])
        assert len(result) > 0


class TestCaptureTargz:
    def test_returns_bytes(self):
        state = build_pipeline_state("minimal")
        result = capture_tar_gz(state["components"])
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestCaptureMegaJson:
    def test_returns_bytes(self):
        state = build_pipeline_state("minimal")
        result = capture_mega_json(state["components"])
        assert isinstance(result, bytes)

    def test_is_valid_json(self):
        state = build_pipeline_state("minimal")
        result = capture_mega_json(state["components"])
        parsed = json.loads(result.decode("utf-8"))
        assert isinstance(parsed, dict)


class TestCapturePickle:
    def test_returns_bytes(self):
        state = build_pipeline_state("minimal")
        result = capture_pickle(state["components"])
        assert isinstance(result, bytes)

    def test_is_valid_pickle(self):
        state = build_pipeline_state("minimal")
        result = capture_pickle(state["components"])
        loaded = pickle.loads(result)
        assert isinstance(loaded, dict)


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------

class TestExtractComponent:
    def test_extract_from_cdxf(self):
        state = build_pipeline_state("full")
        bundle = capture_cdxf(state["components"])
        result = extract_component(bundle, "cdxf", "adapter_config.json")
        assert result["success"]
        assert len(result["text"]) > 0

    def test_extract_from_tar_gz(self):
        state = build_pipeline_state("full")
        bundle = capture_tar_gz(state["components"])
        result = extract_component(bundle, "tar_gz", "adapter_config.json")
        assert result["success"]

    def test_extract_from_mega_json(self):
        state = build_pipeline_state("full")
        bundle = capture_mega_json(state["components"])
        result = extract_component(bundle, "mega_json", "adapter_config.json")
        assert result["success"]

    def test_extract_from_pickle(self):
        state = build_pipeline_state("full")
        bundle = capture_pickle(state["components"])
        result = extract_component(bundle, "pickle", "adapter_config.json")
        assert result["success"]

    def test_nonexistent_component(self):
        state = build_pipeline_state("minimal")
        bundle = capture_mega_json(state["components"])
        result = extract_component(bundle, "mega_json", "nonexistent.txt")
        assert not result["success"]


# ---------------------------------------------------------------------------
# State diffing
# ---------------------------------------------------------------------------

class TestDiffStates:
    def test_identical_states_no_diff(self):
        state = build_pipeline_state("full")
        bundle1 = capture_cdxf(state["components"])
        bundle2 = capture_cdxf(state["components"])
        result = diff_states(bundle1, bundle2, "cdxf")
        assert result["success"]
        assert result["n_diffs"] == 0

    def test_different_states_has_diff(self):
        base = build_pipeline_state("full")
        variant = build_pipeline_state("full", variant="lr_change")
        b1 = capture_cdxf(base["components"])
        b2 = capture_cdxf(variant["components"])
        result = diff_states(b1, b2, "cdxf")
        assert result["success"]
        assert result["n_diffs"] >= 1

    def test_diff_mega_json(self):
        base = build_pipeline_state("full")
        variant = build_pipeline_state("full", variant="lr_change")
        b1 = capture_mega_json(base["components"])
        b2 = capture_mega_json(variant["components"])
        result = diff_states(b1, b2, "mega_json")
        assert result["success"]
        assert result["n_diffs"] >= 1

    def test_diff_returns_changed_components(self):
        base = build_pipeline_state("full")
        variant = build_pipeline_state("full", variant="lr_change")
        b1 = capture_cdxf(base["components"])
        b2 = capture_cdxf(variant["components"])
        result = diff_states(b1, b2, "cdxf")
        assert "changed_components" in result
        assert len(result["changed_components"]) >= 1

    def test_diff_result_has_required_fields(self):
        base = build_pipeline_state("minimal")
        b1 = capture_cdxf(base["components"])
        b2 = capture_cdxf(base["components"])
        result = diff_states(b1, b2, "cdxf")
        assert "success" in result
        assert "n_diffs" in result
        assert "changed_components" in result


# ---------------------------------------------------------------------------
# Cross-language readability
# ---------------------------------------------------------------------------

class TestAssessCrossLanguage:
    def test_returns_dict(self):
        result = assess_cross_language()
        assert isinstance(result, dict)

    def test_has_all_methods(self):
        result = assess_cross_language()
        for method in CAPTURE_METHODS:
            assert method in result

    def test_cdxf_is_cross_language(self):
        """CDXF (CBOR) is readable by any CBOR library."""
        result = assess_cross_language()
        assert result["cdxf"]["cross_language"] is True

    def test_pickle_is_not_cross_language(self):
        """Pickle is Python-only."""
        result = assess_cross_language()
        assert result["pickle"]["cross_language"] is False

    def test_json_is_cross_language(self):
        result = assess_cross_language()
        assert result["mega_json"]["cross_language"] is True


# ---------------------------------------------------------------------------
# Full measurement
# ---------------------------------------------------------------------------

class TestMeasurePipeline:
    def test_returns_result_dict(self):
        result = measure_pipeline("minimal")
        assert isinstance(result, dict)

    def test_result_has_all_methods(self):
        result = measure_pipeline("minimal")
        for method in CAPTURE_METHODS:
            assert method in result

    def test_result_has_complexity(self):
        result = measure_pipeline("full")
        assert result["complexity"] == "full"
        assert result["n_components"] == 8

    def test_each_method_has_size(self):
        result = measure_pipeline("minimal")
        for method in CAPTURE_METHODS:
            assert "capture_size" in result[method]
            assert result[method]["capture_size"] > 0

    def test_cdxf_has_metadata_count(self):
        result = measure_pipeline("full")
        assert "metadata_preserved" in result["cdxf"]

    def test_diff_results_included(self):
        result = measure_pipeline("full")
        assert "diff_detected" in result


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_capture_methods_has_four(self):
        assert len(CAPTURE_METHODS) == 4

    def test_pipeline_components_has_eight(self):
        assert len(PIPELINE_COMPONENTS) == 8
