"""Tests for EXP-015: LangGraph Stateful Agent — Config Handoff Fidelity.

Build a real LangGraph StateGraph where ML agents read/modify configs
passed via state. LangGraph serializes state as JSON (default).
Compare JSON state vs CDXF-enhanced state for metadata survival.

TDD: These tests are written BEFORE the implementation.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp015 import (
    GRAPH_CONFIGS,
    STATE_MODES,
    build_initial_config,
    build_langgraph_pipeline,
    count_config_metadata,
    extract_config_text,
    run_experiment,
    run_graph,
    run_graph_with_checkpoints,
    serialize_config_for_state,
)


# ===========================================================================
# Constants — Protocol compliance
# ===========================================================================


class TestProtocolConstants:
    """Verify experiment constants."""

    def test_state_modes(self):
        """Two modes: json (LangGraph default) vs cdxf-enhanced."""
        assert "json_default" in STATE_MODES
        assert "cdxf_enhanced" in STATE_MODES
        assert len(STATE_MODES) == 2

    def test_graph_configs(self):
        """Multiple graph topologies to test."""
        assert len(GRAPH_CONFIGS) >= 2


# ===========================================================================
# Config construction
# ===========================================================================


class TestBuildInitialConfig:
    """Tests for the initial ML config with rich metadata."""

    def test_returns_text_and_format(self):
        text, fmt = build_initial_config()
        assert isinstance(text, str)
        assert fmt == "yaml"
        assert len(text) > 0

    def test_has_comments(self):
        text, fmt = build_initial_config()
        meta = count_config_metadata(text, fmt)
        assert meta["comments"] >= 10, (
            "Config needs substantial comments for meaningful test"
        )

    def test_has_training_params(self):
        text, _ = build_initial_config()
        assert "learning_rate" in text
        assert "batch_size" in text or "per_device" in text


# ===========================================================================
# State serialization
# ===========================================================================


class TestSerializeConfigForState:
    """Tests for serializing config into LangGraph state values."""

    def test_json_default_returns_dict(self):
        text, fmt = build_initial_config()
        result = serialize_config_for_state(text, fmt, "json_default")
        assert isinstance(result, (dict, str))

    def test_cdxf_enhanced_returns_string(self):
        """CDXF stores config as base64 CDXF binary in state."""
        text, fmt = build_initial_config()
        result = serialize_config_for_state(text, fmt, "cdxf_enhanced")
        assert isinstance(result, str)

    def test_json_default_is_json_serializable(self):
        text, fmt = build_initial_config()
        result = serialize_config_for_state(text, fmt, "json_default")
        # Must survive JSON round-trip (LangGraph requirement)
        json_str = json.dumps(result)
        restored = json.loads(json_str)
        assert restored == result

    def test_cdxf_enhanced_is_json_serializable(self):
        text, fmt = build_initial_config()
        result = serialize_config_for_state(text, fmt, "cdxf_enhanced")
        json_str = json.dumps(result)
        restored = json.loads(json_str)
        assert restored == result


class TestExtractConfigText:
    """Tests for extracting config text back from state values."""

    def test_json_default_round_trip(self):
        text, fmt = build_initial_config()
        serialized = serialize_config_for_state(text, fmt, "json_default")
        extracted = extract_config_text(serialized, fmt, "json_default")
        assert isinstance(extracted, str)
        assert len(extracted) > 0

    def test_cdxf_enhanced_round_trip(self):
        text, fmt = build_initial_config()
        serialized = serialize_config_for_state(text, fmt, "cdxf_enhanced")
        extracted = extract_config_text(serialized, fmt, "cdxf_enhanced")
        assert isinstance(extracted, str)
        assert len(extracted) > 0

    def test_json_default_loses_comments(self):
        """JSON default: YAML comments should be lost."""
        text, fmt = build_initial_config()
        serialized = serialize_config_for_state(text, fmt, "json_default")
        extracted = extract_config_text(serialized, fmt, "json_default")
        meta = count_config_metadata(extracted, fmt)
        assert meta["comments"] == 0

    def test_cdxf_enhanced_preserves_comments(self):
        """CDXF enhanced: YAML comments should survive."""
        text, fmt = build_initial_config()
        original_meta = count_config_metadata(text, fmt)
        serialized = serialize_config_for_state(text, fmt, "cdxf_enhanced")
        extracted = extract_config_text(serialized, fmt, "cdxf_enhanced")
        meta = count_config_metadata(extracted, fmt)
        assert meta["comments"] >= original_meta["comments"] * 0.9


# ===========================================================================
# Metadata counting
# ===========================================================================


class TestCountConfigMetadata:
    """Tests for metadata counting."""

    def test_yaml_comments(self):
        text = "# top\nkey: value  # inline\n"
        result = count_config_metadata(text, "yaml")
        assert result["comments"] >= 2

    def test_json_no_comments(self):
        text = '{"key": "value"}'
        result = count_config_metadata(text, "json")
        assert result["comments"] == 0

    def test_returns_total(self):
        text = "# comment\nkey: value\n"
        result = count_config_metadata(text, "yaml")
        assert "total" in result


# ===========================================================================
# LangGraph pipeline
# ===========================================================================


class TestBuildLanggraphPipeline:
    """Tests for LangGraph StateGraph construction."""

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_returns_compiled_graph(self, mode):
        graph = build_langgraph_pipeline(mode)
        assert graph is not None
        # LangGraph compiled graphs have an invoke method
        assert hasattr(graph, "invoke")

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_graph_has_nodes(self, mode):
        graph = build_langgraph_pipeline(mode)
        # Should have multiple agent nodes
        assert hasattr(graph, "get_graph")


class TestRunGraph:
    """Tests for executing the LangGraph pipeline."""

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_returns_result_dict(self, mode):
        result = run_graph(mode)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_has_node_trace(self, mode):
        result = run_graph(mode)
        assert "node_trace" in result
        assert len(result["node_trace"]) >= 2

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_has_metadata_at_each_node(self, mode):
        result = run_graph(mode)
        for node_info in result["node_trace"]:
            assert "node_name" in node_info
            assert "comments_after" in node_info

    def test_json_default_loses_comments(self):
        result = run_graph("json_default")
        assert result["initial_comments"] > 0
        assert result["final_comments"] == 0

    def test_cdxf_enhanced_preserves_comments(self):
        result = run_graph("cdxf_enhanced")
        assert result["initial_comments"] > 0
        assert result["final_comments"] >= result["initial_comments"] * 0.9

    def test_has_surviving_fraction(self):
        result = run_graph("json_default")
        assert "surviving_fraction" in result


# ===========================================================================
# Checkpoint/restore cycle
# ===========================================================================


class TestRunGraphWithCheckpoints:
    """Tests for checkpoint serialization fidelity."""

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_returns_result_dict(self, mode):
        result = run_graph_with_checkpoints(mode)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_has_checkpoint_data(self, mode):
        result = run_graph_with_checkpoints(mode)
        assert "checkpoints" in result
        assert len(result["checkpoints"]) >= 1

    def test_json_default_checkpoint_loses_comments(self):
        """After checkpoint/restore, JSON state loses comments."""
        result = run_graph_with_checkpoints("json_default")
        assert result["comments_after_restore"] == 0

    def test_cdxf_enhanced_checkpoint_preserves(self):
        """After checkpoint/restore, CDXF state preserves comments."""
        result = run_graph_with_checkpoints("cdxf_enhanced")
        assert result["comments_after_restore"] >= (
            result["initial_comments"] * 0.9
        )


# ===========================================================================
# Hypothesis validation
# ===========================================================================


class TestHypothesisValidation:
    """Validate that CDXF improves LangGraph state fidelity."""

    def test_cdxf_outperforms_json_default(self):
        json_result = run_graph("json_default")
        cdxf_result = run_graph("cdxf_enhanced")
        assert cdxf_result["surviving_fraction"] > (
            json_result["surviving_fraction"]
        )

    def test_json_default_zero_survival(self):
        result = run_graph("json_default")
        assert result["surviving_fraction"] == 0.0

    def test_cdxf_enhanced_near_perfect(self):
        result = run_graph("cdxf_enhanced")
        assert result["surviving_fraction"] >= 0.9


# ===========================================================================
# Full experiment (integration)
# ===========================================================================


class TestRunExperiment:
    """Integration tests for the full experiment."""

    @pytest.fixture(scope="class")
    def results(self):
        return run_experiment()

    def test_returns_dict(self, results):
        assert isinstance(results, dict)

    def test_has_experiment_id(self, results):
        assert results["experiment"] == "EXP-015"

    def test_has_timestamp(self, results):
        assert "timestamp" in results

    def test_has_both_modes(self, results):
        r = results["results"]
        has_json = any("json_default" in k for k in r)
        has_cdxf = any("cdxf_enhanced" in k for k in r)
        assert has_json
        assert has_cdxf

    def test_has_checkpoint_results(self, results):
        assert "checkpoint_results" in results

    def test_has_summary(self, results):
        assert "summary" in results



# ===========================================================================
# Enhanced: Scaling experiment (multi-size configs)
# ===========================================================================


class TestScalingExperiment:
    """Test metadata fidelity across different config sizes."""

    @pytest.fixture(scope="class")
    def scaling_results(self):
        from benchmarks.src.run_exp015 import run_scaling_experiment
        return run_scaling_experiment()

    def test_returns_dict(self, scaling_results):
        assert "scaling_results" in scaling_results

    def test_all_sizes_present(self, scaling_results):
        sizes = {r["config_size"] for r in scaling_results["scaling_results"]}
        assert sizes == {"small", "medium", "large", "xlarge"}

    def test_both_modes_per_size(self, scaling_results):
        for size in ["small", "medium", "large", "xlarge"]:
            modes = {
                r["mode"] for r in scaling_results["scaling_results"]
                if r["config_size"] == size
            }
            assert modes == {"json_default", "cdxf_enhanced"}

    @pytest.mark.parametrize("size", ["small", "medium", "large", "xlarge"])
    def test_json_default_loses_all(self, size, scaling_results):
        rows = [
            r for r in scaling_results["scaling_results"]
            if r["config_size"] == size and r["mode"] == "json_default"
        ]
        assert len(rows) == 1
        assert rows[0]["surviving_fraction"] == 0.0

    @pytest.mark.parametrize("size", ["small", "medium", "large", "xlarge"])
    def test_cdxf_preserves_all(self, size, scaling_results):
        rows = [
            r for r in scaling_results["scaling_results"]
            if r["config_size"] == size and r["mode"] == "cdxf_enhanced"
        ]
        assert len(rows) == 1
        assert rows[0]["surviving_fraction"] >= 0.9


# ===========================================================================
# Enhanced: Timing experiment
# ===========================================================================


class TestTimingExperiment:
    """Test overhead measurement."""

    @pytest.fixture(scope="class")
    def timing_results(self):
        from benchmarks.src.run_exp015 import run_timing_experiment
        return run_timing_experiment(n_iterations=5)

    def test_returns_dict(self, timing_results):
        assert "timings_seconds" in timing_results
        assert "overhead" in timing_results

    def test_all_timings_positive(self, timing_results):
        for k, v in timing_results["timings_seconds"].items():
            assert v > 0, f"{k} timing is not positive"

    def test_has_serialize_timings(self, timing_results):
        t = timing_results["timings_seconds"]
        assert "json_default_serialize" in t
        assert "cdxf_enhanced_serialize" in t

    def test_has_extract_timings(self, timing_results):
        t = timing_results["timings_seconds"]
        assert "json_default_extract" in t
        assert "cdxf_enhanced_extract" in t

    def test_has_pipeline_timings(self, timing_results):
        t = timing_results["timings_seconds"]
        assert "json_default_pipeline" in t
        assert "cdxf_enhanced_pipeline" in t

    def test_overhead_is_computed(self, timing_results):
        o = timing_results["overhead"]
        assert "serialize_overhead_ms" in o
        assert "extract_overhead_ms" in o
        assert "pipeline_overhead_ms" in o


# ===========================================================================
# Enhanced: Data integrity experiment
# ===========================================================================


class TestIntegrityExperiment:
    """Test that agent modifications are correctly applied."""

    @pytest.fixture(scope="class")
    def integrity_results(self):
        from benchmarks.src.run_exp015 import run_integrity_experiment
        return run_integrity_experiment()

    def test_returns_dict(self, integrity_results):
        assert "integrity_results" in integrity_results

    def test_all_configs_tested(self, integrity_results):
        configs = {r["graph_config"]
                   for r in integrity_results["integrity_results"]}
        assert len(configs) >= 2

    def test_both_modes_tested(self, integrity_results):
        modes = {r["mode"]
                 for r in integrity_results["integrity_results"]}
        assert modes == {"json_default", "cdxf_enhanced"}

    def test_cdxf_integrity_passes(self, integrity_results):
        """CDXF must preserve both metadata AND data values."""
        cdxf_rows = [
            r for r in integrity_results["integrity_results"]
            if r["mode"] == "cdxf_enhanced"
        ]
        for row in cdxf_rows:
            assert row["integrity_passed"], (
                f"{row['graph_config']}: {row['failures']}"
            )

    def test_json_default_integrity_passes(self, integrity_results):
        """JSON default should also preserve data values (just not comments)."""
        json_rows = [
            r for r in integrity_results["integrity_results"]
            if r["mode"] == "json_default"
        ]
        for row in json_rows:
            assert row["integrity_passed"], (
                f"{row['graph_config']}: {row['failures']}"
            )
