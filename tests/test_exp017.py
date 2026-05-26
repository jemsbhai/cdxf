"""Tests for EXP-017: CrewAI Pipeline — ML Config Handoff Fidelity.

Build a real CrewAI crew with FakeLLM where ML agents pass annotated
YAML configs as task outputs. CrewAI serializes task results as strings
via TaskOutput.raw. Compare:
  - json_default: agent parses YAML → dict, modifies, re-emits → comments lost
  - cdxf_enhanced: agent encodes to CDXF, passes base64, decodes → comments preserved

TDD: These tests are written BEFORE the implementation.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp017 import (
    CREW_CONFIGS,
    STATE_MODES,
    FakeLLM,
    build_initial_config,
    count_config_metadata,
    extract_config_from_output,
    serialize_config_for_output,
    build_crew_pipeline,
    run_crew,
    run_experiment,
)


# ===========================================================================
# Constants — Protocol compliance
# ===========================================================================


class TestProtocolConstants:
    """Verify experiment constants."""

    def test_state_modes(self):
        """Two modes: json (standard) vs cdxf-enhanced."""
        assert "json_default" in STATE_MODES
        assert "cdxf_enhanced" in STATE_MODES
        assert len(STATE_MODES) == 2

    def test_crew_configs(self):
        """Multiple crew topologies to test."""
        assert len(CREW_CONFIGS) >= 2

    def test_crew_configs_have_required_fields(self):
        """Each config must have name, description, roles."""
        for cc in CREW_CONFIGS:
            assert "name" in cc
            assert "description" in cc
            assert "roles" in cc
            assert len(cc["roles"]) >= 4


# ===========================================================================
# FakeLLM
# ===========================================================================


class TestFakeLLM:
    """Tests for the deterministic FakeLLM."""

    def test_is_base_llm_subclass(self):
        from crewai.llms.base_llm import BaseLLM
        llm = FakeLLM(mode="json_default")
        assert isinstance(llm, BaseLLM)

    def test_call_returns_string(self):
        llm = FakeLLM(mode="json_default")
        result = llm.call("test message")
        assert isinstance(result, str)

    def test_json_default_mode_returns_yaml(self):
        """In json_default mode, FakeLLM should return YAML text (lossy)."""
        llm = FakeLLM(mode="json_default")
        text, fmt = build_initial_config()
        result = llm.call(f"Process this config:\n{text}")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cdxf_enhanced_mode_returns_base64(self):
        """In cdxf_enhanced mode, FakeLLM should return base64 CDXF."""
        llm = FakeLLM(mode="cdxf_enhanced")
        text, fmt = build_initial_config()
        result = llm.call(f"Process this config:\n{text}")
        assert isinstance(result, str)
        assert len(result) > 0


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

    def test_identical_to_exp015(self):
        """Same config as EXP-015 for cross-experiment comparability."""
        from benchmarks.src.run_exp015 import (
            build_initial_config as build_015,
        )
        text_015, _ = build_015()
        text_017, _ = build_initial_config()
        assert text_015 == text_017


# ===========================================================================
# Config serialization for task output
# ===========================================================================


class TestSerializeConfigForOutput:
    """Tests for serializing config into CrewAI task output strings."""

    def test_json_default_returns_string(self):
        text, fmt = build_initial_config()
        result = serialize_config_for_output(text, fmt, "json_default")
        assert isinstance(result, str)

    def test_cdxf_enhanced_returns_string(self):
        text, fmt = build_initial_config()
        result = serialize_config_for_output(text, fmt, "cdxf_enhanced")
        assert isinstance(result, str)

    def test_json_default_loses_comments(self):
        """JSON default: parse YAML → dict → yaml.dump → comments lost."""
        text, fmt = build_initial_config()
        serialized = serialize_config_for_output(text, fmt, "json_default")
        meta = count_config_metadata(serialized, fmt)
        assert meta["comments"] == 0

    def test_cdxf_enhanced_preserves_comments(self):
        """CDXF enhanced: encode → base64 → decode → comments survive."""
        text, fmt = build_initial_config()
        serialized = serialize_config_for_output(text, fmt, "cdxf_enhanced")
        extracted = extract_config_from_output(serialized, fmt, "cdxf_enhanced")
        meta = count_config_metadata(extracted, fmt)
        original_meta = count_config_metadata(text, fmt)
        assert meta["comments"] == original_meta["comments"]


class TestExtractConfigFromOutput:
    """Tests for extracting config text from task output strings."""

    def test_json_default_round_trip(self):
        text, fmt = build_initial_config()
        serialized = serialize_config_for_output(text, fmt, "json_default")
        extracted = extract_config_from_output(serialized, fmt, "json_default")
        assert isinstance(extracted, str)
        assert len(extracted) > 0

    def test_cdxf_enhanced_round_trip(self):
        text, fmt = build_initial_config()
        serialized = serialize_config_for_output(text, fmt, "cdxf_enhanced")
        extracted = extract_config_from_output(serialized, fmt, "cdxf_enhanced")
        assert isinstance(extracted, str)
        assert len(extracted) > 0

    def test_json_default_loses_comments(self):
        text, fmt = build_initial_config()
        serialized = serialize_config_for_output(text, fmt, "json_default")
        extracted = extract_config_from_output(serialized, fmt, "json_default")
        meta = count_config_metadata(extracted, fmt)
        assert meta["comments"] == 0

    def test_cdxf_enhanced_preserves_comments(self):
        text, fmt = build_initial_config()
        original_meta = count_config_metadata(text, fmt)
        serialized = serialize_config_for_output(text, fmt, "cdxf_enhanced")
        extracted = extract_config_from_output(serialized, fmt, "cdxf_enhanced")
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

    def test_no_comments(self):
        text = "key: value\nother: 123\n"
        result = count_config_metadata(text, "yaml")
        assert result["comments"] == 0

    def test_returns_total(self):
        text = "# comment\nkey: value\n"
        result = count_config_metadata(text, "yaml")
        assert "total" in result


# ===========================================================================
# CrewAI pipeline construction
# ===========================================================================


class TestBuildCrewPipeline:
    """Tests for CrewAI crew construction."""

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_returns_crew(self, mode):
        from crewai import Crew
        crew, tasks = build_crew_pipeline(mode)
        assert isinstance(crew, Crew)

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_has_agents(self, mode):
        crew, tasks = build_crew_pipeline(mode)
        assert len(crew.agents) >= 4

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_has_tasks(self, mode):
        crew, tasks = build_crew_pipeline(mode)
        assert len(tasks) >= 4

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_tasks_are_sequential(self, mode):
        """Each task (after first) should have context from prior task."""
        crew, tasks = build_crew_pipeline(mode)
        for i in range(1, len(tasks)):
            assert tasks[i].context is not None
            assert len(tasks[i].context) >= 1


# ===========================================================================
# Crew execution
# ===========================================================================


class TestRunCrew:
    """Tests for executing the CrewAI pipeline."""

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_returns_result_dict(self, mode):
        result = run_crew(mode)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_has_required_fields(self, mode):
        result = run_crew(mode)
        assert "mode" in result
        assert "crew_config" in result
        assert "initial_comments" in result
        assert "final_comments" in result
        assert "surviving_fraction" in result
        assert "n_agents" in result

    def test_json_default_loses_comments(self):
        result = run_crew("json_default")
        assert result["initial_comments"] > 0
        assert result["final_comments"] == 0

    def test_cdxf_enhanced_preserves_comments(self):
        result = run_crew("cdxf_enhanced")
        assert result["initial_comments"] > 0
        assert result["final_comments"] >= result["initial_comments"] * 0.9

    def test_json_default_zero_survival(self):
        result = run_crew("json_default")
        assert result["surviving_fraction"] == 0.0

    def test_cdxf_enhanced_near_perfect(self):
        result = run_crew("cdxf_enhanced")
        assert result["surviving_fraction"] >= 0.9


# ===========================================================================
# Hypothesis validation
# ===========================================================================


class TestHypothesisValidation:
    """Validate that CDXF improves CrewAI config fidelity."""

    def test_cdxf_outperforms_json_default(self):
        json_result = run_crew("json_default")
        cdxf_result = run_crew("cdxf_enhanced")
        assert cdxf_result["surviving_fraction"] > (
            json_result["surviving_fraction"]
        )

    def test_consistent_with_exp015(self):
        """Same directional result as EXP-015 (LangGraph)."""
        cdxf_result = run_crew("cdxf_enhanced")
        json_result = run_crew("json_default")
        # Same pattern: json=0%, cdxf≥90%
        assert json_result["surviving_fraction"] == 0.0
        assert cdxf_result["surviving_fraction"] >= 0.9

    @pytest.mark.parametrize("crew_config_idx", [0, 1])
    def test_fidelity_across_topologies(self, crew_config_idx):
        """CDXF fidelity holds across different crew sizes."""
        cc = CREW_CONFIGS[crew_config_idx]
        result = run_crew("cdxf_enhanced", crew_config=cc)
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
        assert results["experiment"] == "EXP-017"

    def test_has_timestamp(self, results):
        assert "timestamp" in results

    def test_has_framework_info(self, results):
        assert results["framework"] == "crewai"
        assert "framework_version" in results

    def test_has_both_modes(self, results):
        r = results["results"]
        has_json = any("json_default" in k for k in r)
        has_cdxf = any("cdxf_enhanced" in k for k in r)
        assert has_json
        assert has_cdxf

    def test_has_summary(self, results):
        assert "summary" in results

    def test_output_files_exist(self, results):
        output_dir = Path("benchmarks/results/exp_017")
        assert (output_dir / "exp_017_results.json").exists()
        assert (output_dir / "mode_comparison.csv").exists()



# ===========================================================================
# Enhanced: Scaling experiment (multi-size configs)
# ===========================================================================


class TestScalingExperiment:
    """Test metadata fidelity across different config sizes."""

    @pytest.fixture(scope="class")
    def scaling_results(self):
        from benchmarks.src.run_exp017 import run_scaling_experiment
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
        from benchmarks.src.run_exp017 import run_timing_experiment
        return run_timing_experiment(n_iterations=3)

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
        from benchmarks.src.run_exp017 import run_integrity_experiment
        return run_integrity_experiment()

    def test_returns_dict(self, integrity_results):
        assert "integrity_results" in integrity_results

    def test_all_configs_tested(self, integrity_results):
        configs = {r["crew_config"]
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
                f"{row['crew_config']}: {row['failures']}"
            )

    def test_json_default_integrity_passes(self, integrity_results):
        """JSON default should also preserve data values (just not comments)."""
        json_rows = [
            r for r in integrity_results["integrity_results"]
            if r["mode"] == "json_default"
        ]
        for row in json_rows:
            assert row["integrity_passed"], (
                f"{row['crew_config']}: {row['failures']}"
            )
