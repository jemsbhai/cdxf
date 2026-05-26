"""Tests for EXP-018: AutoGen Group Chat — ML Config Handoff Fidelity.

Build a real AutoGen RoundRobinGroupChat with FakeChatCompletionClient
where ML agents pass annotated YAML configs as TextMessages. Compare:
  - json_default: agent parses YAML → dict, modifies, re-emits → comments lost
  - cdxf_enhanced: agent encodes to CDXF, passes base64, decodes → comments preserved

TDD: These tests are written BEFORE the implementation.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp018 import (
    GROUP_CONFIGS,
    STATE_MODES,
    FakeChatCompletionClient,
    build_initial_config,
    count_config_metadata,
    extract_config_from_output,
    serialize_config_for_output,
    build_group_chat,
    run_group_chat,
    run_scaling_experiment,
    run_timing_experiment,
    run_integrity_experiment,
    run_experiment,
)


# ===========================================================================
# Constants — Protocol compliance
# ===========================================================================


class TestProtocolConstants:
    """Verify experiment constants."""

    def test_state_modes(self):
        assert "json_default" in STATE_MODES
        assert "cdxf_enhanced" in STATE_MODES
        assert len(STATE_MODES) == 2

    def test_group_configs(self):
        assert len(GROUP_CONFIGS) >= 2

    def test_group_configs_have_required_fields(self):
        for gc in GROUP_CONFIGS:
            assert "name" in gc
            assert "description" in gc
            assert "roles" in gc
            assert len(gc["roles"]) >= 4


# ===========================================================================
# FakeChatCompletionClient
# ===========================================================================


class TestFakeChatCompletionClient:
    """Tests for the deterministic fake model client."""

    def test_is_chat_completion_client(self):
        from autogen_core.models import ChatCompletionClient
        client = FakeChatCompletionClient(mode="json_default")
        assert isinstance(client, ChatCompletionClient)

    def test_create_returns_create_result(self):
        from autogen_core.models import CreateResult
        client = FakeChatCompletionClient(mode="json_default")
        from autogen_core.models import UserMessage
        result = asyncio.run(client.create([UserMessage(content="test", source="user")]))
        assert isinstance(result, CreateResult)

    def test_create_result_has_string_content(self):
        client = FakeChatCompletionClient(mode="json_default")
        from autogen_core.models import UserMessage
        result = asyncio.run(client.create([UserMessage(content="test", source="user")]))
        assert isinstance(result.content, str)


# ===========================================================================
# Config construction
# ===========================================================================


class TestBuildInitialConfig:
    """Tests for the initial ML config."""

    def test_returns_text_and_format(self):
        text, fmt = build_initial_config()
        assert isinstance(text, str)
        assert fmt == "yaml"

    def test_has_comments(self):
        text, fmt = build_initial_config()
        meta = count_config_metadata(text, fmt)
        assert meta["comments"] >= 10

    def test_identical_to_exp015(self):
        from benchmarks.src.run_exp015 import (
            build_initial_config as build_015,
        )
        text_015, _ = build_015()
        text_018, _ = build_initial_config()
        assert text_015 == text_018


# ===========================================================================
# Config serialization
# ===========================================================================


class TestSerializeConfigForOutput:
    """Tests for serializing config for AutoGen messages."""

    def test_json_default_returns_string(self):
        text, fmt = build_initial_config()
        result = serialize_config_for_output(text, fmt, "json_default")
        assert isinstance(result, str)

    def test_cdxf_enhanced_returns_string(self):
        text, fmt = build_initial_config()
        result = serialize_config_for_output(text, fmt, "cdxf_enhanced")
        assert isinstance(result, str)

    def test_json_default_loses_comments(self):
        text, fmt = build_initial_config()
        serialized = serialize_config_for_output(text, fmt, "json_default")
        meta = count_config_metadata(serialized, fmt)
        assert meta["comments"] == 0

    def test_cdxf_enhanced_preserves_comments(self):
        text, fmt = build_initial_config()
        original_meta = count_config_metadata(text, fmt)
        serialized = serialize_config_for_output(text, fmt, "cdxf_enhanced")
        extracted = extract_config_from_output(serialized, fmt, "cdxf_enhanced")
        meta = count_config_metadata(extracted, fmt)
        assert meta["comments"] == original_meta["comments"]


class TestExtractConfigFromOutput:
    """Tests for extracting config from message text."""

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


# ===========================================================================
# Group chat construction
# ===========================================================================


class TestBuildGroupChat:
    """Tests for AutoGen group chat construction."""

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_returns_team_and_agents(self, mode):
        from autogen_agentchat.teams import RoundRobinGroupChat
        team, agents = build_group_chat(mode)
        assert isinstance(team, RoundRobinGroupChat)
        assert len(agents) >= 4


# ===========================================================================
# Group chat execution
# ===========================================================================


class TestRunGroupChat:
    """Tests for executing the AutoGen group chat."""

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_returns_result_dict(self, mode):
        result = asyncio.run(run_group_chat(mode))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("mode", STATE_MODES)
    def test_has_required_fields(self, mode):
        result = asyncio.run(run_group_chat(mode))
        assert "mode" in result
        assert "group_config" in result
        assert "initial_comments" in result
        assert "final_comments" in result
        assert "surviving_fraction" in result
        assert "n_agents" in result

    def test_json_default_loses_comments(self):
        result = asyncio.run(run_group_chat("json_default"))
        assert result["initial_comments"] > 0
        assert result["final_comments"] == 0

    def test_cdxf_enhanced_preserves_comments(self):
        result = asyncio.run(run_group_chat("cdxf_enhanced"))
        assert result["initial_comments"] > 0
        assert result["final_comments"] >= result["initial_comments"] * 0.9

    def test_json_default_zero_survival(self):
        result = asyncio.run(run_group_chat("json_default"))
        assert result["surviving_fraction"] == 0.0

    def test_cdxf_enhanced_near_perfect(self):
        result = asyncio.run(run_group_chat("cdxf_enhanced"))
        assert result["surviving_fraction"] >= 0.9


# ===========================================================================
# Hypothesis validation
# ===========================================================================


class TestHypothesisValidation:
    """Validate that CDXF improves AutoGen config fidelity."""

    def test_cdxf_outperforms_json_default(self):
        json_result = asyncio.run(run_group_chat("json_default"))
        cdxf_result = asyncio.run(run_group_chat("cdxf_enhanced"))
        assert cdxf_result["surviving_fraction"] > json_result["surviving_fraction"]

    def test_consistent_with_exp015_and_017(self):
        json_result = asyncio.run(run_group_chat("json_default"))
        cdxf_result = asyncio.run(run_group_chat("cdxf_enhanced"))
        assert json_result["surviving_fraction"] == 0.0
        assert cdxf_result["surviving_fraction"] >= 0.9

    @pytest.mark.parametrize("group_config_idx", [0, 1])
    def test_fidelity_across_topologies(self, group_config_idx):
        gc = GROUP_CONFIGS[group_config_idx]
        result = asyncio.run(run_group_chat("cdxf_enhanced", group_config=gc))
        assert result["surviving_fraction"] >= 0.9


# ===========================================================================
# Enhanced: Scaling
# ===========================================================================


class TestScalingExperiment:
    """Test fidelity across config sizes."""

    @pytest.fixture(scope="class")
    def scaling_results(self):
        return asyncio.run(run_scaling_experiment())

    def test_returns_dict(self, scaling_results):
        assert "scaling_results" in scaling_results

    def test_all_sizes_present(self, scaling_results):
        sizes = {r["config_size"] for r in scaling_results["scaling_results"]}
        assert sizes == {"small", "medium", "large", "xlarge"}

    @pytest.mark.parametrize("size", ["small", "medium", "large", "xlarge"])
    def test_cdxf_preserves_all(self, size, scaling_results):
        rows = [
            r for r in scaling_results["scaling_results"]
            if r["config_size"] == size and r["mode"] == "cdxf_enhanced"
        ]
        assert len(rows) == 1
        assert rows[0]["surviving_fraction"] >= 0.9


# ===========================================================================
# Enhanced: Timing
# ===========================================================================


class TestTimingExperiment:
    """Test overhead measurement."""

    @pytest.fixture(scope="class")
    def timing_results(self):
        return asyncio.run(run_timing_experiment(n_iterations=3))

    def test_returns_dict(self, timing_results):
        assert "timings_seconds" in timing_results
        assert "overhead" in timing_results

    def test_all_timings_positive(self, timing_results):
        for k, v in timing_results["timings_seconds"].items():
            assert v > 0, f"{k} timing is not positive"


# ===========================================================================
# Enhanced: Integrity
# ===========================================================================


class TestIntegrityExperiment:
    """Test data integrity."""

    @pytest.fixture(scope="class")
    def integrity_results(self):
        return asyncio.run(run_integrity_experiment())

    def test_returns_dict(self, integrity_results):
        assert "integrity_results" in integrity_results

    def test_cdxf_integrity_passes(self, integrity_results):
        cdxf_rows = [
            r for r in integrity_results["integrity_results"]
            if r["mode"] == "cdxf_enhanced"
        ]
        for row in cdxf_rows:
            assert row["integrity_passed"], (
                f"{row['group_config']}: {row['failures']}"
            )

    def test_json_default_integrity_passes(self, integrity_results):
        json_rows = [
            r for r in integrity_results["integrity_results"]
            if r["mode"] == "json_default"
        ]
        for row in json_rows:
            assert row["integrity_passed"], (
                f"{row['group_config']}: {row['failures']}"
            )


# ===========================================================================
# Full experiment (integration)
# ===========================================================================


class TestRunExperiment:
    """Integration tests for the full experiment."""

    @pytest.fixture(scope="class")
    def results(self):
        return asyncio.run(run_experiment())

    def test_returns_dict(self, results):
        assert isinstance(results, dict)

    def test_has_experiment_id(self, results):
        assert results["experiment"] == "EXP-018"

    def test_has_framework_info(self, results):
        assert results["framework"] == "autogen-agentchat"

    def test_has_both_modes(self, results):
        r = results["results"]
        has_json = any("json_default" in k for k in r)
        has_cdxf = any("cdxf_enhanced" in k for k in r)
        assert has_json
        assert has_cdxf

    def test_has_enhancements(self, results):
        assert "scaling" in results
        assert "timing" in results
        assert "integrity" in results
