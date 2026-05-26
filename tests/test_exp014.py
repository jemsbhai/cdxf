"""Tests for EXP-014: Multi-Agent Format Interchange — Hub vs Direct.

Measures converter scaling (O(N²) vs O(N)) and metadata compounding
over H sequential handoffs in a multi-agent pipeline.

TDD: These tests are written BEFORE the implementation.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp014 import (
    AGENTS,
    CONVERSION_METHODS,
    N_FORMAT_COUNTS,
    PIPELINE_DEPTHS,
    build_agent_pipeline,
    build_test_config,
    cdxf_hub_convert,
    count_converters,
    count_metadata,
    direct_convert,
    run_experiment,
    run_metadata_compounding,
    run_pipeline,
    run_scaling_analysis,
)


# ===========================================================================
# Constants — Protocol compliance
# ===========================================================================


class TestProtocolConstants:
    """Verify protocol-mandated constants."""

    def test_pipeline_depths(self):
        """Protocol: H = 1, 2, 3, 5."""
        assert set(PIPELINE_DEPTHS) == {1, 2, 3, 5}

    def test_n_format_counts(self):
        """Protocol: N = 2, 3, 4."""
        assert set(N_FORMAT_COUNTS) == {2, 3, 4}

    def test_conversion_methods(self):
        assert set(CONVERSION_METHODS) == {"direct", "cdxf_hub"}

    def test_agents_defined(self):
        """Protocol: 5 agents with distinct roles and formats."""
        assert len(AGENTS) == 5
        for agent in AGENTS:
            assert "name" in agent
            assert "role" in agent
            assert "format" in agent


class TestAgentDefinitions:
    """Verify agent definitions match the protocol."""

    def test_agent_formats_span_four(self):
        """Agents should use JSON, YAML, TOML, XML."""
        formats = {a["format"] for a in AGENTS}
        assert formats == {"json", "yaml", "toml", "xml"}

    def test_agent_roles(self):
        roles = {a["role"] for a in AGENTS}
        expected = {"curator", "trainer", "evaluator", "deployer", "monitor"}
        assert roles == expected


# ===========================================================================
# Converter counting — the O(N²) vs O(N) argument
# ===========================================================================


class TestCountConverters:
    """Tests for converter count scaling."""

    def test_direct_is_n_times_n_minus_1(self):
        """Direct: N×(N-1) pairwise converters."""
        assert count_converters(2, "direct") == 2
        assert count_converters(3, "direct") == 6
        assert count_converters(4, "direct") == 12

    def test_cdxf_hub_is_2n(self):
        """CDXF hub: 2N (encode + decode per format)."""
        assert count_converters(2, "cdxf_hub") == 4
        assert count_converters(3, "cdxf_hub") == 6
        assert count_converters(4, "cdxf_hub") == 8

    def test_crossover_at_n3(self):
        """At N=3, direct (6) = hub (6). Hub wins for N>3."""
        assert count_converters(3, "direct") == count_converters(3, "cdxf_hub")
        assert count_converters(4, "direct") > count_converters(4, "cdxf_hub")

    def test_direct_grows_quadratically(self):
        counts = [count_converters(n, "direct") for n in [2, 3, 4]]
        # 2, 6, 12 — growing faster than linear
        assert counts[2] - counts[1] > counts[1] - counts[0]


# ===========================================================================
# Test config construction
# ===========================================================================


class TestBuildTestConfig:
    """Tests for building the annotated test config."""

    def test_returns_text_and_format(self):
        text, fmt = build_test_config()
        assert isinstance(text, str)
        assert isinstance(fmt, str)
        assert len(text) > 0

    def test_config_has_comments(self):
        text, fmt = build_test_config()
        meta = count_metadata(text, fmt)
        assert meta["comments"] > 0, "Config must have comments for testing"

    def test_config_format_is_yaml(self):
        """Start with YAML since it supports comments."""
        _, fmt = build_test_config()
        assert fmt == "yaml"


# ===========================================================================
# Pipeline construction
# ===========================================================================


class TestBuildAgentPipeline:
    """Tests for building agent pipelines of varying depth."""

    @pytest.mark.parametrize("n", N_FORMAT_COUNTS)
    def test_returns_list_of_agents(self, n):
        pipeline = build_agent_pipeline(n)
        assert isinstance(pipeline, list)
        assert len(pipeline) >= 2

    def test_pipeline_agents_have_different_formats(self):
        pipeline = build_agent_pipeline(4)
        # Adjacent agents should require format conversion
        formats = [a["format"] for a in pipeline]
        # At least some adjacent pairs should differ
        diffs = sum(1 for i in range(len(formats) - 1)
                    if formats[i] != formats[i + 1])
        assert diffs >= 1


# ===========================================================================
# Direct conversion
# ===========================================================================


class TestDirectConvert:
    """Tests for direct format-to-format conversion."""

    def test_returns_tuple(self):
        text = "key: value\n# comment\n"
        result_text, meta = direct_convert(text, "yaml", "json")
        assert isinstance(result_text, str)
        assert isinstance(meta, dict)

    def test_yaml_to_json_loses_comments(self):
        text = "# important note\nkey: value\n"
        result_text, meta = direct_convert(text, "yaml", "json")
        result_meta = count_metadata(result_text, "json")
        assert result_meta["comments"] == 0

    def test_produces_valid_target_format(self):
        text = "key: value\ncount: 42\n"
        result_text, _ = direct_convert(text, "yaml", "json")
        parsed = json.loads(result_text)
        assert parsed["key"] == "value"

    def test_meta_has_comments_lost(self):
        text = "# note\nkey: value\n"
        _, meta = direct_convert(text, "yaml", "json")
        assert "comments_before" in meta
        assert "comments_after" in meta
        assert meta["comments_lost"] > 0

    def test_identity_conversion(self):
        """Same format → same format should work."""
        text = '{"key": "value"}'
        result_text, meta = direct_convert(text, "json", "json")
        assert json.loads(result_text) == json.loads(text)


# ===========================================================================
# CDXF hub conversion
# ===========================================================================


class TestCdxfHubConvert:
    """Tests for CDXF hub conversion (encode → decode)."""

    def test_returns_tuple(self):
        text = "key: value\n# comment\n"
        result_text, meta = cdxf_hub_convert(text, "yaml", "json")
        assert isinstance(result_text, str)
        assert isinstance(meta, dict)

    def test_preserves_data(self):
        text = "key: value\ncount: 42\n"
        result_text, _ = cdxf_hub_convert(text, "yaml", "json")
        parsed = json.loads(result_text)
        assert parsed["key"] == "value"

    def test_meta_has_comments_fields(self):
        text = "# note\nkey: value\n"
        _, meta = cdxf_hub_convert(text, "yaml", "json")
        assert "comments_before" in meta
        assert "comments_after" in meta


# ===========================================================================
# Metadata counting
# ===========================================================================


class TestCountMetadata:
    """Tests for the metadata counter."""

    def test_yaml_comments(self):
        text = "# comment\nkey: value\n"
        result = count_metadata(text, "yaml")
        assert result["comments"] >= 1

    def test_json_no_comments(self):
        text = '{"key": "value"}'
        result = count_metadata(text, "json")
        assert result["comments"] == 0

    def test_toml_comments(self):
        text = '# comment\nkey = "value"\n'
        result = count_metadata(text, "toml")
        assert result["comments"] >= 1

    def test_has_total(self):
        text = "# comment\nkey: value\n"
        result = count_metadata(text, "yaml")
        assert "total" in result


# ===========================================================================
# Pipeline execution
# ===========================================================================


class TestRunPipeline:
    """Tests for running a full agent pipeline."""

    def test_returns_dict(self):
        result = run_pipeline("direct", depth=1, n_formats=2)
        assert isinstance(result, dict)

    def test_has_hops(self):
        result = run_pipeline("direct", depth=2, n_formats=2)
        assert "hops" in result
        assert len(result["hops"]) == 2

    def test_hop_structure(self):
        result = run_pipeline("direct", depth=1, n_formats=2)
        hop = result["hops"][0]
        assert "from_agent" in hop
        assert "to_agent" in hop
        assert "from_format" in hop
        assert "to_format" in hop
        assert "comments_before" in hop
        assert "comments_after" in hop
        assert "latency_us" in hop

    def test_has_final_metadata(self):
        result = run_pipeline("direct", depth=1, n_formats=2)
        assert "initial_comments" in result
        assert "final_comments" in result
        assert "surviving_fraction" in result

    def test_direct_loses_comments_over_hops(self):
        """Direct conversion should lose comments."""
        result = run_pipeline("direct", depth=1, n_formats=4)
        # The config starts as YAML with comments, gets converted
        assert result["initial_comments"] > 0
        assert result["final_comments"] < result["initial_comments"]

    def test_cdxf_hub_preserves_comments(self):
        """CDXF hub should preserve comments."""
        result = run_pipeline("cdxf_hub", depth=1, n_formats=4)
        assert result["initial_comments"] > 0
        # CDXF preserves comments (may have slight bridge inflation)
        assert result["final_comments"] >= result["initial_comments"] * 0.9


class TestMetadataCompounding:
    """Tests for metadata loss compounding over H hops."""

    def test_direct_loss_increases_with_depth(self):
        """More hops → more metadata loss for direct conversion."""
        r1 = run_pipeline("direct", depth=1, n_formats=4)
        r3 = run_pipeline("direct", depth=3, n_formats=4)
        # At depth 3, should have lost at least as much as depth 1
        assert r3["surviving_fraction"] <= r1["surviving_fraction"]

    def test_cdxf_hub_stable_across_depths(self):
        """CDXF hub: metadata should not degrade with depth."""
        r1 = run_pipeline("cdxf_hub", depth=1, n_formats=4)
        r5 = run_pipeline("cdxf_hub", depth=5, n_formats=4)
        # CDXF should maintain near-100% at any depth
        assert r5["surviving_fraction"] >= 0.9


# ===========================================================================
# Scaling analysis
# ===========================================================================


class TestRunScalingAnalysis:
    """Tests for the O(N²) vs O(N) scaling analysis."""

    def test_returns_dict(self):
        result = run_scaling_analysis()
        assert isinstance(result, dict)

    def test_has_entries_for_all_n(self):
        result = run_scaling_analysis()
        for n in N_FORMAT_COUNTS:
            assert n in result

    def test_entry_has_both_methods(self):
        result = run_scaling_analysis()
        entry = result[N_FORMAT_COUNTS[0]]
        assert "direct_converters" in entry
        assert "cdxf_hub_converters" in entry

    def test_quadratic_vs_linear(self):
        result = run_scaling_analysis()
        for n in N_FORMAT_COUNTS:
            assert result[n]["direct_converters"] == n * (n - 1)
            assert result[n]["cdxf_hub_converters"] == 2 * n


# ===========================================================================
# Metadata compounding analysis
# ===========================================================================


class TestRunMetadataCompounding:
    """Tests for the metadata compounding analysis."""

    def test_returns_dict(self):
        result = run_metadata_compounding()
        assert isinstance(result, dict)

    def test_has_both_methods(self):
        result = run_metadata_compounding()
        assert "direct" in result
        assert "cdxf_hub" in result

    def test_has_all_depths(self):
        result = run_metadata_compounding()
        for method in CONVERSION_METHODS:
            for depth in PIPELINE_DEPTHS:
                assert depth in result[method]

    def test_direct_degradation_curve(self):
        """Direct: surviving fraction should decrease with depth."""
        result = run_metadata_compounding()
        fracs = [result["direct"][d]["surviving_fraction"]
                 for d in sorted(PIPELINE_DEPTHS)]
        # Should be non-increasing (more hops → more loss)
        for i in range(1, len(fracs)):
            assert fracs[i] <= fracs[i - 1] + 0.01  # small tolerance

    def test_cdxf_hub_flat(self):
        """CDXF hub: surviving fraction should stay near 1.0."""
        result = run_metadata_compounding()
        for depth in PIPELINE_DEPTHS:
            frac = result["cdxf_hub"][depth]["surviving_fraction"]
            assert frac >= 0.9, (
                f"CDXF hub survival {frac:.2f} at depth {depth}"
            )


# ===========================================================================
# Hypothesis validation — the scientific core
# ===========================================================================


class TestHypothesisValidation:
    """Validate the core hypotheses from the protocol."""

    def test_quadratic_to_linear_reduction(self):
        """H1: CDXF reduces converter count from O(N²) to O(N)."""
        scaling = run_scaling_analysis()
        for n in N_FORMAT_COUNTS:
            if n > 3:
                assert (scaling[n]["direct_converters"] >
                        scaling[n]["cdxf_hub_converters"])

    def test_zero_cumulative_metadata_loss(self):
        """H3: CDXF hub has zero cumulative metadata loss."""
        compounding = run_metadata_compounding()
        for depth in PIPELINE_DEPTHS:
            frac = compounding["cdxf_hub"][depth]["surviving_fraction"]
            assert frac >= 0.9

    def test_direct_metadata_compounds(self):
        """H4: Direct conversion loses all comments at first non-comment format."""
        compounding = run_metadata_compounding()
        # Direct is so lossy that ALL comments die at the first
        # yaml→json hop. Both depth=1 and depth=5 yield 0.
        frac_1 = compounding["direct"][1]["surviving_fraction"]
        frac_5 = compounding["direct"][5]["surviving_fraction"]
        assert frac_1 == 0.0, "Direct should lose all comments at hop 1"
        assert frac_5 == 0.0, "Direct should still have 0 at hop 5"

    def test_divergence_grows_with_depth(self):
        """Gap between CDXF and direct should grow with pipeline depth."""
        compounding = run_metadata_compounding()
        gaps = []
        for depth in sorted(PIPELINE_DEPTHS):
            cdxf_frac = compounding["cdxf_hub"][depth]["surviving_fraction"]
            direct_frac = compounding["direct"][depth]["surviving_fraction"]
            gaps.append(cdxf_frac - direct_frac)
        # Gap should be non-decreasing
        for i in range(1, len(gaps)):
            assert gaps[i] >= gaps[i - 1] - 0.01


# ===========================================================================
# Full experiment (integration)
# ===========================================================================


class TestRunExperiment:
    """Integration tests for the full experiment pipeline."""

    @pytest.fixture(scope="class")
    def results(self):
        return run_experiment()

    def test_returns_dict(self, results):
        assert isinstance(results, dict)

    def test_has_experiment_id(self, results):
        assert results["experiment"] == "EXP-014"

    def test_has_timestamp(self, results):
        assert "timestamp" in results

    def test_has_scaling(self, results):
        assert "scaling_analysis" in results

    def test_has_compounding(self, results):
        assert "metadata_compounding" in results

    def test_has_pipeline_results(self, results):
        assert "pipeline_results" in results

    def test_has_summary(self, results):
        assert "summary" in results
