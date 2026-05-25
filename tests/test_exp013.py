"""Tests for EXP-013: Agent Workflow State Persistence Across Sessions.

Measures cumulative information loss when agent state is serialized across
K sequential session boundaries. CDXF should maintain zero loss regardless
of K, while lossy formats degrade.

TDD: These tests are written BEFORE the implementation.
"""

import json
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp013 import (
    SESSION_BOUNDARIES,
    STATE_COMPLEXITIES,
    STATE_FORMATS,
    add_result_file,
    build_initial_state,
    count_all_metadata,
    count_metadata_constructs,
    deserialize_state,
    modify_one_config,
    run_degradation_loop,
    run_experiment,
    serialize_state,
)


# ===========================================================================
# Constants — Protocol compliance
# ===========================================================================


class TestProtocolConstants:
    """Verify protocol-mandated constants."""

    def test_session_boundaries(self):
        """Protocol: K = 1, 2, 5, 10, 20."""
        assert set(SESSION_BOUNDARIES) == {1, 2, 5, 10, 20}

    def test_state_complexities(self):
        """Protocol: small (3), medium (8), large (15)."""
        assert "small" in STATE_COMPLEXITIES
        assert "medium" in STATE_COMPLEXITIES
        assert "large" in STATE_COMPLEXITIES
        assert STATE_COMPLEXITIES["small"] == 3
        assert STATE_COMPLEXITIES["medium"] == 8
        assert STATE_COMPLEXITIES["large"] == 15

    def test_state_formats(self):
        """Protocol: CDXF, mega-JSON, Pickle, tar.gz."""
        assert set(STATE_FORMATS) == {"cdxf", "json_mega", "pickle", "tar_gz"}


# ===========================================================================
# Initial state construction
# ===========================================================================


class TestBuildInitialState:
    """Tests for building the initial agent state."""

    @pytest.mark.parametrize("complexity", ["small", "medium", "large"])
    def test_returns_list(self, complexity):
        state = build_initial_state(complexity)
        assert isinstance(state, list)

    @pytest.mark.parametrize("complexity,expected", [
        ("small", 3), ("medium", 8), ("large", 15),
    ])
    def test_correct_component_count(self, complexity, expected):
        state = build_initial_state(complexity)
        assert len(state) == expected

    def test_component_structure(self):
        state = build_initial_state("small")
        for comp in state:
            assert "name" in comp
            assert "filename" in comp
            assert "format" in comp
            assert "text" in comp
            assert isinstance(comp["text"], str)
            assert len(comp["text"]) > 0

    def test_has_multiple_formats(self):
        """State should include files in different formats."""
        state = build_initial_state("medium")
        formats = {c["format"] for c in state}
        assert len(formats) >= 2

    def test_has_yaml_with_comments(self):
        """At least some YAML files should contain comments."""
        state = build_initial_state("medium")
        yaml_files = [c for c in state if c["format"] == "yaml"]
        assert len(yaml_files) >= 1
        has_comments = any("#" in c["text"] for c in yaml_files)
        assert has_comments, "YAML files should contain comments for testing"

    def test_has_toml_with_comments(self):
        """At least some TOML files should contain comments."""
        state = build_initial_state("medium")
        toml_files = [c for c in state if c["format"] == "toml"]
        if toml_files:  # small complexity may not have TOML
            has_comments = any("#" in c["text"] for c in toml_files)
            assert has_comments

    def test_deterministic(self):
        """Same complexity should produce same state."""
        a = build_initial_state("medium")
        b = build_initial_state("medium")
        assert len(a) == len(b)
        for ca, cb in zip(a, b):
            assert ca["text"] == cb["text"]


# ===========================================================================
# Metadata counting
# ===========================================================================


class TestCountMetadataConstructs:
    """Tests for counting format-specific metadata constructs."""

    def test_yaml_comments(self):
        text = "# top comment\nkey: value  # inline\n"
        result = count_metadata_constructs(text, "yaml")
        assert result["comments"] >= 2

    def test_toml_comments(self):
        text = '# header\nkey = "value"  # inline\n'
        result = count_metadata_constructs(text, "toml")
        assert result["comments"] >= 2

    def test_json_no_comments(self):
        text = '{"key": "value"}'
        result = count_metadata_constructs(text, "json")
        assert result["comments"] == 0

    def test_returns_dict_with_total(self):
        text = "# comment\nkey: value\n"
        result = count_metadata_constructs(text, "yaml")
        assert "comments" in result
        assert "anchors" in result
        assert "typed_temporals" in result
        assert "total" in result
        assert result["total"] >= result["comments"]

    def test_yaml_anchors(self):
        text = "defaults: &defaults\n  timeout: 30\ndev:\n  <<: *defaults\n"
        result = count_metadata_constructs(text, "yaml")
        assert result["anchors"] >= 1

    def test_empty_text(self):
        result = count_metadata_constructs("", "yaml")
        assert result["total"] == 0


class TestCountAllMetadata:
    """Tests for counting metadata across all files in a state."""

    def test_returns_dict(self):
        state = build_initial_state("small")
        result = count_all_metadata(state)
        assert isinstance(result, dict)

    def test_has_totals(self):
        state = build_initial_state("medium")
        result = count_all_metadata(state)
        assert "total_comments" in result
        assert "total_anchors" in result
        assert "total_typed_temporals" in result
        assert "grand_total" in result

    def test_medium_state_has_nonzero_comments(self):
        """Medium state should have substantial comments."""
        state = build_initial_state("medium")
        result = count_all_metadata(state)
        assert result["total_comments"] > 0

    def test_per_file_breakdown(self):
        state = build_initial_state("small")
        result = count_all_metadata(state)
        assert "per_file" in result
        assert len(result["per_file"]) == len(state)


# ===========================================================================
# Serialization / Deserialization round-trip
# ===========================================================================


class TestSerializeState:
    """Tests for serializing agent state."""

    @pytest.mark.parametrize("fmt", STATE_FORMATS)
    def test_returns_bytes(self, fmt):
        state = build_initial_state("small")
        data = serialize_state(state, fmt)
        assert isinstance(data, bytes)
        assert len(data) > 0

    @pytest.mark.parametrize("fmt", STATE_FORMATS)
    def test_deterministic(self, fmt):
        state = build_initial_state("small")
        a = serialize_state(state, fmt)
        b = serialize_state(state, fmt)
        assert a == b


class TestDeserializeState:
    """Tests for deserializing agent state."""

    @pytest.mark.parametrize("fmt", STATE_FORMATS)
    def test_returns_list(self, fmt):
        state = build_initial_state("small")
        data = serialize_state(state, fmt)
        result = deserialize_state(data, fmt)
        assert isinstance(result, list)

    @pytest.mark.parametrize("fmt", STATE_FORMATS)
    def test_preserves_file_count(self, fmt):
        state = build_initial_state("small")
        data = serialize_state(state, fmt)
        result = deserialize_state(data, fmt)
        assert len(result) == len(state)

    @pytest.mark.parametrize("fmt", STATE_FORMATS)
    def test_preserves_filenames(self, fmt):
        state = build_initial_state("small")
        data = serialize_state(state, fmt)
        result = deserialize_state(data, fmt)
        original_names = {c["filename"] for c in state}
        result_names = {c["filename"] for c in result}
        assert original_names == result_names


class TestRoundTripFidelity:
    """Tests for metadata preservation through serialize→deserialize."""

    def test_cdxf_preserves_comments(self):
        """CDXF must preserve all comments through round-trip."""
        state = build_initial_state("medium")
        original = count_all_metadata(state)

        data = serialize_state(state, "cdxf")
        restored = deserialize_state(data, "cdxf")
        after = count_all_metadata(restored)

        assert after["total_comments"] >= original["total_comments"] * 0.95, (
            f"CDXF lost comments: {original['total_comments']} -> "
            f"{after['total_comments']}"
        )

    def test_json_mega_loses_comments(self):
        """JSON mega-JSON must lose YAML/TOML comments."""
        state = build_initial_state("medium")
        original = count_all_metadata(state)
        assert original["total_comments"] > 0  # precondition

        data = serialize_state(state, "json_mega")
        restored = deserialize_state(data, "json_mega")
        after = count_all_metadata(restored)

        assert after["total_comments"] == 0, (
            f"JSON mega should lose all comments but kept "
            f"{after['total_comments']}"
        )

    def test_pickle_loses_comments(self):
        """Pickle must lose comments (parses to Python objects)."""
        state = build_initial_state("medium")
        original = count_all_metadata(state)
        assert original["total_comments"] > 0

        data = serialize_state(state, "pickle")
        restored = deserialize_state(data, "pickle")
        after = count_all_metadata(restored)

        assert after["total_comments"] == 0

    def test_tar_gz_preserves_comments(self):
        """tar.gz preserves original text files (comments survive)."""
        state = build_initial_state("medium")
        original = count_all_metadata(state)

        data = serialize_state(state, "tar_gz")
        restored = deserialize_state(data, "tar_gz")
        after = count_all_metadata(restored)

        assert after["total_comments"] == original["total_comments"]


# ===========================================================================
# Session simulation — modify + add
# ===========================================================================


class TestModifyOneConfig:
    """Tests for the per-session modification step."""

    def test_returns_list(self):
        state = build_initial_state("small")
        result = modify_one_config(state, session_k=1)
        assert isinstance(result, list)

    def test_preserves_file_count(self):
        state = build_initial_state("small")
        result = modify_one_config(state, session_k=1)
        assert len(result) == len(state)

    def test_modifies_exactly_one_file(self):
        state = build_initial_state("medium")
        result = modify_one_config(state, session_k=1)
        changed = sum(
            1 for a, b in zip(state, result)
            if a["text"] != b["text"]
        )
        assert changed == 1

    def test_different_sessions_can_modify_different_files(self):
        """Ensure different K values modify different files (for tar.gz degradation)."""
        state = build_initial_state("medium")
        r1 = modify_one_config(state, session_k=1)
        r2 = modify_one_config(state, session_k=2)
        changed_1 = [i for i, (a, b) in enumerate(zip(state, r1))
                     if a["text"] != b["text"]]
        changed_2 = [i for i, (a, b) in enumerate(zip(state, r2))
                     if a["text"] != b["text"]]
        # Different sessions should modify different files (round-robin)
        if len(state) > 1:
            assert changed_1 != changed_2


class TestAddResultFile:
    """Tests for adding a result file each session."""

    def test_adds_one_file(self):
        state = build_initial_state("small")
        result = add_result_file(state, session_k=1)
        assert len(result) == len(state) + 1

    def test_result_file_has_comments(self):
        """Added result files should contain YAML comments (for degradation tracking)."""
        state = build_initial_state("small")
        result = add_result_file(state, session_k=1)
        added = result[-1]
        assert added["format"] == "yaml"
        meta = count_metadata_constructs(added["text"], added["format"])
        assert meta["comments"] > 0

    def test_different_sessions_add_different_files(self):
        state = build_initial_state("small")
        r1 = add_result_file(state, session_k=1)
        r2 = add_result_file(state, session_k=2)
        assert r1[-1]["filename"] != r2[-1]["filename"]


# ===========================================================================
# Degradation loop — the scientific core
# ===========================================================================


class TestRunDegradationLoop:
    """Tests for the full degradation simulation loop."""

    def test_returns_list(self):
        result = run_degradation_loop("small", "cdxf", max_k=2)
        assert isinstance(result, list)

    def test_one_entry_per_k(self):
        result = run_degradation_loop("small", "cdxf", max_k=3)
        ks = [r["k"] for r in result]
        assert ks == [0, 1, 2, 3]

    def test_entry_structure(self):
        result = run_degradation_loop("small", "cdxf", max_k=1)
        for entry in result:
            assert "k" in entry
            assert "total_comments" in entry
            assert "original_comments_surviving" in entry
            assert "state_size_bytes" in entry
            assert "n_files" in entry

    def test_k0_is_initial_state(self):
        """K=0 should reflect the initial state metadata."""
        result = run_degradation_loop("medium", "cdxf", max_k=1)
        k0 = result[0]
        assert k0["k"] == 0
        assert k0["total_comments"] > 0

    def test_cdxf_preserves_comments_across_boundaries(self):
        """CDXF: comments should NOT decrease across boundaries."""
        result = run_degradation_loop("medium", "cdxf", max_k=5)
        initial_comments = result[0]["original_comments_surviving"]
        for entry in result[1:]:
            assert entry["original_comments_surviving"] >= initial_comments * 0.95, (
                f"CDXF lost comments at K={entry['k']}: "
                f"{initial_comments} -> {entry['original_comments_surviving']}"
            )

    def test_json_mega_loses_all_comments_at_k1(self):
        """JSON mega: all comments should be lost after first boundary."""
        result = run_degradation_loop("medium", "json_mega", max_k=2)
        assert result[0]["total_comments"] > 0  # precondition
        assert result[1]["original_comments_surviving"] == 0

    def test_pickle_loses_all_comments_at_k1(self):
        """Pickle: all comments should be lost after first boundary."""
        result = run_degradation_loop("medium", "pickle", max_k=2)
        assert result[0]["total_comments"] > 0
        assert result[1]["original_comments_surviving"] == 0

    def test_file_count_grows(self):
        """Each session adds a result file, so file count should grow."""
        result = run_degradation_loop("small", "cdxf", max_k=3)
        initial_files = result[0]["n_files"]
        assert result[-1]["n_files"] == initial_files + 3

    def test_state_size_recorded(self):
        result = run_degradation_loop("small", "cdxf", max_k=1)
        for entry in result:
            assert entry["state_size_bytes"] > 0


class TestDegradationHypothesis:
    """Validate the core hypothesis: CDXF = zero loss, lossy = degradation."""

    def test_cdxf_flat_line(self):
        """CDXF: surviving fraction should be ~1.0 at all K."""
        result = run_degradation_loop("medium", "cdxf", max_k=10)
        initial = result[0]["original_comments_surviving"]
        for entry in result[1:]:
            frac = entry["original_comments_surviving"] / max(1, initial)
            assert frac >= 0.95, (
                f"CDXF surviving fraction {frac:.2f} at K={entry['k']}"
            )

    def test_json_mega_immediate_total_loss(self):
        """JSON mega: surviving fraction = 0 for all K >= 1."""
        result = run_degradation_loop("medium", "json_mega", max_k=5)
        for entry in result[1:]:
            assert entry["original_comments_surviving"] == 0

    def test_lossy_vs_lossless_diverge(self):
        """CDXF and JSON mega should produce different outcomes."""
        cdxf = run_degradation_loop("medium", "cdxf", max_k=5)
        jmeg = run_degradation_loop("medium", "json_mega", max_k=5)
        # At K=5, CDXF should have comments, JSON should have 0
        assert cdxf[-1]["original_comments_surviving"] > 0
        assert jmeg[-1]["original_comments_surviving"] == 0


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
        assert results["experiment"] == "EXP-013"

    def test_has_timestamp(self, results):
        assert "timestamp" in results

    def test_has_degradation_results(self, results):
        assert "degradation" in results

    def test_has_all_state_formats(self, results):
        deg = results["degradation"]
        for sf in STATE_FORMATS:
            assert any(sf in key for key in deg), (
                f"Missing state format {sf} in results"
            )

    def test_has_summary(self, results):
        assert "summary" in results
