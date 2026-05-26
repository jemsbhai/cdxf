"""Tests for the shared config corpus module.

Verifies config comment counts, data integrity checking,
and timing utilities used across EXP-015, EXP-016, EXP-017.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.config_corpus import (
    YAML_CONFIGS,
    XML_CONFIG,
    TOML_CONFIG,
    ROLE_MODIFICATIONS,
    EXPECTED_VALUES_AFTER_4AGENT,
    EXPECTED_VALUES_AFTER_6AGENT,
    count_config_metadata,
    verify_data_integrity,
    Timer,
)


# ===========================================================================
# YAML config corpus
# ===========================================================================


class TestYAMLConfigs:
    """Verify YAML configs have expected properties."""

    @pytest.mark.parametrize("size", ["small", "medium", "large", "xlarge"])
    def test_has_required_fields(self, size):
        cfg = YAML_CONFIGS[size]
        assert "name" in cfg
        assert "text" in cfg
        assert "expected_comments" in cfg
        assert "format" in cfg
        assert cfg["format"] == "yaml"

    @pytest.mark.parametrize("size", ["small", "medium", "large", "xlarge"])
    def test_comment_counts_match(self, size):
        """Verify declared comment count matches actual count."""
        cfg = YAML_CONFIGS[size]
        meta = count_config_metadata(cfg["text"], "yaml")
        assert meta["comments"] == cfg["expected_comments"], (
            f"{size}: counted {meta['comments']}, "
            f"expected {cfg['expected_comments']}"
        )

    @pytest.mark.parametrize("size", ["small", "medium", "large", "xlarge"])
    def test_is_valid_yaml(self, size):
        """Each config must be parseable YAML."""
        import yaml
        cfg = YAML_CONFIGS[size]
        parsed = yaml.safe_load(cfg["text"])
        assert isinstance(parsed, dict)

    def test_sizes_are_graduated(self):
        """Comment counts increase with size."""
        counts = [
            YAML_CONFIGS[s]["expected_comments"]
            for s in ["small", "medium", "large", "xlarge"]
        ]
        assert counts == sorted(counts)
        assert counts[0] < counts[-1]

    def test_large_identical_to_exp015(self):
        """Large config must match EXP-015 for cross-experiment comparability."""
        from benchmarks.src.run_exp015 import (
            build_initial_config as build_015,
        )
        text_015, _ = build_015()
        assert YAML_CONFIGS["large"]["text"] == text_015


# ===========================================================================
# Cross-format configs
# ===========================================================================


class TestCrossFormatConfigs:
    """Verify XML and TOML configs."""

    def test_xml_comment_count(self):
        meta = count_config_metadata(XML_CONFIG["text"], "xml")
        assert meta["comments"] == XML_CONFIG["expected_comments"]

    def test_toml_comment_count(self):
        meta = count_config_metadata(TOML_CONFIG["text"], "toml")
        assert meta["comments"] == TOML_CONFIG["expected_comments"]

    def test_toml_is_valid(self):
        import tomlkit
        parsed = tomlkit.parse(TOML_CONFIG["text"])
        assert "model" in parsed


# ===========================================================================
# Comment counting
# ===========================================================================


class TestCountConfigMetadata:
    """Verify the shared comment counter."""

    def test_yaml_full_line(self):
        meta = count_config_metadata("# comment\nkey: val\n", "yaml")
        assert meta["comments"] == 1

    def test_yaml_inline(self):
        meta = count_config_metadata("key: val  # inline\n", "yaml")
        assert meta["comments"] == 1

    def test_yaml_no_comments(self):
        meta = count_config_metadata("key: val\n", "yaml")
        assert meta["comments"] == 0

    def test_xml_comments(self):
        meta = count_config_metadata(
            "<!-- A -->\n<x>1</x>\n<!-- B -->", "xml"
        )
        assert meta["comments"] == 2

    def test_toml_comments(self):
        meta = count_config_metadata(
            "# Comment\nkey = 1\n# Another\n", "toml"
        )
        assert meta["comments"] == 2

    def test_empty_string(self):
        meta = count_config_metadata("", "yaml")
        assert meta["comments"] == 0


# ===========================================================================
# Data integrity verification
# ===========================================================================


class TestVerifyDataIntegrity:
    """Verify the data integrity checker."""

    def test_correct_values_pass(self):
        config = (
            "data:\n  num_proc: 12\n"
            "training:\n  num_epochs: 4\n"
            "evaluation:\n  min_accuracy: 0.5\n"
            "deployment:\n  max_num_seqs: 80\n"
        )
        result = verify_data_integrity(config, EXPECTED_VALUES_AFTER_4AGENT)
        assert result["passed"] is True
        assert result["checks"] == 4
        assert len(result["failures"]) == 0

    def test_wrong_value_fails(self):
        config = (
            "data:\n  num_proc: 8\n"    # wrong — should be 12
            "training:\n  num_epochs: 4\n"
            "evaluation:\n  min_accuracy: 0.5\n"
            "deployment:\n  max_num_seqs: 80\n"
        )
        result = verify_data_integrity(config, EXPECTED_VALUES_AFTER_4AGENT)
        assert result["passed"] is False
        assert len(result["failures"]) >= 1

    def test_missing_key_fails(self):
        config = "training:\n  num_epochs: 4\n"
        result = verify_data_integrity(
            config, {"num_epochs": 4, "missing_key": 99}
        )
        assert result["passed"] is False

    def test_invalid_yaml_fails(self):
        result = verify_data_integrity(":::bad", {"a": 1})
        assert result["passed"] is False

    def test_6agent_expected_values(self):
        """6-agent expected values are a superset of 4-agent."""
        for k, v in EXPECTED_VALUES_AFTER_4AGENT.items():
            assert k in EXPECTED_VALUES_AFTER_6AGENT
            assert EXPECTED_VALUES_AFTER_6AGENT[k] == v


# ===========================================================================
# Timer utility
# ===========================================================================


class TestTimer:
    """Verify the timing utility."""

    def test_measure_records_time(self):
        timer = Timer()
        with timer.measure("test"):
            _ = sum(range(1000))
        assert "test" in timer.results
        assert timer.results["test"] > 0

    def test_measure_n_records_mean(self):
        timer = Timer()
        t = timer.measure_n("sum", lambda: sum(range(1000)), n=5)
        assert t > 0
        assert "sum" in timer.results

    def test_summary(self):
        timer = Timer()
        with timer.measure("a"):
            pass
        with timer.measure("b"):
            pass
        s = timer.summary()
        assert "a" in s
        assert "b" in s


# ===========================================================================
# Role modifications
# ===========================================================================


class TestRoleModifications:
    """Verify shared role modifications are consistent."""

    def test_has_all_roles(self):
        expected = {"data_curator", "trainer", "evaluator", "deployer",
                    "monitor", "reviewer"}
        assert set(ROLE_MODIFICATIONS.keys()) == expected

    def test_each_role_has_patterns(self):
        for role, mods in ROLE_MODIFICATIONS.items():
            assert len(mods) >= 1, f"{role} has no modifications"
