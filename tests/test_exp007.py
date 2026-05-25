"""Tests for EXP-007: Cross-Framework Configuration Migration.

Tests the migration scenario corpus, metadata counting, direct conversion,
CDXF hub conversion, and migration measurement functions.

TDD: These tests are written BEFORE the implementation.
"""

import json
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp007 import (
    build_migration_scenarios,
    count_metadata,
    direct_convert,
    cdxf_hub_convert,
    measure_migration,
    compute_converter_counts,
    SUPPORTED_FORMATS,
)


# ---------------------------------------------------------------------------
# Scenario corpus
# ---------------------------------------------------------------------------

class TestBuildMigrationScenarios:
    def test_returns_list(self):
        scenarios = build_migration_scenarios()
        assert isinstance(scenarios, list)

    def test_has_8_scenarios(self):
        """Protocol specifies exactly 8 migration scenarios."""
        scenarios = build_migration_scenarios()
        assert len(scenarios) == 8

    def test_scenario_required_fields(self):
        scenarios = build_migration_scenarios()
        required = {"name", "source_format", "target_format",
                    "source_text", "motivation"}
        for s in scenarios:
            missing = required - set(s.keys())
            assert not missing, f"Scenario {s.get('name', '?')} missing: {missing}"

    def test_source_text_not_empty(self):
        scenarios = build_migration_scenarios()
        for s in scenarios:
            assert isinstance(s["source_text"], str)
            assert len(s["source_text"]) > 0, f"Empty source for {s['name']}"

    def test_formats_are_valid(self):
        scenarios = build_migration_scenarios()
        valid = {"json", "yaml", "toml", "xml"}
        for s in scenarios:
            assert s["source_format"] in valid, (
                f"Invalid source_format '{s['source_format']}' in {s['name']}"
            )
            assert s["target_format"] in valid, (
                f"Invalid target_format '{s['target_format']}' in {s['name']}"
            )

    def test_source_target_differ(self):
        """Each scenario must convert between different formats."""
        scenarios = build_migration_scenarios()
        for s in scenarios:
            assert s["source_format"] != s["target_format"], (
                f"Same source/target in {s['name']}"
            )

    def test_has_yaml_to_json_scenario(self):
        """Protocol requires PyTorch Lightning YAML -> HF JSON."""
        scenarios = build_migration_scenarios()
        yaml_to_json = [s for s in scenarios
                        if s["source_format"] == "yaml"
                        and s["target_format"] == "json"]
        assert len(yaml_to_json) >= 1

    def test_has_toml_source(self):
        """Protocol requires at least one TOML source (pyproject.toml)."""
        scenarios = build_migration_scenarios()
        toml_sources = [s for s in scenarios if s["source_format"] == "toml"]
        assert len(toml_sources) >= 1

    def test_has_xml_source(self):
        """Protocol requires ONNX XML metadata scenario."""
        scenarios = build_migration_scenarios()
        xml_sources = [s for s in scenarios if s["source_format"] == "xml"]
        assert len(xml_sources) >= 1

    def test_scenarios_have_metadata(self):
        """Source configs should contain metadata (comments, etc.)
        so we can measure survival."""
        scenarios = build_migration_scenarios()
        metadata_count = 0
        for s in scenarios:
            meta = count_metadata(s["source_text"], s["source_format"])
            if meta["total"] > 0:
                metadata_count += 1
        # At least half should have metadata
        assert metadata_count >= 4, (
            f"Only {metadata_count}/8 scenarios have metadata"
        )

    def test_source_text_is_parseable(self):
        """All source texts must actually parse in their declared format."""
        scenarios = build_migration_scenarios()
        for s in scenarios:
            # If count_metadata doesn't raise, it parsed fine
            meta = count_metadata(s["source_text"], s["source_format"])
            assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# Metadata counting
# ---------------------------------------------------------------------------

class TestCountMetadata:
    def test_yaml_comments(self):
        text = "# This is a comment\nkey: value\n# Another comment\n"
        meta = count_metadata(text, "yaml")
        assert meta["comments"] >= 2

    def test_yaml_anchors(self):
        text = "defaults: &defaults\n  lr: 0.001\nrun1:\n  <<: *defaults\n"
        meta = count_metadata(text, "yaml")
        assert meta["anchors"] >= 1

    def test_yaml_no_metadata(self):
        text = "key: value\nother: 42\n"
        meta = count_metadata(text, "yaml")
        assert meta["comments"] == 0
        assert meta["anchors"] == 0

    def test_toml_comments(self):
        text = "# TOML comment\n[section]\nkey = 'value'\n"
        meta = count_metadata(text, "toml")
        assert meta["comments"] >= 1

    def test_toml_temporal(self):
        text = '[dates]\ncreated = 2026-01-15T10:30:00\n'
        meta = count_metadata(text, "toml")
        assert meta["temporal_values"] >= 1

    def test_xml_comments(self):
        text = '<?xml version="1.0"?>\n<!-- A comment -->\n<root>data</root>\n'
        meta = count_metadata(text, "xml")
        assert meta["comments"] >= 1

    def test_xml_processing_instructions(self):
        text = '<?xml version="1.0"?>\n<?custom instruction?>\n<root/>\n'
        meta = count_metadata(text, "xml")
        assert meta["processing_instructions"] >= 1

    def test_json_no_comments(self):
        """JSON has no comment syntax — metadata count should be 0."""
        text = '{"key": "value", "num": 42}'
        meta = count_metadata(text, "json")
        assert meta["total"] == 0

    def test_returns_total(self):
        text = "# comment\nkey: value\n"
        meta = count_metadata(text, "yaml")
        assert "total" in meta
        assert meta["total"] >= meta["comments"]

    def test_invalid_format_raises(self):
        with pytest.raises((ValueError, KeyError)):
            count_metadata("data", "invalid_format")


# ---------------------------------------------------------------------------
# Direct conversion
# ---------------------------------------------------------------------------

class TestDirectConvert:
    def test_yaml_to_json_data_preserved(self):
        source = "learning_rate: 0.001\nepochs: 10\nmodel: bert-base\n"
        result = direct_convert(source, "yaml", "json")
        assert result["success"]
        parsed = json.loads(result["output_text"])
        assert parsed["learning_rate"] == 0.001
        assert parsed["epochs"] == 10
        assert parsed["model"] == "bert-base"

    def test_yaml_to_json_comments_lost(self):
        source = "# Learning rate tuned via grid search\nlr: 0.001\n"
        result = direct_convert(source, "yaml", "json")
        assert result["success"]
        # JSON cannot represent comments
        assert "grid search" not in result["output_text"] or \
            result["metadata_survived"]["comments"] == 0

    def test_toml_to_json_data_preserved(self):
        source = '[training]\nlr = 0.001\nepochs = 10\n'
        result = direct_convert(source, "toml", "json")
        assert result["success"]
        parsed = json.loads(result["output_text"])
        assert parsed["training"]["lr"] == 0.001

    def test_xml_to_json_data_preserved(self):
        source = '<?xml version="1.0"?>\n<config><lr>0.001</lr></config>'
        result = direct_convert(source, "xml", "json")
        assert result["success"]
        assert "0.001" in result["output_text"]

    def test_json_to_yaml_data_preserved(self):
        source = '{"lr": 0.001, "epochs": 10}'
        result = direct_convert(source, "json", "yaml")
        assert result["success"]
        assert "0.001" in result["output_text"]

    def test_json_to_toml_data_preserved(self):
        source = '{"training": {"lr": 0.001, "epochs": 10}}'
        result = direct_convert(source, "json", "toml")
        assert result["success"]
        assert "0.001" in result["output_text"]

    def test_result_has_required_fields(self):
        source = "key: value\n"
        result = direct_convert(source, "yaml", "json")
        required = {"success", "output_text", "metadata_survived",
                    "conversion_time_ns"}
        assert required.issubset(set(result.keys()))

    def test_unsupported_conversion_handled(self):
        """Converting XML to TOML may not be directly supported;
        the function should handle this gracefully."""
        source = '<?xml version="1.0"?>\n<root><key>val</key></root>'
        result = direct_convert(source, "xml", "toml")
        # Should either succeed or fail gracefully (no exception)
        assert "success" in result


# ---------------------------------------------------------------------------
# CDXF hub conversion
# ---------------------------------------------------------------------------

class TestCdxfHubConvert:
    def test_yaml_to_json_data_preserved(self):
        source = "lr: 0.001\nepochs: 10\nmodel: bert-base\n"
        result = cdxf_hub_convert(source, "yaml", "json")
        assert result["success"]
        parsed = json.loads(result["output_text"])
        assert parsed["lr"] == 0.001
        assert parsed["epochs"] == 10

    def test_yaml_to_json_via_cdxf_roundtrips(self):
        """CDXF hub should produce valid output."""
        source = "key: value\nnum: 42\n"
        result = cdxf_hub_convert(source, "yaml", "json")
        assert result["success"]
        parsed = json.loads(result["output_text"])
        assert parsed["key"] == "value"
        assert parsed["num"] == 42

    def test_metadata_preserved_in_cdxf(self):
        """CDXF should preserve comments internally even if target
        format can't represent them."""
        source = "# Important comment\nlr: 0.001\n"
        result = cdxf_hub_convert(source, "yaml", "json")
        assert result["success"]
        # The CDXF intermediate should have preserved the comment
        assert result["cdxf_metadata"]["comments"] >= 1

    def test_toml_to_json_via_cdxf(self):
        source = '# Comment\n[section]\nkey = "value"\n'
        result = cdxf_hub_convert(source, "toml", "json")
        assert result["success"]
        assert result["cdxf_metadata"]["comments"] >= 1

    def test_json_to_yaml_via_cdxf(self):
        source = '{"lr": 0.001, "epochs": 10}'
        result = cdxf_hub_convert(source, "json", "yaml")
        assert result["success"]
        assert "0.001" in result["output_text"]

    def test_xml_to_json_via_cdxf(self):
        source = '<?xml version="1.0"?>\n<!-- Config -->\n<config><lr>0.001</lr></config>'
        result = cdxf_hub_convert(source, "xml", "json")
        assert result["success"]

    def test_result_has_required_fields(self):
        source = "key: value\n"
        result = cdxf_hub_convert(source, "yaml", "json")
        required = {"success", "output_text", "cdxf_metadata",
                    "cdxf_size_bytes", "conversion_time_ns"}
        assert required.issubset(set(result.keys()))

    def test_cdxf_size_is_positive(self):
        source = "key: value\nnum: 42\n"
        result = cdxf_hub_convert(source, "yaml", "json")
        assert result["success"]
        assert result["cdxf_size_bytes"] > 0


# ---------------------------------------------------------------------------
# Migration measurement
# ---------------------------------------------------------------------------

class TestMeasureMigration:
    def test_returns_result_dict(self):
        scenarios = build_migration_scenarios()
        result = measure_migration(scenarios[0])
        assert isinstance(result, dict)

    def test_result_has_both_methods(self):
        scenarios = build_migration_scenarios()
        result = measure_migration(scenarios[0])
        assert "direct" in result
        assert "cdxf_hub" in result

    def test_result_has_scenario_info(self):
        scenarios = build_migration_scenarios()
        result = measure_migration(scenarios[0])
        assert "name" in result
        assert "source_format" in result
        assert "target_format" in result

    def test_result_has_source_metadata_count(self):
        scenarios = build_migration_scenarios()
        result = measure_migration(scenarios[0])
        assert "source_metadata" in result

    def test_cdxf_preserves_more_metadata_than_direct(self):
        """Core hypothesis: CDXF hub preserves metadata that direct
        conversion loses."""
        scenarios = build_migration_scenarios()
        # Find a scenario with comments in source
        for s in scenarios:
            meta = count_metadata(s["source_text"], s["source_format"])
            if meta["comments"] > 0 and s["target_format"] == "json":
                result = measure_migration(s)
                cdxf_meta = result["cdxf_hub"].get("cdxf_metadata", {})
                # CDXF preserves comments in the intermediate representation
                assert cdxf_meta.get("comments", 0) > 0, (
                    f"CDXF should preserve comments for {s['name']}"
                )
                break
        else:
            pytest.skip("No YAML/TOML→JSON scenario with comments found")


# ---------------------------------------------------------------------------
# Converter count calculation
# ---------------------------------------------------------------------------

class TestComputeConverterCounts:
    def test_2_formats(self):
        result = compute_converter_counts(2)
        assert result["direct"] == 2  # 2*(2-1) = 2
        assert result["cdxf_hub"] == 4  # 2*2 = 4

    def test_3_formats(self):
        result = compute_converter_counts(3)
        assert result["direct"] == 6  # 3*2 = 6
        assert result["cdxf_hub"] == 6  # 2*3 = 6

    def test_4_formats(self):
        result = compute_converter_counts(4)
        assert result["direct"] == 12  # 4*3 = 12
        assert result["cdxf_hub"] == 8  # 2*4 = 8

    def test_5_formats(self):
        result = compute_converter_counts(5)
        assert result["direct"] == 20  # 5*4 = 20
        assert result["cdxf_hub"] == 10  # 2*5 = 10

    def test_hub_cheaper_at_3_plus(self):
        """For N >= 3, CDXF hub requires fewer converters."""
        for n in range(3, 10):
            result = compute_converter_counts(n)
            assert result["cdxf_hub"] <= result["direct"], (
                f"Hub should be cheaper at N={n}"
            )

    def test_crossover_at_3(self):
        """At N=3, direct == hub (breakeven). Above 3, hub wins."""
        r2 = compute_converter_counts(2)
        r3 = compute_converter_counts(3)
        r4 = compute_converter_counts(4)
        assert r2["direct"] < r2["cdxf_hub"]  # 2 < 4, direct cheaper
        assert r3["direct"] == r3["cdxf_hub"]  # 6 == 6, breakeven
        assert r4["direct"] > r4["cdxf_hub"]  # 12 > 8, hub cheaper

    def test_returns_required_fields(self):
        result = compute_converter_counts(4)
        assert "direct" in result
        assert "cdxf_hub" in result
        assert "n_formats" in result
        assert result["n_formats"] == 4


# ---------------------------------------------------------------------------
# Supported formats constant
# ---------------------------------------------------------------------------

class TestSupportedFormats:
    def test_has_four_formats(self):
        assert len(SUPPORTED_FORMATS) == 4

    def test_contains_json_yaml_xml_toml(self):
        assert "json" in SUPPORTED_FORMATS
        assert "yaml" in SUPPORTED_FORMATS
        assert "xml" in SUPPORTED_FORMATS
        assert "toml" in SUPPORTED_FORMATS
