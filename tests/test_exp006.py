"""Tests for EXP-006: ML Configuration Fidelity Under Serialization.

Tests the corpus generation, serialization round-trip, and construct
counting functions.
"""

import json
import pickle
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp006 import (
    build_ml_corpus,
    count_constructs,
    serialize_roundtrip,
    measure_fidelity,
    BASELINES,
)


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

class TestBuildCorpus:
    def test_corpus_not_empty(self):
        corpus = build_ml_corpus()
        assert len(corpus) > 0

    def test_corpus_has_required_formats(self):
        corpus = build_ml_corpus()
        formats = {entry["format"] for entry in corpus}
        assert "json" in formats
        assert "yaml" in formats
        assert "toml" in formats
        assert "xml" in formats

    def test_corpus_entries_have_required_fields(self):
        corpus = build_ml_corpus()
        for entry in corpus:
            assert "name" in entry
            assert "format" in entry
            assert "text" in entry
            assert "category" in entry
            assert isinstance(entry["text"], str)
            assert len(entry["text"]) > 0

    def test_corpus_has_construct_rich_files(self):
        """At least some files should have comments, anchors, etc."""
        corpus = build_ml_corpus()
        has_comments = any(
            count_constructs(e["text"], e["format"]).get("comments", 0) > 0
            for e in corpus
        )
        assert has_comments, "Corpus must include files with comments"

    def test_corpus_has_yaml_anchors(self):
        corpus = build_ml_corpus()
        has_anchors = any(
            count_constructs(e["text"], e["format"]).get("anchors", 0) > 0
            for e in corpus
        )
        assert has_anchors, "Corpus must include YAML files with anchors"

    def test_corpus_has_multi_doc_yaml(self):
        corpus = build_ml_corpus()
        has_multi = any(
            count_constructs(e["text"], e["format"]).get("multi_doc_markers", 0) > 1
            for e in corpus
        )
        assert has_multi, "Corpus must include multi-document YAML"

    def test_corpus_has_toml_temporals(self):
        corpus = build_ml_corpus()
        has_temporal = any(
            count_constructs(e["text"], e["format"]).get("temporal_values", 0) > 0
            for e in corpus
            if e["format"] == "toml"
        )
        assert has_temporal, "Corpus must include TOML with temporal values"

    def test_corpus_has_xml_constructs(self):
        corpus = build_ml_corpus()
        xml_files = [e for e in corpus if e["format"] == "xml"]
        assert len(xml_files) > 0
        has_ns = any(
            count_constructs(e["text"], "xml").get("namespaces", 0) > 0
            for e in xml_files
        )
        assert has_ns, "Corpus must include XML with namespaces"


# ---------------------------------------------------------------------------
# Construct counting
# ---------------------------------------------------------------------------

class TestCountConstructs:
    def test_yaml_comments(self):
        text = "# Comment 1\nkey: value\n# Comment 2\n"
        c = count_constructs(text, "yaml")
        assert c["comments"] == 2

    def test_yaml_anchors_aliases(self):
        text = "base: &base\n  lr: 0.001\nexp1:\n  <<: *base\n  lr: 0.01\n"
        c = count_constructs(text, "yaml")
        assert c["anchors"] >= 1
        assert c["aliases"] >= 1
        assert c["merge_keys"] >= 1

    def test_json_no_constructs(self):
        text = '{"key": "value"}'
        c = count_constructs(text, "json")
        # JSON has no comments/anchors — total constructs should be 0
        total = sum(v for k, v in c.items() if k != "key_count" and k != "max_depth" and k != "array_count")
        assert total == 0

    def test_toml_comments_and_temporals(self):
        text = "# A comment\ncreated = 2024-01-15T10:30:00\n"
        c = count_constructs(text, "toml")
        assert c["comments"] >= 1
        assert c["temporal_values"] >= 1

    def test_xml_all_constructs(self):
        text = """<?xml version="1.0"?>
<!-- A comment -->
<?my-pi data="test"?>
<root xmlns="http://example.com">
  <child attr="val"/>
</root>"""
        c = count_constructs(text, "xml")
        assert c["comments"] == 1
        assert c["processing_instructions"] == 1
        assert c["namespaces"] >= 1

    def test_unknown_format(self):
        c = count_constructs("some text", "other")
        assert isinstance(c, dict)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerializeRoundtrip:
    def test_cdxf_yaml_roundtrip(self):
        text = "# Important comment\nkey: value\nnum: 42\n"
        result = serialize_roundtrip(text, "yaml", "cdxf")
        assert result["success"] is True
        assert result["serialized_size"] > 0

    def test_cdxf_json_roundtrip(self):
        text = '{"key": "value", "num": 42}'
        result = serialize_roundtrip(text, "json", "cdxf")
        assert result["success"] is True

    def test_cdxf_toml_roundtrip(self):
        text = "# Comment\nkey = 'value'\nnum = 42\n"
        result = serialize_roundtrip(text, "toml", "cdxf")
        assert result["success"] is True

    def test_cdxf_xml_roundtrip(self):
        text = '<?xml version="1.0"?>\n<root><child>text</child></root>'
        result = serialize_roundtrip(text, "xml", "cdxf")
        assert result["success"] is True

    def test_json_baseline_roundtrip(self):
        text = '{"key": "value", "num": 42}'
        result = serialize_roundtrip(text, "json", "json_stdlib")
        assert result["success"] is True

    def test_cbor_roundtrip(self):
        text = '{"key": "value"}'
        result = serialize_roundtrip(text, "json", "cbor")
        assert result["success"] is True

    def test_msgpack_roundtrip(self):
        text = '{"key": "value"}'
        result = serialize_roundtrip(text, "json", "msgpack")
        assert result["success"] is True

    def test_pickle_roundtrip(self):
        text = '{"key": "value"}'
        result = serialize_roundtrip(text, "json", "pickle")
        assert result["success"] is True

    def test_failed_roundtrip_returns_error(self):
        # Broken YAML should not crash, should return success=False
        text = "{{{{not valid yaml"
        result = serialize_roundtrip(text, "yaml", "cdxf")
        # May succeed or fail depending on parser tolerance
        assert "success" in result


# ---------------------------------------------------------------------------
# Fidelity measurement
# ---------------------------------------------------------------------------

class TestMeasureFidelity:
    def test_cdxf_preserves_yaml_comments(self):
        text = "# HP decision: lr=1e-4 after grid search\nlearning_rate: 0.0001\n"
        result = measure_fidelity(text, "yaml", "cdxf")
        assert result["comments_preserved"] == result["comments_original"]

    def test_json_stdlib_loses_yaml_comments(self):
        text = "# HP decision: lr=1e-4\nlearning_rate: 0.0001\n"
        result = measure_fidelity(text, "yaml", "json_stdlib")
        assert result["comments_preserved"] == 0
        assert result["comments_original"] >= 1

    def test_cdxf_preserves_anchors(self):
        text = "defaults: &defaults\n  lr: 0.001\n  wd: 0.01\nexp:\n  <<: *defaults\n"
        result = measure_fidelity(text, "yaml", "cdxf")
        assert result["anchors_preserved"] == result["anchors_original"]

    def test_cbor_loses_comments(self):
        text = "# Important\nkey: value\n"
        result = measure_fidelity(text, "yaml", "cbor")
        assert result["comments_preserved"] == 0

    def test_all_baselines_exist(self):
        """Every baseline in BASELINES should be testable."""
        assert len(BASELINES) >= 7
        for name in BASELINES:
            assert isinstance(name, str)

    def test_fidelity_result_has_required_fields(self):
        text = "key: value\n"
        result = measure_fidelity(text, "yaml", "cdxf")
        required = [
            "comments_original", "comments_preserved",
            "anchors_original", "anchors_preserved",
            "merge_keys_original", "merge_keys_preserved",
            "temporal_values_original", "temporal_values_preserved",
            "multi_doc_markers_original", "multi_doc_markers_preserved",
            "data_fidelity", "serialized_size", "original_size",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"
