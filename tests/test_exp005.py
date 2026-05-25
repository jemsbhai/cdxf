"""Tests for EXP-005 utility functions.

Tests the format classification, construct counting, and analysis
functions used by the HuggingFace Format Census.
"""

import json
import pytest
import sys
from pathlib import Path

# Add benchmarks/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks" / "src"))

from run_exp005 import (
    classify_format,
    count_yaml_constructs,
    count_json_constructs,
    count_toml_constructs,
    count_xml_constructs,
    detect_yaml_frontmatter,
    shannon_entropy,
    should_download,
)


# ---------------------------------------------------------------------------
# classify_format
# ---------------------------------------------------------------------------

class TestClassifyFormat:
    def test_json(self):
        assert classify_format("config.json") == "json"
        assert classify_format("adapter_config.json") == "json"

    def test_jsonl(self):
        assert classify_format("data.jsonl") == "jsonl"

    def test_yaml(self):
        assert classify_format("config.yaml") == "yaml"
        assert classify_format("config.yml") == "yaml"

    def test_toml(self):
        assert classify_format("pyproject.toml") == "toml"

    def test_xml(self):
        assert classify_format("pom.xml") == "xml"

    def test_ini_variants(self):
        assert classify_format("setup.cfg") == "ini"
        assert classify_format("config.ini") == "ini"
        assert classify_format("app.conf") == "ini"

    def test_markdown_special(self):
        assert classify_format("README.md") == "markdown"
        assert classify_format("MODEL_CARD.md") == "markdown"

    def test_case_insensitive(self):
        assert classify_format("CONFIG.JSON") == "json"
        assert classify_format("Config.YAML") == "yaml"

    def test_other(self):
        assert classify_format("model.safetensors") == "other"
        assert classify_format("script.py") == "other"


# ---------------------------------------------------------------------------
# count_yaml_constructs
# ---------------------------------------------------------------------------

class TestCountYamlConstructs:
    def test_comments_full_line(self):
        text = "# This is a comment\nkey: value\n# Another comment\n"
        result = count_yaml_constructs(text)
        assert result["comments"] == 2

    def test_comments_inline(self):
        text = "key: value  # inline comment\n"
        result = count_yaml_constructs(text)
        assert result["comments"] == 1

    def test_no_false_positive_hash_in_string(self):
        text = 'url: "https://example.com/#fragment"\n'
        result = count_yaml_constructs(text)
        # The # is inside quotes, should not count
        assert result["comments"] == 0

    def test_anchors(self):
        text = "defaults: &defaults\n  key: value\noverride:\n  <<: *defaults\n"
        result = count_yaml_constructs(text)
        assert result["anchors"] == 1
        assert result["aliases"] == 1

    def test_merge_keys(self):
        text = "<<: *base\nkey: value\n"
        result = count_yaml_constructs(text)
        assert result["merge_keys"] == 1

    def test_multi_doc(self):
        text = "---\nkey: value\n---\nkey2: value2\n...\n"
        result = count_yaml_constructs(text)
        assert result["multi_doc_markers"] == 3  # ---, ---, ...

    def test_empty(self):
        result = count_yaml_constructs("")
        assert all(v == 0 for v in result.values())

    def test_complex_yaml(self):
        text = """# Training config for BERT fine-tuning
# Author: researcher@lab.edu
model: &model_name bert-base-uncased
training:
  <<: *defaults  # merge base config
  learning_rate: 1e-4  # tuned via grid search
  epochs: 3
---
# Evaluation config
model: *model_name
"""
        result = count_yaml_constructs(text)
        assert result["comments"] >= 4  # at least 4 comment lines
        assert result["anchors"] >= 1
        assert result["aliases"] >= 1
        assert result["merge_keys"] >= 1
        assert result["multi_doc_markers"] >= 1


# ---------------------------------------------------------------------------
# count_json_constructs
# ---------------------------------------------------------------------------

class TestCountJsonConstructs:
    def test_simple(self):
        text = '{"key": "value", "num": 42}'
        result = count_json_constructs(text)
        assert result["key_count"] == 2
        assert result["max_depth"] == 1

    def test_nested(self):
        text = '{"a": {"b": {"c": 1}}}'
        result = count_json_constructs(text)
        assert result["max_depth"] == 3

    def test_arrays(self):
        text = '{"items": [1, 2, 3], "nested": [[1], [2]]}'
        result = count_json_constructs(text)
        assert result["array_count"] >= 3  # outer + 2 inner

    def test_invalid_json(self):
        result = count_json_constructs("not json at all")
        assert result["max_depth"] == 0

    def test_empty_object(self):
        result = count_json_constructs("{}")
        assert result["key_count"] == 0
        assert result["max_depth"] == 0


# ---------------------------------------------------------------------------
# count_toml_constructs
# ---------------------------------------------------------------------------

class TestCountTomlConstructs:
    def test_comments(self):
        text = "# Comment\nkey = 'value'\n# Another\n"
        result = count_toml_constructs(text)
        assert result["comments"] == 2

    def test_sections(self):
        text = "[package]\nname = 'foo'\n[dependencies]\nbar = '1.0'\n"
        result = count_toml_constructs(text)
        assert result["sections"] == 2

    def test_temporal(self):
        text = 'created = 2024-01-15T10:30:00\nupdated = 2024-06-01\n'
        result = count_toml_constructs(text)
        assert result["temporal_values"] >= 1

    def test_inline_tables(self):
        text = 'server = {host = "localhost", port = 8080}\n'
        result = count_toml_constructs(text)
        assert result["inline_tables"] == 1


# ---------------------------------------------------------------------------
# count_xml_constructs
# ---------------------------------------------------------------------------

class TestCountXmlConstructs:
    def test_comments(self):
        text = "<!-- comment --><root/>"
        result = count_xml_constructs(text)
        assert result["comments"] == 1

    def test_processing_instructions(self):
        text = '<?xml version="1.0"?><?xsl-stylesheet href="style.xsl"?><root/>'
        result = count_xml_constructs(text)
        # <?xml is excluded, <?xsl-stylesheet is counted
        assert result["processing_instructions"] == 1

    def test_namespaces(self):
        text = '<root xmlns="http://example.com" xmlns:xsi="http://w3.org/xsi"><child/></root>'
        result = count_xml_constructs(text)
        assert result["namespaces"] == 2

    def test_elements_and_attrs(self):
        text = '<root attr1="a" attr2="b"><child attr3="c"/></root>'
        result = count_xml_constructs(text)
        assert result["elements"] == 2  # root, child
        assert result["attributes"] == 3


# ---------------------------------------------------------------------------
# detect_yaml_frontmatter
# ---------------------------------------------------------------------------

class TestDetectYamlFrontmatter:
    def test_with_frontmatter(self):
        text = "---\ntitle: My Model\ntags:\n  - nlp\n---\n# Model Card\nContent here."
        result = detect_yaml_frontmatter(text)
        assert result is not None
        assert result["frontmatter_size_bytes"] > 0

    def test_without_frontmatter(self):
        text = "# Just a README\n\nNo frontmatter here."
        result = detect_yaml_frontmatter(text)
        assert result is None

    def test_frontmatter_with_comments(self):
        text = "---\n# This is a comment\ntitle: Test\n---\nBody"
        result = detect_yaml_frontmatter(text)
        assert result is not None
        assert result["comments"] >= 1

    def test_unclosed_frontmatter(self):
        text = "---\ntitle: Test\nNo closing marker"
        result = detect_yaml_frontmatter(text)
        assert result is None


# ---------------------------------------------------------------------------
# shannon_entropy
# ---------------------------------------------------------------------------

class TestShannonEntropy:
    def test_single_format(self):
        assert shannon_entropy({"json": 10}) == 0.0

    def test_uniform_two(self):
        e = shannon_entropy({"json": 5, "yaml": 5})
        assert abs(e - 1.0) < 0.01  # log2(2) = 1.0

    def test_uniform_four(self):
        e = shannon_entropy({"json": 5, "yaml": 5, "toml": 5, "xml": 5})
        assert abs(e - 2.0) < 0.01  # log2(4) = 2.0

    def test_skewed(self):
        e = shannon_entropy({"json": 100, "yaml": 1})
        assert e < 0.1  # very low entropy

    def test_empty(self):
        assert shannon_entropy({}) == 0.0


# ---------------------------------------------------------------------------
# should_download
# ---------------------------------------------------------------------------

class TestShouldDownload:
    def test_config_files(self):
        assert should_download("config.json", 100) is True
        assert should_download("training_args.yaml", 500) is True
        assert should_download("pyproject.toml", 200) is True

    def test_special_files(self):
        assert should_download("README.md", 5000) is True
        assert should_download("MODEL_CARD.md", 3000) is True

    def test_skip_weights(self):
        assert should_download("model.safetensors", 1_000_000_000) is False
        assert should_download("pytorch_model.bin", 500_000_000) is False

    def test_skip_images(self):
        assert should_download("logo.png", 50000) is False

    def test_skip_too_large(self):
        assert should_download("huge_config.json", 2_000_000) is False

    def test_accept_normal_size(self):
        assert should_download("config.json", 999_999) is True

    def test_unknown_extension(self):
        assert should_download("script.py", 100) is False
