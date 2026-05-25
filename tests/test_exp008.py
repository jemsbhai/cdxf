"""Tests for EXP-008: LoRA Adapter Registry Metadata Bundle.

Tests corpus generation, bundling methods (CDXF, tar.gz, mega-JSON, Pickle),
round-trip fidelity, cross-format emission, and component extraction.

TDD: These tests are written BEFORE the implementation.
"""

import gzip
import json
import pickle
import tarfile
import io
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp008 import (
    build_adapter_corpus,
    bundle_cdxf_multidoc,
    bundle_tar_gz,
    bundle_mega_json,
    bundle_pickle,
    extract_component,
    cross_format_emit,
    measure_bundle,
    BUNDLE_METHODS,
)


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

class TestBuildAdapterCorpus:
    def test_returns_list(self):
        corpus = build_adapter_corpus()
        assert isinstance(corpus, list)

    def test_has_at_least_10_adapters(self):
        """Protocol targets 50, but synthetic corpus should have >= 10."""
        corpus = build_adapter_corpus()
        assert len(corpus) >= 10

    def test_adapter_has_required_fields(self):
        corpus = build_adapter_corpus()
        for adapter in corpus:
            assert "name" in adapter
            assert "components" in adapter
            assert isinstance(adapter["components"], list)
            assert len(adapter["components"]) >= 2

    def test_component_has_required_fields(self):
        corpus = build_adapter_corpus()
        for adapter in corpus:
            for comp in adapter["components"]:
                assert "filename" in comp
                assert "format" in comp
                assert "text" in comp
                assert isinstance(comp["text"], str)
                assert len(comp["text"]) > 0

    def test_every_adapter_has_adapter_config(self):
        """Every LoRA adapter must have adapter_config.json."""
        corpus = build_adapter_corpus()
        for adapter in corpus:
            filenames = [c["filename"] for c in adapter["components"]]
            assert "adapter_config.json" in filenames, (
                f"Adapter {adapter['name']} missing adapter_config.json"
            )

    def test_has_yaml_components_with_comments(self):
        """At least some adapters should have YAML with comments."""
        corpus = build_adapter_corpus()
        has_yaml_comments = False
        for adapter in corpus:
            for comp in adapter["components"]:
                if comp["format"] == "yaml" and "#" in comp["text"]:
                    has_yaml_comments = True
                    break
        assert has_yaml_comments

    def test_has_markdown_with_frontmatter(self):
        """At least some adapters should have README with YAML frontmatter."""
        corpus = build_adapter_corpus()
        has_frontmatter = False
        for adapter in corpus:
            for comp in adapter["components"]:
                if comp["filename"] == "README.md" and "---" in comp["text"]:
                    has_frontmatter = True
                    break
        assert has_frontmatter

    def test_components_are_parseable(self):
        """JSON components should parse; YAML components should parse."""
        corpus = build_adapter_corpus()
        for adapter in corpus:
            for comp in adapter["components"]:
                if comp["format"] == "json":
                    json.loads(comp["text"])  # should not raise


# ---------------------------------------------------------------------------
# CDXF multi-doc bundling
# ---------------------------------------------------------------------------

class TestBundleCdxfMultidoc:
    def test_returns_bytes(self):
        corpus = build_adapter_corpus()
        result = bundle_cdxf_multidoc(corpus[0]["components"])
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_size_is_reasonable(self):
        """CDXF bundle should not be wildly larger than sum of inputs."""
        corpus = build_adapter_corpus()
        components = corpus[0]["components"]
        total_input = sum(len(c["text"].encode()) for c in components)
        cdxf_bytes = bundle_cdxf_multidoc(components)
        # Allow up to 3x (binary overhead for small files is acceptable)
        assert len(cdxf_bytes) < total_input * 3


# ---------------------------------------------------------------------------
# tar.gz bundling
# ---------------------------------------------------------------------------

class TestBundleTargz:
    def test_returns_bytes(self):
        corpus = build_adapter_corpus()
        result = bundle_tar_gz(corpus[0]["components"])
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_is_valid_gzip(self):
        corpus = build_adapter_corpus()
        result = bundle_tar_gz(corpus[0]["components"])
        # Should decompress without error
        gzip.decompress(result)


# ---------------------------------------------------------------------------
# mega-JSON bundling
# ---------------------------------------------------------------------------

class TestBundleMegaJson:
    def test_returns_bytes(self):
        corpus = build_adapter_corpus()
        result = bundle_mega_json(corpus[0]["components"])
        assert isinstance(result, bytes)

    def test_is_valid_json(self):
        corpus = build_adapter_corpus()
        result = bundle_mega_json(corpus[0]["components"])
        parsed = json.loads(result.decode("utf-8"))
        assert isinstance(parsed, dict)

    def test_all_components_present(self):
        corpus = build_adapter_corpus()
        components = corpus[0]["components"]
        result = bundle_mega_json(components)
        parsed = json.loads(result.decode("utf-8"))
        for comp in components:
            assert comp["filename"] in parsed


# ---------------------------------------------------------------------------
# Pickle bundling
# ---------------------------------------------------------------------------

class TestBundlePickle:
    def test_returns_bytes(self):
        corpus = build_adapter_corpus()
        result = bundle_pickle(corpus[0]["components"])
        assert isinstance(result, bytes)

    def test_is_valid_pickle(self):
        corpus = build_adapter_corpus()
        result = bundle_pickle(corpus[0]["components"])
        loaded = pickle.loads(result)
        assert isinstance(loaded, dict)


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------

class TestExtractComponent:
    def test_extract_from_cdxf(self):
        corpus = build_adapter_corpus()
        components = corpus[0]["components"]
        bundle = bundle_cdxf_multidoc(components)
        result = extract_component(bundle, "cdxf", "adapter_config.json")
        assert result["success"]
        assert len(result["text"]) > 0

    def test_extract_from_tar_gz(self):
        corpus = build_adapter_corpus()
        components = corpus[0]["components"]
        bundle = bundle_tar_gz(components)
        result = extract_component(bundle, "tar_gz", "adapter_config.json")
        assert result["success"]
        assert len(result["text"]) > 0

    def test_extract_from_mega_json(self):
        corpus = build_adapter_corpus()
        components = corpus[0]["components"]
        bundle = bundle_mega_json(components)
        result = extract_component(bundle, "mega_json", "adapter_config.json")
        assert result["success"]

    def test_extract_from_pickle(self):
        corpus = build_adapter_corpus()
        components = corpus[0]["components"]
        bundle = bundle_pickle(components)
        result = extract_component(bundle, "pickle", "adapter_config.json")
        assert result["success"]

    def test_extract_nonexistent_fails_gracefully(self):
        corpus = build_adapter_corpus()
        bundle = bundle_mega_json(corpus[0]["components"])
        result = extract_component(bundle, "mega_json", "nonexistent.txt")
        assert not result["success"]

    def test_extract_result_has_timing(self):
        corpus = build_adapter_corpus()
        bundle = bundle_mega_json(corpus[0]["components"])
        result = extract_component(bundle, "mega_json", "adapter_config.json")
        assert "time_ns" in result
        assert result["time_ns"] > 0


# ---------------------------------------------------------------------------
# Cross-format emission (CDXF only)
# ---------------------------------------------------------------------------

class TestCrossFormatEmit:
    def test_yaml_to_json(self):
        """Extract a YAML component from CDXF and emit as JSON."""
        corpus = build_adapter_corpus()
        # Find an adapter with a YAML component
        for adapter in corpus:
            yaml_comps = [c for c in adapter["components"]
                          if c["format"] == "yaml"]
            if yaml_comps:
                bundle = bundle_cdxf_multidoc(adapter["components"])
                result = cross_format_emit(
                    bundle, yaml_comps[0]["filename"], "json"
                )
                assert result["success"], f"Failed: {result.get('error')}"
                # Output should be valid JSON
                json.loads(result["text"])
                break
        else:
            pytest.skip("No YAML component found in corpus")

    def test_json_to_yaml(self):
        """Extract a JSON component from CDXF and emit as YAML."""
        corpus = build_adapter_corpus()
        adapter = corpus[0]  # First adapter always has adapter_config.json
        bundle = bundle_cdxf_multidoc(adapter["components"])
        result = cross_format_emit(
            bundle, "adapter_config.json", "yaml"
        )
        assert result["success"], f"Failed: {result.get('error')}"

    def test_unsupported_component_handled(self):
        """Non-existent component should fail gracefully."""
        corpus = build_adapter_corpus()
        bundle = bundle_cdxf_multidoc(corpus[0]["components"])
        result = cross_format_emit(
            bundle, "nonexistent.yaml", "json"
        )
        assert not result["success"]


# ---------------------------------------------------------------------------
# Full measurement
# ---------------------------------------------------------------------------

class TestMeasureBundle:
    def test_returns_result_dict(self):
        corpus = build_adapter_corpus()
        result = measure_bundle(corpus[0])
        assert isinstance(result, dict)

    def test_result_has_all_methods(self):
        corpus = build_adapter_corpus()
        result = measure_bundle(corpus[0])
        for method in BUNDLE_METHODS:
            assert method in result, f"Missing method: {method}"

    def test_result_has_adapter_info(self):
        corpus = build_adapter_corpus()
        result = measure_bundle(corpus[0])
        assert "name" in result
        assert "n_components" in result
        assert "total_input_size" in result

    def test_each_method_has_size(self):
        corpus = build_adapter_corpus()
        result = measure_bundle(corpus[0])
        for method in BUNDLE_METHODS:
            assert "bundle_size" in result[method]
            assert result[method]["bundle_size"] > 0

    def test_each_method_has_size_ratio(self):
        corpus = build_adapter_corpus()
        result = measure_bundle(corpus[0])
        for method in BUNDLE_METHODS:
            assert "size_vs_sum" in result[method]

    def test_cdxf_reports_fidelity(self):
        corpus = build_adapter_corpus()
        result = measure_bundle(corpus[0])
        assert "round_trip_fidelity" in result["cdxf"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestBundleMethods:
    def test_has_four_methods(self):
        assert len(BUNDLE_METHODS) == 4

    def test_contains_expected(self):
        assert "cdxf" in BUNDLE_METHODS
        assert "tar_gz" in BUNDLE_METHODS
        assert "mega_json" in BUNDLE_METHODS
        assert "pickle" in BUNDLE_METHODS
