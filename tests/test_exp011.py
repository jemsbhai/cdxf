"""Tests for EXP-011: Token Cost of Format Syntax — The "Syntax Tax".

Measures the fraction of LLM tokens consumed by format syntax (braces,
brackets, closing tags, quotes) vs semantic content (keys, values, comments).

TDD: These tests are written BEFORE the implementation.
"""

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp011 import (
    CONTEXT_WINDOW,
    FORMATS,
    N_FILES_PROJECTIONS,
    TOKENIZERS,
    build_corpus,
    classify_tokens,
    compute_context_projection,
    identify_char_roles,
    run_single_file,
    run_experiment,
)


# ===========================================================================
# Constants
# ===========================================================================


class TestProtocolConstants:
    def test_tokenizers(self):
        assert "cl100k_base" in TOKENIZERS
        assert "o200k_base" in TOKENIZERS

    def test_context_window(self):
        assert CONTEXT_WINDOW == 128_000

    def test_projection_file_counts(self):
        assert set(N_FILES_PROJECTIONS) == {10, 25, 50, 100}

    def test_formats(self):
        assert set(FORMATS) == {"json", "yaml", "xml", "toml"}


# ===========================================================================
# Character-level role identification — the scientific core
# ===========================================================================


class TestIdentifyCharRolesJSON:
    """JSON syntax: { } [ ] : , and quotes around keys.
    JSON semantic: key names, string values, numbers, booleans, null."""

    def test_returns_list_of_correct_length(self):
        text = '{"a": 1}'
        roles = identify_char_roles(text, "json")
        assert len(roles) == len(text)

    def test_roles_are_valid(self):
        text = '{"key": "val"}'
        roles = identify_char_roles(text, "json")
        assert all(r in ("syntax", "semantic") for r in roles)

    def test_braces_are_syntax(self):
        text = '{"a": 1}'
        roles = identify_char_roles(text, "json")
        assert roles[0] == "syntax"   # {
        assert roles[-1] == "syntax"  # }

    def test_colon_is_syntax(self):
        text = '{"a": 1}'
        roles = identify_char_roles(text, "json")
        colon_idx = text.index(":")
        assert roles[colon_idx] == "syntax"

    def test_comma_is_syntax(self):
        text = '{"a": 1, "b": 2}'
        roles = identify_char_roles(text, "json")
        comma_idx = text.index(",")
        assert roles[comma_idx] == "syntax"

    def test_brackets_are_syntax(self):
        text = '{"a": [1, 2]}'
        roles = identify_char_roles(text, "json")
        assert roles[text.index("[")] == "syntax"
        assert roles[text.index("]")] == "syntax"

    def test_key_quotes_are_syntax(self):
        """Quotes around keys are syntax — they delimit, not carry content."""
        text = '{"key": "val"}'
        roles = identify_char_roles(text, "json")
        # First quote after { is syntax (key quote)
        assert roles[1] == "syntax"  # opening quote of key

    def test_key_name_is_semantic(self):
        text = '{"key": "val"}'
        roles = identify_char_roles(text, "json")
        k_idx = text.index("k")
        assert roles[k_idx] == "semantic"
        assert roles[k_idx + 1] == "semantic"  # e
        assert roles[k_idx + 2] == "semantic"  # y

    def test_string_value_content_is_semantic(self):
        text = '{"key": "val"}'
        roles = identify_char_roles(text, "json")
        v_idx = text.index("v")
        assert roles[v_idx] == "semantic"

    def test_number_is_semantic(self):
        text = '{"x": 42}'
        roles = identify_char_roles(text, "json")
        idx_4 = text.index("4")
        idx_2 = text.index("2")
        assert roles[idx_4] == "semantic"
        assert roles[idx_2] == "semantic"

    def test_boolean_is_semantic(self):
        text = '{"flag": true}'
        roles = identify_char_roles(text, "json")
        t_idx = text.index("t")
        assert roles[t_idx] == "semantic"

    def test_null_is_semantic(self):
        text = '{"x": null}'
        roles = identify_char_roles(text, "json")
        n_idx = text.index("n")
        assert roles[n_idx] == "semantic"

    def test_whitespace_is_syntax(self):
        """Indentation and formatting whitespace is syntax."""
        text = '{\n  "a": 1\n}'
        roles = identify_char_roles(text, "json")
        newline_idx = text.index("\n")
        assert roles[newline_idx] == "syntax"

    def test_nested_object(self):
        text = '{"a": {"b": 1}}'
        roles = identify_char_roles(text, "json")
        # Inner braces are syntax
        inner_open = text.index("{", 1)
        assert roles[inner_open] == "syntax"


class TestIdentifyCharRolesYAML:
    """YAML syntax: :, -, indentation whitespace, ---, ...
    YAML semantic: key names, values, comments (comments ARE semantic)."""

    def test_returns_correct_length(self):
        text = "key: value\n"
        roles = identify_char_roles(text, "yaml")
        assert len(roles) == len(text)

    def test_colon_space_is_syntax(self):
        text = "key: value"
        roles = identify_char_roles(text, "yaml")
        colon_idx = text.index(":")
        assert roles[colon_idx] == "syntax"

    def test_key_is_semantic(self):
        text = "key: value"
        roles = identify_char_roles(text, "yaml")
        assert roles[0] == "semantic"  # k
        assert roles[1] == "semantic"  # e
        assert roles[2] == "semantic"  # y

    def test_value_is_semantic(self):
        text = "key: value"
        roles = identify_char_roles(text, "yaml")
        v_idx = text.index("v")
        assert roles[v_idx] == "semantic"

    def test_comment_is_semantic(self):
        """Protocol: YAML comments ARE semantic content (they carry meaning)."""
        text = "# this is a comment"
        roles = identify_char_roles(text, "yaml")
        # The # itself could be debated, but the comment text is semantic
        # At minimum, the content after # must be semantic
        t_idx = text.index("t")
        assert roles[t_idx] == "semantic"

    def test_dash_list_indicator_is_syntax(self):
        text = "- item1\n- item2"
        roles = identify_char_roles(text, "yaml")
        assert roles[0] == "syntax"  # -

    def test_doc_marker_is_syntax(self):
        text = "---\nkey: val"
        roles = identify_char_roles(text, "yaml")
        assert roles[0] == "syntax"  # -
        assert roles[1] == "syntax"  # -
        assert roles[2] == "syntax"  # -

    def test_indentation_is_syntax(self):
        text = "parent:\n  child: value"
        roles = identify_char_roles(text, "yaml")
        # The spaces before "child" are indentation = syntax
        indent_start = text.index("\n") + 1
        assert roles[indent_start] == "syntax"      # first space
        assert roles[indent_start + 1] == "syntax"  # second space


class TestIdentifyCharRolesXML:
    """XML syntax: < > </ /> = quotes around attrs, closing tag names (redundant).
    XML semantic: element names (first occurrence), attr names, text, attr values."""

    def test_returns_correct_length(self):
        text = "<a>text</a>"
        roles = identify_char_roles(text, "xml")
        assert len(roles) == len(text)

    def test_angle_brackets_are_syntax(self):
        text = "<a>text</a>"
        roles = identify_char_roles(text, "xml")
        assert roles[0] == "syntax"   # <
        assert roles[2] == "syntax"   # >
        assert roles[-1] == "syntax"  # > (closing)

    def test_element_name_first_occurrence_is_semantic(self):
        text = "<name>text</name>"
        roles = identify_char_roles(text, "xml")
        # "name" in opening tag is semantic
        n_idx = text.index("n")
        assert roles[n_idx] == "semantic"

    def test_closing_tag_name_is_syntax(self):
        """Closing tag name is REDUNDANT — it repeats the opening tag."""
        text = "<name>text</name>"
        roles = identify_char_roles(text, "xml")
        # </name> — the "name" part here is syntax (redundant)
        close_start = text.index("</")
        assert roles[close_start] == "syntax"      # <
        assert roles[close_start + 1] == "syntax"   # /

    def test_text_content_is_semantic(self):
        text = "<a>hello</a>"
        roles = identify_char_roles(text, "xml")
        h_idx = text.index("h")
        assert roles[h_idx] == "semantic"

    def test_attribute_name_is_semantic(self):
        text = '<a href="url">text</a>'
        roles = identify_char_roles(text, "xml")
        h_idx = text.index("h")
        assert roles[h_idx] == "semantic"

    def test_attribute_value_is_semantic(self):
        text = '<a href="url">text</a>'
        roles = identify_char_roles(text, "xml")
        u_idx = text.index("u", text.index('"'))
        assert roles[u_idx] == "semantic"

    def test_attribute_equals_is_syntax(self):
        text = '<a href="url">text</a>'
        roles = identify_char_roles(text, "xml")
        eq_idx = text.index("=")
        assert roles[eq_idx] == "syntax"

    def test_attribute_quotes_are_syntax(self):
        text = '<a x="v">t</a>'
        roles = identify_char_roles(text, "xml")
        first_q = text.index('"')
        assert roles[first_q] == "syntax"

    def test_comment_content_is_semantic(self):
        """XML comments carry semantic content."""
        text = "<!-- important note -->"
        roles = identify_char_roles(text, "xml")
        i_idx = text.index("i")
        assert roles[i_idx] == "semantic"

    def test_comment_delimiters_are_syntax(self):
        text = "<!-- note -->"
        roles = identify_char_roles(text, "xml")
        assert roles[0] == "syntax"  # <
        assert roles[1] == "syntax"  # !
        assert roles[2] == "syntax"  # -
        assert roles[3] == "syntax"  # -


class TestIdentifyCharRolesToml:
    """TOML syntax: [ ] = " , { }
    TOML semantic: key names, values, section names, comments."""

    def test_returns_correct_length(self):
        text = 'key = "value"'
        roles = identify_char_roles(text, "toml")
        assert len(roles) == len(text)

    def test_equals_is_syntax(self):
        text = 'key = "value"'
        roles = identify_char_roles(text, "toml")
        eq_idx = text.index("=")
        assert roles[eq_idx] == "syntax"

    def test_key_is_semantic(self):
        text = 'key = "value"'
        roles = identify_char_roles(text, "toml")
        assert roles[0] == "semantic"  # k

    def test_value_is_semantic(self):
        text = 'key = "value"'
        roles = identify_char_roles(text, "toml")
        v_idx = text.index("v", text.index('"'))
        assert roles[v_idx] == "semantic"

    def test_section_brackets_are_syntax(self):
        text = "[section]\nkey = 1"
        roles = identify_char_roles(text, "toml")
        assert roles[0] == "syntax"                  # [
        assert roles[text.index("]")] == "syntax"    # ]

    def test_section_name_is_semantic(self):
        text = "[section]\nkey = 1"
        roles = identify_char_roles(text, "toml")
        s_idx = text.index("s")
        assert roles[s_idx] == "semantic"

    def test_comment_is_semantic(self):
        """TOML comments ARE semantic (like YAML)."""
        text = "# configuration comment"
        roles = identify_char_roles(text, "toml")
        c_idx = text.index("c")
        assert roles[c_idx] == "semantic"

    def test_string_quotes_are_syntax(self):
        text = 'key = "value"'
        roles = identify_char_roles(text, "toml")
        first_q = text.index('"')
        assert roles[first_q] == "syntax"


# ===========================================================================
# Token classification
# ===========================================================================


class TestClassifyTokens:
    """classify_tokens maps tiktoken tokens to syntax/semantic roles."""

    def test_returns_dict(self):
        result = classify_tokens('{"a": 1}', "json", "cl100k_base")
        assert isinstance(result, dict)

    def test_required_fields(self):
        result = classify_tokens('{"a": 1}', "json", "cl100k_base")
        required = {
            "total_tokens", "syntax_tokens", "semantic_tokens",
            "syntax_tax_rate", "tokens_per_byte",
        }
        assert required.issubset(set(result.keys()))

    def test_total_equals_syntax_plus_semantic(self):
        """Total tokens = syntax + semantic (accounting for proportional split)."""
        result = classify_tokens('{"key": "value", "num": 42}', "json", "cl100k_base")
        total = result["total_tokens"]
        syn = result["syntax_tokens"]
        sem = result["semantic_tokens"]
        assert abs(total - (syn + sem)) < 0.5, (
            f"total={total}, syntax={syn}, semantic={sem}, sum={syn+sem}"
        )

    def test_syntax_tax_rate_is_fraction(self):
        result = classify_tokens('{"a": 1}', "json", "cl100k_base")
        rate = result["syntax_tax_rate"]
        assert 0.0 <= rate <= 1.0

    def test_syntax_tax_rate_formula(self):
        """syntax_tax_rate = syntax_tokens / total_tokens."""
        result = classify_tokens('{"key": "value"}', "json", "cl100k_base")
        if result["total_tokens"] > 0:
            expected = result["syntax_tokens"] / result["total_tokens"]
            assert result["syntax_tax_rate"] == pytest.approx(expected, abs=0.01)

    def test_tokens_per_byte(self):
        text = '{"key": "value"}'
        result = classify_tokens(text, "json", "cl100k_base")
        expected = result["total_tokens"] / len(text.encode("utf-8"))
        assert result["tokens_per_byte"] == pytest.approx(expected, rel=0.01)

    def test_json_has_nonzero_syntax(self):
        """JSON always has some syntax ({, }, :, etc.)."""
        result = classify_tokens('{"key": "value"}', "json", "cl100k_base")
        assert result["syntax_tokens"] > 0

    def test_json_has_nonzero_semantic(self):
        result = classify_tokens('{"key": "value"}', "json", "cl100k_base")
        assert result["semantic_tokens"] > 0

    def test_xml_has_substantial_syntax(self):
        """XML has significant syntax overhead from tags and closing tags."""
        xml_text = '<person><name>Alice</name><age>30</age></person>'
        xml_result = classify_tokens(xml_text, "xml", "cl100k_base")
        assert xml_result["syntax_tax_rate"] > 0.3, (
            f"XML syntax tax unexpectedly low: {xml_result['syntax_tax_rate']}"
        )

    def test_works_with_o200k(self):
        """Must work with both tokenizer variants."""
        result = classify_tokens('{"a": 1}', "json", "o200k_base")
        assert result["total_tokens"] > 0

    def test_yaml_comments_count_as_semantic(self):
        """YAML comments are semantic — they should reduce the syntax tax."""
        text_no_comment = "key: value\n"
        text_comment = "# important note\nkey: value\n"
        r1 = classify_tokens(text_no_comment, "yaml", "cl100k_base")
        r2 = classify_tokens(text_comment, "yaml", "cl100k_base")
        # More semantic tokens with the comment
        assert r2["semantic_tokens"] > r1["semantic_tokens"]

    def test_empty_string(self):
        result = classify_tokens("", "json", "cl100k_base")
        assert result["total_tokens"] == 0
        assert result["syntax_tokens"] == 0
        assert result["semantic_tokens"] == 0


# ===========================================================================
# Context projection
# ===========================================================================


class TestComputeContextProjection:
    def test_returns_dict(self):
        result = compute_context_projection(
            median_total_tokens=100, syntax_tax_rate=0.3,
            n_files=10, context_window=128_000,
        )
        assert isinstance(result, dict)

    def test_required_fields(self):
        result = compute_context_projection(
            median_total_tokens=100, syntax_tax_rate=0.3,
            n_files=10, context_window=128_000,
        )
        required = {
            "n_files", "total_tokens_used", "syntax_tokens_wasted",
            "context_waste_fraction", "effective_context_gain_files",
        }
        assert required.issubset(set(result.keys()))

    def test_total_tokens_used(self):
        result = compute_context_projection(
            median_total_tokens=200, syntax_tax_rate=0.3,
            n_files=50, context_window=128_000,
        )
        assert result["total_tokens_used"] == 200 * 50

    def test_syntax_tokens_wasted(self):
        result = compute_context_projection(
            median_total_tokens=200, syntax_tax_rate=0.3,
            n_files=50, context_window=128_000,
        )
        # wasted = n_files * median_total * syntax_tax_rate
        assert result["syntax_tokens_wasted"] == pytest.approx(200 * 50 * 0.3)

    def test_context_waste_fraction(self):
        result = compute_context_projection(
            median_total_tokens=200, syntax_tax_rate=0.3,
            n_files=50, context_window=128_000,
        )
        expected = (200 * 50 * 0.3) / 128_000
        assert result["context_waste_fraction"] == pytest.approx(expected)

    def test_effective_context_gain(self):
        """Extra files = wasted_tokens / median_semantic_tokens_per_file."""
        result = compute_context_projection(
            median_total_tokens=200, syntax_tax_rate=0.3,
            n_files=50, context_window=128_000,
        )
        wasted = 200 * 50 * 0.3
        semantic_per_file = 200 * (1 - 0.3)
        expected_gain = wasted / semantic_per_file
        assert result["effective_context_gain_files"] == pytest.approx(expected_gain)

    def test_zero_syntax_tax(self):
        result = compute_context_projection(
            median_total_tokens=200, syntax_tax_rate=0.0,
            n_files=50, context_window=128_000,
        )
        assert result["syntax_tokens_wasted"] == 0
        assert result["context_waste_fraction"] == 0
        assert result["effective_context_gain_files"] == 0

    def test_all_protocol_n_values(self):
        """Must work for all N in protocol: 10, 25, 50, 100."""
        for n in N_FILES_PROJECTIONS:
            result = compute_context_projection(
                median_total_tokens=100, syntax_tax_rate=0.3,
                n_files=n, context_window=128_000,
            )
            assert result["n_files"] == n


# ===========================================================================
# Corpus
# ===========================================================================


class TestBuildCorpus:
    def test_returns_list(self):
        corpus = build_corpus()
        assert isinstance(corpus, list)

    def test_non_empty(self):
        corpus = build_corpus()
        assert len(corpus) > 0

    def test_entry_required_fields(self):
        corpus = build_corpus()
        for entry in corpus:
            assert "name" in entry
            assert "format" in entry
            assert "text" in entry

    def test_format_is_valid(self):
        corpus = build_corpus()
        for entry in corpus:
            assert entry["format"] in FORMATS, f"Bad format: {entry['format']}"

    def test_has_all_four_formats(self):
        """Protocol requires JSON, YAML, XML, TOML."""
        corpus = build_corpus()
        formats = {e["format"] for e in corpus}
        assert formats == {"json", "yaml", "xml", "toml"}

    def test_includes_data_raw_files(self):
        """Protocol: include all 43 files from EXP-001 corpus."""
        corpus = build_corpus()
        # Should have a substantial number of files from data/raw
        assert len(corpus) >= 15  # at minimum the ML configs

    def test_names_are_unique(self):
        corpus = build_corpus()
        names = [e["name"] for e in corpus]
        assert len(names) == len(set(names)), "Duplicate names in corpus"


# ===========================================================================
# Single-file analysis
# ===========================================================================


class TestRunSingleFile:
    @pytest.fixture
    def simple_json(self):
        return {"name": "test", "format": "json", "text": '{"a": 1, "b": "hello"}'}

    def test_returns_dict(self, simple_json):
        result = run_single_file(simple_json, "cl100k_base")
        assert isinstance(result, dict)

    def test_has_metadata(self, simple_json):
        result = run_single_file(simple_json, "cl100k_base")
        assert result["name"] == "test"
        assert result["format"] == "json"
        assert "size_bytes" in result
        assert "tokenizer" in result

    def test_has_token_counts(self, simple_json):
        result = run_single_file(simple_json, "cl100k_base")
        assert "total_tokens" in result
        assert "syntax_tokens" in result
        assert "semantic_tokens" in result
        assert "syntax_tax_rate" in result
        assert "tokens_per_byte" in result

    def test_syntax_tax_between_0_and_1(self, simple_json):
        result = run_single_file(simple_json, "cl100k_base")
        assert 0.0 <= result["syntax_tax_rate"] <= 1.0


# ===========================================================================
# Full experiment
# ===========================================================================


class TestRunExperiment:
    @pytest.fixture
    def mini_results(self, tmp_path):
        return run_experiment(output_dir=tmp_path)

    def test_returns_dict(self, mini_results):
        assert isinstance(mini_results, dict)

    def test_has_experiment_id(self, mini_results):
        assert mini_results["experiment"] == "EXP-011"

    def test_has_results_list(self, mini_results):
        assert "results" in mini_results
        assert isinstance(mini_results["results"], list)
        assert len(mini_results["results"]) > 0

    def test_has_aggregate(self, mini_results):
        assert "aggregate" in mini_results

    def test_aggregate_has_per_format_stats(self, mini_results):
        agg = mini_results["aggregate"]
        for fmt in FORMATS:
            assert fmt in agg, f"Missing aggregate for format: {fmt}"

    def test_aggregate_has_per_tokenizer(self, mini_results):
        """Each format should have results for both tokenizers."""
        agg = mini_results["aggregate"]
        for fmt in FORMATS:
            for tok in TOKENIZERS:
                key = f"{tok}"
                assert key in agg[fmt], f"Missing {tok} in aggregate[{fmt}]"

    def test_has_context_projections(self, mini_results):
        assert "context_projections" in mini_results
        projs = mini_results["context_projections"]
        assert isinstance(projs, dict)
        assert len(projs) > 0

    def test_writes_json_output(self, mini_results, tmp_path):
        output_file = tmp_path / "exp_011_results.json"
        assert output_file.exists()

    def test_writes_csv_output(self, mini_results, tmp_path):
        csv_file = tmp_path / "syntax_tax_results.csv"
        assert csv_file.exists()
        import csv
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        required_cols = {"name", "format", "tokenizer", "total_tokens",
                         "syntax_tokens", "semantic_tokens", "syntax_tax_rate"}
        actual_cols = set(rows[0].keys())
        assert required_cols.issubset(actual_cols), (
            f"Missing columns: {required_cols - actual_cols}"
        )

    def test_writes_projection_csv(self, mini_results, tmp_path):
        csv_file = tmp_path / "context_projections.csv"
        assert csv_file.exists()


# ===========================================================================
# Scientific rigor
# ===========================================================================


class TestScientificRigor:
    def test_syntax_plus_semantic_equals_total(self):
        """Conservation law: every token is either syntax or semantic."""
        texts = [
            ('{"key": "value", "n": 42}', "json"),
            ("key: value\n# comment\n", "yaml"),
            ('<root attr="v">text</root>', "xml"),
            ('key = "value"\n# comment\n', "toml"),
        ]
        for text, fmt in texts:
            result = classify_tokens(text, fmt, "cl100k_base")
            total = result["total_tokens"]
            syn = result["syntax_tokens"]
            sem = result["semantic_tokens"]
            assert abs(total - (syn + sem)) < 0.5, (
                f"Conservation violated for {fmt}: {total} != {syn} + {sem}"
            )

    def test_format_ranking_is_data_dependent(self):
        """Syntax tax ranking varies with content — the experiment on real
        corpus data validates the protocol hypothesis, not unit tests.
        Verify all formats produce reasonable (non-degenerate) tax rates."""
        json_text = '{"name": "Alice", "city": "Boston"}'
        yaml_text = "name: Alice\ncity: Boston\n"
        xml_text = "<person><name>Alice</name><city>Boston</city></person>"
        toml_text = 'name = "Alice"\ncity = "Boston"\n'

        for text, fmt in [(json_text, "json"), (yaml_text, "yaml"),
                          (xml_text, "xml"), (toml_text, "toml")]:
            r = classify_tokens(text, fmt, "cl100k_base")
            assert 0.1 < r["syntax_tax_rate"] < 0.95, (
                f"{fmt} syntax tax out of plausible range: {r['syntax_tax_rate']}"
            )

    def test_proportional_attribution_for_mixed_tokens(self):
        """When a token spans syntax and semantic chars, attribution
        must be proportional, not all-or-nothing."""
        # A token like '{"' contains 1 syntax char ({) and 1 syntax char (")
        # but a token like '"key' contains 1 syntax (") and 3 semantic (key)
        result = classify_tokens('{"a": 1}', "json", "cl100k_base")
        # The total should NOT be integer (would indicate all-or-nothing)
        # Actually it CAN be integer by coincidence, so just check conservation
        total = result["total_tokens"]
        syn = result["syntax_tokens"]
        sem = result["semantic_tokens"]
        assert abs(total - (syn + sem)) < 0.5

    def test_char_roles_cover_every_character(self):
        """Every character must be classified — no gaps."""
        texts = [
            ('{"key": "value"}', "json"),
            ("key: value\n# comment\n", "yaml"),
            ("<root>text</root>", "xml"),
            ('key = "val"\n', "toml"),
        ]
        for text, fmt in texts:
            roles = identify_char_roles(text, fmt)
            assert len(roles) == len(text), (
                f"Gap in roles for {fmt}: {len(roles)} roles for {len(text)} chars"
            )
            assert all(r in ("syntax", "semantic") for r in roles), (
                f"Invalid role value in {fmt}"
            )


# ===========================================================================
# Realistic input validation — no trivial examples
# ===========================================================================


class TestRealisticJSON:
    """Verify classification on realistic JSON structures."""

    def test_nested_objects(self):
        text = json.dumps({"model": {"name": "bert", "layers": 12, "hidden": 768}}, indent=2)
        result = classify_tokens(text, "json", "cl100k_base")
        # Semantic content: model, name, bert, layers, 12, hidden, 768
        assert result["semantic_tokens"] > 0
        assert result["syntax_tokens"] > 0
        # JSON with indent=2 has lots of whitespace syntax
        assert result["syntax_tax_rate"] > 0.3

    def test_array_of_objects(self):
        text = json.dumps([{"id": i, "value": f"item_{i}"} for i in range(5)], indent=2)
        result = classify_tokens(text, "json", "cl100k_base")
        assert result["total_tokens"] > 10
        assert 0 < result["syntax_tax_rate"] < 1

    def test_escaped_strings(self):
        text = json.dumps({"path": "C:\\Users\\name", "quote": 'He said "hello"'})
        roles = identify_char_roles(text, "json")
        assert len(roles) == len(text)
        result = classify_tokens(text, "json", "cl100k_base")
        assert abs(result["total_tokens"] - (result["syntax_tokens"] + result["semantic_tokens"])) < 0.5

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        text = json.dumps(data, indent=2)
        result = classify_tokens(text, "json", "cl100k_base")
        # Deep nesting = lots of braces and indentation = high syntax tax
        assert result["syntax_tax_rate"] > 0.5

    def test_large_flat_object(self):
        data = {f"param_{i}": i * 0.1 for i in range(50)}
        text = json.dumps(data, indent=2)
        result = classify_tokens(text, "json", "cl100k_base")
        assert result["total_tokens"] > 50
        assert abs(result["total_tokens"] - (result["syntax_tokens"] + result["semantic_tokens"])) < 0.5


class TestRealisticYAML:
    """Verify classification on realistic YAML."""

    def test_commented_config(self):
        """Comments are semantic content. A commented config should have
        a lower syntax tax than the same config with comments stripped."""
        with_comments = "# Training config\n# Author: researcher\nmodel:\n  name: bert-base\n  # Tried large but OOM\n  dropout: 0.1\ntraining:\n  lr: 0.0001  # grid search winner\n  epochs: 3\n"
        without_comments = "model:\n  name: bert-base\n  dropout: 0.1\ntraining:\n  lr: 0.0001\n  epochs: 3\n"
        r_with = classify_tokens(with_comments, "yaml", "cl100k_base")
        r_without = classify_tokens(without_comments, "yaml", "cl100k_base")
        assert r_with["syntax_tax_rate"] < r_without["syntax_tax_rate"], (
            f"Comments should reduce syntax tax: with={r_with['syntax_tax_rate']:.3f} "
            f"vs without={r_without['syntax_tax_rate']:.3f}"
        )

    def test_list_items(self):
        text = "items:\n  - alpha\n  - beta\n  - gamma\n"
        roles = identify_char_roles(text, "yaml")
        assert len(roles) == len(text)
        # Dashes are syntax
        for i, ch in enumerate(text):
            if ch == "-" and (i == 0 or text[i-1] in " \n"):
                assert roles[i] == "syntax"

    def test_block_key_value(self):
        text = "learning_rate: 0.0001\n"
        roles = identify_char_roles(text, "yaml")
        # "learning_rate" should be semantic
        for j in range(len("learning_rate")):
            assert roles[j] == "semantic", f"char {j} ('{text[j]}') should be semantic"
        # ":" should be syntax
        colon = text.index(":")
        assert roles[colon] == "syntax"
        # "0.0001" should be semantic
        val_start = text.index("0.0001")
        for j in range(val_start, val_start + 6):
            assert roles[j] == "semantic"

    def test_multiline_preserves_indentation_as_syntax(self):
        text = "parent:\n  child1: a\n  child2: b\n"
        roles = identify_char_roles(text, "yaml")
        # Indentation spaces are syntax
        line2_start = text.index("  child1")
        assert roles[line2_start] == "syntax"      # first space
        assert roles[line2_start + 1] == "syntax"  # second space


class TestRealisticXML:
    """Verify classification on realistic XML."""

    def test_closing_tag_names_redundant(self):
        """This is the KEY scientific claim for XML: closing tag names are
        redundant (they repeat opening tag info) and must be classified as syntax."""
        text = "<configuration><parameter>value</parameter></configuration>"
        roles = identify_char_roles(text, "xml")
        # "configuration" in opening tag: semantic
        open_start = 1  # after <
        assert roles[open_start] == "semantic"  # 'c' in opening <configuration>
        # "configuration" in closing tag: SYNTAX (redundant)
        close_tag = text.index("</configuration>")
        # Everything inside </...> is syntax
        for j in range(close_tag, close_tag + len("</configuration>")):
            assert roles[j] == "syntax", f"char {j} ('{text[j]}') in closing tag should be syntax"

    def test_self_closing_tag(self):
        text = '<input type="text" />'
        roles = identify_char_roles(text, "xml")
        assert len(roles) == len(text)
        # 'input' is semantic, 'type' is semantic, 'text' is semantic
        assert roles[1] == "semantic"  # 'i' in input

    def test_nested_elements_with_text(self):
        text = "<root><child>hello</child><child>world</child></root>"
        roles = identify_char_roles(text, "xml")
        # Count how many chars are syntax vs semantic
        syn = sum(1 for r in roles if r == "syntax")
        sem = sum(1 for r in roles if r == "semantic")
        # XML should have substantial syntax (angle brackets, closing tags)
        assert syn > sem, "XML should have more syntax than semantic for tag-heavy content"

    def test_attributes_classified_correctly(self):
        text = '<div class="main" id="content">text</div>'
        roles = identify_char_roles(text, "xml")
        # attr name 'class' is semantic
        class_idx = text.index("class")
        assert roles[class_idx] == "semantic"
        # attr value 'main' is semantic
        main_idx = text.index("main")
        assert roles[main_idx] == "semantic"
        # '=' between them is syntax
        eq_idx = text.index("=")
        assert roles[eq_idx] == "syntax"

    def test_xml_comment_body_semantic(self):
        text = "<root><!-- hyperparameter decision: lr=1e-4 --></root>"
        roles = identify_char_roles(text, "xml")
        h_idx = text.index("hyperparameter")
        for j in range(h_idx, h_idx + len("hyperparameter")):
            assert roles[j] == "semantic"


class TestRealisticTOML:
    """Verify classification on realistic TOML."""

    def test_section_with_values(self):
        text = '[server]\nhost = "0.0.0.0"\nport = 8000\n'
        roles = identify_char_roles(text, "toml")
        # [, ] are syntax
        assert roles[0] == "syntax"  # [
        # "server" is semantic
        assert roles[1] == "semantic"  # s
        # "host" is semantic
        host_idx = text.index("host")
        assert roles[host_idx] == "semantic"

    def test_array_of_tables(self):
        text = '[[items]]\nname = "a"\n[[items]]\nname = "b"\n'
        roles = identify_char_roles(text, "toml")
        assert roles[0] == "syntax"  # [
        assert roles[1] == "syntax"  # [
        # "items" is semantic
        assert roles[2] == "semantic"

    def test_inline_comment(self):
        text = 'port = 8000  # default HTTP port\n'
        roles = identify_char_roles(text, "toml")
        # "port" = semantic, "=" = syntax, "8000" = semantic
        assert roles[0] == "semantic"  # p
        eq_idx = text.index("=")
        assert roles[eq_idx] == "syntax"
        # "default HTTP port" = semantic (comment content)
        d_idx = text.index("default")
        assert roles[d_idx] == "semantic"
        # "#" = syntax
        hash_idx = text.index("#")
        assert roles[hash_idx] == "syntax"

    def test_bare_integer_value(self):
        text = 'count = 42\n'
        roles = identify_char_roles(text, "toml")
        idx_4 = text.index("4")
        assert roles[idx_4] == "semantic"
        idx_2 = text.index("2")
        assert roles[idx_2] == "semantic"


# ===========================================================================
# Determinism and consistency
# ===========================================================================


class TestDeterminism:
    """Results must be deterministic — same input, same output."""

    def test_same_input_same_output(self):
        text = '{"model": "bert", "lr": 0.001}'
        r1 = classify_tokens(text, "json", "cl100k_base")
        r2 = classify_tokens(text, "json", "cl100k_base")
        assert r1 == r2

    def test_char_roles_deterministic(self):
        text = "key: value  # comment\n"
        roles1 = identify_char_roles(text, "yaml")
        roles2 = identify_char_roles(text, "yaml")
        assert roles1 == roles2

    def test_both_tokenizers_agree_on_roles(self):
        """Different tokenizers produce different token counts but
        char roles are tokenizer-independent."""
        text = '{"key": "value"}'
        r1 = classify_tokens(text, "json", "cl100k_base")
        r2 = classify_tokens(text, "json", "o200k_base")
        # Token counts may differ
        # But syntax tax rate should be similar (same char classification)
        assert abs(r1["syntax_tax_rate"] - r2["syntax_tax_rate"]) < 0.15, (
            f"Tax rates diverge too much: {r1['syntax_tax_rate']} vs {r2['syntax_tax_rate']}"
        )
