"""Tests for EXP-012: Agentic Tool Schema Consolidation Overhead.

Measures token savings from consolidating N format-specific tools into CDXF
universal tools. Uses canonical tool-calling schemas from real LLM providers:
OpenAI (Chat Completions + Responses API), Anthropic Claude, Google Gemini,
and Mistral.

TDD: These tests are written BEFORE the implementation.
"""

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp012 import (
    CONTEXT_WINDOW,
    N_FORMAT_COUNTS,
    SCHEMA_FORMATS,
    SESSION_CALL_COUNTS,
    SUPPORTED_FORMATS,
    TOKENIZERS,
    build_cdxf_tools,
    build_corpus,
    build_format_specific_tools,
    compute_savings,
    compute_session_overhead,
    measure_call_result_tokens,
    run_experiment,
    scaling_analysis,
    tokenize_tools,
)


# ===========================================================================
# Constants — Protocol compliance
# ===========================================================================


class TestProtocolConstants:
    """Verify protocol-mandated constants match the experimental design."""

    def test_tokenizers(self):
        assert "cl100k_base" in TOKENIZERS
        assert "o200k_base" in TOKENIZERS
        assert len(TOKENIZERS) == 2

    def test_context_window(self):
        assert CONTEXT_WINDOW == 128_000

    def test_schema_formats_are_real_providers(self):
        """Must use canonical schemas from actual LLM providers."""
        assert "openai_chatcomp" in SCHEMA_FORMATS
        assert "openai_responses" in SCHEMA_FORMATS
        assert "anthropic" in SCHEMA_FORMATS
        assert "gemini" in SCHEMA_FORMATS
        assert "mistral" in SCHEMA_FORMATS
        assert len(SCHEMA_FORMATS) == 5

    def test_supported_formats(self):
        """Protocol: JSON, YAML, XML, TOML."""
        assert set(SUPPORTED_FORMATS) == {"json", "yaml", "xml", "toml"}

    def test_n_format_counts(self):
        """Protocol: scaling analysis for N = {2, 3, 4, 5, 6}."""
        assert set(N_FORMAT_COUNTS) == {2, 3, 4, 5, 6}

    def test_session_call_counts(self):
        """Protocol: 5, 20, 100 tool calls."""
        assert set(SESSION_CALL_COUNTS) == {5, 20, 100}


# ===========================================================================
# Format-specific tool construction
# ===========================================================================


class TestBuildFormatSpecificTools:
    """Tests for building format-specific tool definitions."""

    def test_returns_list(self):
        tools = build_format_specific_tools(["json", "yaml"], "openai_chatcomp")
        assert isinstance(tools, list)

    def test_two_tools_per_format(self):
        """Each format gets parse + emit = 2 tools."""
        tools = build_format_specific_tools(["json"], "openai_chatcomp")
        assert len(tools) == 2

    def test_four_formats_yields_eight_tools(self):
        """Protocol: 4 formats -> 8 tools."""
        tools = build_format_specific_tools(
            ["json", "yaml", "xml", "toml"], "openai_chatcomp"
        )
        assert len(tools) == 8

    def test_six_formats_yields_twelve_tools(self):
        """Scaling test: 6 formats -> 12 tools."""
        fmts = ["json", "yaml", "xml", "toml", "ini", "csv"]
        tools = build_format_specific_tools(fmts, "openai_chatcomp")
        assert len(tools) == 12

    def test_descriptions_are_realistic(self):
        """Descriptions should mention the format and parsing/emitting."""
        tools = build_format_specific_tools(["json"], "openai_chatcomp")
        descriptions = [_extract_description(t, "openai_chatcomp") for t in tools]
        combined = " ".join(descriptions).lower()
        assert "json" in combined
        assert any(w in combined for w in ["parse", "read", "emit", "write",
                                            "serialize", "deserialize",
                                            "encode", "decode", "convert"])


class TestBuildFormatSpecificToolsScaling:
    """Verify 2N scaling property for format-specific tools."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_tool_count_equals_2n(self, n):
        fmts = SUPPORTED_FORMATS[:n] if n <= 4 else (
            list(SUPPORTED_FORMATS) + ["ini", "csv"]
        )[:n]
        tools = build_format_specific_tools(fmts, "openai_chatcomp")
        assert len(tools) == 2 * n


# ===========================================================================
# Canonical schema structure — one class per provider
# ===========================================================================


class TestOpenAIChatCompletionsSchema:
    """Validate canonical OpenAI Chat Completions tool format.

    Format: {"type": "function", "function": {"name", "description", "parameters"}}
    Used by: OpenAI Chat Completions API, legacy integrations.
    """

    def test_top_level_type_is_function(self):
        tools = build_format_specific_tools(["json"], "openai_chatcomp")
        for tool in tools:
            assert tool.get("type") == "function"

    def test_nested_function_key(self):
        tools = build_format_specific_tools(["json"], "openai_chatcomp")
        for tool in tools:
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_parameters_is_json_schema_object(self):
        tools = build_format_specific_tools(["json"], "openai_chatcomp")
        for tool in tools:
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params


class TestOpenAIResponsesSchema:
    """Validate canonical OpenAI Responses API tool format.

    Format: {"type": "function", "name", "description", "parameters"}
    Used by: OpenAI Responses API (newer, flat structure).
    """

    def test_top_level_type_is_function(self):
        tools = build_format_specific_tools(["json"], "openai_responses")
        for tool in tools:
            assert tool.get("type") == "function"

    def test_flat_structure_no_function_wrapper(self):
        """Responses API: name/description/parameters at top level, no 'function' key."""
        tools = build_format_specific_tools(["json"], "openai_responses")
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            # Should NOT have a nested "function" key
            assert "function" not in tool

    def test_parameters_is_json_schema_object(self):
        tools = build_format_specific_tools(["json"], "openai_responses")
        for tool in tools:
            assert tool["parameters"]["type"] == "object"
            assert "properties" in tool["parameters"]


class TestAnthropicSchema:
    """Validate canonical Anthropic Claude tool format.

    Format: {"name", "description", "input_schema"}
    Used by: Anthropic Messages API.
    """

    def test_has_name_description_input_schema(self):
        tools = build_format_specific_tools(["json"], "anthropic")
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_no_type_field(self):
        """Anthropic format does not have a top-level 'type' field."""
        tools = build_format_specific_tools(["json"], "anthropic")
        for tool in tools:
            assert "type" not in tool

    def test_no_parameters_key(self):
        """Anthropic uses 'input_schema', not 'parameters'."""
        tools = build_format_specific_tools(["json"], "anthropic")
        for tool in tools:
            assert "parameters" not in tool

    def test_input_schema_is_json_schema_object(self):
        tools = build_format_specific_tools(["json"], "anthropic")
        for tool in tools:
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema


class TestGeminiSchema:
    """Validate canonical Google Gemini tool format.

    Format: {"function_declarations": [{"name", "description", "parameters"}]}
    Uses OpenAPI-compatible schema. Tools wrapped in function_declarations array.
    """

    def test_top_level_is_function_declarations_wrapper(self):
        """Gemini wraps tools in a single object with function_declarations key."""
        tools = build_format_specific_tools(["json"], "gemini")
        # Gemini returns a single wrapper object, not a list of tools
        assert isinstance(tools, list)
        # Each item is a function declaration (no nesting wrapper per tool)
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool

    def test_parameters_use_openapi_schema(self):
        """Gemini parameters follow OpenAPI schema conventions."""
        tools = build_format_specific_tools(["json"], "gemini")
        for tool in tools:
            params = tool["parameters"]
            assert params["type"] == "object"
            assert "properties" in params


class TestMistralSchema:
    """Validate canonical Mistral tool format.

    Format: same as OpenAI Chat Completions:
    {"type": "function", "function": {"name", "description", "parameters"}}
    """

    def test_same_structure_as_openai_chatcomp(self):
        tools = build_format_specific_tools(["json"], "mistral")
        for tool in tools:
            assert tool.get("type") == "function"
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn


# ===========================================================================
# CDXF universal tool construction
# ===========================================================================


class TestBuildCdxfTools:
    """Tests for building CDXF universal tool definitions."""

    def test_returns_list(self):
        tools = build_cdxf_tools("openai_chatcomp")
        assert isinstance(tools, list)

    def test_exactly_three_tools(self):
        """Protocol: cdxf_encode, cdxf_decode, cdxf_convert = 3."""
        tools = build_cdxf_tools("openai_chatcomp")
        assert len(tools) == 3

    def test_tool_names(self):
        tools = build_cdxf_tools("openai_chatcomp")
        names = {_extract_name(t, "openai_chatcomp") for t in tools}
        assert "cdxf_encode" in names
        assert "cdxf_decode" in names
        assert "cdxf_convert" in names

    @pytest.mark.parametrize("sf", SCHEMA_FORMATS)
    def test_all_providers_produce_three_tools(self, sf):
        tools = build_cdxf_tools(sf)
        assert len(tools) == 3

    @pytest.mark.parametrize("sf", SCHEMA_FORMATS)
    def test_descriptions_mention_cdxf(self, sf):
        tools = build_cdxf_tools(sf)
        for tool in tools:
            desc = _extract_description(tool, sf)
            assert "cdxf" in desc.lower() or "universal" in desc.lower()

    def test_encode_tool_accepts_format_param(self):
        """The encode tool should accept source_format as a parameter."""
        tools = build_cdxf_tools("openai_chatcomp")
        encode_tool = [t for t in tools
                       if _extract_name(t, "openai_chatcomp") == "cdxf_encode"][0]
        params = _extract_parameters(encode_tool, "openai_chatcomp")
        props = params.get("properties", {})
        param_names = set(props.keys())
        assert any("format" in p or "source" in p for p in param_names)

    def test_convert_tool_accepts_source_and_target(self):
        """The convert tool should accept both source and target formats."""
        tools = build_cdxf_tools("openai_chatcomp")
        convert_tool = [t for t in tools
                        if _extract_name(t, "openai_chatcomp") == "cdxf_convert"][0]
        params = _extract_parameters(convert_tool, "openai_chatcomp")
        props = params.get("properties", {})
        param_names_lower = {p.lower() for p in props.keys()}
        assert any("source" in p for p in param_names_lower)
        assert any("target" in p for p in param_names_lower)

    def test_count_is_constant_regardless_of_n_formats(self):
        """CDXF tools = O(1), always 3 regardless of format count."""
        for sf in SCHEMA_FORMATS:
            tools = build_cdxf_tools(sf)
            assert len(tools) == 3


# ===========================================================================
# Tokenization
# ===========================================================================


class TestTokenizeTools:
    """Tests for tokenizing tool schemas."""

    def test_returns_positive_integer(self):
        tools = build_cdxf_tools("openai_chatcomp")
        count = tokenize_tools(tools, "cl100k_base")
        assert isinstance(count, int)
        assert count > 0

    def test_more_tools_means_more_tokens(self):
        """8 format-specific tools should use more tokens than 3 CDXF tools."""
        specific = build_format_specific_tools(
            ["json", "yaml", "xml", "toml"], "openai_chatcomp"
        )
        cdxf = build_cdxf_tools("openai_chatcomp")
        specific_tokens = tokenize_tools(specific, "cl100k_base")
        cdxf_tokens = tokenize_tools(cdxf, "cl100k_base")
        assert specific_tokens > cdxf_tokens

    def test_deterministic(self):
        tools = build_cdxf_tools("openai_chatcomp")
        a = tokenize_tools(tools, "cl100k_base")
        b = tokenize_tools(tools, "cl100k_base")
        assert a == b

    def test_different_tokenizers_may_differ(self):
        tools = build_cdxf_tools("openai_chatcomp")
        a = tokenize_tools(tools, "cl100k_base")
        b = tokenize_tools(tools, "o200k_base")
        assert a > 0
        assert b > 0

    @pytest.mark.parametrize("sf", SCHEMA_FORMATS)
    def test_all_providers_tokenizable(self, sf):
        t = tokenize_tools(build_cdxf_tools(sf), "cl100k_base")
        assert t > 0

    def test_empty_list_returns_zero(self):
        count = tokenize_tools([], "cl100k_base")
        assert count == 0

    def test_schema_format_affects_token_count(self):
        """Different provider formats have different structural overhead."""
        counts = {}
        for sf in SCHEMA_FORMATS:
            tools = build_format_specific_tools(["json"], sf)
            counts[sf] = tokenize_tools(tools, "cl100k_base")
        # Not all formats should produce the exact same count
        # (different key names, nesting levels produce different tokens)
        unique_counts = set(counts.values())
        assert len(unique_counts) >= 2, (
            f"Expected different formats to produce different token counts, "
            f"but all produced: {counts}"
        )


# ===========================================================================
# Savings computation
# ===========================================================================


class TestComputeSavings:
    """Tests for computing token savings metrics."""

    def test_returns_dict(self):
        result = compute_savings(1000, 300)
        assert isinstance(result, dict)

    def test_tokens_saved(self):
        result = compute_savings(1000, 300)
        assert result["tokens_saved"] == 700

    def test_savings_fraction_of_context(self):
        result = compute_savings(1000, 300)
        expected = 700 / CONTEXT_WINDOW
        assert abs(result["savings_fraction"] - expected) < 1e-10

    def test_reduction_percentage(self):
        result = compute_savings(1000, 300)
        expected = 70.0
        assert abs(result["reduction_pct"] - expected) < 1e-10

    def test_zero_specific_tokens(self):
        result = compute_savings(0, 0)
        assert result["tokens_saved"] == 0

    def test_includes_both_counts(self):
        result = compute_savings(800, 250)
        assert result["format_specific_tokens"] == 800
        assert result["cdxf_tokens"] == 250


# ===========================================================================
# Corpus construction
# ===========================================================================


class TestBuildCorpus:
    """Tests for building the evaluation corpus from existing data."""

    def test_returns_list(self):
        corpus = build_corpus()
        assert isinstance(corpus, list)

    def test_corpus_not_empty(self):
        corpus = build_corpus()
        assert len(corpus) > 0

    def test_entry_structure(self):
        corpus = build_corpus()
        for name, fmt, text in corpus:
            assert isinstance(name, str)
            assert isinstance(fmt, str)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_formats_in_corpus(self):
        corpus = build_corpus()
        formats = {fmt for _, fmt, _ in corpus}
        assert len(formats) >= 2

    def test_minimum_corpus_size(self):
        """Protocol: 20 sample configs. Should have at least 20."""
        corpus = build_corpus()
        assert len(corpus) >= 20


# ===========================================================================
# Call result token measurement
# ===========================================================================


class TestMeasureCallResultTokens:
    """Tests for measuring tokens consumed by tool call results."""

    def test_returns_dict(self):
        text = '{"learning_rate": 0.001, "epochs": 10}'
        result = measure_call_result_tokens(text, "json", "cl100k_base")
        assert isinstance(result, dict)

    def test_has_total_tokens(self):
        text = '{"key": "value"}'
        result = measure_call_result_tokens(text, "json", "cl100k_base")
        assert "total_tokens" in result
        assert result["total_tokens"] > 0

    def test_has_format(self):
        text = '{"key": "value"}'
        result = measure_call_result_tokens(text, "json", "cl100k_base")
        assert result["format"] == "json"

    def test_has_size_bytes(self):
        text = '{"key": "value"}'
        result = measure_call_result_tokens(text, "json", "cl100k_base")
        assert "size_bytes" in result
        assert result["size_bytes"] == len(text.encode("utf-8"))

    def test_larger_text_more_tokens(self):
        small = '{"a": 1}'
        large = json.dumps({f"key_{i}": f"value_{i}" for i in range(50)})
        r_small = measure_call_result_tokens(small, "json", "cl100k_base")
        r_large = measure_call_result_tokens(large, "json", "cl100k_base")
        assert r_large["total_tokens"] > r_small["total_tokens"]

    def test_deterministic(self):
        text = '{"a": 1, "b": 2}'
        a = measure_call_result_tokens(text, "json", "cl100k_base")
        b = measure_call_result_tokens(text, "json", "cl100k_base")
        assert a["total_tokens"] == b["total_tokens"]


# ===========================================================================
# Session overhead projection
# ===========================================================================


class TestComputeSessionOverhead:
    """Tests for projecting total session overhead."""

    def test_returns_dict(self):
        result = compute_session_overhead(500, 100, 5)
        assert isinstance(result, dict)

    def test_overhead_formula(self):
        result = compute_session_overhead(500, 100, 5)
        expected = 500 + 5 * 100
        assert result["total_tokens"] == expected

    def test_includes_components(self):
        result = compute_session_overhead(500, 100, 10)
        assert result["schema_tokens"] == 500
        assert result["avg_call_result_tokens"] == 100
        assert result["n_calls"] == 10

    def test_fraction_of_context(self):
        result = compute_session_overhead(500, 100, 5)
        total = 500 + 5 * 100
        expected_frac = total / CONTEXT_WINDOW
        assert abs(result["fraction_of_context"] - expected_frac) < 1e-10

    def test_zero_calls(self):
        result = compute_session_overhead(500, 100, 0)
        assert result["total_tokens"] == 500

    @pytest.mark.parametrize("n_calls", [5, 20, 100])
    def test_session_sizes(self, n_calls):
        result = compute_session_overhead(500, 100, n_calls)
        assert result["total_tokens"] == 500 + n_calls * 100


# ===========================================================================
# Scaling analysis
# ===========================================================================


class TestScalingAnalysis:
    """Tests for the O(N) vs O(1) scaling analysis."""

    def test_returns_dict(self):
        result = scaling_analysis("openai_chatcomp", "cl100k_base")
        assert isinstance(result, dict)

    def test_has_entries_for_all_n(self):
        result = scaling_analysis("openai_chatcomp", "cl100k_base")
        for n in N_FORMAT_COUNTS:
            assert n in result

    def test_entry_structure(self):
        result = scaling_analysis("openai_chatcomp", "cl100k_base")
        entry = result[N_FORMAT_COUNTS[0]]
        assert "format_specific_tokens" in entry
        assert "cdxf_tokens" in entry
        assert "tokens_saved" in entry
        assert "n_specific_tools" in entry

    def test_specific_tools_grow_with_n(self):
        result = scaling_analysis("openai_chatcomp", "cl100k_base")
        tokens = [result[n]["format_specific_tokens"]
                  for n in sorted(N_FORMAT_COUNTS)]
        for i in range(1, len(tokens)):
            assert tokens[i] > tokens[i - 1]

    def test_cdxf_tokens_constant(self):
        result = scaling_analysis("openai_chatcomp", "cl100k_base")
        cdxf_counts = [result[n]["cdxf_tokens"] for n in N_FORMAT_COUNTS]
        assert len(set(cdxf_counts)) == 1

    def test_specific_tool_count_is_2n(self):
        result = scaling_analysis("openai_chatcomp", "cl100k_base")
        for n in N_FORMAT_COUNTS:
            assert result[n]["n_specific_tools"] == 2 * n

    def test_savings_grow_with_n(self):
        result = scaling_analysis("openai_chatcomp", "cl100k_base")
        savings = [result[n]["tokens_saved"] for n in sorted(N_FORMAT_COUNTS)]
        for i in range(1, len(savings)):
            assert savings[i] > savings[i - 1]


# ===========================================================================
# Hypothesis validation — the scientific core
# ===========================================================================


class TestHypothesisValidation:
    """Validate the experimental hypothesis: CDXF consolidation saves tokens."""

    @pytest.mark.parametrize("sf", SCHEMA_FORMATS)
    def test_cdxf_uses_fewer_tokens_than_format_specific(self, sf):
        """Core hypothesis: 3 CDXF tools < 8 format-specific tools (all providers)."""
        specific = build_format_specific_tools(
            ["json", "yaml", "xml", "toml"], sf
        )
        cdxf = build_cdxf_tools(sf)
        s_tokens = tokenize_tools(specific, "cl100k_base")
        c_tokens = tokenize_tools(cdxf, "cl100k_base")
        assert c_tokens < s_tokens, (
            f"CDXF ({c_tokens}) should be < format-specific ({s_tokens}) "
            f"for provider {sf}"
        )

    @pytest.mark.parametrize("sf", SCHEMA_FORMATS)
    def test_savings_are_nontrivial(self, sf):
        """Savings should be at least 20% reduction (3 vs 8 tools)."""
        specific = build_format_specific_tools(
            ["json", "yaml", "xml", "toml"], sf
        )
        cdxf = build_cdxf_tools(sf)
        s_tokens = tokenize_tools(specific, "cl100k_base")
        c_tokens = tokenize_tools(cdxf, "cl100k_base")
        reduction_pct = (s_tokens - c_tokens) / s_tokens * 100
        assert reduction_pct >= 20.0, (
            f"Expected >=20% reduction for {sf}, got {reduction_pct:.1f}%"
        )

    def test_savings_scale_linearly_with_n(self):
        result = scaling_analysis("openai_chatcomp", "cl100k_base")
        ns = sorted(N_FORMAT_COUNTS)
        savings = [result[n]["tokens_saved"] for n in ns]
        assert savings[-1] > 2.0 * savings[0]


# ===========================================================================
# Schema quality — realistic, not trivial
# ===========================================================================


class TestSchemaQuality:
    """Ensure tool schemas are realistic, not trivially short."""

    @pytest.mark.parametrize("sf", SCHEMA_FORMATS)
    def test_format_specific_min_token_count(self, sf):
        """Each format-specific tool should be at least ~30 tokens."""
        tools = build_format_specific_tools(["json"], sf)
        for tool in tools:
            tokens = tokenize_tools([tool], "cl100k_base")
            assert tokens >= 30, (
                f"Tool too short ({tokens} tokens) for provider {sf}. "
                "Use realistic descriptions and parameter schemas."
            )

    @pytest.mark.parametrize("sf", SCHEMA_FORMATS)
    def test_cdxf_tool_min_token_count(self, sf):
        """Each CDXF tool should be at least ~30 tokens."""
        tools = build_cdxf_tools(sf)
        for tool in tools:
            tokens = tokenize_tools([tool], "cl100k_base")
            assert tokens >= 30, (
                f"Tool too short ({tokens} tokens) for provider {sf}."
            )

    def test_parameter_schemas_have_types(self):
        """Parameters should have type annotations (spot-check openai_chatcomp)."""
        tools = build_format_specific_tools(["json"], "openai_chatcomp")
        for tool in tools:
            params = _extract_parameters(tool, "openai_chatcomp")
            if "properties" in params:
                for prop_name, prop_def in params["properties"].items():
                    assert "type" in prop_def or "enum" in prop_def, (
                        f"Parameter {prop_name} lacks type information"
                    )


# ===========================================================================
# Cross-provider comparison — novel contribution
# ===========================================================================


class TestCrossProviderComparison:
    """Ensure we can compare overhead across providers — key paper contribution."""

    def test_all_providers_produce_valid_savings(self):
        """Every provider format should produce positive savings."""
        for sf in SCHEMA_FORMATS:
            specific = build_format_specific_tools(SUPPORTED_FORMATS, sf)
            cdxf = build_cdxf_tools(sf)
            s = tokenize_tools(specific, "cl100k_base")
            c = tokenize_tools(cdxf, "cl100k_base")
            assert s > c, f"No savings for provider {sf}"

    def test_nested_formats_use_more_tokens_than_flat(self):
        """OpenAI Chat Completions (nested) should use more tokens than
        Anthropic/Responses API (flat) for the same tool definitions."""
        specific_nested = build_format_specific_tools(
            SUPPORTED_FORMATS, "openai_chatcomp"
        )
        specific_flat = build_format_specific_tools(
            SUPPORTED_FORMATS, "openai_responses"
        )
        t_nested = tokenize_tools(specific_nested, "cl100k_base")
        t_flat = tokenize_tools(specific_flat, "cl100k_base")
        assert t_nested > t_flat, (
            f"Expected nested ({t_nested}) > flat ({t_flat})"
        )


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
        assert results["experiment"] == "EXP-012"

    def test_has_timestamp(self, results):
        assert "timestamp" in results

    def test_has_schema_comparison(self, results):
        assert "schema_comparison" in results

    def test_has_scaling_analysis(self, results):
        assert "scaling_analysis" in results

    def test_has_session_overhead(self, results):
        assert "session_overhead" in results

    def test_has_call_result_analysis(self, results):
        assert "call_result_analysis" in results

    def test_has_corpus_size(self, results):
        assert "corpus_size" in results
        assert results["corpus_size"] > 0

    def test_schema_comparison_covers_all_providers(self, results):
        """Every provider × tokenizer should have results."""
        sc = results["schema_comparison"]
        for sf in SCHEMA_FORMATS:
            assert sf in sc, f"Missing provider {sf}"
            for tok in TOKENIZERS:
                assert tok in sc[sf], f"Missing tokenizer {tok} for {sf}"
                entry = sc[sf][tok]
                assert "format_specific_tokens" in entry
                assert "cdxf_tokens" in entry
                assert "tokens_saved" in entry

    def test_session_overhead_structure(self, results):
        so = results["session_overhead"]
        for n_calls in SESSION_CALL_COUNTS:
            assert str(n_calls) in so or n_calls in so


# ===========================================================================
# Helpers
# ===========================================================================


def _extract_name(tool: dict, schema_format: str) -> str:
    """Extract tool name from provider-specific format."""
    if schema_format in ("openai_chatcomp", "mistral"):
        return tool["function"]["name"]
    # openai_responses, anthropic, gemini — flat
    return tool["name"]


def _extract_description(tool: dict, schema_format: str) -> str:
    """Extract description from provider-specific format."""
    if schema_format in ("openai_chatcomp", "mistral"):
        return tool["function"].get("description", "")
    return tool.get("description", "")


def _extract_parameters(tool: dict, schema_format: str) -> dict:
    """Extract parameters/input_schema from provider-specific format."""
    if schema_format in ("openai_chatcomp", "mistral"):
        return tool["function"].get("parameters", {})
    if schema_format == "anthropic":
        return tool.get("input_schema", {})
    # openai_responses, gemini
    return tool.get("parameters", {})
