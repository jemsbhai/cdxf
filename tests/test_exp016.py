"""Tests for EXP-016: MCP Tool Server — CDXF Universal Config Tools.

Build real MCP servers exposing format-specific tools vs CDXF universal
tools. Compare tool count, schema overhead, and metadata fidelity through
actual MCP tool call invocations.

TDD: Tests written BEFORE implementation.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp016 import (
    SUPPORTED_FORMATS,
    build_cdxf_mcp_server,
    build_format_specific_mcp_server,
    call_tool,
    count_comments,
    get_tool_schemas,
    run_experiment,
    tokenize_schemas,
)


# Helper to run async functions in sync tests
def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# MCP server construction
# ===========================================================================


class TestBuildFormatSpecificMcpServer:
    """Tests for format-specific MCP server."""

    def test_returns_server(self):
        server = build_format_specific_mcp_server()
        assert server is not None

    def test_has_eight_tools(self):
        """4 formats × 2 (parse + emit) = 8 tools."""
        server = build_format_specific_mcp_server()
        tools = _run(get_tool_schemas(server))
        assert len(tools) == 8

    def test_tool_names_include_format(self):
        server = build_format_specific_mcp_server()
        tools = _run(get_tool_schemas(server))
        names = {t.name for t in tools}
        for fmt in SUPPORTED_FORMATS:
            assert any(fmt in n for n in names), (
                f"No tool name contains '{fmt}'"
            )

    def test_tools_have_descriptions(self):
        server = build_format_specific_mcp_server()
        tools = _run(get_tool_schemas(server))
        for tool in tools:
            assert tool.description and len(tool.description) > 10

    def test_tools_have_input_schemas(self):
        server = build_format_specific_mcp_server()
        tools = _run(get_tool_schemas(server))
        for tool in tools:
            assert tool.inputSchema is not None
            assert "properties" in tool.inputSchema


class TestBuildCdxfMcpServer:
    """Tests for CDXF universal MCP server."""

    def test_returns_server(self):
        server = build_cdxf_mcp_server()
        assert server is not None

    def test_has_three_tools(self):
        """cdxf_encode, cdxf_decode, cdxf_convert = 3 tools."""
        server = build_cdxf_mcp_server()
        tools = _run(get_tool_schemas(server))
        assert len(tools) == 3

    def test_tool_names(self):
        server = build_cdxf_mcp_server()
        tools = _run(get_tool_schemas(server))
        names = {t.name for t in tools}
        assert names == {"cdxf_encode", "cdxf_decode", "cdxf_convert"}

    def test_tools_have_descriptions(self):
        server = build_cdxf_mcp_server()
        tools = _run(get_tool_schemas(server))
        for tool in tools:
            assert tool.description and len(tool.description) > 10

    def test_tools_have_input_schemas(self):
        server = build_cdxf_mcp_server()
        tools = _run(get_tool_schemas(server))
        for tool in tools:
            assert tool.inputSchema is not None
            assert "properties" in tool.inputSchema


# ===========================================================================
# Schema token comparison
# ===========================================================================


class TestTokenizeSchemas:
    """Tests for comparing schema token overhead."""

    def test_returns_dict(self):
        result = _run(tokenize_schemas())
        assert isinstance(result, dict)

    def test_has_both_servers(self):
        result = _run(tokenize_schemas())
        assert "format_specific" in result
        assert "cdxf_universal" in result

    def test_format_specific_more_tokens(self):
        result = _run(tokenize_schemas())
        assert result["format_specific"]["tokens"] > (
            result["cdxf_universal"]["tokens"]
        )

    def test_has_tool_counts(self):
        result = _run(tokenize_schemas())
        assert result["format_specific"]["n_tools"] == 8
        assert result["cdxf_universal"]["n_tools"] == 3

    def test_has_savings(self):
        result = _run(tokenize_schemas())
        assert "tokens_saved" in result
        assert "reduction_pct" in result
        assert result["tokens_saved"] > 0


# ===========================================================================
# Tool invocation — actual MCP calls
# ===========================================================================


class TestCallToolFormatSpecific:
    """Tests for invoking format-specific MCP tools."""

    def test_parse_json(self):
        server = build_format_specific_mcp_server()
        text = '{"key": "value", "count": 42}'
        result = _run(call_tool(server, "parse_json", {"content": text}))
        assert result["success"] is True
        assert "data" in result

    def test_emit_json(self):
        server = build_format_specific_mcp_server()
        result = _run(call_tool(server, "emit_json", {
            "data": {"key": "value"}
        }))
        assert result["success"] is True
        parsed = json.loads(result["text"])
        assert parsed["key"] == "value"

    def test_parse_yaml(self):
        server = build_format_specific_mcp_server()
        text = "# comment\nkey: value\n"
        result = _run(call_tool(server, "parse_yaml", {"content": text}))
        assert result["success"] is True

    def test_parse_yaml_loses_comments(self):
        """Format-specific parse_yaml uses yaml.safe_load — comments lost."""
        server = build_format_specific_mcp_server()
        text = "# important note\nkey: value\n"
        result = _run(call_tool(server, "parse_yaml", {"content": text}))
        assert result["success"] is True
        # Re-emit and check: comments should be gone
        emit_result = _run(call_tool(server, "emit_yaml", {
            "data": result["data"]
        }))
        assert count_comments(emit_result["text"], "yaml") == 0


class TestCallToolCdxf:
    """Tests for invoking CDXF universal MCP tools."""

    def test_encode_yaml(self):
        server = build_cdxf_mcp_server()
        text = "# comment\nkey: value\n"
        result = _run(call_tool(server, "cdxf_encode", {
            "content": text, "source_format": "yaml"
        }))
        assert result["success"] is True
        assert "cdxf_data" in result

    def test_decode_yaml(self):
        server = build_cdxf_mcp_server()
        # Encode first
        text = "# comment\nkey: value\n"
        enc = _run(call_tool(server, "cdxf_encode", {
            "content": text, "source_format": "yaml"
        }))
        # Decode back
        dec = _run(call_tool(server, "cdxf_decode", {
            "cdxf_data": enc["cdxf_data"], "target_format": "yaml"
        }))
        assert dec["success"] is True
        assert "text" in dec

    def test_encode_decode_preserves_comments(self):
        """CDXF round-trip must preserve YAML comments."""
        server = build_cdxf_mcp_server()
        text = "# important note\n# second comment\nkey: value\ncount: 42\n"
        original_comments = count_comments(text, "yaml")
        assert original_comments >= 2

        enc = _run(call_tool(server, "cdxf_encode", {
            "content": text, "source_format": "yaml"
        }))
        dec = _run(call_tool(server, "cdxf_decode", {
            "cdxf_data": enc["cdxf_data"], "target_format": "yaml"
        }))
        restored_comments = count_comments(dec["text"], "yaml")
        assert restored_comments == original_comments

    def test_convert_yaml_to_json_to_yaml(self):
        """Convert YAML → JSON → YAML, comments should survive via CDXF."""
        server = build_cdxf_mcp_server()
        text = "# note\nkey: value\n"

        # YAML → JSON
        r1 = _run(call_tool(server, "cdxf_convert", {
            "content": text, "source_format": "yaml",
            "target_format": "json"
        }))
        assert r1["success"] is True

        # JSON → YAML (via encode then decode)
        enc = _run(call_tool(server, "cdxf_encode", {
            "content": text, "source_format": "yaml"
        }))
        dec = _run(call_tool(server, "cdxf_decode", {
            "cdxf_data": enc["cdxf_data"], "target_format": "yaml"
        }))
        assert count_comments(dec["text"], "yaml") >= 1

    def test_convert_all_formats(self):
        """CDXF convert should work for all supported formats."""
        server = build_cdxf_mcp_server()
        text = "key: value\ncount: 42\n"
        for target in ["json", "toml"]:
            result = _run(call_tool(server, "cdxf_convert", {
                "content": text, "source_format": "yaml",
                "target_format": target,
            }))
            assert result["success"] is True, (
                f"Convert to {target} failed"
            )


# ===========================================================================
# Fidelity comparison — the scientific core
# ===========================================================================


class TestFidelityComparison:
    """Compare metadata fidelity: format-specific vs CDXF."""

    def test_format_specific_round_trip_loses_comments(self):
        """parse_yaml + emit_yaml loses comments."""
        server = build_format_specific_mcp_server()
        text = "# LR from grid search\nlearning_rate: 0.001\n# Seed\nseed: 42\n"
        original = count_comments(text, "yaml")
        assert original >= 2

        parsed = _run(call_tool(server, "parse_yaml", {"content": text}))
        emitted = _run(call_tool(server, "emit_yaml", {
            "data": parsed["data"]
        }))
        assert count_comments(emitted["text"], "yaml") == 0

    def test_cdxf_round_trip_preserves_comments(self):
        """cdxf_encode + cdxf_decode preserves comments."""
        server = build_cdxf_mcp_server()
        text = "# LR from grid search\nlearning_rate: 0.001\n# Seed\nseed: 42\n"
        original = count_comments(text, "yaml")

        enc = _run(call_tool(server, "cdxf_encode", {
            "content": text, "source_format": "yaml"
        }))
        dec = _run(call_tool(server, "cdxf_decode", {
            "cdxf_data": enc["cdxf_data"], "target_format": "yaml"
        }))
        assert count_comments(dec["text"], "yaml") == original


# ===========================================================================
# Full experiment (integration)
# ===========================================================================


class TestRunExperiment:
    """Integration tests for the full experiment."""

    @pytest.fixture(scope="class")
    def results(self):
        return _run(run_experiment())

    def test_returns_dict(self, results):
        assert isinstance(results, dict)

    def test_has_experiment_id(self, results):
        assert results["experiment"] == "EXP-016"

    def test_has_timestamp(self, results):
        assert "timestamp" in results

    def test_has_schema_comparison(self, results):
        assert "schema_comparison" in results

    def test_has_fidelity_results(self, results):
        assert "fidelity_results" in results

    def test_has_summary(self, results):
        assert "summary" in results
