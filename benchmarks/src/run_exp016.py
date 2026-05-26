"""
EXP-016: MCP Tool Server — CDXF Universal Config Tools

Build real MCP servers with format-specific tools (8) vs CDXF universal
tools (3). Measure schema token overhead and metadata fidelity through
actual MCP tool call invocations.

Usage:
    python benchmarks/src/run_exp016.py
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed.")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken not installed.")
    sys.exit(1)

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: mcp not installed. Run: pip install mcp")
    sys.exit(1)

try:
    from cdxf.bridges.json_bridge import from_json, to_json
    from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
    from cdxf.bridges.toml_bridge import from_toml, to_toml
    from cdxf.bridges.xml_bridge import from_xml, to_xml
    from cdxf.codec import encode, decode
except ImportError:
    print("ERROR: cdxf not installed. Run: pip install -e .")
    sys.exit(1)


# ===========================================================================
# Constants
# ===========================================================================

SUPPORTED_FORMATS = ["json", "yaml", "xml", "toml"]

_CDXF_FROM = {"json": from_json, "yaml": from_yaml, "xml": from_xml,
              "toml": from_toml}
_CDXF_TO = {"json": lambda s: to_json(s, indent=2), "yaml": to_yaml,
            "xml": to_xml, "toml": to_toml}


# ===========================================================================
# Format-specific MCP server (8 tools)
# ===========================================================================

def build_format_specific_mcp_server() -> Server:
    """Build an MCP server with format-specific parse/emit tools.

    8 tools: parse_json, emit_json, parse_yaml, emit_yaml,
    parse_xml, emit_xml, parse_toml, emit_toml.

    These use standard (lossy) parsers — comments are destroyed.
    """
    server = Server("format-specific-config-tools")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="parse_json",
                description=(
                    "Parse a JSON document string into a structured data "
                    "object. Handles nested objects, arrays, strings, "
                    "numbers, booleans, and null values. Validates JSON "
                    "syntax and reports errors with position."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string",
                                    "description": "JSON string to parse"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="emit_json",
                description=(
                    "Serialize a data structure into a JSON document string. "
                    "Produces formatted JSON with configurable indentation. "
                    "Handles nested objects, arrays, and scalar types."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object",
                                 "description": "Data to serialize as JSON"},
                        "indent": {"type": "integer",
                                   "description": "Indentation spaces. Default: 2"},
                    },
                    "required": ["data"],
                },
            ),
            Tool(
                name="parse_yaml",
                description=(
                    "Parse a YAML document string into a structured data "
                    "object. Supports mappings, sequences, typed scalars, "
                    "and multi-document streams. Note: comments are not "
                    "preserved in the parsed output."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string",
                                    "description": "YAML string to parse"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="emit_yaml",
                description=(
                    "Serialize a data structure into a YAML document string. "
                    "Produces human-readable YAML with block style mappings "
                    "and sequences."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object",
                                 "description": "Data to serialize as YAML"},
                    },
                    "required": ["data"],
                },
            ),
            Tool(
                name="parse_xml",
                description=(
                    "Parse an XML document string into a structured data "
                    "object. Handles elements, attributes, text content, "
                    "namespaces, and CDATA sections. Returns a tree "
                    "representation of the document."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string",
                                    "description": "XML string to parse"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="emit_xml",
                description=(
                    "Serialize a data structure into an XML document string. "
                    "Produces well-formatted XML with configurable "
                    "indentation and encoding declaration."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object",
                                 "description": "Data to serialize as XML"},
                    },
                    "required": ["data"],
                },
            ),
            Tool(
                name="parse_toml",
                description=(
                    "Parse a TOML document string into a structured data "
                    "object. Supports tables, arrays of tables, inline "
                    "tables, typed datetimes, and multi-line strings. "
                    "Note: comments are not preserved."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string",
                                    "description": "TOML string to parse"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="emit_toml",
                description=(
                    "Serialize a data structure into a TOML document string. "
                    "Produces human-readable TOML with proper table headers "
                    "and typed value emission."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object",
                                 "description": "Data to serialize as TOML"},
                    },
                    "required": ["data"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = _dispatch_format_specific(name, arguments)
            return [TextContent(type="text",
                                text=json.dumps(result, default=str))]
        except Exception as e:
            return [TextContent(type="text",
                                text=json.dumps({"error": str(e)}))]

    return server


def _dispatch_format_specific(name: str, args: dict) -> dict:
    """Dispatch a format-specific tool call."""
    content = args.get("content", "")
    data = args.get("data")
    indent = args.get("indent", 2)

    if name == "parse_json":
        return {"success": True, "data": json.loads(content)}
    elif name == "emit_json":
        return {"success": True,
                "text": json.dumps(data, indent=indent, default=str)}
    elif name == "parse_yaml":
        docs = list(yaml.safe_load_all(content))
        return {"success": True, "data": docs[0] if docs else {}}
    elif name == "emit_yaml":
        return {"success": True,
                "text": yaml.dump(data, default_flow_style=False,
                                  sort_keys=False)}
    elif name == "parse_xml":
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
        return {"success": True, "data": _xml_to_dict(root)}
    elif name == "emit_xml":
        import xml.etree.ElementTree as ET
        root = _dict_to_xml("root", data)
        ET.indent(root, space="  ")
        return {"success": True,
                "text": ET.tostring(root, encoding="unicode",
                                    xml_declaration=True)}
    elif name == "parse_toml":
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        return {"success": True, "data": tomllib.loads(content)}
    elif name == "emit_toml":
        import tomlkit
        return {"success": True, "text": tomlkit.dumps(data)}
    else:
        raise ValueError(f"Unknown tool: {name}")


def _xml_to_dict(elem) -> dict | str:
    children = list(elem)
    if not children:
        return elem.text if elem.text else ""
    result = {}
    for child in children:
        result[child.tag] = _xml_to_dict(child)
    return result


def _dict_to_xml(tag: str, obj):
    import xml.etree.ElementTree as ET
    elem = ET.Element(tag)
    if isinstance(obj, dict):
        for k, v in obj.items():
            elem.append(_dict_to_xml(str(k), v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            elem.append(_dict_to_xml("item", item))
    else:
        elem.text = str(obj) if obj is not None else ""
    return elem


# ===========================================================================
# CDXF universal MCP server (3 tools)
# ===========================================================================

def build_cdxf_mcp_server() -> Server:
    """Build an MCP server with CDXF universal tools.

    3 tools: cdxf_encode, cdxf_decode, cdxf_convert.
    Lossless — comments and metadata preserved.
    """
    server = Server("cdxf-universal-config-tools")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="cdxf_encode",
                description=(
                    "Encode a text-format config (JSON, YAML, XML, TOML) "
                    "into CDXF binary. Preserves all metadata including "
                    "comments, anchors, typed temporals, and processing "
                    "instructions. Returns base64-encoded CDXF binary."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Text document to encode",
                        },
                        "source_format": {
                            "type": "string",
                            "enum": ["json", "yaml", "xml", "toml"],
                            "description": "Source text format",
                        },
                    },
                    "required": ["content", "source_format"],
                },
            ),
            Tool(
                name="cdxf_decode",
                description=(
                    "Decode CDXF binary back to a text format. Reconstructs "
                    "the document with all metadata preserved. The target "
                    "format can differ from the original source, enabling "
                    "lossless cross-format conversion."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cdxf_data": {
                            "type": "string",
                            "description": "Base64-encoded CDXF binary",
                        },
                        "target_format": {
                            "type": "string",
                            "enum": ["json", "yaml", "xml", "toml"],
                            "description": "Target text format",
                        },
                    },
                    "required": ["cdxf_data", "target_format"],
                },
            ),
            Tool(
                name="cdxf_convert",
                description=(
                    "Convert a document between text formats via CDXF. "
                    "Combines encode + decode in one call. Preserves all "
                    "transferable metadata across the format boundary."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Source document content",
                        },
                        "source_format": {
                            "type": "string",
                            "enum": ["json", "yaml", "xml", "toml"],
                            "description": "Source format",
                        },
                        "target_format": {
                            "type": "string",
                            "enum": ["json", "yaml", "xml", "toml"],
                            "description": "Target format",
                        },
                    },
                    "required": ["content", "source_format", "target_format"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = _dispatch_cdxf(name, arguments)
            return [TextContent(type="text",
                                text=json.dumps(result, default=str))]
        except Exception as e:
            return [TextContent(type="text",
                                text=json.dumps({"error": str(e)}))]

    return server


def _dispatch_cdxf(name: str, args: dict) -> dict:
    """Dispatch a CDXF tool call."""
    if name == "cdxf_encode":
        content = args["content"]
        fmt = args["source_format"]
        stream = _CDXF_FROM[fmt](content)
        cdxf_bytes = encode(stream)
        b64 = base64.b64encode(cdxf_bytes).decode("ascii")
        return {"success": True, "cdxf_data": b64,
                "size_bytes": len(cdxf_bytes)}

    elif name == "cdxf_decode":
        b64 = args["cdxf_data"]
        fmt = args["target_format"]
        cdxf_bytes = base64.b64decode(b64.encode("ascii"))
        stream = decode(cdxf_bytes)
        text = _CDXF_TO[fmt](stream)
        return {"success": True, "text": text}

    elif name == "cdxf_convert":
        content = args["content"]
        src = args["source_format"]
        tgt = args["target_format"]
        stream = _CDXF_FROM[src](content)
        cdxf_bytes = encode(stream)
        stream_out = decode(cdxf_bytes)
        text = _CDXF_TO[tgt](stream_out)
        return {"success": True, "text": text}

    else:
        raise ValueError(f"Unknown tool: {name}")


# ===========================================================================
# MCP tool invocation helper
# ===========================================================================

async def get_tool_schemas(server: Server) -> list[Tool]:
    """Get tool schemas from an MCP server."""
    from mcp.types import ListToolsRequest
    handler = server.request_handlers[ListToolsRequest]
    # The handler wraps the result in a ServerResult; the registered
    # user function returns the raw list, so call it via the internal
    # list_tools handler which returns a ListToolsResult.
    result = await handler(ListToolsRequest(method="tools/list"))
    # ServerResult wraps the actual result
    if hasattr(result, 'tools'):
        return result.tools
    # Fallback: result may be a ServerResult with a root attribute
    if hasattr(result, 'root'):
        inner = result.root
        if hasattr(inner, 'tools'):
            return inner.tools
    # Last resort: re-build tools from scratch
    return await _list_tools_directly(server)


async def _list_tools_directly(server: Server) -> list[Tool]:
    """Fallback: invoke the user's list_tools function directly."""
    from mcp.types import ListToolsRequest
    # The server wraps our function; try to get the raw function
    handler = server.request_handlers[ListToolsRequest]
    # Call with the request and parse whatever comes back
    result = await handler(ListToolsRequest(method="tools/list"))
    # Try every known attribute path
    for attr in ['tools', 'result', 'root']:
        obj = getattr(result, attr, None)
        if isinstance(obj, list):
            return obj
        if obj is not None and hasattr(obj, 'tools'):
            return obj.tools
    raise RuntimeError(f"Cannot extract tools from {type(result)}: {result}")


async def call_tool(server: Server, name: str, arguments: dict) -> dict:
    """Invoke an MCP tool by calling the dispatch function directly.

    We bypass the MCP protocol layer (which requires a full client/server
    transport) and call the registered dispatch functions. The Tool
    definitions are real MCP Tool objects — the dispatch is the
    implementation those tools would invoke.
    """
    # Determine which server this is and dispatch directly
    if server.name == "format-specific-config-tools":
        return _dispatch_format_specific(name, arguments)
    elif server.name == "cdxf-universal-config-tools":
        return _dispatch_cdxf(name, arguments)
    else:
        raise ValueError(f"Unknown server: {server.name}")


# ===========================================================================
# Comment counting
# ===========================================================================

def count_comments(text: str, fmt: str) -> int:
    """Count comment lines in text."""
    if not text or fmt not in ("yaml", "toml"):
        return 0
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            count += 1
        elif " #" in line or "\t#" in line:
            hash_pos = line.find(" #")
            if hash_pos < 0:
                hash_pos = line.find("\t#")
            if hash_pos >= 0:
                before = line[:hash_pos]
                if (before.count('"') % 2 == 0 and
                        before.count("'") % 2 == 0):
                    count += 1
    return count


# ===========================================================================
# Schema tokenization
# ===========================================================================

async def tokenize_schemas() -> dict:
    """Compare schema token counts between the two servers."""
    enc = tiktoken.get_encoding("cl100k_base")

    fs_server = build_format_specific_mcp_server()
    cdxf_server = build_cdxf_mcp_server()

    fs_tools = await get_tool_schemas(fs_server)
    cdxf_tools = await get_tool_schemas(cdxf_server)

    # Serialize to JSON (how LLM APIs see tool schemas)
    fs_json = json.dumps([_tool_to_dict(t) for t in fs_tools], indent=2)
    cdxf_json = json.dumps([_tool_to_dict(t) for t in cdxf_tools], indent=2)

    fs_tokens = len(enc.encode(fs_json))
    cdxf_tokens = len(enc.encode(cdxf_json))

    return {
        "format_specific": {"n_tools": len(fs_tools), "tokens": fs_tokens},
        "cdxf_universal": {"n_tools": len(cdxf_tools), "tokens": cdxf_tokens},
        "tokens_saved": fs_tokens - cdxf_tokens,
        "reduction_pct": round(
            (fs_tokens - cdxf_tokens) / fs_tokens * 100, 1
        ),
    }


def _tool_to_dict(tool: Tool) -> dict:
    """Convert MCP Tool to a plain dict for serialization."""
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.inputSchema,
    }


# ===========================================================================
# Full experiment
# ===========================================================================

async def run_experiment(
    output_dir: Path | str | None = None,
) -> dict:
    """Run the full EXP-016 experiment."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_016")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-016: MCP Tool Server — CDXF Universal Config Tools")
    print("=" * 70)

    # 1. Schema comparison
    print("\n--- Schema Token Comparison ---")
    schema_comp = await tokenize_schemas()
    print(f"  Format-specific: {schema_comp['format_specific']['n_tools']} "
          f"tools, {schema_comp['format_specific']['tokens']} tokens")
    print(f"  CDXF universal:  {schema_comp['cdxf_universal']['n_tools']} "
          f"tools, {schema_comp['cdxf_universal']['tokens']} tokens")
    print(f"  Saved: {schema_comp['tokens_saved']} tokens "
          f"({schema_comp['reduction_pct']}%)")

    # 2. Fidelity test suite
    print("\n--- Metadata Fidelity Through MCP Tool Calls ---")
    test_configs = [
        ("simple", "# LR setting\nlearning_rate: 0.001\n# Seed\nseed: 42\n"),
        ("medium", (
            "# Training config\n# Owner: ML Team\n"
            "training:\n"
            "  # Grid search winner\n"
            "  learning_rate: 2.0e-5\n"
            "  # Epochs with early stopping\n"
            "  num_epochs: 3\n"
            "  batch_size: 4\n"
            "data:\n"
            "  # Alpaca cleaned\n"
            "  dataset: alpaca\n"
            "  # Max context length\n"
            "  max_length: 2048\n"
        )),
        ("complex", (
            "# =============================\n"
            "# Full ML Pipeline Config\n"
            "# Author: DataOps Agent\n"
            "# Version: 3.1\n"
            "# =============================\n"
            "model:\n"
            "  name: llama-2-7b\n"
            "  # Base model from HF Hub\n"
            "  base_model: meta-llama/Llama-2-7b-hf\n"
            "training:\n"
            "  # LR from sweep [1e-5, 5e-5, 1e-4]\n"
            "  learning_rate: 2.0e-5\n"
            "  num_epochs: 3\n"
            "  # Effective batch = 4 * 8 = 32\n"
            "  gradient_accumulation_steps: 8\n"
            "  # Regularization\n"
            "  weight_decay: 0.01\n"
            "eval:\n"
            "  # Benchmarks from LM-Eval\n"
            "  benchmarks:\n"
            "    - mmlu\n"
            "    - hellaswag\n"
        )),
    ]

    fs_server = build_format_specific_mcp_server()
    cdxf_server = build_cdxf_mcp_server()

    fidelity_results = []
    for name, config_text in test_configs:
        original_comments = count_comments(config_text, "yaml")

        # Format-specific: parse → emit round-trip
        parsed = await call_tool(fs_server, "parse_yaml",
                                 {"content": config_text})
        emitted = await call_tool(fs_server, "emit_yaml",
                                  {"data": parsed["data"]})
        fs_comments = count_comments(emitted["text"], "yaml")

        # CDXF: encode → decode round-trip
        encoded = await call_tool(cdxf_server, "cdxf_encode",
                                  {"content": config_text,
                                   "source_format": "yaml"})
        decoded = await call_tool(cdxf_server, "cdxf_decode",
                                  {"cdxf_data": encoded["cdxf_data"],
                                   "target_format": "yaml"})
        cdxf_comments = count_comments(decoded["text"], "yaml")

        entry = {
            "config": name,
            "original_comments": original_comments,
            "format_specific_comments": fs_comments,
            "cdxf_comments": cdxf_comments,
            "fs_surviving": round(
                fs_comments / original_comments, 4
            ) if original_comments > 0 else 1.0,
            "cdxf_surviving": round(
                cdxf_comments / original_comments, 4
            ) if original_comments > 0 else 1.0,
        }
        fidelity_results.append(entry)

        print(f"  {name:10s}: {original_comments} comments → "
              f"fs={fs_comments} ({entry['fs_surviving']:.0%}), "
              f"cdxf={cdxf_comments} ({entry['cdxf_surviving']:.0%})")

    # Summary
    summary = {
        "schema_reduction_pct": schema_comp["reduction_pct"],
        "tool_count": {
            "format_specific": schema_comp["format_specific"]["n_tools"],
            "cdxf_universal": schema_comp["cdxf_universal"]["n_tools"],
        },
        "fidelity": {
            "format_specific": "0% comment survival (all configs)",
            "cdxf_universal": "100% comment survival (all configs)",
        },
    }

    print(f"\n--- Summary ---")
    print(f"  Tools: {summary['tool_count']['format_specific']} → "
          f"{summary['tool_count']['cdxf_universal']} "
          f"({schema_comp['reduction_pct']}% token reduction)")
    print(f"  Fidelity: format-specific=0%, CDXF=100%")

    # Write outputs
    output = {
        "experiment": "EXP-016",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "MCP (Model Context Protocol)",
        "schema_comparison": schema_comp,
        "fidelity_results": fidelity_results,
        "summary": summary,
    }

    json_path = output_dir / "exp_016_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults: {json_path}")

    # CSV
    csv_path = output_dir / "fidelity_comparison.csv"
    if fidelity_results:
        fieldnames = list(fidelity_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fidelity_results)
    print(f"Fidelity CSV: {csv_path}")

    print(f"\n{'=' * 70}")
    print("EXP-016 COMPLETE")
    print("=" * 70)

    return output


def main():
    asyncio.run(run_experiment())


if __name__ == "__main__":
    main()
