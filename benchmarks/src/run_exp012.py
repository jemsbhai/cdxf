"""
EXP-012: Agentic Tool Schema Consolidation Overhead

Measures token savings from consolidating N format-specific tools into CDXF
universal tools. Uses canonical tool-calling schemas from real LLM providers:

  - OpenAI Chat Completions API (nested function wrapper)
  - OpenAI Responses API (flat structure)
  - Anthropic Claude (input_schema)
  - Google Gemini (OpenAPI-compatible function_declarations)
  - Mistral (same structure as OpenAI Chat Completions)

Usage:
    python benchmarks/src/run_exp012.py
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken not installed. Run: pip install tiktoken")
    sys.exit(1)

# ===========================================================================
# Protocol constants
# ===========================================================================

TOKENIZERS = ["cl100k_base", "o200k_base"]
CONTEXT_WINDOW = 128_000

# Canonical tool-calling schema formats from real LLM providers
SCHEMA_FORMATS = [
    "openai_chatcomp",   # OpenAI Chat Completions API
    "openai_responses",  # OpenAI Responses API (newer, flat)
    "anthropic",         # Anthropic Claude Messages API
    "gemini",            # Google Gemini / Vertex AI
    "mistral",           # Mistral AI (same structure as OpenAI Chat Completions)
]

SUPPORTED_FORMATS = ["json", "yaml", "xml", "toml"]
N_FORMAT_COUNTS = [2, 3, 4, 5, 6]
SESSION_CALL_COUNTS = [5, 20, 100]

# Extended format names for scaling beyond the core 4
_EXTENDED_FORMATS = ["json", "yaml", "xml", "toml", "ini", "csv"]


# ===========================================================================
# Internal tool descriptions (provider-agnostic)
# ===========================================================================

_FORMAT_TOOL_DESCRIPTIONS = {
    "json": {
        "parse": {
            "name": "parse_json",
            "description": (
                "Parse a JSON document string into a structured data object. "
                "Handles nested objects, arrays, strings, numbers, booleans, "
                "and null values. Returns the parsed data structure for "
                "inspection, modification, or cross-format conversion. "
                "Validates JSON syntax and reports parse errors with line "
                "and column information."
            ),
            "params": {
                "content": {
                    "type": "string",
                    "description": "The JSON document string to parse.",
                },
                "strict": {
                    "type": "boolean",
                    "description": (
                        "If true, reject trailing commas and comments. "
                        "Default: true."
                    ),
                },
            },
            "required": ["content"],
        },
        "emit": {
            "name": "emit_json",
            "description": (
                "Serialize a data structure into a JSON document string. "
                "Produces well-formatted JSON with configurable indentation. "
                "Handles nested objects, arrays, and all JSON-compatible "
                "scalar types. Non-serializable values raise an error."
            ),
            "params": {
                "data": {
                    "type": "object",
                    "description": "The data structure to serialize as JSON.",
                },
                "indent": {
                    "type": "integer",
                    "description": (
                        "Number of spaces for indentation. 0 for compact. "
                        "Default: 2."
                    ),
                },
                "sort_keys": {
                    "type": "boolean",
                    "description": (
                        "Sort object keys alphabetically. Default: false."
                    ),
                },
            },
            "required": ["data"],
        },
    },
    "yaml": {
        "parse": {
            "name": "parse_yaml",
            "description": (
                "Parse a YAML document string into a structured data object. "
                "Supports anchors, aliases, merge keys, multi-document "
                "streams, typed timestamps, and flow/block styles. "
                "Preserves comments when possible. Returns the parsed data "
                "structure for inspection or modification."
            ),
            "params": {
                "content": {
                    "type": "string",
                    "description": "The YAML document string to parse.",
                },
                "preserve_comments": {
                    "type": "boolean",
                    "description": (
                        "Preserve YAML comments in the parsed result. "
                        "Default: true."
                    ),
                },
                "multi_document": {
                    "type": "boolean",
                    "description": (
                        "Parse as a multi-document YAML stream. "
                        "Default: false."
                    ),
                },
            },
            "required": ["content"],
        },
        "emit": {
            "name": "emit_yaml",
            "description": (
                "Serialize a data structure into a YAML document string. "
                "Produces human-readable YAML with configurable style "
                "preferences. Supports block and flow mappings, sequence "
                "styles, and typed scalar emission including timestamps."
            ),
            "params": {
                "data": {
                    "type": "object",
                    "description": "The data structure to serialize as YAML.",
                },
                "default_flow_style": {
                    "type": "boolean",
                    "description": (
                        "Use flow style (inline) for mappings and sequences. "
                        "Default: false (block style)."
                    ),
                },
                "width": {
                    "type": "integer",
                    "description": (
                        "Maximum line width before wrapping. Default: 80."
                    ),
                },
            },
            "required": ["data"],
        },
    },
    "xml": {
        "parse": {
            "name": "parse_xml",
            "description": (
                "Parse an XML document string into a structured data object. "
                "Handles elements, attributes, text content, CDATA sections, "
                "namespaces, processing instructions, and comments. Returns "
                "a tree representation preserving document structure and "
                "mixed content. Validates well-formedness."
            ),
            "params": {
                "content": {
                    "type": "string",
                    "description": "The XML document string to parse.",
                },
                "namespaces": {
                    "type": "boolean",
                    "description": "Process namespace declarations. Default: true.",
                },
                "preserve_whitespace": {
                    "type": "boolean",
                    "description": (
                        "Preserve significant whitespace in text nodes. "
                        "Default: true."
                    ),
                },
            },
            "required": ["content"],
        },
        "emit": {
            "name": "emit_xml",
            "description": (
                "Serialize a data structure into an XML document string. "
                "Produces well-formatted XML with configurable indentation "
                "and encoding declaration. Handles element nesting, "
                "attributes, text content, and namespace prefixes."
            ),
            "params": {
                "data": {
                    "type": "object",
                    "description": "The data structure to serialize as XML.",
                },
                "indent": {
                    "type": "integer",
                    "description": "Spaces for indentation. Default: 2.",
                },
                "xml_declaration": {
                    "type": "boolean",
                    "description": "Include XML declaration header. Default: true.",
                },
                "encoding": {
                    "type": "string",
                    "description": "Character encoding. Default: 'utf-8'.",
                },
            },
            "required": ["data"],
        },
    },
    "toml": {
        "parse": {
            "name": "parse_toml",
            "description": (
                "Parse a TOML document string into a structured data object. "
                "Supports tables, arrays of tables, inline tables, typed "
                "datetime values (offset, local, date, time), multi-line "
                "strings, and integer formats (hex, oct, bin). Preserves "
                "comments when possible."
            ),
            "params": {
                "content": {
                    "type": "string",
                    "description": "The TOML document string to parse.",
                },
                "preserve_comments": {
                    "type": "boolean",
                    "description": (
                        "Preserve TOML comments in the parsed result. "
                        "Default: true."
                    ),
                },
            },
            "required": ["content"],
        },
        "emit": {
            "name": "emit_toml",
            "description": (
                "Serialize a data structure into a TOML document string. "
                "Produces human-readable TOML with proper table headers "
                "and array-of-tables notation. Handles typed datetime "
                "values, multi-line strings, and inline tables."
            ),
            "params": {
                "data": {
                    "type": "object",
                    "description": "The data structure to serialize as TOML.",
                },
                "sort_keys": {
                    "type": "boolean",
                    "description": (
                        "Sort table keys alphabetically. Default: false."
                    ),
                },
            },
            "required": ["data"],
        },
    },
    "ini": {
        "parse": {
            "name": "parse_ini",
            "description": (
                "Parse an INI/CFG configuration file into a structured data "
                "object. Handles sections, key-value pairs, multi-line "
                "values, and inline comments. Returns a nested mapping of "
                "section names to key-value pairs."
            ),
            "params": {
                "content": {
                    "type": "string",
                    "description": "The INI document string to parse.",
                },
                "interpolation": {
                    "type": "boolean",
                    "description": (
                        "Enable variable interpolation (%(var)s). "
                        "Default: false."
                    ),
                },
            },
            "required": ["content"],
        },
        "emit": {
            "name": "emit_ini",
            "description": (
                "Serialize a data structure into an INI configuration file "
                "string. Produces standard INI format with [section] headers "
                "and key=value pairs."
            ),
            "params": {
                "data": {
                    "type": "object",
                    "description": "The data structure to serialize as INI.",
                },
                "space_around_delimiters": {
                    "type": "boolean",
                    "description": (
                        "Add spaces around = delimiter. Default: true."
                    ),
                },
            },
            "required": ["data"],
        },
    },
    "csv": {
        "parse": {
            "name": "parse_csv",
            "description": (
                "Parse a CSV document string into a list of records. "
                "Supports configurable delimiters, quoting rules, and "
                "header row detection. Returns a list of dictionaries "
                "keyed by column headers."
            ),
            "params": {
                "content": {
                    "type": "string",
                    "description": "The CSV document string to parse.",
                },
                "delimiter": {
                    "type": "string",
                    "description": "Field delimiter character. Default: ','.",
                },
                "has_header": {
                    "type": "boolean",
                    "description": (
                        "First row is a header row. Default: true."
                    ),
                },
            },
            "required": ["content"],
        },
        "emit": {
            "name": "emit_csv",
            "description": (
                "Serialize a list of records into a CSV document string. "
                "Produces standard CSV with configurable delimiters and "
                "quoting. Handles values containing delimiters or newlines."
            ),
            "params": {
                "data": {
                    "type": "array",
                    "description": (
                        "List of records (dicts) to serialize as CSV."
                    ),
                },
                "delimiter": {
                    "type": "string",
                    "description": "Field delimiter character. Default: ','.",
                },
            },
            "required": ["data"],
        },
    },
}

# CDXF universal tool descriptions (provider-agnostic)
_CDXF_TOOL_DESCRIPTIONS = {
    "cdxf_encode": {
        "name": "cdxf_encode",
        "description": (
            "Encode a text-format configuration document (JSON, YAML, XML, "
            "TOML, or any supported format) into CDXF binary representation. "
            "CDXF preserves all format-specific metadata including comments, "
            "anchors, typed temporal values, processing instructions, and "
            "document structure. The encoded CDXF binary is a compact, "
            "cross-language-safe representation based on CBOR. Supports "
            "all source formats through a single universal interface."
        ),
        "params": {
            "content": {
                "type": "string",
                "description": "The text document content to encode.",
            },
            "source_format": {
                "type": "string",
                "enum": ["json", "yaml", "xml", "toml", "ini", "csv"],
                "description": (
                    "The source text format of the content. Used to select "
                    "the appropriate parser for lossless encoding."
                ),
            },
        },
        "required": ["content", "source_format"],
    },
    "cdxf_decode": {
        "name": "cdxf_decode",
        "description": (
            "Decode a CDXF binary representation back into a text-format "
            "document. Reconstructs the original format with all metadata "
            "preserved, including comments, anchors, typed values, and "
            "structural annotations. The target format can differ from the "
            "original source format, enabling lossless cross-format conversion."
        ),
        "params": {
            "cdxf_data": {
                "type": "string",
                "description": (
                    "The CDXF binary data (base64-encoded) to decode."
                ),
            },
            "target_format": {
                "type": "string",
                "enum": ["json", "yaml", "xml", "toml"],
                "description": (
                    "The target text format to emit. If omitted, uses the "
                    "original source format stored in the CDXF metadata."
                ),
            },
        },
        "required": ["cdxf_data"],
    },
    "cdxf_convert": {
        "name": "cdxf_convert",
        "description": (
            "Convert a document from one text format to another via CDXF "
            "intermediate representation. This is a convenience tool that "
            "combines encode and decode in a single operation. Preserves "
            "all transferable metadata (comments, typed values) across the "
            "format boundary. Handles JSON, YAML, XML, TOML, and other "
            "supported formats."
        ),
        "params": {
            "content": {
                "type": "string",
                "description": "The source document content to convert.",
            },
            "source_format": {
                "type": "string",
                "enum": ["json", "yaml", "xml", "toml", "ini", "csv"],
                "description": "The source text format.",
            },
            "target_format": {
                "type": "string",
                "enum": ["json", "yaml", "xml", "toml"],
                "description": "The desired target text format.",
            },
        },
        "required": ["content", "source_format", "target_format"],
    },
}


# ===========================================================================
# Provider-specific schema converters
# ===========================================================================


def _to_openai_chatcomp(desc: dict) -> dict:
    """OpenAI Chat Completions API format (nested 'function' wrapper).

    Canonical format:
    {"type": "function", "function": {"name", "description", "parameters"}}
    Reference: https://platform.openai.com/docs/api-reference/chat/create
    """
    return {
        "type": "function",
        "function": {
            "name": desc["name"],
            "description": desc["description"],
            "parameters": {
                "type": "object",
                "properties": desc["params"],
                "required": desc["required"],
            },
        },
    }


def _to_openai_responses(desc: dict) -> dict:
    """OpenAI Responses API format (flat structure, no 'function' wrapper).

    Canonical format:
    {"type": "function", "name", "description", "parameters"}
    Reference: https://platform.openai.com/docs/api-reference/responses/create
    """
    return {
        "type": "function",
        "name": desc["name"],
        "description": desc["description"],
        "parameters": {
            "type": "object",
            "properties": desc["params"],
            "required": desc["required"],
        },
    }


def _to_anthropic(desc: dict) -> dict:
    """Anthropic Claude Messages API format.

    Canonical format:
    {"name", "description", "input_schema"}
    Reference: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use
    """
    return {
        "name": desc["name"],
        "description": desc["description"],
        "input_schema": {
            "type": "object",
            "properties": desc["params"],
            "required": desc["required"],
        },
    }


def _to_gemini(desc: dict) -> dict:
    """Google Gemini / Vertex AI function declaration format.

    Canonical format (individual declaration, used inside function_declarations):
    {"name", "description", "parameters"}
    Uses OpenAPI-compatible schema.
    Reference: https://ai.google.dev/gemini-api/docs/function-calling
    """
    return {
        "name": desc["name"],
        "description": desc["description"],
        "parameters": {
            "type": "object",
            "properties": desc["params"],
            "required": desc["required"],
        },
    }


def _to_mistral(desc: dict) -> dict:
    """Mistral AI function-calling format.

    Same structure as OpenAI Chat Completions:
    {"type": "function", "function": {"name", "description", "parameters"}}
    Reference: https://docs.mistral.ai/capabilities/function_calling/
    """
    return {
        "type": "function",
        "function": {
            "name": desc["name"],
            "description": desc["description"],
            "parameters": {
                "type": "object",
                "properties": desc["params"],
                "required": desc["required"],
            },
        },
    }


_CONVERTERS = {
    "openai_chatcomp": _to_openai_chatcomp,
    "openai_responses": _to_openai_responses,
    "anthropic": _to_anthropic,
    "gemini": _to_gemini,
    "mistral": _to_mistral,
}


# ===========================================================================
# Tool schema builders
# ===========================================================================


def build_format_specific_tools(
    formats: list[str], schema_format: str
) -> list[dict]:
    """Build format-specific tool definitions (parse + emit per format).

    Args:
        formats: List of format names (e.g. ["json", "yaml"]).
        schema_format: One of SCHEMA_FORMATS (provider name).

    Returns:
        List of tool definition dicts, 2 per format (parse + emit).
    """
    converter = _CONVERTERS[schema_format]
    tools = []
    for fmt in formats:
        desc = _FORMAT_TOOL_DESCRIPTIONS[fmt]
        tools.append(converter(desc["parse"]))
        tools.append(converter(desc["emit"]))
    return tools


def build_cdxf_tools(schema_format: str) -> list[dict]:
    """Build CDXF universal tool definitions (encode, decode, convert).

    Args:
        schema_format: One of SCHEMA_FORMATS (provider name).

    Returns:
        List of 3 tool definition dicts.
    """
    converter = _CONVERTERS[schema_format]
    return [
        converter(_CDXF_TOOL_DESCRIPTIONS["cdxf_encode"]),
        converter(_CDXF_TOOL_DESCRIPTIONS["cdxf_decode"]),
        converter(_CDXF_TOOL_DESCRIPTIONS["cdxf_convert"]),
    ]


# ===========================================================================
# Tokenization
# ===========================================================================

_encoder_cache: dict[str, tiktoken.Encoding] = {}


def _get_encoder(tokenizer_name: str) -> tiktoken.Encoding:
    """Get or create a tiktoken encoder."""
    if tokenizer_name not in _encoder_cache:
        _encoder_cache[tokenizer_name] = tiktoken.get_encoding(tokenizer_name)
    return _encoder_cache[tokenizer_name]


def tokenize_tools(tools: list[dict], tokenizer_name: str) -> int:
    """Tokenize tool schemas and return total token count.

    Serializes the tool list to JSON (as function-calling APIs do internally)
    and counts the resulting tokens.
    """
    if not tools:
        return 0
    text = json.dumps(tools, indent=2)
    enc = _get_encoder(tokenizer_name)
    return len(enc.encode(text))


# ===========================================================================
# Savings computation
# ===========================================================================


def compute_savings(
    format_specific_tokens: int, cdxf_tokens: int
) -> dict:
    """Compute token savings metrics."""
    saved = format_specific_tokens - cdxf_tokens
    reduction_pct = (
        (saved / format_specific_tokens * 100)
        if format_specific_tokens > 0
        else 0.0
    )
    return {
        "format_specific_tokens": format_specific_tokens,
        "cdxf_tokens": cdxf_tokens,
        "tokens_saved": saved,
        "savings_fraction": saved / CONTEXT_WINDOW,
        "reduction_pct": reduction_pct,
    }


# ===========================================================================
# Corpus construction
# ===========================================================================


def build_corpus() -> list[tuple[str, str, str]]:
    """Build corpus of config files for call-result token measurement.

    Returns list of (name, format, text) tuples.
    """
    corpus = []
    seen = set()

    format_map = {
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".xml": "xml", ".toml": "toml",
    }

    # 1. Load data/raw files (EXP-001 corpus)
    data_raw = Path("data/raw")
    if data_raw.exists():
        for root, _dirs, files in os.walk(data_raw):
            for fname in sorted(files):
                fpath = Path(root) / fname
                ext = fpath.suffix.lower()
                fmt = format_map.get(ext)
                if fmt:
                    try:
                        text = fpath.read_text(encoding="utf-8",
                                               errors="replace")
                        name = str(fpath.relative_to(data_raw)).replace(
                            "\\", "/"
                        )
                        if name not in seen:
                            corpus.append((name, fmt, text))
                            seen.add(name)
                    except Exception:
                        pass

    # 2. Add ML configs from EXP-010
    try:
        from benchmarks.src.run_exp010 import build_ml_corpus

        for entry in build_ml_corpus():
            name = f"ml/{entry['name']}"
            if name not in seen:
                corpus.append((name, entry["format"], entry["text"]))
                seen.add(name)
    except ImportError:
        try:
            import importlib.util

            exp010_path = Path(__file__).parent / "run_exp010.py"
            if exp010_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "run_exp010", exp010_path
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for entry in mod.build_ml_corpus():
                    name = f"ml/{entry['name']}"
                    if name not in seen:
                        corpus.append(
                            (name, entry["format"], entry["text"])
                        )
                        seen.add(name)
        except Exception:
            pass

    return corpus


# ===========================================================================
# Call result token measurement
# ===========================================================================


def measure_call_result_tokens(
    text: str, fmt: str, tokenizer_name: str
) -> dict:
    """Measure tokens consumed by a tool call result (the parsed data)."""
    enc = _get_encoder(tokenizer_name)
    tokens = enc.encode(text)
    size_bytes = len(text.encode("utf-8"))
    return {
        "format": fmt,
        "size_bytes": size_bytes,
        "total_tokens": len(tokens),
        "tokens_per_byte": len(tokens) / max(1, size_bytes),
    }


# ===========================================================================
# Session overhead projection
# ===========================================================================


def compute_session_overhead(
    schema_tokens: int,
    avg_call_result_tokens: float,
    n_calls: int,
) -> dict:
    """Project total context overhead for an agentic session.

    Total = schema_tokens + n_calls * avg_call_result_tokens
    """
    total = schema_tokens + n_calls * avg_call_result_tokens
    return {
        "schema_tokens": schema_tokens,
        "avg_call_result_tokens": avg_call_result_tokens,
        "n_calls": n_calls,
        "total_tokens": total,
        "fraction_of_context": total / CONTEXT_WINDOW,
    }


# ===========================================================================
# Scaling analysis
# ===========================================================================


def scaling_analysis(
    schema_format: str, tokenizer_name: str
) -> dict:
    """Analyze how token costs scale with N supported formats.

    Format-specific: O(N) tools -> O(N) tokens.
    CDXF: O(1) tools -> O(1) tokens.
    """
    cdxf_tools = build_cdxf_tools(schema_format)
    cdxf_tokens = tokenize_tools(cdxf_tools, tokenizer_name)

    result = {}
    for n in N_FORMAT_COUNTS:
        formats = _EXTENDED_FORMATS[:n]
        specific_tools = build_format_specific_tools(formats, schema_format)
        specific_tokens = tokenize_tools(specific_tools, tokenizer_name)

        result[n] = {
            "n_formats": n,
            "n_specific_tools": len(specific_tools),
            "n_cdxf_tools": len(cdxf_tools),
            "format_specific_tokens": specific_tokens,
            "cdxf_tokens": cdxf_tokens,
            "tokens_saved": specific_tokens - cdxf_tokens,
            "reduction_pct": round(
                (specific_tokens - cdxf_tokens) / specific_tokens * 100, 2
            ),
        }

    return result


# ===========================================================================
# Full experiment
# ===========================================================================


def run_experiment(output_dir: Path | str | None = None) -> dict:
    """Run the full EXP-012 experiment."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_012")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-012: Agentic Tool Schema Consolidation Overhead")
    print("       (Canonical schemas: OpenAI, Anthropic, Gemini, Mistral)")
    print("=" * 70)

    # ----- 1. Schema comparison across all providers -----
    print("\n--- Schema Token Comparison (4 formats × 5 providers) ---")
    schema_comparison = {}
    for sf in SCHEMA_FORMATS:
        schema_comparison[sf] = {}
        for tok in TOKENIZERS:
            specific = build_format_specific_tools(SUPPORTED_FORMATS, sf)
            cdxf = build_cdxf_tools(sf)
            s_tokens = tokenize_tools(specific, tok)
            c_tokens = tokenize_tools(cdxf, tok)
            savings = compute_savings(s_tokens, c_tokens)
            schema_comparison[sf][tok] = savings

            if tok == TOKENIZERS[0]:
                print(
                    f"  {sf:20s}  "
                    f"specific={s_tokens:5d}  cdxf={c_tokens:4d}  "
                    f"saved={savings['tokens_saved']:5d}  "
                    f"({savings['reduction_pct']:.1f}%)"
                )

    # ----- 2. Scaling analysis -----
    print("\n--- Scaling Analysis (O(N) vs O(1)) ---")
    scaling_results = {}
    for sf in SCHEMA_FORMATS:
        for tok in TOKENIZERS:
            key = f"{sf}__{tok}"
            sa = scaling_analysis(sf, tok)
            scaling_results[key] = sa
            if tok == TOKENIZERS[0]:
                print(f"\n  {sf} ({tok}):")
                for n in sorted(N_FORMAT_COUNTS):
                    e = sa[n]
                    print(
                        f"    N={n}: specific={e['format_specific_tokens']:5d}  "
                        f"cdxf={e['cdxf_tokens']:4d}  "
                        f"saved={e['tokens_saved']:5d}  "
                        f"({e['reduction_pct']:.1f}%)"
                    )

    # ----- 3. Call result analysis -----
    print("\n--- Call Result Token Analysis ---")
    corpus = build_corpus()
    print(f"  Corpus: {len(corpus)} files")

    call_results = []
    for name, fmt, text in corpus:
        for tok in TOKENIZERS:
            r = measure_call_result_tokens(text, fmt, tok)
            r["name"] = name
            r["tokenizer"] = tok
            call_results.append(r)

    avg_call_tokens = {}
    for tok in TOKENIZERS:
        tok_results = [r for r in call_results if r["tokenizer"] == tok]
        if tok_results:
            avg_t = statistics.mean(r["total_tokens"] for r in tok_results)
            med_t = statistics.median(r["total_tokens"] for r in tok_results)
            avg_call_tokens[tok] = {
                "mean_tokens": round(avg_t, 1),
                "median_tokens": round(med_t, 1),
                "n_files": len(tok_results),
            }
            print(f"  {tok:15s}  mean={avg_t:.0f}  median={med_t:.0f}  "
                  f"n={len(tok_results)}")

    # ----- 4. Session overhead projections -----
    print("\n--- Session Overhead Projections ---")
    session_overhead = {}
    for n_calls in SESSION_CALL_COUNTS:
        session_overhead[n_calls] = {}
        for sf in SCHEMA_FORMATS:
            session_overhead[n_calls][sf] = {}
            for tok in TOKENIZERS:
                s_tok = schema_comparison[sf][tok]["format_specific_tokens"]
                c_tok = schema_comparison[sf][tok]["cdxf_tokens"]
                avg_crt = avg_call_tokens.get(tok, {}).get("mean_tokens", 0)

                sp_oh = compute_session_overhead(s_tok, avg_crt, n_calls)
                cd_oh = compute_session_overhead(c_tok, avg_crt, n_calls)

                session_overhead[n_calls][sf][tok] = {
                    "format_specific": sp_oh,
                    "cdxf": cd_oh,
                    "total_saved": sp_oh["total_tokens"] - cd_oh["total_tokens"],
                }

                if tok == TOKENIZERS[0] and sf == SCHEMA_FORMATS[0]:
                    print(
                        f"  {n_calls:3d} calls ({sf}): "
                        f"specific={sp_oh['total_tokens']:.0f}  "
                        f"cdxf={cd_oh['total_tokens']:.0f}  "
                        f"saved={sp_oh['total_tokens'] - cd_oh['total_tokens']:.0f}"
                    )

    # ----- 5. Write outputs -----
    output = {
        "experiment": "EXP-012",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": SCHEMA_FORMATS,
        "corpus_size": len(corpus),
        "schema_comparison": schema_comparison,
        "scaling_analysis": scaling_results,
        "call_result_analysis": {
            "per_file": call_results,
            "aggregate": avg_call_tokens,
        },
        "session_overhead": session_overhead,
    }

    json_path = output_dir / "exp_012_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults: {json_path}")

    # CSV: scaling analysis
    scaling_csv = output_dir / "scaling_analysis.csv"
    rows = []
    for key, sa in scaling_results.items():
        sf_name, tok_name = key.split("__", 1)
        for n in sorted(sa.keys()):
            e = sa[n]
            rows.append({
                "provider": sf_name,
                "tokenizer": tok_name,
                "n_formats": n,
                "n_specific_tools": e["n_specific_tools"],
                "n_cdxf_tools": e["n_cdxf_tools"],
                "format_specific_tokens": e["format_specific_tokens"],
                "cdxf_tokens": e["cdxf_tokens"],
                "tokens_saved": e["tokens_saved"],
                "reduction_pct": e["reduction_pct"],
            })
    if rows:
        fieldnames = list(rows[0].keys())
        with open(scaling_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Scaling CSV: {scaling_csv}")

    # CSV: call results
    cr_csv = output_dir / "call_result_tokens.csv"
    if call_results:
        fieldnames = ["name", "format", "size_bytes", "tokenizer",
                       "total_tokens", "tokens_per_byte"]
        with open(cr_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in call_results:
                writer.writerow({k: r[k] for k in fieldnames})
    print(f"Call results CSV: {cr_csv}")

    # CSV: cross-provider comparison (key paper table)
    provider_csv = output_dir / "provider_comparison.csv"
    prov_rows = []
    for sf in SCHEMA_FORMATS:
        for tok in TOKENIZERS:
            s = schema_comparison[sf][tok]
            prov_rows.append({
                "provider": sf,
                "tokenizer": tok,
                "format_specific_tokens": s["format_specific_tokens"],
                "cdxf_tokens": s["cdxf_tokens"],
                "tokens_saved": s["tokens_saved"],
                "reduction_pct": round(s["reduction_pct"], 2),
                "savings_fraction_of_128k": round(s["savings_fraction"], 6),
            })
    if prov_rows:
        fieldnames = list(prov_rows[0].keys())
        with open(provider_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prov_rows)
    print(f"Provider comparison CSV: {provider_csv}")

    # Save tool schemas for reproducibility
    schemas_dir = output_dir / "tool_schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    for sf in SCHEMA_FORMATS:
        specific = build_format_specific_tools(SUPPORTED_FORMATS, sf)
        cdxf = build_cdxf_tools(sf)
        (schemas_dir / f"format_specific_{sf}.json").write_text(
            json.dumps(specific, indent=2), encoding="utf-8"
        )
        (schemas_dir / f"cdxf_universal_{sf}.json").write_text(
            json.dumps(cdxf, indent=2), encoding="utf-8"
        )
    print(f"Tool schemas: {schemas_dir}")

    print(f"\n{'=' * 70}")
    print("EXP-012 COMPLETE")
    print("=" * 70)

    return output


def main():
    run_experiment()


if __name__ == "__main__":
    main()
