"""CDXF command-line interface.

Usage:
    cdxf encode <file> [-o output]      Text -> CDXF binary
    cdxf decode <file> [-o output] [-f] CDXF binary -> text
    cdxf convert <file> --to <fmt> [-o] Cross-format conversion via CDXF
    cdxf info <file>                    Inspect a file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cdxf.codec import encode, decode
from cdxf.bridges.json_bridge import from_json, to_json
from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
from cdxf.bridges.xml_bridge import from_xml, to_xml
from cdxf.bridges.toml_bridge import from_toml, to_toml
from cdxf.model import (
    Comment, Element, Map, Scalar, Sequence, SourceFormat, Stream,
)


# ===================================================================
# Format detection
# ===================================================================

_EXT_TO_FORMAT = {
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".xhtml": "xml",
    ".html": "xml",
    ".svg": "svg",
    ".toml": "toml",
    ".cdxf": "cdxf",
}

_SOURCE_FORMAT_TO_NAME = {
    SourceFormat.JSON: "json",
    SourceFormat.YAML: "yaml",
    SourceFormat.XML: "xml",
    SourceFormat.TOML: "toml",
}

_NAME_TO_SOURCE_FORMAT = {v: k for k, v in _SOURCE_FORMAT_TO_NAME.items()}

BRIDGES_FROM = {
    "json": from_json,
    "yaml": from_yaml,
    "xml": from_xml,
    "toml": from_toml,
}

BRIDGES_TO = {
    "json": lambda s: to_json(s, indent=2),
    "yaml": to_yaml,
    "xml": to_xml,
    "toml": to_toml,
}


def detect_format(path: Path) -> str | None:
    """Detect format from file extension."""
    return _EXT_TO_FORMAT.get(path.suffix.lower())


def is_binary(path: Path) -> bool:
    """Heuristic: check if a file looks like CDXF binary."""
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        # CDXF streams start with CBOR tag 25700 which encodes as
        # D9 646 4 in CBOR (0xD9 = major type 6, 2-byte tag)
        if len(header) >= 3 and header[0] == 0xD9:
            return True
        return False
    except Exception:
        return False


# ===================================================================
# Commands
# ===================================================================

def cmd_encode(args: argparse.Namespace) -> int:
    """Encode a text file to CDXF binary."""
    src = Path(args.file)
    if not src.exists():
        print(f"Error: file not found: {src}", file=sys.stderr)
        return 1

    fmt = detect_format(src)
    if fmt is None or fmt == "cdxf":
        print(f"Error: cannot detect text format for {src.suffix}", file=sys.stderr)
        return 1

    bridge = BRIDGES_FROM.get(fmt)
    if bridge is None:
        print(f"Error: unsupported format: {fmt}", file=sys.stderr)
        return 1

    text = src.read_text(encoding="utf-8")
    stream = bridge(text)
    binary = encode(stream)

    if args.output == "-":
        sys.stdout.buffer.write(binary)
    else:
        out = Path(args.output) if args.output else src.with_suffix(".cdxf")
        out.write_bytes(binary)

    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """Decode a CDXF binary file to text."""
    src = Path(args.file)
    if not src.exists():
        print(f"Error: file not found: {src}", file=sys.stderr)
        return 1

    binary = src.read_bytes()
    stream = decode(binary)

    # Determine output format
    fmt = args.format
    if fmt is None:
        # Auto-detect from source_format_hint
        if stream.documents:
            hint = stream.documents[0].source_format_hint
            fmt = _SOURCE_FORMAT_TO_NAME.get(hint)
        if fmt is None:
            fmt = "json"  # default fallback

    bridge = BRIDGES_TO.get(fmt)
    if bridge is None:
        print(f"Error: unsupported output format: {fmt}", file=sys.stderr)
        return 1

    text = bridge(stream)

    if args.output == "-":
        sys.stdout.write(text)
    else:
        out = Path(args.output) if args.output else src.with_suffix(f".{fmt}")
        out.write_text(text, encoding="utf-8")

    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert between text formats via CDXF."""
    src = Path(args.file)
    if not src.exists():
        print(f"Error: file not found: {src}", file=sys.stderr)
        return 1

    src_fmt = detect_format(src)
    if src_fmt is None or src_fmt == "cdxf":
        print(f"Error: cannot detect source format for {src.suffix}", file=sys.stderr)
        return 1

    dst_fmt = args.to
    if dst_fmt not in BRIDGES_TO:
        print(f"Error: unsupported target format: {dst_fmt}", file=sys.stderr)
        return 1

    # Parse source
    text = src.read_text(encoding="utf-8")
    stream = BRIDGES_FROM[src_fmt](text)

    # Convert to target
    output = BRIDGES_TO[dst_fmt](stream)

    if args.output == "-":
        sys.stdout.write(output)
    elif args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        out = src.with_suffix(f".{dst_fmt}")
        out.write_text(output, encoding="utf-8")

    return 0


def _count_nodes(node) -> dict:
    """Count node types in a CDXF tree."""
    counts: dict[str, int] = {}

    def walk(n):
        name = type(n).__name__
        counts[name] = counts.get(name, 0) + 1
        if isinstance(n, Map):
            for entry in n.entries:
                if isinstance(entry, Comment):
                    counts["Comment"] = counts.get("Comment", 0) + 1
                else:
                    walk(entry[0])
                    walk(entry[1])
        elif isinstance(n, Sequence):
            for item in n.items:
                walk(item)
        elif isinstance(n, Element):
            for child in n.children:
                walk(child)

    walk(node)
    return counts


def cmd_info(args: argparse.Namespace) -> int:
    """Display information about a file."""
    src = Path(args.file)
    if not src.exists():
        print(f"Error: file not found: {src}", file=sys.stderr)
        return 1

    file_size = src.stat().st_size
    fmt = detect_format(src)

    if fmt == "cdxf" or is_binary(src):
        # Binary CDXF file
        binary = src.read_bytes()
        stream = decode(binary)
        hint = SourceFormat.UNSPECIFIED
        if stream.documents:
            hint = stream.documents[0].source_format_hint
        hint_name = _SOURCE_FORMAT_TO_NAME.get(hint, "unspecified")

        print(f"File:          {src}")
        print(f"Type:          CDXF binary")
        print(f"Size:          {file_size:,} bytes")
        print(f"Documents:     {len(stream.documents)}")
        print(f"Source format: {hint_name}")

        if stream.documents:
            root = stream.documents[0].root
            counts = _count_nodes(root)
            print(f"Node counts:   {counts}")
    else:
        # Text file
        if fmt and fmt in BRIDGES_FROM:
            text = src.read_text(encoding="utf-8")
            stream = BRIDGES_FROM[fmt](text)
            cdxf_binary = encode(stream)

            print(f"File:          {src}")
            print(f"Type:          {fmt.upper()} text")
            print(f"Text size:     {file_size:,} bytes")
            print(f"CDXF size:     {len(cdxf_binary):,} bytes")
            print(f"Ratio:         {len(cdxf_binary) / file_size:.3f}")

            if stream.documents:
                root = stream.documents[0].root
                counts = _count_nodes(root)
                print(f"Node counts:   {counts}")
        else:
            print(f"File:          {src}")
            print(f"Type:          {fmt or 'unknown'}")
            print(f"Size:          {file_size:,} bytes")

    return 0


# ===================================================================
# Argument parser
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdxf",
        description="CDXF — Compact Data Exchange Format. "
                    "Universal binary interchange for JSON, YAML, XML, and TOML.",
    )
    sub = parser.add_subparsers(dest="command")

    # encode
    p_enc = sub.add_parser("encode", help="Encode a text file to CDXF binary")
    p_enc.add_argument("file", help="Input text file (JSON/YAML/XML/TOML)")
    p_enc.add_argument("-o", "--output", default=None,
                       help="Output path (default: <stem>.cdxf, use - for stdout)")

    # decode
    p_dec = sub.add_parser("decode", help="Decode a CDXF binary file to text")
    p_dec.add_argument("file", help="Input CDXF binary file")
    p_dec.add_argument("-o", "--output", default=None,
                       help="Output path (default: <stem>.<fmt>, use - for stdout)")
    p_dec.add_argument("-f", "--format", default=None,
                       choices=["json", "yaml", "xml", "toml"],
                       help="Output format (default: auto-detect from source)")

    # convert
    p_cvt = sub.add_parser("convert", help="Convert between formats via CDXF")
    p_cvt.add_argument("file", help="Input text file")
    p_cvt.add_argument("--to", required=True,
                       choices=["json", "yaml", "xml", "toml"],
                       help="Target format")
    p_cvt.add_argument("-o", "--output", default=None,
                       help="Output path (default: <stem>.<fmt>, use - for stdout)")

    # info
    p_info = sub.add_parser("info", help="Show file information")
    p_info.add_argument("file", help="Input file (text or CDXF binary)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    commands = {
        "encode": cmd_encode,
        "decode": cmd_decode,
        "convert": cmd_convert,
        "info": cmd_info,
    }

    try:
        return commands[args.command](args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def entry_point():
    """Console script entry point."""
    sys.exit(main())


if __name__ == "__main__":
    entry_point()
