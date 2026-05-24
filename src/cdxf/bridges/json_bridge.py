"""JSON bridge — convert between JSON text and CDXF model.

Functions:
    from_json(text) -> Stream
    to_json(stream, indent=None) -> str
"""

from __future__ import annotations

import json
from typing import Any

from cdxf.model import (
    Comment,
    Document,
    Map,
    Scalar,
    ScalarType,
    Sequence,
    SourceFormat,
    Stream,
)


def from_json(text: str) -> Stream:
    """Parse JSON text into a CDXF Stream.

    Parameters
    ----------
    text : str
        Valid JSON text.

    Returns
    -------
    Stream
        A single-document CDXF Stream with source_format_hint=JSON.
    """
    raw = json.loads(text)
    root = _from_native(raw)
    doc = Document(root=root, source_format_hint=SourceFormat.JSON)
    return Stream(documents=[doc])


def to_json(stream: Stream, *, indent: int | None = None) -> str:
    """Convert a CDXF Stream to JSON text.

    Uses the first document in the stream. Comments and CDXF-specific
    annotations are silently dropped (JSON cannot represent them).

    Parameters
    ----------
    stream : Stream
        A CDXF Stream.
    indent : int or None
        If set, pretty-print with this indentation level.

    Returns
    -------
    str
        Valid JSON text.
    """
    if not stream.documents:
        return "null"
    root = stream.documents[0].root
    native = _to_native(root)
    return json.dumps(native, indent=indent, ensure_ascii=False)


# -------------------------------------------------------------------
# Internal: Python native types ↔ CDXF model
# -------------------------------------------------------------------

def _from_native(value: Any) -> Scalar | Map | Sequence:
    """Convert a Python value (from json.loads) to a CDXF node."""
    if value is None:
        return Scalar(ScalarType.NULL, None)
    if isinstance(value, bool):
        return Scalar(ScalarType.BOOLEAN, value)
    if isinstance(value, int):
        return Scalar(ScalarType.INTEGER, value)
    if isinstance(value, float):
        return Scalar(ScalarType.FLOAT, value)
    if isinstance(value, str):
        return Scalar(ScalarType.STRING, value)
    if isinstance(value, dict):
        entries = []
        for k, v in value.items():
            key_node = Scalar(ScalarType.STRING, k)
            value_node = _from_native(v)
            entries.append((key_node, value_node))
        return Map(entries=entries)
    if isinstance(value, list):
        items = [_from_native(item) for item in value]
        return Sequence(items=items)
    raise ValueError(f"Unsupported JSON value type: {type(value)}")


def _to_native(node) -> Any:
    """Convert a CDXF node to a Python native value for json.dumps."""
    if isinstance(node, Scalar):
        return node.value
    if isinstance(node, Map):
        result = {}
        for entry in node.entries:
            # Skip comments — JSON can't represent them
            if isinstance(entry, Comment):
                continue
            key, value = entry
            # JSON keys must be strings
            key_str = key.value if isinstance(key, Scalar) else str(key)
            result[key_str] = _to_native(value)
        return result
    if isinstance(node, Sequence):
        return [
            _to_native(item)
            for item in node.items
            if not isinstance(item, Comment)
        ]
    # Fallback for types JSON can't represent
    if hasattr(node, "value"):
        return node.value
    return str(node)
