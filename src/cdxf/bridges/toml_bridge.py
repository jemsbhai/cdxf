"""TOML bridge — convert between TOML text and CDXF model.

Uses tomlkit for parsing and serialization, which preserves comments,
formatting, and distinguishes all four TOML temporal types.

Functions:
    from_toml(text) -> Stream
    to_toml(stream) -> str
"""

from __future__ import annotations

from datetime import date, datetime, time

import tomlkit
from tomlkit import items as toml_items

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


# ===================================================================
# Public API
# ===================================================================

def from_toml(text: str) -> Stream:
    """Parse TOML text into a CDXF Stream.

    Parameters
    ----------
    text : str
        Valid TOML text.

    Returns
    -------
    Stream
        A single-document CDXF Stream with source_format_hint=TOML.
    """
    doc = tomlkit.parse(text)
    converter = _TomlToModel()
    root = converter.convert_container(doc)
    return Stream(documents=[Document(root=root, source_format_hint=SourceFormat.TOML)])


def to_toml(stream: Stream) -> str:
    """Convert a CDXF Stream to TOML text.

    Parameters
    ----------
    stream : Stream
        A CDXF Stream whose first document's root is a Map.

    Returns
    -------
    str
        Valid TOML text.
    """
    if not stream.documents:
        return ""
    root = stream.documents[0].root
    converter = _ModelToToml()
    doc = converter.convert(root)
    return tomlkit.dumps(doc)


# ===================================================================
# Helpers
# ===================================================================

def _get_body(container) -> list:
    """Extract the body list from a tomlkit container.

    TOMLDocument is a Container with a .body property.
    Table and InlineTable are Items that wrap a Container via .value.
    """
    # TOMLDocument / Container — has .body directly
    if hasattr(container, "body") and not isinstance(container, toml_items.Item):
        return container.body
    # Table / InlineTable — body is on the inner Container
    if hasattr(container, "value") and hasattr(container.value, "body"):
        return container.value.body
    return []


# ===================================================================
# TOML → CDXF model converter
# ===================================================================

class _TomlToModel:
    """Convert a tomlkit document tree to CDXF model nodes."""

    def convert_container(self, container) -> Map:
        """Convert a tomlkit Container (Document/Table) to a CDXF Map."""
        entries: list = []
        body = _get_body(container)

        for key, item in body:
            if key is None:
                # Standalone comment or whitespace
                if isinstance(item, toml_items.Comment):
                    text = self._comment_text(item.trivia.comment)
                    if text:
                        entries.append(Comment(text))
                continue

            # Convert the key-value pair
            key_str = str(key).strip()
            key_node = Scalar(ScalarType.STRING, key_str)
            value_node = self._convert_item(item)
            entries.append((key_node, value_node))

            # Extract inline comment (on the same line as the value)
            if hasattr(item, "trivia") and item.trivia.comment:
                text = self._comment_text(item.trivia.comment)
                if text:
                    entries.append(Comment(text))

        return Map(entries=entries)

    def _convert_item(self, item) -> Map | Sequence | Scalar:
        """Convert a tomlkit Item to the appropriate CDXF node."""
        if isinstance(item, (toml_items.Table, toml_items.InlineTable)):
            return self.convert_container(item)
        if isinstance(item, toml_items.AoT):
            return Sequence(
                items=[self.convert_container(table) for table in item.body]
            )
        if isinstance(item, toml_items.Array):
            return Sequence(
                items=[self._convert_native(v) for v in item.unwrap()]
            )
        # Scalar value — unwrap from tomlkit wrapper
        return self._convert_scalar(item)

    def _convert_scalar(self, item) -> Scalar:
        """Convert a tomlkit scalar Item to a CDXF Scalar."""
        value = item.unwrap() if hasattr(item, "unwrap") else item
        return self._convert_native(value)

    def _convert_native(self, value) -> Map | Sequence | Scalar:
        """Convert a Python native value to a CDXF node.

        Used for array elements and unwrapped scalars where we have
        raw Python types rather than tomlkit Items.
        """
        if isinstance(value, bool):
            return Scalar(ScalarType.BOOLEAN, value)
        if isinstance(value, int):
            return Scalar(ScalarType.INTEGER, value)
        if isinstance(value, float):
            return Scalar(ScalarType.FLOAT, value)
        if isinstance(value, str):
            return Scalar(ScalarType.STRING, value)
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                return Scalar(ScalarType.TIMESTAMP_OFFSET, value)
            return Scalar(ScalarType.TIMESTAMP_LOCAL, value)
        if isinstance(value, date):
            return Scalar(ScalarType.DATE, value)
        if isinstance(value, time):
            return Scalar(ScalarType.TIME, value)
        if isinstance(value, dict):
            entries = [
                (Scalar(ScalarType.STRING, k), self._convert_native(v))
                for k, v in value.items()
            ]
            return Map(entries=entries)
        if isinstance(value, list):
            return Sequence(items=[self._convert_native(v) for v in value])
        return Scalar(ScalarType.STRING, str(value))

    @staticmethod
    def _comment_text(raw: str) -> str | None:
        """Extract clean text from a tomlkit comment string like '# text'."""
        if not raw:
            return None
        text = raw.strip().lstrip("#").strip()
        return text if text else None


# ===================================================================
# CDXF model → TOML converter
# ===================================================================

class _ModelToToml:
    """Convert CDXF model nodes to a tomlkit Document."""

    def convert(self, node) -> tomlkit.TOMLDocument:
        """Convert a CDXF Map to a tomlkit Document."""
        doc = tomlkit.document()
        if isinstance(node, Map):
            self._populate_table(doc, node)
        return doc

    def _populate_table(self, table, map_node: Map) -> None:
        """Populate a tomlkit table/document from a CDXF Map's entries."""
        for entry in map_node.entries:
            if isinstance(entry, Comment):
                continue  # Comments are not reconstructed in to_toml
            key, value = entry
            key_str = key.value if isinstance(key, Scalar) else str(key)
            table.add(key_str, self._convert_value(value))

    def _convert_value(self, node):
        """Convert a CDXF node to a tomlkit-compatible value."""
        if isinstance(node, Map):
            t = tomlkit.table()
            self._populate_table(t, node)
            return t
        if isinstance(node, Sequence):
            return self._convert_sequence(node)
        if isinstance(node, Scalar):
            return self._convert_scalar(node)
        return None

    def _convert_sequence(self, seq: Sequence):
        """Convert a CDXF Sequence to either an AoT or an Array."""
        items = [i for i in seq.items if not isinstance(i, Comment)]

        # If all items are Maps, produce an Array of Tables (AoT)
        if items and all(isinstance(i, Map) for i in items):
            aot = tomlkit.aot()
            for item in items:
                t = tomlkit.table()
                self._populate_table(t, item)
                aot.append(t)
            return aot

        # Otherwise, a regular array
        arr = tomlkit.array()
        for item in items:
            if isinstance(item, Scalar):
                arr.append(item.value)
            elif isinstance(item, Sequence):
                inner = self._convert_sequence(item)
                arr.append(inner)
            else:
                arr.append(str(item))
        return arr

    @staticmethod
    def _convert_scalar(scalar: Scalar):
        """Convert a CDXF Scalar to a Python value for tomlkit."""
        return scalar.value
