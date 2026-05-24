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
    """Parse TOML text into a CDXF Stream."""
    doc = tomlkit.parse(text)
    converter = _TomlToModel()
    root = converter.convert_container(doc)
    return Stream(documents=[Document(root=root, source_format_hint=SourceFormat.TOML)])


def to_toml(stream: Stream) -> str:
    """Convert a CDXF Stream to TOML text."""
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
    """Extract the body list from a tomlkit container."""
    # Direct Container (TOMLDocument)
    if hasattr(container, "body") and not isinstance(container, toml_items.Item):
        return container.body
    # Table / InlineTable — unwrap through .value
    if isinstance(container, (toml_items.Table, toml_items.InlineTable)):
        inner = container.value
        if hasattr(inner, "body"):
            return inner.body
    # Generic Item with .value wrapping a Container
    if hasattr(container, "value"):
        val = container.value
        if hasattr(val, "body"):
            return val.body
    return []


def _unwrap_key(key) -> str:
    """Extract the raw key string from a tomlkit Key object.

    tomlkit Key objects may include TOML quoting in their str()
    representation. The .key property returns the unquoted value.
    """
    if hasattr(key, "key"):
        return str(key.key)
    return str(key).strip().strip('"').strip("'")


def _merge_maps(base: Map, addition: Map) -> Map:
    """Merge two CDXF Maps, combining entries under the same key.

    TOML allows split table definitions (e.g., [workspace] appearing
    twice with different sub-keys). This function merges them into a
    single Map without losing any entries.
    """
    # Build a dict of key -> index for efficient lookup
    key_index: dict[str, int] = {}
    for i, entry in enumerate(base.entries):
        if isinstance(entry, Comment):
            continue
        k, _ = entry
        if isinstance(k, Scalar):
            key_index[k.value] = i

    for entry in addition.entries:
        if isinstance(entry, Comment):
            base.entries.append(entry)
            continue
        k, v = entry
        k_str = k.value if isinstance(k, Scalar) else str(k)

        if k_str in key_index:
            # Key exists — if both are Maps, merge recursively
            existing_entry = base.entries[key_index[k_str]]
            _, existing_val = existing_entry
            if isinstance(existing_val, Map) and isinstance(v, Map):
                merged = _merge_maps(existing_val, v)
                base.entries[key_index[k_str]] = (k, merged)
            else:
                # Overwrite (last wins)
                base.entries[key_index[k_str]] = (k, v)
        else:
            key_index[k_str] = len(base.entries)
            base.entries.append((k, v))

    return base


# ===================================================================
# TOML -> CDXF model converter
# ===================================================================

class _TomlToModel:
    """Convert a tomlkit document tree to CDXF model nodes."""

    def convert_container(self, container) -> Map:
        """Convert a tomlkit Container (Document/Table) to a CDXF Map.

        Handles TOML's split table definitions by merging entries that
        share the same key (e.g., [workspace] appearing in multiple
        sections).
        """
        entries: list = []
        body = _get_body(container)

        # Track keys found via body iteration, mapping to entry index
        found_keys: dict[str, int] = {}

        for key, item in body:
            if key is None:
                if isinstance(item, toml_items.Comment):
                    text = self._comment_text(item.trivia.comment)
                    if text:
                        entries.append(Comment(text))
                continue

            key_str = _unwrap_key(key)
            key_node = Scalar(ScalarType.STRING, key_str)
            value_node = self._convert_item(item)

            if key_str in found_keys:
                # Duplicate key — merge Maps (split table definitions)
                idx = found_keys[key_str]
                existing_entry = entries[idx]
                _, existing_val = existing_entry
                if isinstance(existing_val, Map) and isinstance(value_node, Map):
                    merged = _merge_maps(existing_val, value_node)
                    entries[idx] = (key_node, merged)
                else:
                    entries[idx] = (key_node, value_node)
            else:
                found_keys[key_str] = len(entries)
                entries.append((key_node, value_node))

            # Extract inline comment
            if hasattr(item, "trivia") and item.trivia.comment:
                text = self._comment_text(item.trivia.comment)
                if text:
                    entries.append(Comment(text))

        # Fallback: dict-based iteration for any keys body missed
        if hasattr(container, "items") and callable(container.items):
            for k, v in container.items():
                k_str = str(k)
                if k_str not in found_keys:
                    key_node = Scalar(ScalarType.STRING, k_str)
                    if isinstance(v, toml_items.Item):
                        value_node = self._convert_item(v)
                    else:
                        value_node = self._convert_native(v)
                    found_keys[k_str] = len(entries)
                    entries.append((key_node, value_node))

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
        return self._convert_scalar(item)

    def _convert_scalar(self, item) -> Scalar:
        value = item.unwrap() if hasattr(item, "unwrap") else item
        return self._convert_native(value)

    def _convert_native(self, value) -> Map | Sequence | Scalar:
        """Convert a Python native value to a CDXF node."""
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
        if not raw:
            return None
        text = raw.strip().lstrip("#").strip()
        return text if text else None


# ===================================================================
# CDXF model -> TOML converter
# ===================================================================

class _ModelToToml:
    """Convert CDXF model nodes to a tomlkit Document."""

    def convert(self, node) -> tomlkit.TOMLDocument:
        doc = tomlkit.document()
        if isinstance(node, Map):
            self._populate_table(doc, node)
        return doc

    def _populate_table(self, table, map_node: Map) -> None:
        for entry in map_node.entries:
            if isinstance(entry, Comment):
                continue
            key, value = entry
            key_str = key.value if isinstance(key, Scalar) else str(key)
            table.add(key_str, self._convert_value(value))

    def _convert_value(self, node):
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
        items = [i for i in seq.items if not isinstance(i, Comment)]

        if items and all(isinstance(i, Map) for i in items):
            aot = tomlkit.aot()
            for item in items:
                t = tomlkit.table()
                self._populate_table(t, item)
                aot.append(t)
            return aot

        arr = tomlkit.array()
        for item in items:
            if isinstance(item, Scalar):
                arr.append(item.value)
            elif isinstance(item, Sequence):
                inner = self._convert_sequence(item)
                arr.append(inner)
            elif isinstance(item, Map):
                # Inline table inside an array (mixed array)
                it = tomlkit.inline_table()
                self._populate_table(it, item)
                arr.append(it)
            else:
                arr.append(str(item))
        return arr

    @staticmethod
    def _convert_scalar(scalar: Scalar):
        return scalar.value
