"""YAML bridge — convert between YAML text and CDXF model.

Uses ruamel.yaml in round-trip mode to preserve anchors, aliases,
comments, tags, and multi-document streams.

Functions:
    from_yaml(text) -> Stream
    to_yaml(stream) -> str
"""

from __future__ import annotations

import io
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarfloat import ScalarFloat

from cdxf.model import (
    Alias as CdxfAlias,
    Anchor,
    Comment,
    Document,
    Map,
    Scalar,
    ScalarType,
    Sequence,
    SourceFormat,
    Stream,
    TagAnnotation,
)


# ===================================================================
# Public API
# ===================================================================

def from_yaml(text: str) -> Stream:
    """Parse YAML text into a CDXF Stream."""
    yaml = YAML()
    yaml.preserve_quotes = True

    documents = []
    for raw_doc in yaml.load_all(text):
        converter = _YamlToModel()
        root = converter.convert(raw_doc)
        doc = Document(root=root, source_format_hint=SourceFormat.YAML)
        documents.append(doc)

    if not documents:
        documents = [Document(
            root=Scalar(ScalarType.NULL, None),
            source_format_hint=SourceFormat.YAML,
        )]

    return Stream(documents=documents)


def to_yaml(stream: Stream) -> str:
    """Convert a CDXF Stream to YAML text."""
    yaml = YAML()
    yaml.default_flow_style = False

    # Register a clean float representer so plain Python floats
    # don't hit ruamel's buggy ScalarFloat path.
    def _represent_plain_float(representer, data):
        if data != data:
            return representer.represent_scalar('tag:yaml.org,2002:float', '.nan')
        if data == float('inf'):
            return representer.represent_scalar('tag:yaml.org,2002:float', '.inf')
        if data == float('-inf'):
            return representer.represent_scalar('tag:yaml.org,2002:float', '-.inf')
        # Use str() for clean output (e.g. 0.0001 not 1e-04)
        s = str(data)
        if '.' not in s and 'e' not in s and 'E' not in s:
            s += '.0'
        return representer.represent_scalar('tag:yaml.org,2002:float', s)

    yaml.representer.add_representer(float, _represent_plain_float)
    yaml.representer.add_representer(ScalarFloat, _represent_plain_float)

    buf = io.StringIO()

    if len(stream.documents) > 1:
        docs = [_ModelToYaml().convert(doc.root) for doc in stream.documents]
        yaml.dump_all(docs, buf)
    elif stream.documents:
        native = _ModelToYaml().convert(stream.documents[0].root)
        yaml.dump(native, buf)
    else:
        return ""

    return buf.getvalue()


# ===================================================================
# Helpers
# ===================================================================

def _get_anchor_name(value: Any) -> str | None:
    """Safely extract anchor name from a ruamel.yaml object."""
    try:
        a = value.anchor
        if a is not None and hasattr(a, "value") and a.value:
            return a.value
    except (AttributeError, TypeError):
        pass
    return None


def _extract_ca_comments(ca, key=None, seen_ids: set | None = None) -> list[str]:
    """Extract comment texts from a ruamel.yaml Comment attribute.

    If key is given, extract comments associated with that key in
    ca.items. Otherwise extract top-level comments from ca.comment.

    Uses seen_ids to skip CommentToken objects that have already been
    extracted (ruamel.yaml shares the same token object between a
    parent map's ca.items[key][3] and the child map's ca.comment).
    """
    results = []
    if seen_ids is None:
        seen_ids = set()

    def _collect_token(token):
        """Collect text from a single CommentToken, deduplicating by id."""
        if token is None:
            return
        tid = id(token)
        if tid in seen_ids:
            return
        if hasattr(token, "value"):
            text = token.value.strip().lstrip("#").strip()
            if text:
                seen_ids.add(tid)
                results.append(text)

    def _collect_item(item):
        """Collect from a single item (may be token, list, or None)."""
        if item is None:
            return
        if isinstance(item, list):
            for sub in item:
                _collect_token(sub)
        else:
            _collect_token(item)

    if key is not None:
        # Per-key comments from ca.items
        try:
            items_dict = ca.items
            if key in items_dict:
                comment_list = items_dict[key]
                if comment_list:
                    for item in comment_list:
                        _collect_item(item)
        except (AttributeError, TypeError, KeyError):
            pass
    else:
        # Top-level comment on the node
        try:
            if ca.comment:
                for item in ca.comment:
                    _collect_item(item)
        except (AttributeError, TypeError):
            pass

    return results


# ===================================================================
# YAML → CDXF model converter
# ===================================================================

class _YamlToModel:
    """Stateful converter tracking object identity for anchor/alias."""

    def __init__(self):
        self._seen: dict[int, str] = {}  # id -> anchor_name
        self._seen_comment_ids: set[int] = set()  # dedup CommentTokens

    def convert(self, value: Any) -> Any:
        if value is None:
            return Scalar(ScalarType.NULL, None)

        obj_id = id(value)

        # Check if this is an alias (same object seen before)
        if obj_id in self._seen:
            return CdxfAlias(self._seen[obj_id])

        # Check for anchor
        anchor_name = _get_anchor_name(value)
        if anchor_name:
            self._seen[obj_id] = anchor_name

        # Convert based on type
        if isinstance(value, CommentedMap):
            node = self._convert_mapping(value)
        elif isinstance(value, CommentedSeq):
            node = self._convert_sequence(value)
        elif isinstance(value, dict):
            node = self._convert_mapping(value)
        elif isinstance(value, list):
            node = self._convert_sequence(value)
        else:
            node = self._convert_scalar(value)

        # Apply anchor
        if anchor_name and hasattr(node, "anchor"):
            node.anchor = Anchor(anchor_name)

        return node

    def _convert_scalar(self, value: Any) -> Scalar:
        if isinstance(value, bool):
            return Scalar(ScalarType.BOOLEAN, value)
        if isinstance(value, int):
            return Scalar(ScalarType.INTEGER, int(value))
        if isinstance(value, float):
            return Scalar(ScalarType.FLOAT, float(value))
        if isinstance(value, str):
            return Scalar(ScalarType.STRING, str(value))
        if isinstance(value, bytes):
            return Scalar(ScalarType.BYTE_STRING, value)
        return Scalar(ScalarType.STRING, str(value))

    def _convert_mapping(self, mapping) -> Map:
        entries = []

        # --- Extract top-level comments ---
        if hasattr(mapping, "ca") and mapping.ca:
            for text in _extract_ca_comments(mapping.ca,
                                             seen_ids=self._seen_comment_ids):
                entries.append(Comment(text))

        # --- Handle merge keys (ruamel.yaml stores them in .merge) ---
        merged_obj_ids = set()
        if hasattr(mapping, "merge") and mapping.merge:
            for merge_pair in mapping.merge:
                if isinstance(merge_pair, tuple):
                    _, merged_map = merge_pair
                else:
                    merged_map = merge_pair

                # Track which object ids came from the merge
                merged_obj_ids.add(id(merged_map))

                merge_key = Scalar(
                    ScalarType.STRING, "<<",
                    tag=TagAnnotation("tag:yaml.org,2002:merge"),
                )
                merge_value = self.convert(merged_map)
                entries.append((merge_key, merge_value))

        # --- Determine which keys to skip (from merge, not overridden) ---
        keys_from_merge = set()
        if hasattr(mapping, "merge") and mapping.merge:
            for merge_pair in mapping.merge:
                if isinstance(merge_pair, tuple):
                    _, merged_map = merge_pair
                else:
                    merged_map = merge_pair
                if hasattr(merged_map, "keys"):
                    for k in merged_map.keys():
                        # Skip if the value is the same object (not overridden)
                        try:
                            if k in mapping and mapping[k] is merged_map[k]:
                                keys_from_merge.add(k)
                        except (KeyError, TypeError):
                            pass

        # --- Process own keys ---
        keys = list(mapping.keys()) if hasattr(mapping, "keys") else []
        for key in keys:
            if key in keys_from_merge:
                continue  # This key came from merge, skip it

            value = mapping[key]

            # Per-key comments
            if hasattr(mapping, "ca") and mapping.ca:
                for text in _extract_ca_comments(mapping.ca, key=key,
                                                 seen_ids=self._seen_comment_ids):
                    entries.append(Comment(text))

            key_node = self._convert_key(key)
            value_node = self.convert(value)
            entries.append((key_node, value_node))

        return Map(entries=entries)

    def _convert_sequence(self, seq) -> Sequence:
        items = []

        for idx, value in enumerate(seq):
            # Per-item comments
            if hasattr(seq, "ca") and seq.ca:
                for text in _extract_ca_comments(seq.ca, key=idx,
                                                 seen_ids=self._seen_comment_ids):
                    items.append(Comment(text))

            items.append(self.convert(value))

        return Sequence(items=items)

    def _convert_key(self, key) -> Scalar:
        if isinstance(key, bool):
            return Scalar(ScalarType.BOOLEAN, key)
        if isinstance(key, int):
            return Scalar(ScalarType.INTEGER, int(key))
        if isinstance(key, float):
            return Scalar(ScalarType.FLOAT, float(key))
        if isinstance(key, str):
            return Scalar(ScalarType.STRING, str(key))
        return Scalar(ScalarType.STRING, str(key))


# ===================================================================
# CDXF model → YAML converter
# ===================================================================

class _ModelToYaml:
    """Convert CDXF model nodes to ruamel.yaml native types."""

    def __init__(self):
        self._anchored: dict[str, Any] = {}

    def convert(self, node) -> Any:
        if isinstance(node, CdxfAlias):
            if node.name in self._anchored:
                return self._anchored[node.name]
            return None

        result = self._convert_inner(node)

        # Apply anchor
        if hasattr(node, "anchor") and node.anchor is not None:
            anchor_name = node.anchor.name
            if hasattr(result, "yaml_set_anchor"):
                result.yaml_set_anchor(anchor_name, always_dump=True)
            self._anchored[anchor_name] = result

        return result

    def _convert_inner(self, node) -> Any:
        if isinstance(node, Scalar):
            return self._convert_scalar(node)
        if isinstance(node, Map):
            return self._convert_map(node)
        if isinstance(node, Sequence):
            return self._convert_sequence(node)
        if isinstance(node, Comment):
            return None
        return None

    def _convert_scalar(self, scalar: Scalar) -> Any:
        if scalar.scalar_type == ScalarType.NULL:
            return None
        if scalar.scalar_type == ScalarType.BOOLEAN:
            return scalar.value
        if scalar.scalar_type == ScalarType.INTEGER:
            return ScalarInt(int(scalar.value))
        if scalar.scalar_type == ScalarType.FLOAT:
            return ScalarFloat(float(scalar.value))
        if scalar.scalar_type == ScalarType.STRING:
            return scalar.value
        if scalar.scalar_type == ScalarType.BYTE_STRING:
            return scalar.value
        # Temporal types — pass through as Python datetime objects
        # (ruamel.yaml can serialize these natively)
        return scalar.value

    def _convert_map(self, map_node: Map) -> CommentedMap:
        result = CommentedMap()
        merge_pairs = []
        pending_comments = []  # Buffer comments to attach to the next key

        for entry in map_node.entries:
            if isinstance(entry, Comment):
                pending_comments.append(entry.text)
                continue
            key, value = entry
            key_native = key.value if isinstance(key, Scalar) else key

            # Detect YAML merge key: key is "<<" with merge tag
            is_merge = (
                key_native == "<<"
                and isinstance(key, Scalar)
                and key.tag is not None
                and key.tag.uri == "tag:yaml.org,2002:merge"
            )

            value_native = self.convert(value)

            if is_merge:
                # Collect merge pairs for add_yaml_merge
                if isinstance(value_native, list):
                    # Sequence of maps
                    for idx, m in enumerate(value_native):
                        merge_pairs.append((idx, m))
                else:
                    merge_pairs.append((0, value_native))
            else:
                result[key_native] = value_native
                # Attach any buffered comments to this key
                if pending_comments:
                    for comment_text in pending_comments:
                        result.yaml_set_comment_before_after_key(
                            key_native,
                            before=comment_text,
                        )
                    pending_comments = []

        # If there are trailing comments with no following key,
        # attach them as end-of-map comments
        if pending_comments and result:
            last_key = list(result.keys())[-1]
            for comment_text in pending_comments:
                result.yaml_set_comment_before_after_key(
                    last_key,
                    after=comment_text,
                )

        # Apply merge keys via ruamel.yaml's merge API so that
        # the dumper emits proper <<: *alias syntax
        if merge_pairs:
            result.add_yaml_merge(merge_pairs)

        return result

    def _convert_sequence(self, seq: Sequence) -> CommentedSeq:
        result = CommentedSeq()
        pending_comments = []
        for item in seq.items:
            if isinstance(item, Comment):
                pending_comments.append(item.text)
                continue
            result.append(self.convert(item))
            if pending_comments:
                idx = len(result) - 1
                for comment_text in pending_comments:
                    result.yaml_set_comment_before_after_key(
                        idx,
                        before=comment_text,
                    )
                pending_comments = []
        return result
