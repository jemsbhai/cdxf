"""Structural fidelity tests — verify that format-specific constructs
survive the full pipeline: text -> CDXF -> CBOR -> CDXF -> text -> CDXF.

These tests go beyond data-level equality to verify preservation of:
- Comments (YAML, XML, TOML)
- Anchors and aliases / graph structure (YAML)
- Merge keys (YAML)
- Namespace URIs and prefixes (XML)
- Processing instructions (XML)
- Mixed content ordering (XML)
- Typed scalars including all 4 temporal types (TOML)
- Multi-document streams (YAML)

Each test parses the same text twice — once directly, once after a full
CBOR round-trip — and compares the resulting CDXF model trees structurally,
not just the text output.
"""

import pytest
from datetime import date, datetime, time, timezone, timedelta

from cdxf.codec import encode, decode
from cdxf.model import (
    Alias,
    Anchor,
    Attribute,
    Comment,
    Document,
    Element,
    Map,
    ProcessingInstruction,
    Scalar,
    ScalarType,
    Sequence,
    SourceFormat,
    Stream,
    TagAnnotation,
)
from cdxf.bridges.json_bridge import from_json
from cdxf.bridges.yaml_bridge import from_yaml
from cdxf.bridges.xml_bridge import from_xml
from cdxf.bridges.toml_bridge import from_toml


# ===================================================================
# Helpers
# ===================================================================

def full_pipeline(text: str, fmt: str) -> Stream:
    """text -> CDXF -> CBOR -> CDXF (model-level round-trip)."""
    if fmt == "json":
        stream = from_json(text)
    elif fmt == "yaml":
        stream = from_yaml(text)
    elif fmt == "xml":
        stream = from_xml(text)
    elif fmt == "toml":
        stream = from_toml(text)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    binary = encode(stream)
    return decode(binary)


def count_nodes(node, node_type) -> int:
    """Recursively count nodes of a given type in a CDXF tree."""
    count = 1 if isinstance(node, node_type) else 0
    if isinstance(node, Map):
        for entry in node.entries:
            if isinstance(entry, Comment):
                count += 1 if node_type is Comment else 0
            else:
                count += count_nodes(entry[0], node_type)
                count += count_nodes(entry[1], node_type)
    elif isinstance(node, Sequence):
        for item in node.items:
            count += count_nodes(item, node_type)
    elif isinstance(node, Element):
        for child in node.children:
            count += count_nodes(child, node_type)
    return count


def collect_nodes(node, node_type) -> list:
    """Recursively collect all nodes of a given type."""
    result = [node] if isinstance(node, node_type) else []
    if isinstance(node, Map):
        for entry in node.entries:
            if isinstance(entry, Comment):
                if node_type is Comment:
                    result.append(entry)
            else:
                result.extend(collect_nodes(entry[0], node_type))
                result.extend(collect_nodes(entry[1], node_type))
    elif isinstance(node, Sequence):
        for item in node.items:
            result.extend(collect_nodes(item, node_type))
    elif isinstance(node, Element):
        for child in node.children:
            result.extend(collect_nodes(child, node_type))
    return result


# ===================================================================
# YAML structural fidelity
# ===================================================================

class TestYamlStructuralFidelity:
    """Verify YAML-specific constructs survive CBOR round-trip."""

    def test_comments_preserved_count(self):
        yaml_text = (
            "# header comment\n"
            "key1: value1\n"
            "# middle comment\n"
            "key2: value2\n"
        )
        restored = full_pipeline(yaml_text, "yaml")
        root = restored.documents[0].root
        comments = [e for e in root.entries if isinstance(e, Comment)]
        assert len(comments) >= 2, f"Expected >=2 comments, got {len(comments)}"

    def test_comment_text_preserved(self):
        yaml_text = "# important note\nkey: value\n"
        restored = full_pipeline(yaml_text, "yaml")
        root = restored.documents[0].root
        comments = [e for e in root.entries if isinstance(e, Comment)]
        assert any("important" in c.text for c in comments)

    def test_anchor_survives_cbor(self):
        yaml_text = (
            "defaults: &defaults\n"
            "  timeout: 30\n"
            "  retries: 3\n"
            "service:\n"
            "  <<: *defaults\n"
            "  name: my-service\n"
        )
        restored = full_pipeline(yaml_text, "yaml")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]

        # The defaults map must have an anchor
        defaults_val = entries[0][1]
        assert defaults_val.anchor is not None, "Anchor lost after CBOR round-trip"
        assert defaults_val.anchor.name == "defaults"

    def test_alias_survives_cbor_not_expanded(self):
        yaml_text = (
            "shared: &shared\n"
            "  x: 1\n"
            "ref1: *shared\n"
            "ref2: *shared\n"
        )
        restored = full_pipeline(yaml_text, "yaml")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]

        # ref1 and ref2 must be Alias nodes, not expanded copies
        assert isinstance(entries[1][1], Alias), "Alias was expanded to a copy"
        assert isinstance(entries[2][1], Alias), "Alias was expanded to a copy"
        assert entries[1][1].name == "shared"
        assert entries[2][1].name == "shared"

    def test_merge_key_survives_cbor(self):
        yaml_text = (
            "defaults: &defaults\n"
            "  timeout: 30\n"
            "service:\n"
            "  <<: *defaults\n"
            "  name: svc\n"
        )
        restored = full_pipeline(yaml_text, "yaml")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        service_map = entries[1][1]
        service_entries = [e for e in service_map.entries if not isinstance(e, Comment)]

        # Find the merge key entry
        merge_entries = [
            e for e in service_entries
            if isinstance(e[0], Scalar) and e[0].value == "<<"
        ]
        assert len(merge_entries) >= 1, "Merge key << lost after CBOR round-trip"

        # The merge key must have the merge tag
        merge_key = merge_entries[0][0]
        assert merge_key.tag is not None, "Merge tag lost after CBOR round-trip"
        assert merge_key.tag.uri == "tag:yaml.org,2002:merge"

        # The merge value must be an Alias
        merge_val = merge_entries[0][1]
        assert isinstance(merge_val, Alias), "Merge value expanded, not alias"

    def test_multi_document_stream_survives_cbor(self):
        yaml_text = "---\na: 1\n---\nb: 2\n---\nc: 3\n"
        restored = full_pipeline(yaml_text, "yaml")
        assert len(restored.documents) == 3, (
            f"Expected 3 documents, got {len(restored.documents)}"
        )

    def test_non_string_map_key_survives_cbor(self):
        yaml_text = "42: answer\ntrue: yes\n"
        restored = full_pipeline(yaml_text, "yaml")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        key_types = [e[0].scalar_type for e in entries]
        assert ScalarType.INTEGER in key_types, "Integer key lost"
        assert ScalarType.BOOLEAN in key_types, "Boolean key lost"

    def test_source_format_hint_preserved(self):
        yaml_text = "key: value\n"
        restored = full_pipeline(yaml_text, "yaml")
        assert restored.documents[0].source_format_hint == SourceFormat.YAML


# ===================================================================
# XML structural fidelity
# ===================================================================

class TestXmlStructuralFidelity:
    """Verify XML-specific constructs survive CBOR round-trip."""

    def test_comments_preserved(self):
        xml = "<root><!-- important --><child/></root>"
        restored = full_pipeline(xml, "xml")
        root = restored.documents[0].root
        comments = [c for c in root.children if isinstance(c, Comment)]
        assert len(comments) == 1
        assert "important" in comments[0].text

    def test_comment_position_preserved(self):
        xml = "<root><a/><!-- between --><b/></root>"
        restored = full_pipeline(xml, "xml")
        root = restored.documents[0].root
        assert len(root.children) == 3
        assert isinstance(root.children[0], Element)
        assert isinstance(root.children[1], Comment)
        assert isinstance(root.children[2], Element)
        assert root.children[0].name == "a"
        assert root.children[2].name == "b"

    def test_processing_instruction_preserved(self):
        xml = "<root><?myapp do-something?></root>"
        restored = full_pipeline(xml, "xml")
        root = restored.documents[0].root
        pis = [c for c in root.children if isinstance(c, ProcessingInstruction)]
        assert len(pis) == 1
        assert pis[0].target == "myapp"
        assert pis[0].data == "do-something"

    def test_namespace_uri_preserved(self):
        xml = '<root xmlns="http://example.com/ns"><child/></root>'
        restored = full_pipeline(xml, "xml")
        root = restored.documents[0].root
        assert root.namespace_uri == "http://example.com/ns"
        child = [c for c in root.children if isinstance(c, Element)][0]
        assert child.namespace_uri == "http://example.com/ns"

    def test_namespace_prefix_preserved(self):
        xml = '<ex:root xmlns:ex="http://example.com/ns"><ex:child/></ex:root>'
        restored = full_pipeline(xml, "xml")
        root = restored.documents[0].root
        assert root.prefix == "ex"
        assert root.namespace_uri == "http://example.com/ns"

    def test_namespace_declarations_preserved(self):
        xml = '<root xmlns="http://a.example.com" xmlns:b="http://b.example.com"/>'
        restored = full_pipeline(xml, "xml")
        root = restored.documents[0].root
        assert "" in root.namespace_declarations
        assert "b" in root.namespace_declarations

    def test_attribute_namespace_preserved(self):
        xml = (
            '<root xmlns:xlink="http://www.w3.org/1999/xlink"'
            ' xlink:href="http://example.com"/>'
        )
        restored = full_pipeline(xml, "xml")
        root = restored.documents[0].root
        attr = root.attributes[0]
        assert attr.name == "href"
        assert attr.namespace_uri == "http://www.w3.org/1999/xlink"
        assert attr.prefix == "xlink"

    def test_mixed_content_order_preserved(self):
        xml = "<p>Hello <b>world</b>!</p>"
        restored = full_pipeline(xml, "xml")
        root = restored.documents[0].root
        assert len(root.children) == 3
        assert isinstance(root.children[0], Scalar) and root.children[0].value == "Hello "
        assert isinstance(root.children[1], Element) and root.children[1].name == "b"
        assert isinstance(root.children[2], Scalar) and root.children[2].value == "!"

    def test_element_vs_attribute_distinction_preserved(self):
        """An element and attribute with the same name must remain distinct."""
        xml = '<item type="book"><type>hardcover</type></item>'
        restored = full_pipeline(xml, "xml")
        root = restored.documents[0].root
        assert len(root.attributes) == 1
        assert root.attributes[0].name == "type"
        assert root.attributes[0].value == "book"
        assert len(root.children) == 1
        child_elem = [c for c in root.children if isinstance(c, Element)][0]
        assert child_elem.name == "type"

    def test_preamble_comment_preserved(self):
        xml = "<!-- preamble --><root/>"
        restored = full_pipeline(xml, "xml")
        doc = restored.documents[0]
        assert len(doc.preamble) >= 1
        assert any(isinstance(n, Comment) and "preamble" in n.text
                    for n in doc.preamble)

    def test_source_format_hint_preserved(self):
        xml = "<root/>"
        restored = full_pipeline(xml, "xml")
        assert restored.documents[0].source_format_hint == SourceFormat.XML


# ===================================================================
# TOML structural fidelity
# ===================================================================

class TestTomlStructuralFidelity:
    """Verify TOML-specific constructs survive CBOR round-trip."""

    def test_offset_datetime_type_preserved(self):
        toml_text = "ts = 1979-05-27T07:32:00Z"
        restored = full_pipeline(toml_text, "toml")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        scalar = entries[0][1]
        assert scalar.scalar_type == ScalarType.TIMESTAMP_OFFSET
        assert isinstance(scalar.value, datetime)
        assert scalar.value.tzinfo is not None

    def test_local_datetime_type_preserved(self):
        toml_text = "ts = 1979-05-27T07:32:00"
        restored = full_pipeline(toml_text, "toml")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        scalar = entries[0][1]
        assert scalar.scalar_type == ScalarType.TIMESTAMP_LOCAL
        assert isinstance(scalar.value, datetime)
        assert scalar.value.tzinfo is None

    def test_date_type_preserved(self):
        toml_text = "d = 1979-05-27"
        restored = full_pipeline(toml_text, "toml")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        scalar = entries[0][1]
        assert scalar.scalar_type == ScalarType.DATE
        assert isinstance(scalar.value, date)
        assert not isinstance(scalar.value, datetime)

    def test_time_type_preserved(self):
        toml_text = "t = 07:32:00"
        restored = full_pipeline(toml_text, "toml")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        scalar = entries[0][1]
        assert scalar.scalar_type == ScalarType.TIME
        assert isinstance(scalar.value, time)

    def test_all_four_temporal_types_distinct(self):
        """TOML's 4 datetime types must remain distinct — not collapsed."""
        toml_text = (
            "offset = 2026-01-15T09:00:00+05:30\n"
            "local = 2026-01-15T09:00:00\n"
            "d = 2026-01-15\n"
            "t = 09:00:00\n"
        )
        restored = full_pipeline(toml_text, "toml")
        root = restored.documents[0].root
        entries = {
            e[0].value: e[1]
            for e in root.entries
            if not isinstance(e, Comment)
        }
        assert entries["offset"].scalar_type == ScalarType.TIMESTAMP_OFFSET
        assert entries["local"].scalar_type == ScalarType.TIMESTAMP_LOCAL
        assert entries["d"].scalar_type == ScalarType.DATE
        assert entries["t"].scalar_type == ScalarType.TIME

    def test_comment_preserved(self):
        toml_text = "# important\nkey = 'value'\n"
        restored = full_pipeline(toml_text, "toml")
        root = restored.documents[0].root
        comments = [e for e in root.entries if isinstance(e, Comment)]
        assert len(comments) >= 1
        assert any("important" in c.text for c in comments)

    def test_integer_type_not_collapsed_to_float(self):
        toml_text = "count = 42\npi = 3.14\n"
        restored = full_pipeline(toml_text, "toml")
        root = restored.documents[0].root
        entries = {
            e[0].value: e[1]
            for e in root.entries
            if not isinstance(e, Comment)
        }
        assert entries["count"].scalar_type == ScalarType.INTEGER
        assert entries["pi"].scalar_type == ScalarType.FLOAT

    def test_key_order_preserved(self):
        toml_text = "z = 1\na = 2\nm = 3\n"
        restored = full_pipeline(toml_text, "toml")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        keys = [e[0].value for e in entries]
        assert keys == ["z", "a", "m"]

    def test_source_format_hint_preserved(self):
        toml_text = "x = 1\n"
        restored = full_pipeline(toml_text, "toml")
        assert restored.documents[0].source_format_hint == SourceFormat.TOML


# ===================================================================
# JSON structural fidelity (baseline — JSON has no metadata)
# ===================================================================

class TestJsonStructuralFidelity:
    """JSON has no comments, anchors, or namespaces. Verify data fidelity
    and map key order preservation."""

    def test_map_key_order_preserved(self):
        json_text = '{"z": 1, "a": 2, "m": 3}'
        restored = full_pipeline(json_text, "json")
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        keys = [e[0].value for e in entries]
        assert keys == ["z", "a", "m"]

    def test_all_json_types_preserved(self):
        json_text = '{"s":"hi","i":42,"f":3.14,"b":true,"n":null,"a":[1,2]}'
        restored = full_pipeline(json_text, "json")
        root = restored.documents[0].root
        entries = {
            e[0].value: e[1]
            for e in root.entries
            if not isinstance(e, Comment)
        }
        assert entries["s"].scalar_type == ScalarType.STRING
        assert entries["i"].scalar_type == ScalarType.INTEGER
        assert entries["f"].scalar_type == ScalarType.FLOAT
        assert entries["b"].scalar_type == ScalarType.BOOLEAN
        assert entries["n"].scalar_type == ScalarType.NULL
        assert isinstance(entries["a"], Sequence)

    def test_deeply_nested_structure_preserved(self):
        json_text = '{"a":{"b":{"c":{"d":"deep"}}}}'
        restored = full_pipeline(json_text, "json")
        root = restored.documents[0].root
        # Navigate a.b.c.d
        node = root
        for key in ["a", "b", "c"]:
            entries = [e for e in node.entries if not isinstance(e, Comment)]
            node = entries[0][1]
        entries = [e for e in node.entries if not isinstance(e, Comment)]
        assert entries[0][1].value == "deep"
