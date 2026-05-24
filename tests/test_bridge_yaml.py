"""Tests for the YAML bridge — from_yaml() and to_yaml().

These tests are critical: they validate CDXF's core differentiating
claims versus CBOR, MessagePack, and Ion.
"""

import pytest

from cdxf.model import (
    Alias,
    Anchor,
    Comment,
    Directive,
    Document,
    Map,
    Scalar,
    ScalarType,
    Sequence,
    SourceFormat,
    Stream,
    TagAnnotation,
)
from cdxf.bridges.yaml_bridge import from_yaml, to_yaml


# ===================================================================
# from_yaml: basic types
# ===================================================================

class TestFromYamlBasicTypes:
    def test_string(self):
        stream = from_yaml("hello")
        assert stream.documents[0].root.value == "hello"

    def test_integer(self):
        stream = from_yaml("42")
        root = stream.documents[0].root
        assert root.scalar_type == ScalarType.INTEGER
        assert root.value == 42

    def test_float(self):
        stream = from_yaml("3.14")
        root = stream.documents[0].root
        assert root.scalar_type == ScalarType.FLOAT
        assert root.value == pytest.approx(3.14)

    def test_boolean_true(self):
        stream = from_yaml("true")
        root = stream.documents[0].root
        assert root.scalar_type == ScalarType.BOOLEAN
        assert root.value is True

    def test_boolean_false(self):
        stream = from_yaml("false")
        root = stream.documents[0].root
        assert root.value is False

    def test_null_explicit(self):
        stream = from_yaml("null")
        root = stream.documents[0].root
        assert root.scalar_type == ScalarType.NULL

    def test_null_tilde(self):
        stream = from_yaml("~")
        root = stream.documents[0].root
        assert root.scalar_type == ScalarType.NULL

    def test_source_format_hint(self):
        stream = from_yaml("hello")
        assert stream.documents[0].source_format_hint == SourceFormat.YAML


# ===================================================================
# from_yaml: mappings and sequences
# ===================================================================

class TestFromYamlStructures:
    def test_simple_mapping(self):
        stream = from_yaml("name: Alice\nage: 30")
        root = stream.documents[0].root
        assert isinstance(root, Map)
        keys = [e[0].value for e in root.entries if not isinstance(e, Comment)]
        assert "name" in keys
        assert "age" in keys

    def test_simple_sequence(self):
        stream = from_yaml("- 1\n- 2\n- 3")
        root = stream.documents[0].root
        assert isinstance(root, Sequence)
        values = [i.value for i in root.items if not isinstance(i, Comment)]
        assert values == [1, 2, 3]

    def test_nested_mapping(self):
        yaml_text = "server:\n  host: localhost\n  port: 8080"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        assert isinstance(root, Map)

    def test_mapping_key_order_preserved(self):
        yaml_text = "z: 1\na: 2\nm: 3"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        keys = [e[0].value for e in root.entries if not isinstance(e, Comment)]
        assert keys == ["z", "a", "m"]

    def test_sequence_of_mappings(self):
        yaml_text = "- name: Alice\n  age: 30\n- name: Bob\n  age: 25"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        assert isinstance(root, Sequence)
        items = [i for i in root.items if not isinstance(i, Comment)]
        assert len(items) == 2
        assert isinstance(items[0], Map)


# ===================================================================
# CLAIM: Anchor/alias preservation
# ===================================================================

class TestFromYamlAnchorsAliases:
    def test_anchor_and_alias(self):
        yaml_text = "defaults: &defaults\n  timeout: 30\nproduction:\n  ref: *defaults"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]

        # defaults should have an anchor
        defaults_val = entries[0][1]
        assert defaults_val.anchor is not None
        assert defaults_val.anchor.name == "defaults"

        # ref should be an Alias, NOT an expanded copy
        production = entries[1][1]
        prod_entries = [e for e in production.entries if not isinstance(e, Comment)]
        ref_val = prod_entries[0][1]
        assert isinstance(ref_val, Alias), (
            "Alias was expanded to a copy — this violates CDXF's core claim"
        )
        assert ref_val.name == "defaults"

    def test_multiple_aliases_same_anchor(self):
        yaml_text = "- &val 42\n- *val\n- *val"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        items = [i for i in root.items if not isinstance(i, Comment)]

        assert items[0].anchor is not None
        assert isinstance(items[1], Alias)
        assert isinstance(items[2], Alias)

    def test_anchor_on_mapping(self):
        yaml_text = "base: &base\n  x: 1\n  y: 2\nref: *base"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].anchor.name == "base"
        assert isinstance(entries[1][1], Alias)

    def test_anchor_on_sequence(self):
        yaml_text = "items: &list\n  - a\n  - b\nref: *list"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].anchor.name == "list"
        assert isinstance(entries[1][1], Alias)


# ===================================================================
# CLAIM: Merge key preservation
# ===================================================================

class TestFromYamlMergeKeys:
    def test_merge_key_preserved(self):
        yaml_text = (
            "defaults: &defaults\n"
            "  timeout: 30\n"
            "  retries: 3\n"
            "production:\n"
            "  <<: *defaults\n"
            "  timeout: 60\n"
        )
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]

        production = entries[1][1]
        prod_entries = [e for e in production.entries if not isinstance(e, Comment)]

        # Find the merge key entry
        merge_entry = None
        for entry in prod_entries:
            if isinstance(entry, tuple) and entry[0].value == "<<":
                merge_entry = entry
                break

        assert merge_entry is not None, "Merge key << not found"
        assert isinstance(merge_entry[1], Alias), (
            "Merge key value should be an Alias, not expanded"
        )


# ===================================================================
# CLAIM: Comment preservation
# ===================================================================

class TestFromYamlComments:
    def test_comment_before_mapping_entry(self):
        yaml_text = "# server config\nhost: localhost"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        has_comment = any(isinstance(e, Comment) for e in root.entries)
        assert has_comment, "Comment before entry was lost"

    def test_comment_between_entries(self):
        yaml_text = "a: 1\n# between\nb: 2"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        comments = [e for e in root.entries if isinstance(e, Comment)]
        assert len(comments) >= 1
        assert any("between" in c.text for c in comments)

    def test_comment_in_sequence(self):
        yaml_text = "- 1\n# mid comment\n- 2"
        stream = from_yaml(yaml_text)
        root = stream.documents[0].root
        comments = [i for i in root.items if isinstance(i, Comment)]
        assert len(comments) >= 1


# ===================================================================
# CLAIM: Multi-document stream
# ===================================================================

class TestFromYamlMultiDocument:
    def test_two_documents(self):
        yaml_text = "---\nhello\n---\nworld"
        stream = from_yaml(yaml_text)
        assert len(stream.documents) >= 2

    def test_three_documents(self):
        yaml_text = "---\n1\n---\n2\n---\n3"
        stream = from_yaml(yaml_text)
        assert len(stream.documents) >= 3
        values = [d.root.value for d in stream.documents]
        assert values == [1, 2, 3]

    def test_mixed_document_types(self):
        yaml_text = "---\nhello\n---\n- 1\n- 2\n---\nkey: val"
        stream = from_yaml(yaml_text)
        assert isinstance(stream.documents[0].root, Scalar)
        assert isinstance(stream.documents[1].root, Sequence)
        assert isinstance(stream.documents[2].root, Map)


# ===================================================================
# to_yaml: CDXF model → YAML text
# ===================================================================

class TestToYaml:
    def test_simple_mapping(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "name"), Scalar(ScalarType.STRING, "Alice")),
            (Scalar(ScalarType.STRING, "age"), Scalar(ScalarType.INTEGER, 30)),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_yaml(stream)
        assert "name" in text
        assert "Alice" in text
        assert "30" in text

    def test_simple_sequence(self):
        root = Sequence(items=[
            Scalar(ScalarType.INTEGER, 1),
            Scalar(ScalarType.INTEGER, 2),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_yaml(stream)
        assert "- 1" in text
        assert "- 2" in text

    def test_null_output(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "val"), Scalar(ScalarType.NULL, None)),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_yaml(stream)
        # Both 'val:' and 'val: null' are valid YAML null
        assert 'val' in text

    def test_anchor_alias_output(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "src"),
             Scalar(ScalarType.INTEGER, 42, anchor=Anchor("val"))),
            (Scalar(ScalarType.STRING, "ref"), Alias("val")),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_yaml(stream)
        assert "&val" in text
        assert "*val" in text

    def test_multi_document_output(self):
        stream = Stream(documents=[
            Document(root=Scalar(ScalarType.STRING, "first")),
            Document(root=Scalar(ScalarType.STRING, "second")),
        ])
        text = to_yaml(stream)
        assert "---" in text
        assert "first" in text
        assert "second" in text


# ===================================================================
# Round-trip: YAML → CDXF → YAML (semantic equivalence)
# ===================================================================

class TestYamlRoundTrip:
    def test_simple_roundtrip(self):
        yaml_text = "name: Alice\nage: 30"
        stream = from_yaml(yaml_text)
        output = to_yaml(stream)
        restored = from_yaml(output)

        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        values = {e[0].value: e[1].value for e in entries}
        assert values["name"] == "Alice"
        assert values["age"] == 30

    def test_anchor_alias_roundtrip(self):
        yaml_text = "src: &val 42\nref: *val"
        stream = from_yaml(yaml_text)
        output = to_yaml(stream)
        restored = from_yaml(output)

        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].anchor is not None
        assert isinstance(entries[1][1], Alias)


# ===================================================================
# Full pipeline: YAML → CDXF model → CBOR → CDXF model → YAML
# ===================================================================

class TestYamlFullPipeline:
    def test_full_pipeline_with_anchors(self):
        from cdxf.codec import encode, decode

        yaml_text = "defaults: &defaults\n  timeout: 30\nproduction:\n  ref: *defaults"
        stream = from_yaml(yaml_text)
        binary = encode(stream)
        restored_stream = decode(binary)
        output = to_yaml(restored_stream)
        final = from_yaml(output)

        root = final.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].anchor is not None
        assert isinstance(entries[1][1].entries[0][1], Alias)
