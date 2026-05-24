"""Tests for CDXF CBOR encoding and decoding — every node kind."""

import math
from datetime import date, datetime, time, timezone, timedelta

import cbor2
import pytest

from cdxf.model import (
    Alias,
    Anchor,
    Attribute,
    Comment,
    Directive,
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
from cdxf.codec import encode, decode, Encoder
from cdxf import tags


# ===================================================================
# Helpers
# ===================================================================

def roundtrip(node):
    """Encode a CDXF node to bytes and decode it back."""
    data = encode(node)
    return decode(data)


def assert_scalar_roundtrip(scalar_type, value):
    """Assert a scalar survives round-trip with type and value preserved."""
    original = Stream(documents=[Document(root=Scalar(scalar_type, value))])
    restored = roundtrip(original)
    root = restored.documents[0].root
    assert isinstance(root, Scalar)
    assert root.scalar_type == scalar_type
    if scalar_type == ScalarType.FLOAT and math.isnan(value):
        assert math.isnan(root.value)
    else:
        assert root.value == value


# ===================================================================
# Scalar encoding — every scalar subtype
# ===================================================================

class TestScalarEncoding:
    def test_null(self):
        assert_scalar_roundtrip(ScalarType.NULL, None)

    def test_boolean_true(self):
        assert_scalar_roundtrip(ScalarType.BOOLEAN, True)

    def test_boolean_false(self):
        assert_scalar_roundtrip(ScalarType.BOOLEAN, False)

    def test_integer_zero(self):
        assert_scalar_roundtrip(ScalarType.INTEGER, 0)

    def test_integer_positive(self):
        assert_scalar_roundtrip(ScalarType.INTEGER, 42)

    def test_integer_negative(self):
        assert_scalar_roundtrip(ScalarType.INTEGER, -999)

    def test_integer_large(self):
        assert_scalar_roundtrip(ScalarType.INTEGER, 2**63)

    def test_integer_very_large_bignum(self):
        assert_scalar_roundtrip(ScalarType.INTEGER, 2**128)

    def test_integer_negative_bignum(self):
        assert_scalar_roundtrip(ScalarType.INTEGER, -(2**128))

    def test_float_normal(self):
        assert_scalar_roundtrip(ScalarType.FLOAT, 3.14)

    def test_float_zero(self):
        assert_scalar_roundtrip(ScalarType.FLOAT, 0.0)

    def test_float_negative(self):
        assert_scalar_roundtrip(ScalarType.FLOAT, -273.15)

    def test_float_infinity(self):
        assert_scalar_roundtrip(ScalarType.FLOAT, float("inf"))

    def test_float_negative_infinity(self):
        assert_scalar_roundtrip(ScalarType.FLOAT, float("-inf"))

    def test_float_nan(self):
        assert_scalar_roundtrip(ScalarType.FLOAT, float("nan"))

    def test_string_empty(self):
        assert_scalar_roundtrip(ScalarType.STRING, "")

    def test_string_ascii(self):
        assert_scalar_roundtrip(ScalarType.STRING, "hello world")

    def test_string_unicode(self):
        assert_scalar_roundtrip(ScalarType.STRING, "こんにちは世界 🌍")

    def test_string_with_special_chars(self):
        assert_scalar_roundtrip(ScalarType.STRING, 'line1\nline2\ttab"quote')

    def test_byte_string(self):
        assert_scalar_roundtrip(ScalarType.BYTE_STRING, b"\x00\x01\x02\xff")

    def test_byte_string_empty(self):
        assert_scalar_roundtrip(ScalarType.BYTE_STRING, b"")

    def test_timestamp_offset(self):
        assert_scalar_roundtrip(
            ScalarType.TIMESTAMP_OFFSET,
            datetime(2024, 5, 27, 7, 32, tzinfo=timezone(timedelta(hours=-4))),
        )

    def test_timestamp_offset_utc(self):
        assert_scalar_roundtrip(
            ScalarType.TIMESTAMP_OFFSET,
            datetime(2024, 5, 27, 7, 32, tzinfo=timezone.utc),
        )

    def test_timestamp_local(self):
        assert_scalar_roundtrip(
            ScalarType.TIMESTAMP_LOCAL,
            datetime(2024, 5, 27, 7, 32),
        )

    def test_date(self):
        assert_scalar_roundtrip(ScalarType.DATE, date(2024, 5, 27))

    def test_time(self):
        assert_scalar_roundtrip(
            ScalarType.TIME, time(7, 32, 0, 999000),
        )

    def test_time_no_fraction(self):
        assert_scalar_roundtrip(ScalarType.TIME, time(7, 32, 0))


# ===================================================================
# Map encoding
# ===================================================================

class TestMapEncoding:
    def test_empty_map(self):
        original = Stream(documents=[Document(root=Map())])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert isinstance(root, Map)
        assert len(root.entries) == 0

    def test_map_with_string_keys(self):
        original = Stream(documents=[Document(root=Map(entries=[
            (Scalar(ScalarType.STRING, "a"), Scalar(ScalarType.INTEGER, 1)),
            (Scalar(ScalarType.STRING, "b"), Scalar(ScalarType.INTEGER, 2)),
        ]))])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert isinstance(root, Map)
        assert len(root.entries) == 2

    def test_map_preserves_key_order(self):
        original = Stream(documents=[Document(root=Map(entries=[
            (Scalar(ScalarType.STRING, "z"), Scalar(ScalarType.INTEGER, 1)),
            (Scalar(ScalarType.STRING, "a"), Scalar(ScalarType.INTEGER, 2)),
            (Scalar(ScalarType.STRING, "m"), Scalar(ScalarType.INTEGER, 3)),
        ]))])
        restored = roundtrip(original)
        keys = [e[0].value for e in restored.documents[0].root.entries]
        assert keys == ["z", "a", "m"]

    def test_map_with_non_string_key(self):
        original = Stream(documents=[Document(root=Map(entries=[
            (Scalar(ScalarType.INTEGER, 42), Scalar(ScalarType.STRING, "answer")),
        ]))])
        restored = roundtrip(original)
        key = restored.documents[0].root.entries[0][0]
        assert key.scalar_type == ScalarType.INTEGER
        assert key.value == 42

    def test_map_nested(self):
        inner = Map(entries=[
            (Scalar(ScalarType.STRING, "x"), Scalar(ScalarType.INTEGER, 1)),
        ])
        original = Stream(documents=[Document(root=Map(entries=[
            (Scalar(ScalarType.STRING, "inner"), inner),
        ]))])
        restored = roundtrip(original)
        inner_restored = restored.documents[0].root.entries[0][1]
        assert isinstance(inner_restored, Map)
        assert inner_restored.entries[0][1].value == 1

    def test_map_with_interleaved_comments(self):
        original = Stream(documents=[Document(root=Map(entries=[
            (Scalar(ScalarType.STRING, "key1"), Scalar(ScalarType.INTEGER, 1)),
            Comment("between entries"),
            (Scalar(ScalarType.STRING, "key2"), Scalar(ScalarType.INTEGER, 2)),
        ]))])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert len(root.entries) == 3
        assert isinstance(root.entries[1], Comment)
        assert root.entries[1].text == "between entries"


# ===================================================================
# Sequence encoding
# ===================================================================

class TestSequenceEncoding:
    def test_empty_sequence(self):
        original = Stream(documents=[Document(root=Sequence())])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert isinstance(root, Sequence)
        assert len(root.items) == 0

    def test_sequence_with_items(self):
        original = Stream(documents=[Document(root=Sequence(items=[
            Scalar(ScalarType.INTEGER, 1),
            Scalar(ScalarType.INTEGER, 2),
            Scalar(ScalarType.INTEGER, 3),
        ]))])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert len(root.items) == 3
        assert [i.value for i in root.items] == [1, 2, 3]

    def test_sequence_preserves_order(self):
        original = Stream(documents=[Document(root=Sequence(items=[
            Scalar(ScalarType.STRING, "c"),
            Scalar(ScalarType.STRING, "a"),
            Scalar(ScalarType.STRING, "b"),
        ]))])
        restored = roundtrip(original)
        values = [i.value for i in restored.documents[0].root.items]
        assert values == ["c", "a", "b"]

    def test_sequence_heterogeneous_types(self):
        original = Stream(documents=[Document(root=Sequence(items=[
            Scalar(ScalarType.INTEGER, 1),
            Scalar(ScalarType.STRING, "hello"),
            Scalar(ScalarType.BOOLEAN, True),
            Scalar(ScalarType.NULL, None),
        ]))])
        restored = roundtrip(original)
        items = restored.documents[0].root.items
        assert items[0].scalar_type == ScalarType.INTEGER
        assert items[1].scalar_type == ScalarType.STRING
        assert items[2].scalar_type == ScalarType.BOOLEAN
        assert items[3].scalar_type == ScalarType.NULL

    def test_sequence_with_interleaved_comments(self):
        original = Stream(documents=[Document(root=Sequence(items=[
            Scalar(ScalarType.INTEGER, 1),
            Comment("between items"),
            Scalar(ScalarType.INTEGER, 2),
        ]))])
        restored = roundtrip(original)
        items = restored.documents[0].root.items
        assert len(items) == 3
        assert isinstance(items[1], Comment)
        assert items[1].text == "between items"

    def test_sequence_nested(self):
        inner = Sequence(items=[Scalar(ScalarType.INTEGER, 1)])
        original = Stream(documents=[Document(root=Sequence(items=[inner]))])
        restored = roundtrip(original)
        assert isinstance(restored.documents[0].root.items[0], Sequence)


# ===================================================================
# Comment encoding
# ===================================================================

class TestCommentEncoding:
    def test_comment_text_preserved(self):
        original = Stream(documents=[Document(root=Map(entries=[
            Comment("this is a comment"),
            (Scalar(ScalarType.STRING, "key"), Scalar(ScalarType.INTEGER, 1)),
        ]))])
        restored = roundtrip(original)
        assert isinstance(restored.documents[0].root.entries[0], Comment)
        assert restored.documents[0].root.entries[0].text == "this is a comment"

    def test_empty_comment(self):
        original = Stream(documents=[Document(root=Sequence(items=[
            Comment(""),
            Scalar(ScalarType.INTEGER, 1),
        ]))])
        restored = roundtrip(original)
        assert restored.documents[0].root.items[0].text == ""

    def test_comment_with_special_chars(self):
        text = "TODO: fix this! @see https://example.com #123\n  indented"
        original = Stream(documents=[Document(root=Sequence(items=[
            Comment(text),
        ]))])
        restored = roundtrip(original)
        assert restored.documents[0].root.items[0].text == text


# ===================================================================
# Element encoding (XML)
# ===================================================================

class TestElementEncoding:
    def test_simple_element(self):
        original = Stream(documents=[Document(root=Element(name="div"))])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert isinstance(root, Element)
        assert root.name == "div"
        assert root.namespace_uri is None
        assert len(root.attributes) == 0
        assert len(root.children) == 0

    def test_element_with_namespace(self):
        original = Stream(documents=[Document(root=Element(
            name="html",
            namespace_uri="http://www.w3.org/1999/xhtml",
        ))])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert root.namespace_uri == "http://www.w3.org/1999/xhtml"

    def test_element_with_simple_attributes(self):
        original = Stream(documents=[Document(root=Element(
            name="div",
            attributes=[
                Attribute(name="class", value="main"),
                Attribute(name="id", value="content"),
            ],
        ))])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert len(root.attributes) == 2
        attr_names = {a.name for a in root.attributes}
        assert attr_names == {"class", "id"}

    def test_element_with_namespaced_attribute(self):
        original = Stream(documents=[Document(root=Element(
            name="a",
            attributes=[
                Attribute(
                    name="href",
                    value="doc.xml",
                    namespace_uri="http://www.w3.org/1999/xlink",
                    prefix="xlink",
                ),
            ],
        ))])
        restored = roundtrip(original)
        attr = restored.documents[0].root.attributes[0]
        assert attr.name == "href"
        assert attr.value == "doc.xml"
        assert attr.namespace_uri == "http://www.w3.org/1999/xlink"

    def test_element_with_text_children(self):
        original = Stream(documents=[Document(root=Element(
            name="p",
            children=[Scalar(ScalarType.STRING, "Hello world")],
        ))])
        restored = roundtrip(original)
        children = restored.documents[0].root.children
        assert len(children) == 1
        assert isinstance(children[0], Scalar)
        assert children[0].value == "Hello world"

    def test_element_with_child_elements(self):
        original = Stream(documents=[Document(root=Element(
            name="div",
            children=[
                Element(name="span"),
                Element(name="p"),
            ],
        ))])
        restored = roundtrip(original)
        children = restored.documents[0].root.children
        assert len(children) == 2
        assert children[0].name == "span"
        assert children[1].name == "p"

    def test_element_mixed_content(self):
        """Text interleaved with child elements — the XML mixed content test."""
        original = Stream(documents=[Document(root=Element(
            name="p",
            children=[
                Scalar(ScalarType.STRING, "Hello "),
                Element(name="b", children=[Scalar(ScalarType.STRING, "world")]),
                Scalar(ScalarType.STRING, "!"),
            ],
        ))])
        restored = roundtrip(original)
        children = restored.documents[0].root.children
        assert len(children) == 3
        assert isinstance(children[0], Scalar) and children[0].value == "Hello "
        assert isinstance(children[1], Element) and children[1].name == "b"
        assert isinstance(children[2], Scalar) and children[2].value == "!"
        # Inner element's text child
        assert children[1].children[0].value == "world"

    def test_element_with_namespace_declarations(self):
        ns = {"": "http://www.w3.org/1999/xhtml", "xs": "http://www.w3.org/2001/XMLSchema"}
        original = Stream(documents=[Document(root=Element(
            name="root",
            namespace_declarations=ns,
        ))])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert root.namespace_declarations[""] == "http://www.w3.org/1999/xhtml"
        assert root.namespace_declarations["xs"] == "http://www.w3.org/2001/XMLSchema"

    def test_element_with_comment_children(self):
        original = Stream(documents=[Document(root=Element(
            name="div",
            children=[
                Comment("a comment"),
                Element(name="p"),
            ],
        ))])
        restored = roundtrip(original)
        children = restored.documents[0].root.children
        assert isinstance(children[0], Comment)
        assert children[0].text == "a comment"

    def test_deeply_nested_elements(self):
        """5-level deep nesting."""
        leaf = Element(name="leaf", children=[Scalar(ScalarType.STRING, "deep")])
        node = leaf
        for name in ["d4", "d3", "d2", "d1", "root"]:
            node = Element(name=name, children=[node])
        original = Stream(documents=[Document(root=node)])
        restored = roundtrip(original)

        current = restored.documents[0].root
        for expected_name in ["root", "d1", "d2", "d3", "d4", "leaf"]:
            assert current.name == expected_name
            if expected_name != "leaf":
                current = current.children[0]
        assert current.children[0].value == "deep"


# ===================================================================
# ProcessingInstruction encoding
# ===================================================================

class TestProcessingInstructionEncoding:
    def test_pi_with_data(self):
        pi = ProcessingInstruction(target="xml-stylesheet", data='type="text/xsl"')
        original = Stream(documents=[Document(
            root=Element(name="root"),
            preamble=[pi],
        )])
        restored = roundtrip(original)
        pi_restored = restored.documents[0].preamble[0]
        assert isinstance(pi_restored, ProcessingInstruction)
        assert pi_restored.target == "xml-stylesheet"
        assert pi_restored.data == 'type="text/xsl"'

    def test_pi_without_data(self):
        pi = ProcessingInstruction(target="page-break")
        original = Stream(documents=[Document(
            root=Element(name="root"),
            preamble=[pi],
        )])
        restored = roundtrip(original)
        assert restored.documents[0].preamble[0].data is None


# ===================================================================
# Directive encoding
# ===================================================================

class TestDirectiveEncoding:
    def test_yaml_directive(self):
        d = Directive(name="YAML", parameters=["1.2"])
        original = Stream(documents=[Document(
            root=Map(),
            preamble=[d],
        )])
        restored = roundtrip(original)
        d_restored = restored.documents[0].preamble[0]
        assert isinstance(d_restored, Directive)
        assert d_restored.name == "YAML"
        assert d_restored.parameters == ["1.2"]

    def test_tag_directive(self):
        d = Directive(name="TAG", parameters=["!custom!", "tag:example.com,2024:"])
        original = Stream(documents=[Document(root=Map(), preamble=[d])])
        restored = roundtrip(original)
        d_restored = restored.documents[0].preamble[0]
        assert d_restored.parameters == ["!custom!", "tag:example.com,2024:"]

    def test_doctype_directive(self):
        d = Directive(name="DOCTYPE", parameters=["html", "SYSTEM", "about:legacy-compat"])
        original = Stream(documents=[Document(root=Element(name="html"), preamble=[d])])
        restored = roundtrip(original)
        assert restored.documents[0].preamble[0].name == "DOCTYPE"


# ===================================================================
# Anchor / Alias encoding
# ===================================================================

class TestAnchorAliasEncoding:
    def test_anchor_on_scalar(self):
        original = Stream(documents=[Document(root=Map(entries=[
            (
                Scalar(ScalarType.STRING, "value"),
                Scalar(ScalarType.INTEGER, 42, anchor=Anchor("myval")),
            ),
        ]))])
        restored = roundtrip(original)
        val = restored.documents[0].root.entries[0][1]
        assert val.anchor is not None
        assert val.anchor.name == "myval"
        assert val.value == 42

    def test_anchor_on_map(self):
        original = Stream(documents=[Document(root=Map(entries=[
            (
                Scalar(ScalarType.STRING, "defaults"),
                Map(
                    entries=[
                        (Scalar(ScalarType.STRING, "timeout"), Scalar(ScalarType.INTEGER, 30)),
                    ],
                    anchor=Anchor("defaults"),
                ),
            ),
        ]))])
        restored = roundtrip(original)
        defaults = restored.documents[0].root.entries[0][1]
        assert defaults.anchor.name == "defaults"
        assert isinstance(defaults, Map)

    def test_alias_reference(self):
        original = Stream(documents=[Document(root=Map(entries=[
            (
                Scalar(ScalarType.STRING, "source"),
                Scalar(ScalarType.INTEGER, 42, anchor=Anchor("val")),
            ),
            (
                Scalar(ScalarType.STRING, "ref"),
                Alias("val"),
            ),
        ]))])
        restored = roundtrip(original)
        ref = restored.documents[0].root.entries[1][1]
        assert isinstance(ref, Alias)
        assert ref.name == "val"

    def test_anchor_on_sequence(self):
        seq = Sequence(
            items=[Scalar(ScalarType.INTEGER, 1), Scalar(ScalarType.INTEGER, 2)],
            anchor=Anchor("myseq"),
        )
        original = Stream(documents=[Document(root=Map(entries=[
            (Scalar(ScalarType.STRING, "seq"), seq),
            (Scalar(ScalarType.STRING, "ref"), Alias("myseq")),
        ]))])
        restored = roundtrip(original)
        assert restored.documents[0].root.entries[0][1].anchor.name == "myseq"
        assert isinstance(restored.documents[0].root.entries[1][1], Alias)


# ===================================================================
# Tag annotation encoding
# ===================================================================

class TestTagAnnotationEncoding:
    def test_yaml_tag_on_scalar(self):
        original = Stream(documents=[Document(root=Scalar(
            ScalarType.STRING,
            "42",
            tag=TagAnnotation("tag:yaml.org,2002:int"),
        ))])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert root.tag is not None
        assert root.tag.uri == "tag:yaml.org,2002:int"
        assert root.value == "42"

    def test_custom_tag(self):
        original = Stream(documents=[Document(root=Scalar(
            ScalarType.STRING,
            "custom-value",
            tag=TagAnnotation("urn:example:mytype"),
        ))])
        restored = roundtrip(original)
        assert restored.documents[0].root.tag.uri == "urn:example:mytype"

    def test_merge_tag(self):
        original = Stream(documents=[Document(root=Scalar(
            ScalarType.STRING,
            "<<",
            tag=TagAnnotation("tag:yaml.org,2002:merge"),
        ))])
        restored = roundtrip(original)
        root = restored.documents[0].root
        assert root.tag.uri == "tag:yaml.org,2002:merge"
        assert root.value == "<<"


# ===================================================================
# Document encoding
# ===================================================================

class TestDocumentEncoding:
    def test_minimal_document(self):
        original = Stream(documents=[Document(root=Scalar(ScalarType.NULL, None))])
        restored = roundtrip(original)
        assert len(restored.documents) == 1
        assert restored.documents[0].root.scalar_type == ScalarType.NULL

    def test_document_source_format_hint(self):
        for fmt in SourceFormat:
            original = Stream(documents=[Document(
                root=Map(),
                source_format_hint=fmt,
            )])
            restored = roundtrip(original)
            assert restored.documents[0].source_format_hint == fmt

    def test_document_allows_cycles(self):
        original = Stream(documents=[Document(
            root=Map(),
            allows_cycles=True,
        )])
        restored = roundtrip(original)
        assert restored.documents[0].allows_cycles is True

    def test_document_with_preamble_and_postamble(self):
        original = Stream(documents=[Document(
            root=Map(),
            preamble=[Comment("header"), Directive(name="YAML", parameters=["1.2"])],
            postamble=[Comment("footer")],
        )])
        restored = roundtrip(original)
        doc = restored.documents[0]
        assert len(doc.preamble) == 2
        assert isinstance(doc.preamble[0], Comment)
        assert isinstance(doc.preamble[1], Directive)
        assert len(doc.postamble) == 1
        assert doc.postamble[0].text == "footer"


# ===================================================================
# Stream encoding
# ===================================================================

class TestStreamEncoding:
    def test_single_document_stream(self):
        original = Stream(documents=[
            Document(root=Scalar(ScalarType.STRING, "hello")),
        ])
        restored = roundtrip(original)
        assert len(restored.documents) == 1
        assert restored.documents[0].root.value == "hello"

    def test_multi_document_stream(self):
        original = Stream(documents=[
            Document(root=Scalar(ScalarType.INTEGER, 1)),
            Document(root=Scalar(ScalarType.INTEGER, 2)),
            Document(root=Scalar(ScalarType.INTEGER, 3)),
        ])
        restored = roundtrip(original)
        assert len(restored.documents) == 3
        values = [d.root.value for d in restored.documents]
        assert values == [1, 2, 3]

    def test_empty_stream(self):
        original = Stream(documents=[])
        restored = roundtrip(original)
        assert len(restored.documents) == 0
