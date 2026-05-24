"""Tests for CDXF core data model — the 9 node kinds and annotations."""

import pytest
from cdxf.model import (
    Stream,
    Document,
    Map,
    Sequence,
    Scalar,
    Element,
    Attribute,
    Comment,
    ProcessingInstruction,
    Directive,
    TagAnnotation,
    Anchor,
    Alias,
    ScalarType,
    SourceFormat,
)


# ---------------------------------------------------------------------------
# ScalarType enum
# ---------------------------------------------------------------------------

class TestScalarType:
    def test_all_types_exist(self):
        expected = {
            "NULL", "BOOLEAN", "INTEGER", "FLOAT", "DECIMAL",
            "STRING", "BYTE_STRING", "TIMESTAMP_OFFSET",
            "TIMESTAMP_LOCAL", "DATE", "TIME",
        }
        actual = {member.name for member in ScalarType}
        assert actual == expected

    def test_values_are_unique(self):
        values = [member.value for member in ScalarType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# SourceFormat enum
# ---------------------------------------------------------------------------

class TestSourceFormat:
    def test_all_formats_exist(self):
        expected = {"UNSPECIFIED", "JSON", "YAML", "XML", "TOML"}
        actual = {member.name for member in SourceFormat}
        assert actual == expected


# ---------------------------------------------------------------------------
# Scalar
# ---------------------------------------------------------------------------

class TestScalar:
    def test_null_scalar(self):
        s = Scalar(ScalarType.NULL, None)
        assert s.scalar_type == ScalarType.NULL
        assert s.value is None

    def test_boolean_scalar(self):
        s = Scalar(ScalarType.BOOLEAN, True)
        assert s.scalar_type == ScalarType.BOOLEAN
        assert s.value is True

    def test_integer_scalar(self):
        s = Scalar(ScalarType.INTEGER, 42)
        assert s.value == 42

    def test_large_integer_scalar(self):
        big = 2**128
        s = Scalar(ScalarType.INTEGER, big)
        assert s.value == big

    def test_negative_integer_scalar(self):
        s = Scalar(ScalarType.INTEGER, -999)
        assert s.value == -999

    def test_float_scalar(self):
        s = Scalar(ScalarType.FLOAT, 3.14)
        assert s.value == pytest.approx(3.14)

    def test_float_special_values(self):
        import math
        s_inf = Scalar(ScalarType.FLOAT, float("inf"))
        assert math.isinf(s_inf.value)

        s_nan = Scalar(ScalarType.FLOAT, float("nan"))
        assert math.isnan(s_nan.value)

    def test_string_scalar(self):
        s = Scalar(ScalarType.STRING, "hello world")
        assert s.value == "hello world"

    def test_empty_string_scalar(self):
        s = Scalar(ScalarType.STRING, "")
        assert s.value == ""

    def test_unicode_string_scalar(self):
        s = Scalar(ScalarType.STRING, "こんにちは世界 🌍")
        assert s.value == "こんにちは世界 🌍"

    def test_byte_string_scalar(self):
        data = b"\x00\x01\x02\xff"
        s = Scalar(ScalarType.BYTE_STRING, data)
        assert s.value == data

    def test_timestamp_offset_scalar(self):
        s = Scalar(ScalarType.TIMESTAMP_OFFSET, "2024-05-27T07:32:00-04:00")
        assert s.value == "2024-05-27T07:32:00-04:00"

    def test_timestamp_local_scalar(self):
        s = Scalar(ScalarType.TIMESTAMP_LOCAL, "2024-05-27T07:32:00")
        assert s.value == "2024-05-27T07:32:00"

    def test_date_scalar(self):
        s = Scalar(ScalarType.DATE, "2024-05-27")
        assert s.value == "2024-05-27"

    def test_time_scalar(self):
        s = Scalar(ScalarType.TIME, "07:32:00.999")
        assert s.value == "07:32:00.999"

    def test_scalar_default_annotations(self):
        s = Scalar(ScalarType.STRING, "test")
        assert s.tag is None
        assert s.anchor is None

    def test_scalar_with_tag(self):
        s = Scalar(ScalarType.STRING, "42", tag=TagAnnotation("tag:yaml.org,2002:int"))
        assert s.tag.uri == "tag:yaml.org,2002:int"

    def test_scalar_with_anchor(self):
        s = Scalar(ScalarType.INTEGER, 30, anchor=Anchor("defaults"))
        assert s.anchor.name == "defaults"


# ---------------------------------------------------------------------------
# Comment
# ---------------------------------------------------------------------------

class TestComment:
    def test_comment_text(self):
        c = Comment("this is a comment")
        assert c.text == "this is a comment"

    def test_empty_comment(self):
        c = Comment("")
        assert c.text == ""


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

class TestMap:
    def test_empty_map(self):
        m = Map()
        assert len(m.entries) == 0

    def test_map_with_string_key_entries(self):
        key = Scalar(ScalarType.STRING, "name")
        value = Scalar(ScalarType.STRING, "Alice")
        m = Map(entries=[(key, value)])
        assert len(m.entries) == 1
        assert m.entries[0] == (key, value)

    def test_map_with_multiple_entries(self):
        entries = [
            (Scalar(ScalarType.STRING, "a"), Scalar(ScalarType.INTEGER, 1)),
            (Scalar(ScalarType.STRING, "b"), Scalar(ScalarType.INTEGER, 2)),
            (Scalar(ScalarType.STRING, "c"), Scalar(ScalarType.INTEGER, 3)),
        ]
        m = Map(entries=entries)
        assert len(m.entries) == 3

    def test_map_preserves_insertion_order(self):
        entries = [
            (Scalar(ScalarType.STRING, "z"), Scalar(ScalarType.INTEGER, 1)),
            (Scalar(ScalarType.STRING, "a"), Scalar(ScalarType.INTEGER, 2)),
            (Scalar(ScalarType.STRING, "m"), Scalar(ScalarType.INTEGER, 3)),
        ]
        m = Map(entries=entries)
        keys = [e[0].value for e in m.entries]
        assert keys == ["z", "a", "m"]

    def test_map_with_interleaved_comments(self):
        entries = [
            (Scalar(ScalarType.STRING, "key1"), Scalar(ScalarType.INTEGER, 1)),
            Comment("a comment between entries"),
            (Scalar(ScalarType.STRING, "key2"), Scalar(ScalarType.INTEGER, 2)),
        ]
        m = Map(entries=entries)
        assert len(m.entries) == 3
        assert isinstance(m.entries[1], Comment)

    def test_map_with_non_string_key(self):
        """YAML allows non-string keys."""
        key = Scalar(ScalarType.INTEGER, 42)
        value = Scalar(ScalarType.STRING, "the answer")
        m = Map(entries=[(key, value)])
        assert m.entries[0][0].value == 42

    def test_map_with_anchor(self):
        m = Map(
            entries=[(Scalar(ScalarType.STRING, "x"), Scalar(ScalarType.INTEGER, 1))],
            anchor=Anchor("mymap"),
        )
        assert m.anchor.name == "mymap"


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

class TestSequence:
    def test_empty_sequence(self):
        s = Sequence()
        assert len(s.items) == 0

    def test_sequence_with_items(self):
        items = [
            Scalar(ScalarType.INTEGER, 1),
            Scalar(ScalarType.INTEGER, 2),
            Scalar(ScalarType.INTEGER, 3),
        ]
        s = Sequence(items=items)
        assert len(s.items) == 3
        assert s.items[0].value == 1

    def test_sequence_preserves_order(self):
        items = [
            Scalar(ScalarType.STRING, "c"),
            Scalar(ScalarType.STRING, "a"),
            Scalar(ScalarType.STRING, "b"),
        ]
        s = Sequence(items=items)
        values = [i.value for i in s.items]
        assert values == ["c", "a", "b"]

    def test_sequence_with_interleaved_comments(self):
        items = [
            Scalar(ScalarType.INTEGER, 1),
            Comment("between items"),
            Scalar(ScalarType.INTEGER, 2),
        ]
        s = Sequence(items=items)
        assert len(s.items) == 3
        assert isinstance(s.items[1], Comment)

    def test_nested_sequence(self):
        inner = Sequence(items=[Scalar(ScalarType.INTEGER, 1)])
        outer = Sequence(items=[inner])
        assert isinstance(outer.items[0], Sequence)

    def test_sequence_with_anchor(self):
        s = Sequence(
            items=[Scalar(ScalarType.INTEGER, 1)],
            anchor=Anchor("myseq"),
        )
        assert s.anchor.name == "myseq"


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------

class TestAlias:
    def test_alias_by_name(self):
        a = Alias("defaults")
        assert a.name == "defaults"

    def test_alias_equality(self):
        a1 = Alias("x")
        a2 = Alias("x")
        assert a1.name == a2.name


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------

class TestAnchor:
    def test_anchor_name(self):
        a = Anchor("myanchor")
        assert a.name == "myanchor"


# ---------------------------------------------------------------------------
# TagAnnotation
# ---------------------------------------------------------------------------

class TestTagAnnotation:
    def test_tag_uri(self):
        t = TagAnnotation("tag:yaml.org,2002:int")
        assert t.uri == "tag:yaml.org,2002:int"

    def test_custom_tag(self):
        t = TagAnnotation("urn:example:custom-type")
        assert t.uri == "urn:example:custom-type"


# ---------------------------------------------------------------------------
# Element (XML)
# ---------------------------------------------------------------------------

class TestElement:
    def test_simple_element(self):
        e = Element(name="div")
        assert e.name == "div"
        assert e.namespace_uri is None
        assert e.prefix is None
        assert len(e.attributes) == 0
        assert len(e.children) == 0

    def test_element_with_namespace(self):
        e = Element(
            name="html",
            namespace_uri="http://www.w3.org/1999/xhtml",
            prefix="xhtml",
        )
        assert e.namespace_uri == "http://www.w3.org/1999/xhtml"
        assert e.prefix == "xhtml"

    def test_element_with_attributes(self):
        attrs = [Attribute(name="class", value="main")]
        e = Element(name="div", attributes=attrs)
        assert len(e.attributes) == 1
        assert e.attributes[0].name == "class"
        assert e.attributes[0].value == "main"

    def test_element_with_children(self):
        child = Element(name="span")
        e = Element(name="div", children=[child])
        assert len(e.children) == 1
        assert isinstance(e.children[0], Element)

    def test_element_mixed_content(self):
        """XML mixed content: text interleaved with child elements."""
        children = [
            Scalar(ScalarType.STRING, "Hello "),
            Element(name="b", children=[Scalar(ScalarType.STRING, "world")]),
            Scalar(ScalarType.STRING, "!"),
        ]
        e = Element(name="p", children=children)
        assert len(e.children) == 3
        assert isinstance(e.children[0], Scalar)
        assert isinstance(e.children[1], Element)
        assert isinstance(e.children[2], Scalar)

    def test_element_with_namespace_declarations(self):
        ns = {"": "http://www.w3.org/1999/xhtml", "xs": "http://www.w3.org/2001/XMLSchema"}
        e = Element(name="root", namespace_declarations=ns)
        assert e.namespace_declarations[""] == "http://www.w3.org/1999/xhtml"
        assert e.namespace_declarations["xs"] == "http://www.w3.org/2001/XMLSchema"

    def test_element_with_comment_children(self):
        children = [
            Comment("a comment"),
            Element(name="child"),
        ]
        e = Element(name="parent", children=children)
        assert isinstance(e.children[0], Comment)


# ---------------------------------------------------------------------------
# Attribute (XML)
# ---------------------------------------------------------------------------

class TestAttribute:
    def test_simple_attribute(self):
        a = Attribute(name="id", value="main")
        assert a.name == "id"
        assert a.value == "main"
        assert a.namespace_uri is None

    def test_namespaced_attribute(self):
        a = Attribute(
            name="href",
            value="doc.xml",
            namespace_uri="http://www.w3.org/1999/xlink",
            prefix="xlink",
        )
        assert a.namespace_uri == "http://www.w3.org/1999/xlink"
        assert a.prefix == "xlink"


# ---------------------------------------------------------------------------
# ProcessingInstruction
# ---------------------------------------------------------------------------

class TestProcessingInstruction:
    def test_pi_with_data(self):
        pi = ProcessingInstruction(
            target="xml-stylesheet",
            data='type="text/xsl" href="style.xsl"',
        )
        assert pi.target == "xml-stylesheet"
        assert pi.data == 'type="text/xsl" href="style.xsl"'

    def test_pi_without_data(self):
        pi = ProcessingInstruction(target="page-break")
        assert pi.target == "page-break"
        assert pi.data is None


# ---------------------------------------------------------------------------
# Directive
# ---------------------------------------------------------------------------

class TestDirective:
    def test_yaml_directive(self):
        d = Directive(name="YAML", parameters=["1.2"])
        assert d.name == "YAML"
        assert d.parameters == ["1.2"]

    def test_tag_directive(self):
        d = Directive(name="TAG", parameters=["!custom!", "tag:example.com,2024:"])
        assert d.name == "TAG"
        assert len(d.parameters) == 2

    def test_doctype_directive(self):
        d = Directive(name="DOCTYPE", parameters=["html", "SYSTEM", "about:legacy-compat"])
        assert d.name == "DOCTYPE"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class TestDocument:
    def test_minimal_document(self):
        root = Scalar(ScalarType.NULL, None)
        doc = Document(root=root)
        assert doc.root is root
        assert doc.source_format_hint == SourceFormat.UNSPECIFIED
        assert doc.allows_cycles is False
        assert len(doc.preamble) == 0
        assert len(doc.postamble) == 0

    def test_document_with_source_hint(self):
        root = Map()
        doc = Document(root=root, source_format_hint=SourceFormat.JSON)
        assert doc.source_format_hint == SourceFormat.JSON

    def test_document_with_cycles_allowed(self):
        root = Map()
        doc = Document(root=root, allows_cycles=True)
        assert doc.allows_cycles is True

    def test_document_with_preamble(self):
        root = Map()
        preamble = [Comment("file header"), Directive(name="YAML", parameters=["1.2"])]
        doc = Document(root=root, preamble=preamble)
        assert len(doc.preamble) == 2
        assert isinstance(doc.preamble[0], Comment)
        assert isinstance(doc.preamble[1], Directive)

    def test_document_with_postamble(self):
        root = Map()
        postamble = [Comment("end of file")]
        doc = Document(root=root, postamble=postamble)
        assert len(doc.postamble) == 1


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

class TestStream:
    def test_empty_stream(self):
        s = Stream()
        assert len(s.documents) == 0

    def test_single_document_stream(self):
        doc = Document(root=Scalar(ScalarType.STRING, "hello"))
        s = Stream(documents=[doc])
        assert len(s.documents) == 1

    def test_multi_document_stream(self):
        docs = [
            Document(root=Scalar(ScalarType.INTEGER, 1)),
            Document(root=Scalar(ScalarType.INTEGER, 2)),
            Document(root=Scalar(ScalarType.INTEGER, 3)),
        ]
        s = Stream(documents=docs)
        assert len(s.documents) == 3

    def test_stream_preserves_document_order(self):
        docs = [
            Document(root=Scalar(ScalarType.STRING, "first")),
            Document(root=Scalar(ScalarType.STRING, "second")),
        ]
        s = Stream(documents=docs)
        assert s.documents[0].root.value == "first"
        assert s.documents[1].root.value == "second"


# ---------------------------------------------------------------------------
# Composition: realistic document structures
# ---------------------------------------------------------------------------

class TestComposition:
    def test_json_like_document(self):
        """A typical JSON document: map of scalars."""
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "name"), Scalar(ScalarType.STRING, "Alice")),
            (Scalar(ScalarType.STRING, "age"), Scalar(ScalarType.INTEGER, 30)),
            (Scalar(ScalarType.STRING, "active"), Scalar(ScalarType.BOOLEAN, True)),
        ])
        doc = Document(root=root, source_format_hint=SourceFormat.JSON)
        stream = Stream(documents=[doc])

        assert len(stream.documents) == 1
        assert isinstance(stream.documents[0].root, Map)
        assert len(stream.documents[0].root.entries) == 3

    def test_yaml_with_anchors_and_aliases(self):
        """YAML document with anchor/alias graph structure."""
        defaults = Map(
            entries=[
                (Scalar(ScalarType.STRING, "timeout"), Scalar(ScalarType.INTEGER, 30)),
                (Scalar(ScalarType.STRING, "retries"), Scalar(ScalarType.INTEGER, 3)),
            ],
            anchor=Anchor("defaults"),
        )
        production = Map(entries=[
            (
                Scalar(ScalarType.STRING, "<<", tag=TagAnnotation("tag:yaml.org,2002:merge")),
                Alias("defaults"),
            ),
            (Scalar(ScalarType.STRING, "timeout"), Scalar(ScalarType.INTEGER, 60)),
        ])
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "defaults"), defaults),
            (Scalar(ScalarType.STRING, "production"), production),
        ])
        doc = Document(root=root, source_format_hint=SourceFormat.YAML)
        stream = Stream(documents=[doc])

        # Verify anchor is on defaults map
        defaults_node = stream.documents[0].root.entries[0][1]
        assert defaults_node.anchor.name == "defaults"

        # Verify alias in production
        production_node = stream.documents[0].root.entries[1][1]
        merge_value = production_node.entries[0][1]
        assert isinstance(merge_value, Alias)
        assert merge_value.name == "defaults"

    def test_xml_mixed_content(self):
        """XML document with mixed content and namespaces."""
        p_element = Element(
            name="p",
            children=[
                Scalar(ScalarType.STRING, "Hello "),
                Element(name="b", children=[Scalar(ScalarType.STRING, "world")]),
                Scalar(ScalarType.STRING, "!"),
            ],
        )
        body = Element(
            name="body",
            attributes=[Attribute(name="class", value="main")],
            children=[p_element],
        )
        html = Element(
            name="html",
            namespace_uri="http://www.w3.org/1999/xhtml",
            children=[body],
            namespace_declarations={"": "http://www.w3.org/1999/xhtml"},
        )
        doc = Document(root=html, source_format_hint=SourceFormat.XML)
        stream = Stream(documents=[doc])

        root_elem = stream.documents[0].root
        assert isinstance(root_elem, Element)
        assert root_elem.name == "html"
        assert root_elem.namespace_uri == "http://www.w3.org/1999/xhtml"

    def test_toml_with_datetimes(self):
        """TOML document with typed datetime scalars."""
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "server"), Map(entries=[
                (Scalar(ScalarType.STRING, "host"), Scalar(ScalarType.STRING, "localhost")),
                (Scalar(ScalarType.STRING, "port"), Scalar(ScalarType.INTEGER, 8080)),
                (Scalar(ScalarType.STRING, "started"),
                 Scalar(ScalarType.TIMESTAMP_OFFSET, "2024-05-27T07:32:00-04:00")),
                (Scalar(ScalarType.STRING, "cert_expiry"),
                 Scalar(ScalarType.DATE, "2025-12-31")),
            ])),
        ])
        doc = Document(root=root, source_format_hint=SourceFormat.TOML)
        stream = Stream(documents=[doc])

        server_map = stream.documents[0].root.entries[0][1]
        assert isinstance(server_map, Map)
        cert_entry = server_map.entries[3]
        assert cert_entry[1].scalar_type == ScalarType.DATE
        assert cert_entry[1].value == "2025-12-31"

    def test_yaml_multi_document_stream(self):
        """YAML multi-document stream."""
        docs = [
            Document(
                root=Scalar(ScalarType.STRING, "doc1"),
                preamble=[Directive(name="YAML", parameters=["1.2"])],
            ),
            Document(root=Scalar(ScalarType.STRING, "doc2")),
            Document(root=Scalar(ScalarType.STRING, "doc3")),
        ]
        stream = Stream(documents=docs)
        assert len(stream.documents) == 3
        assert stream.documents[0].preamble[0].name == "YAML"
