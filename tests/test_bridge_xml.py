"""Tests for the XML bridge — from_xml() and to_xml().

These tests validate CDXF's ability to losslessly represent XML's
information model, including constructs no other binary format
preserves together: elements, attributes, namespaces, mixed content,
comments, and processing instructions.

TDD: This file is written BEFORE the implementation.
"""

import pytest

from cdxf.model import (
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
)

# The bridge module under test — does not exist yet.
from cdxf.bridges.xml_bridge import from_xml, to_xml


# ===================================================================
# from_xml: basic elements
# ===================================================================

class TestFromXmlBasicElements:
    """Verify that XML elements map to CDXF Element nodes."""

    def test_empty_element(self):
        stream = from_xml("<root/>")
        root = stream.documents[0].root
        assert isinstance(root, Element)
        assert root.name == "root"
        assert root.children == []

    def test_element_with_text(self):
        stream = from_xml("<greeting>hello</greeting>")
        root = stream.documents[0].root
        assert isinstance(root, Element)
        assert root.name == "greeting"
        assert len(root.children) == 1
        child = root.children[0]
        assert isinstance(child, Scalar)
        assert child.scalar_type == ScalarType.STRING
        assert child.value == "hello"

    def test_nested_elements(self):
        xml = "<parent><child>text</child></parent>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        assert root.name == "parent"
        assert len(root.children) == 1
        child = root.children[0]
        assert isinstance(child, Element)
        assert child.name == "child"
        assert child.children[0].value == "text"

    def test_multiple_children(self):
        xml = "<root><a>1</a><b>2</b><c>3</c></root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        elements = [c for c in root.children if isinstance(c, Element)]
        assert len(elements) == 3
        assert [e.name for e in elements] == ["a", "b", "c"]

    def test_deeply_nested(self):
        xml = "<a><b><c><d>deep</d></c></b></a>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        node = root
        for name in ["a", "b", "c", "d"]:
            assert node.name == name
            if name == "d":
                assert node.children[0].value == "deep"
            else:
                node = node.children[0]

    def test_source_format_hint(self):
        stream = from_xml("<root/>")
        assert stream.documents[0].source_format_hint == SourceFormat.XML

    def test_single_document_stream(self):
        stream = from_xml("<root/>")
        assert len(stream.documents) == 1


# ===================================================================
# from_xml: attributes
# ===================================================================

class TestFromXmlAttributes:
    """Verify that XML attributes map to CDXF Attribute nodes."""

    def test_single_attribute(self):
        stream = from_xml('<item id="42"/>')
        root = stream.documents[0].root
        assert len(root.attributes) == 1
        attr = root.attributes[0]
        assert isinstance(attr, Attribute)
        assert attr.name == "id"
        assert attr.value == "42"

    def test_multiple_attributes(self):
        stream = from_xml('<point x="1" y="2" z="3"/>')
        root = stream.documents[0].root
        assert len(root.attributes) == 3
        names = {a.name for a in root.attributes}
        assert names == {"x", "y", "z"}

    def test_attribute_values_are_strings(self):
        """XML attributes are always strings, even if they look numeric."""
        stream = from_xml('<item count="100" active="true"/>')
        root = stream.documents[0].root
        for attr in root.attributes:
            assert isinstance(attr.value, str)

    def test_attribute_with_special_chars(self):
        stream = from_xml('<msg text="a&amp;b&lt;c"/>')
        root = stream.documents[0].root
        attr = root.attributes[0]
        assert attr.value == "a&b<c"

    def test_element_with_attributes_and_children(self):
        xml = '<div class="main"><p>hello</p></div>'
        stream = from_xml(xml)
        root = stream.documents[0].root
        assert root.name == "div"
        assert len(root.attributes) == 1
        assert root.attributes[0].value == "main"
        assert len(root.children) == 1
        assert root.children[0].name == "p"


# ===================================================================
# from_xml: mixed content
# ===================================================================

class TestFromXmlMixedContent:
    """Verify that XML mixed content is preserved as interleaved
    Scalar(STRING) and Element children — a key differentiator."""

    def test_text_and_element_interleaved(self):
        xml = "<p>Hello <b>world</b>!</p>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        assert root.name == "p"
        # Expect: Scalar("Hello "), Element("b"), Scalar("!")
        assert len(root.children) == 3
        assert isinstance(root.children[0], Scalar)
        assert root.children[0].value == "Hello "
        assert isinstance(root.children[1], Element)
        assert root.children[1].name == "b"
        assert isinstance(root.children[2], Scalar)
        assert root.children[2].value == "!"

    def test_multiple_text_segments(self):
        xml = "<p>a<em>b</em>c<em>d</em>e</p>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        # a, em(b), c, em(d), e
        assert len(root.children) == 5
        texts = [c.value for c in root.children if isinstance(c, Scalar)]
        assert texts == ["a", "c", "e"]

    def test_tail_text_after_child(self):
        """Text after a closing tag but before the parent's close."""
        xml = "<root><child/>tail</root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        assert len(root.children) == 2
        assert isinstance(root.children[0], Element)
        assert isinstance(root.children[1], Scalar)
        assert root.children[1].value == "tail"


# ===================================================================
# from_xml: namespaces
# ===================================================================

class TestFromXmlNamespaces:
    """Verify namespace handling: URIs, prefixes, declarations."""

    def test_default_namespace(self):
        xml = '<root xmlns="http://example.com/ns"/>'
        stream = from_xml(xml)
        root = stream.documents[0].root
        assert root.namespace_uri == "http://example.com/ns"

    def test_prefixed_namespace(self):
        xml = '<ns:root xmlns:ns="http://example.com/ns"/>'
        stream = from_xml(xml)
        root = stream.documents[0].root
        assert root.namespace_uri == "http://example.com/ns"
        assert root.prefix == "ns"
        assert root.name == "root"

    def test_namespace_declarations_recorded(self):
        xml = '<root xmlns="http://example.com/default" xmlns:x="http://example.com/x"/>'
        stream = from_xml(xml)
        root = stream.documents[0].root
        # namespace_declarations should contain the bindings
        assert "" in root.namespace_declarations or "x" in root.namespace_declarations

    def test_child_inherits_namespace(self):
        xml = '<root xmlns="http://example.com/ns"><child/></root>'
        stream = from_xml(xml)
        root = stream.documents[0].root
        child = root.children[0]
        assert child.namespace_uri == "http://example.com/ns"

    def test_attribute_with_namespace(self):
        xml = '<root xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="http://example.com"/>'
        stream = from_xml(xml)
        root = stream.documents[0].root
        attr = root.attributes[0]
        assert attr.name == "href"
        assert attr.namespace_uri == "http://www.w3.org/1999/xlink"
        assert attr.prefix == "xlink"

    def test_multiple_namespace_prefixes(self):
        xml = (
            '<root xmlns:a="http://a.example.com" xmlns:b="http://b.example.com">'
            '<a:x/><b:y/>'
            '</root>'
        )
        stream = from_xml(xml)
        root = stream.documents[0].root
        elements = [c for c in root.children if isinstance(c, Element)]
        assert elements[0].namespace_uri == "http://a.example.com"
        assert elements[0].prefix == "a"
        assert elements[1].namespace_uri == "http://b.example.com"
        assert elements[1].prefix == "b"


# ===================================================================
# from_xml: comments
# ===================================================================

class TestFromXmlComments:
    """Verify comment preservation — a key CDXF differentiator.

    Note: stdlib ElementTree drops comments. Our expat-based parser
    must capture them.
    """

    def test_comment_in_element(self):
        xml = "<root><!-- hello --></root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        comments = [c for c in root.children if isinstance(c, Comment)]
        assert len(comments) == 1
        assert comments[0].text.strip() == "hello"

    def test_comment_between_elements(self):
        xml = "<root><a/><!-- between --><b/></root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        # Order must be: Element(a), Comment, Element(b)
        assert len(root.children) == 3
        assert isinstance(root.children[0], Element)
        assert isinstance(root.children[1], Comment)
        assert isinstance(root.children[2], Element)

    def test_comment_before_root(self):
        """Comments before the root element go in Document preamble."""
        xml = "<!-- preamble comment --><root/>"
        stream = from_xml(xml)
        doc = stream.documents[0]
        preamble_comments = [n for n in doc.preamble if isinstance(n, Comment)]
        assert len(preamble_comments) >= 1
        assert "preamble" in preamble_comments[0].text

    def test_comment_after_root(self):
        """Comments after the root element go in Document postamble."""
        xml = "<root/><!-- postamble comment -->"
        stream = from_xml(xml)
        doc = stream.documents[0]
        postamble_comments = [n for n in doc.postamble if isinstance(n, Comment)]
        assert len(postamble_comments) >= 1
        assert "postamble" in postamble_comments[0].text

    def test_multiple_comments(self):
        xml = "<root><!-- first --><!-- second --></root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        comments = [c for c in root.children if isinstance(c, Comment)]
        assert len(comments) == 2

    def test_comment_with_mixed_content(self):
        xml = "<p>text<!-- comment --><b>bold</b></p>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        # text, comment, bold
        assert len(root.children) == 3
        assert isinstance(root.children[0], Scalar)
        assert isinstance(root.children[1], Comment)
        assert isinstance(root.children[2], Element)


# ===================================================================
# from_xml: processing instructions
# ===================================================================

class TestFromXmlProcessingInstructions:
    """Verify PI preservation."""

    def test_pi_in_element(self):
        xml = "<root><?myapp do-something?></root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        pis = [c for c in root.children if isinstance(c, ProcessingInstruction)]
        assert len(pis) == 1
        assert pis[0].target == "myapp"
        assert pis[0].data == "do-something"

    def test_pi_before_root(self):
        xml = "<?xml-stylesheet type='text/xsl' href='style.xsl'?><root/>"
        stream = from_xml(xml)
        doc = stream.documents[0]
        pis = [n for n in doc.preamble if isinstance(n, ProcessingInstruction)]
        assert len(pis) >= 1
        assert pis[0].target == "xml-stylesheet"

    def test_pi_with_no_data(self):
        xml = "<root><?target?></root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        pis = [c for c in root.children if isinstance(c, ProcessingInstruction)]
        assert len(pis) == 1
        assert pis[0].target == "target"
        assert pis[0].data is None or pis[0].data == ""


# ===================================================================
# from_xml: XML declaration and DOCTYPE
# ===================================================================

class TestFromXmlDeclaration:
    """Verify XML declaration and DOCTYPE handling."""

    def test_xml_declaration_ignored(self):
        """The <?xml ...?> declaration is not a PI per XML spec."""
        xml = '<?xml version="1.0" encoding="UTF-8"?><root/>'
        stream = from_xml(xml)
        root = stream.documents[0].root
        assert root.name == "root"
        # XML declaration should NOT appear as a PI
        pis = [n for n in stream.documents[0].preamble
               if isinstance(n, ProcessingInstruction) and n.target == "xml"]
        assert len(pis) == 0


# ===================================================================
# from_xml: CDATA sections
# ===================================================================

class TestFromXmlCdata:
    """CDATA sections should be treated as text content."""

    def test_cdata_becomes_text(self):
        xml = "<root><![CDATA[<not>&xml</not>]]></root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        text_children = [c for c in root.children if isinstance(c, Scalar)]
        assert len(text_children) == 1
        assert text_children[0].value == "<not>&xml</not>"

    def test_cdata_adjacent_to_text(self):
        """CDATA next to regular text should merge or preserve order."""
        xml = "<root>before<![CDATA[ middle ]]>after</root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        # All text content should be present (possibly merged into one scalar
        # or as separate scalars — both are acceptable)
        all_text = "".join(
            c.value for c in root.children if isinstance(c, Scalar)
        )
        assert "before" in all_text
        assert " middle " in all_text
        assert "after" in all_text


# ===================================================================
# from_xml: empty and whitespace-only text
# ===================================================================

class TestFromXmlWhitespace:
    """Verify whitespace handling in element content."""

    def test_whitespace_only_text_preserved(self):
        """Whitespace-only text nodes between elements should be preserved."""
        xml = "<root>\n  <child/>\n</root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        # At minimum the child element must be present
        elements = [c for c in root.children if isinstance(c, Element)]
        assert len(elements) == 1

    def test_significant_whitespace_in_text(self):
        xml = "<pre>  indented  </pre>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        text = root.children[0]
        assert text.value == "  indented  "


# ===================================================================
# from_xml: entity references
# ===================================================================

class TestFromXmlEntities:
    """Entity references are expanded per the information model spec."""

    def test_predefined_entities(self):
        xml = "<root>&amp;&lt;&gt;&apos;&quot;</root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        text = root.children[0].value
        assert text == "&<>'\""

    def test_numeric_character_reference(self):
        xml = "<root>&#65;&#x42;</root>"
        stream = from_xml(xml)
        root = stream.documents[0].root
        text = root.children[0].value
        assert text == "AB"


# ===================================================================
# to_xml: CDXF model → XML text
# ===================================================================

class TestToXml:
    """Verify conversion from CDXF model to XML text."""

    def test_simple_element(self):
        root = Element(name="root")
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert "<root" in text

    def test_element_with_text(self):
        root = Element(
            name="msg",
            children=[Scalar(ScalarType.STRING, "hello")]
        )
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert "<msg>" in text
        assert "hello" in text
        assert "</msg>" in text

    def test_element_with_attributes(self):
        root = Element(
            name="item",
            attributes=[
                Attribute(name="id", value="42"),
                Attribute(name="type", value="book"),
            ]
        )
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert 'id="42"' in text
        assert 'type="book"' in text

    def test_nested_elements(self):
        child = Element(name="child", children=[
            Scalar(ScalarType.STRING, "text")
        ])
        root = Element(name="parent", children=[child])
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert "<parent>" in text
        assert "<child>" in text
        assert "text" in text

    def test_mixed_content_output(self):
        root = Element(name="p", children=[
            Scalar(ScalarType.STRING, "Hello "),
            Element(name="b", children=[Scalar(ScalarType.STRING, "world")]),
            Scalar(ScalarType.STRING, "!"),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert "Hello " in text
        assert "<b>world</b>" in text
        assert "!" in text

    def test_comment_output(self):
        root = Element(name="root", children=[
            Comment("a comment"),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert "<!--" in text
        assert "a comment" in text
        assert "-->" in text

    def test_pi_output(self):
        root = Element(name="root", children=[
            ProcessingInstruction(target="app", data="action"),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert "<?app" in text
        assert "action" in text
        assert "?>" in text

    def test_namespace_output(self):
        root = Element(
            name="root",
            namespace_uri="http://example.com/ns",
            namespace_declarations={"": "http://example.com/ns"},
        )
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert 'xmlns="http://example.com/ns"' in text

    def test_prefixed_namespace_output(self):
        root = Element(
            name="root",
            namespace_uri="http://example.com/ns",
            prefix="ex",
            namespace_declarations={"ex": "http://example.com/ns"},
        )
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert "ex:root" in text
        assert 'xmlns:ex="http://example.com/ns"' in text

    def test_attribute_namespace_output(self):
        root = Element(
            name="root",
            attributes=[
                Attribute(
                    name="href",
                    value="http://example.com",
                    namespace_uri="http://www.w3.org/1999/xlink",
                    prefix="xlink",
                ),
            ],
            namespace_declarations={
                "xlink": "http://www.w3.org/1999/xlink",
            },
        )
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert 'xlink:href="http://example.com"' in text

    def test_empty_stream(self):
        stream = Stream(documents=[])
        text = to_xml(stream)
        assert text == ""

    def test_special_chars_escaped(self):
        root = Element(
            name="root",
            children=[Scalar(ScalarType.STRING, "a<b&c")]
        )
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert "&lt;" in text
        assert "&amp;" in text

    def test_attribute_special_chars_escaped(self):
        root = Element(
            name="root",
            attributes=[Attribute(name="val", value='a"b')]
        )
        stream = Stream(documents=[Document(root=root)])
        text = to_xml(stream)
        assert "&quot;" in text or "'" in text  # either quoting strategy ok


# ===================================================================
# Round-trip: XML → CDXF → XML (semantic equivalence)
# ===================================================================

class TestXmlRoundTrip:
    """Verify that XML → CDXF → XML preserves the essential structure."""

    def test_simple_roundtrip(self):
        xml = "<root><child>text</child></root>"
        stream = from_xml(xml)
        output = to_xml(stream)
        restored = from_xml(output)
        root = restored.documents[0].root
        assert root.name == "root"
        assert root.children[0].name == "child"
        assert root.children[0].children[0].value == "text"

    def test_attributes_roundtrip(self):
        xml = '<item id="42" type="book"/>'
        stream = from_xml(xml)
        output = to_xml(stream)
        restored = from_xml(output)
        root = restored.documents[0].root
        attr_map = {a.name: a.value for a in root.attributes}
        assert attr_map["id"] == "42"
        assert attr_map["type"] == "book"

    def test_mixed_content_roundtrip(self):
        xml = "<p>Hello <b>world</b>!</p>"
        stream = from_xml(xml)
        output = to_xml(stream)
        restored = from_xml(output)
        root = restored.documents[0].root
        assert len(root.children) == 3
        assert root.children[0].value == "Hello "
        assert root.children[1].name == "b"
        assert root.children[2].value == "!"

    def test_namespace_roundtrip(self):
        xml = '<root xmlns="http://example.com/ns"><child/></root>'
        stream = from_xml(xml)
        output = to_xml(stream)
        restored = from_xml(output)
        root = restored.documents[0].root
        assert root.namespace_uri == "http://example.com/ns"

    def test_comment_roundtrip(self):
        xml = "<root><!-- important --><child/></root>"
        stream = from_xml(xml)
        output = to_xml(stream)
        restored = from_xml(output)
        root = restored.documents[0].root
        comments = [c for c in root.children if isinstance(c, Comment)]
        assert len(comments) >= 1
        assert "important" in comments[0].text

    def test_pi_roundtrip(self):
        xml = "<root><?app action?></root>"
        stream = from_xml(xml)
        output = to_xml(stream)
        restored = from_xml(output)
        root = restored.documents[0].root
        pis = [c for c in root.children if isinstance(c, ProcessingInstruction)]
        assert len(pis) >= 1
        assert pis[0].target == "app"


# ===================================================================
# Full pipeline: XML → CDXF model → CBOR → CDXF model → XML
# ===================================================================

class TestXmlFullPipeline:
    """End-to-end: XML text → CDXF → binary → CDXF → XML text."""

    def test_pipeline_simple_element(self):
        from cdxf.codec import encode, decode

        xml = "<root><child>hello</child></root>"
        stream = from_xml(xml)
        binary = encode(stream)
        restored = decode(binary)
        output = to_xml(restored)
        final = from_xml(output)

        root = final.documents[0].root
        assert root.name == "root"
        assert root.children[0].name == "child"
        assert root.children[0].children[0].value == "hello"

    def test_pipeline_with_attributes(self):
        from cdxf.codec import encode, decode

        xml = '<item id="42" type="book">content</item>'
        stream = from_xml(xml)
        binary = encode(stream)
        restored = decode(binary)
        output = to_xml(restored)
        final = from_xml(output)

        root = final.documents[0].root
        attr_map = {a.name: a.value for a in root.attributes}
        assert attr_map["id"] == "42"
        assert root.children[0].value == "content"

    def test_pipeline_with_namespaces(self):
        from cdxf.codec import encode, decode

        xml = '<root xmlns="http://example.com/ns"><child/></root>'
        stream = from_xml(xml)
        binary = encode(stream)
        restored = decode(binary)
        output = to_xml(restored)
        final = from_xml(output)

        root = final.documents[0].root
        assert root.namespace_uri == "http://example.com/ns"

    def test_pipeline_with_mixed_content(self):
        from cdxf.codec import encode, decode

        xml = "<p>Hello <b>world</b>!</p>"
        stream = from_xml(xml)
        binary = encode(stream)
        restored = decode(binary)
        output = to_xml(restored)
        final = from_xml(output)

        root = final.documents[0].root
        assert root.name == "p"
        texts = [c.value for c in root.children if isinstance(c, Scalar)]
        assert "Hello " in texts
        assert "!" in texts

    def test_pipeline_with_comments(self):
        from cdxf.codec import encode, decode

        xml = "<root><!-- preserved --><child/></root>"
        stream = from_xml(xml)
        binary = encode(stream)
        restored = decode(binary)
        output = to_xml(restored)
        final = from_xml(output)

        root = final.documents[0].root
        comments = [c for c in root.children if isinstance(c, Comment)]
        assert len(comments) >= 1
        assert "preserved" in comments[0].text

    def test_pipeline_with_pi(self):
        from cdxf.codec import encode, decode

        xml = "<root><?myapp action?></root>"
        stream = from_xml(xml)
        binary = encode(stream)
        restored = decode(binary)
        output = to_xml(restored)
        final = from_xml(output)

        root = final.documents[0].root
        pis = [c for c in root.children if isinstance(c, ProcessingInstruction)]
        assert len(pis) >= 1
        assert pis[0].target == "myapp"

    def test_pipeline_complex_document(self):
        """A realistic XML document through the full pipeline."""
        from cdxf.codec import encode, decode

        xml = (
            '<catalog xmlns="http://example.com/catalog">'
            '<!-- Book listing -->'
            '<book id="1">'
            '<title>The Art of Code</title>'
            '<author>Jane Doe</author>'
            '</book>'
            '<book id="2">'
            '<title>Data Structures</title>'
            '<author>John Smith</author>'
            '</book>'
            '</catalog>'
        )
        stream = from_xml(xml)
        binary = encode(stream)
        restored = decode(binary)
        output = to_xml(restored)
        final = from_xml(output)

        root = final.documents[0].root
        assert root.name == "catalog"
        assert root.namespace_uri == "http://example.com/catalog"

        # Find book elements (skip comments)
        books = [c for c in root.children if isinstance(c, Element)]
        assert len(books) == 2

        # Check first book
        book1 = books[0]
        assert any(a.value == "1" for a in book1.attributes)
