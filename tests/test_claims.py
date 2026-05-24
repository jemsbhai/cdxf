"""Tests backing CDXF's publishable claims.

Each test class corresponds to a specific claim made in the paper.
Failure of any test in this file means a paper claim is unsupported.
"""

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
# CLAIM 1: Lossless round-trip for JSON
# ===================================================================

class TestClaimJsonRoundTrip:
    """Every JSON construct survives CDXF encode/decode with exact fidelity."""

    def test_empty_object(self):
        original = Stream(documents=[Document(
            root=Map(), source_format_hint=SourceFormat.JSON,
        )])
        restored = decode(encode(original))
        assert isinstance(restored.documents[0].root, Map)
        assert len(restored.documents[0].root.entries) == 0

    def test_empty_array(self):
        original = Stream(documents=[Document(
            root=Sequence(), source_format_hint=SourceFormat.JSON,
        )])
        restored = decode(encode(original))
        assert isinstance(restored.documents[0].root, Sequence)
        assert len(restored.documents[0].root.items) == 0

    def test_nested_objects_and_arrays(self):
        """Complex nested JSON structure."""
        inner_arr = Sequence(items=[
            Scalar(ScalarType.INTEGER, 1),
            Scalar(ScalarType.STRING, "two"),
            Scalar(ScalarType.NULL, None),
        ])
        inner_obj = Map(entries=[
            (Scalar(ScalarType.STRING, "nested"), Scalar(ScalarType.BOOLEAN, True)),
        ])
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "array"), inner_arr),
            (Scalar(ScalarType.STRING, "object"), inner_obj),
            (Scalar(ScalarType.STRING, "number"), Scalar(ScalarType.FLOAT, 3.14)),
            (Scalar(ScalarType.STRING, "null"), Scalar(ScalarType.NULL, None)),
        ])
        original = Stream(documents=[Document(root=root, source_format_hint=SourceFormat.JSON)])
        restored = decode(encode(original))

        r = restored.documents[0].root
        assert len(r.entries) == 4

        arr = r.entries[0][1]
        assert isinstance(arr, Sequence)
        assert len(arr.items) == 3
        assert arr.items[2].scalar_type == ScalarType.NULL

        obj = r.entries[1][1]
        assert isinstance(obj, Map)
        assert obj.entries[0][1].value is True

    def test_all_json_scalar_types(self):
        """JSON has: string, number (int/float), boolean, null."""
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "str"), Scalar(ScalarType.STRING, "hello")),
            (Scalar(ScalarType.STRING, "int"), Scalar(ScalarType.INTEGER, 42)),
            (Scalar(ScalarType.STRING, "float"), Scalar(ScalarType.FLOAT, 2.718)),
            (Scalar(ScalarType.STRING, "bool_t"), Scalar(ScalarType.BOOLEAN, True)),
            (Scalar(ScalarType.STRING, "bool_f"), Scalar(ScalarType.BOOLEAN, False)),
            (Scalar(ScalarType.STRING, "null"), Scalar(ScalarType.NULL, None)),
        ])
        original = Stream(documents=[Document(root=root, source_format_hint=SourceFormat.JSON)])
        restored = decode(encode(original))
        entries = {e[0].value: e[1] for e in restored.documents[0].root.entries}

        assert entries["str"].value == "hello"
        assert entries["int"].value == 42
        assert entries["float"].value == pytest.approx(2.718)
        assert entries["bool_t"].value is True
        assert entries["bool_f"].value is False
        assert entries["null"].value is None


# ===================================================================
# CLAIM 2: Zero overhead for JSON-model data
# ===================================================================

class TestClaimZeroOverhead:
    """CDXF-encoded JSON-model data in shorthand mode must be
    byte-identical to standard CBOR."""

    def test_simple_map_shorthand(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "key"), Scalar(ScalarType.STRING, "value")),
        ])
        # Shorthand encoding: no Stream/Document wrappers
        encoder = Encoder(shorthand=True)
        cdxf_bytes = encoder.encode(
            Stream(documents=[Document(root=root, source_format_hint=SourceFormat.JSON)])
        )
        # Pure CBOR encoding of the same map
        cbor_bytes = cbor2.dumps({"key": "value"})
        assert cdxf_bytes == cbor_bytes

    def test_nested_structure_shorthand(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "a"), Scalar(ScalarType.INTEGER, 1)),
            (Scalar(ScalarType.STRING, "b"), Sequence(items=[
                Scalar(ScalarType.INTEGER, 2),
                Scalar(ScalarType.INTEGER, 3),
            ])),
        ])
        encoder = Encoder(shorthand=True)
        cdxf_bytes = encoder.encode(
            Stream(documents=[Document(root=root, source_format_hint=SourceFormat.JSON)])
        )
        cbor_bytes = cbor2.dumps({"a": 1, "b": [2, 3]})
        assert cdxf_bytes == cbor_bytes

    def test_scalar_array_shorthand(self):
        root = Sequence(items=[
            Scalar(ScalarType.INTEGER, 1),
            Scalar(ScalarType.STRING, "two"),
            Scalar(ScalarType.BOOLEAN, True),
            Scalar(ScalarType.NULL, None),
        ])
        encoder = Encoder(shorthand=True)
        cdxf_bytes = encoder.encode(
            Stream(documents=[Document(root=root, source_format_hint=SourceFormat.JSON)])
        )
        cbor_bytes = cbor2.dumps([1, "two", True, None])
        assert cdxf_bytes == cbor_bytes


# ===================================================================
# CLAIM 3: YAML DAG preservation (anchors & aliases)
# ===================================================================

class TestClaimYamlDagPreservation:
    """YAML anchors and aliases must survive round-trip as graph
    structure, NOT as duplicated subtrees. This is CDXF's key
    differentiator versus CBOR, MessagePack, and Ion."""

    def test_anchor_alias_preserved_not_duplicated(self):
        """The alias must remain an Alias node, not a copy of the
        anchored subtree."""
        defaults = Map(
            entries=[
                (Scalar(ScalarType.STRING, "timeout"), Scalar(ScalarType.INTEGER, 30)),
                (Scalar(ScalarType.STRING, "retries"), Scalar(ScalarType.INTEGER, 3)),
            ],
            anchor=Anchor("defaults"),
        )
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "defaults"), defaults),
            (Scalar(ScalarType.STRING, "production"), Map(entries=[
                (Scalar(ScalarType.STRING, "ref"), Alias("defaults")),
            ])),
        ])
        original = Stream(documents=[Document(root=root, source_format_hint=SourceFormat.YAML)])
        restored = decode(encode(original))

        # The anchored node must have its anchor
        defaults_restored = restored.documents[0].root.entries[0][1]
        assert defaults_restored.anchor is not None
        assert defaults_restored.anchor.name == "defaults"

        # The alias must still be an Alias, not a deep copy
        ref = restored.documents[0].root.entries[1][1].entries[0][1]
        assert isinstance(ref, Alias), (
            "Alias was expanded into a copy — DAG structure lost. "
            "This violates CDXF's core differentiator vs CBOR."
        )
        assert ref.name == "defaults"

    def test_multiple_aliases_to_same_anchor(self):
        """Multiple aliases referencing the same anchor."""
        shared = Scalar(ScalarType.STRING, "shared-value", anchor=Anchor("shared"))
        root = Sequence(items=[
            shared,
            Alias("shared"),
            Alias("shared"),
        ])
        original = Stream(documents=[Document(root=root, source_format_hint=SourceFormat.YAML)])
        restored = decode(encode(original))

        items = restored.documents[0].root.items
        assert items[0].anchor.name == "shared"
        assert isinstance(items[1], Alias) and items[1].name == "shared"
        assert isinstance(items[2], Alias) and items[2].name == "shared"

    def test_nested_anchors(self):
        """Anchors at multiple levels of nesting."""
        inner = Map(
            entries=[(Scalar(ScalarType.STRING, "x"), Scalar(ScalarType.INTEGER, 1))],
            anchor=Anchor("inner"),
        )
        outer = Map(
            entries=[(Scalar(ScalarType.STRING, "child"), inner)],
            anchor=Anchor("outer"),
        )
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "data"), outer),
            (Scalar(ScalarType.STRING, "ref_outer"), Alias("outer")),
            (Scalar(ScalarType.STRING, "ref_inner"), Alias("inner")),
        ])
        original = Stream(documents=[Document(root=root)])
        restored = decode(encode(original))

        entries = restored.documents[0].root.entries
        assert entries[0][1].anchor.name == "outer"
        assert entries[0][1].entries[0][1].anchor.name == "inner"
        assert isinstance(entries[1][1], Alias) and entries[1][1].name == "outer"
        assert isinstance(entries[2][1], Alias) and entries[2][1].name == "inner"


# ===================================================================
# CLAIM 4: XML attribute-vs-element distinction preserved
# ===================================================================

class TestClaimXmlAttributeElementDistinction:
    """Attributes and child elements must remain structurally distinct
    after round-trip. This is what CBOR and Ion lose."""

    def test_attribute_not_confused_with_child(self):
        """An element with both an attribute and a child element of the
        same name must preserve both distinctly."""
        original = Stream(documents=[Document(root=Element(
            name="item",
            attributes=[Attribute(name="type", value="book")],
            children=[Element(
                name="type",
                children=[Scalar(ScalarType.STRING, "hardcover")],
            )],
        ), source_format_hint=SourceFormat.XML)])
        restored = decode(encode(original))

        root = restored.documents[0].root
        # Attribute "type" = "book"
        assert len(root.attributes) == 1
        assert root.attributes[0].name == "type"
        assert root.attributes[0].value == "book"
        # Child element "type" containing "hardcover"
        assert len(root.children) == 1
        assert root.children[0].name == "type"
        assert root.children[0].children[0].value == "hardcover"

    def test_attribute_order_independent(self):
        """Attributes are unordered; their presence matters, not position."""
        original = Stream(documents=[Document(root=Element(
            name="div",
            attributes=[
                Attribute(name="class", value="main"),
                Attribute(name="id", value="root"),
                Attribute(name="style", value="color:red"),
            ],
        ))])
        restored = decode(encode(original))
        attr_map = {a.name: a.value for a in restored.documents[0].root.attributes}
        assert attr_map == {"class": "main", "id": "root", "style": "color:red"}


# ===================================================================
# CLAIM 5: XML mixed content preservation
# ===================================================================

class TestClaimXmlMixedContent:
    """Text interleaved with child elements must preserve order and
    content exactly."""

    def test_text_element_text_pattern(self):
        """The classic <p>Hello <b>world</b>!</p> pattern."""
        original = Stream(documents=[Document(root=Element(
            name="p",
            children=[
                Scalar(ScalarType.STRING, "Hello "),
                Element(name="b", children=[Scalar(ScalarType.STRING, "world")]),
                Scalar(ScalarType.STRING, "!"),
            ],
        ), source_format_hint=SourceFormat.XML)])
        restored = decode(encode(original))

        children = restored.documents[0].root.children
        assert len(children) == 3
        assert isinstance(children[0], Scalar) and children[0].value == "Hello "
        assert isinstance(children[1], Element) and children[1].name == "b"
        assert isinstance(children[2], Scalar) and children[2].value == "!"

    def test_complex_mixed_content(self):
        """Multiple elements and text nodes interleaved."""
        original = Stream(documents=[Document(root=Element(
            name="p",
            children=[
                Scalar(ScalarType.STRING, "Start "),
                Element(name="em", children=[Scalar(ScalarType.STRING, "emphasized")]),
                Scalar(ScalarType.STRING, " middle "),
                Element(name="a", attributes=[Attribute(name="href", value="http://example.com")],
                        children=[Scalar(ScalarType.STRING, "link")]),
                Scalar(ScalarType.STRING, " end."),
            ],
        ))])
        restored = decode(encode(original))

        children = restored.documents[0].root.children
        assert len(children) == 5
        assert children[0].value == "Start "
        assert children[1].name == "em"
        assert children[2].value == " middle "
        assert children[3].name == "a"
        assert children[3].attributes[0].value == "http://example.com"
        assert children[4].value == " end."


# ===================================================================
# CLAIM 6: Comment preservation with correct positioning
# ===================================================================

class TestClaimCommentPreservation:
    """Comments must survive round-trip with text and relative position
    preserved. JSON has no comments; YAML, XML, and TOML do."""

    def test_yaml_style_comments_in_map(self):
        root = Map(entries=[
            Comment("Server configuration"),
            (Scalar(ScalarType.STRING, "host"), Scalar(ScalarType.STRING, "localhost")),
            Comment("Default port"),
            (Scalar(ScalarType.STRING, "port"), Scalar(ScalarType.INTEGER, 8080)),
        ])
        original = Stream(documents=[Document(root=root, source_format_hint=SourceFormat.YAML)])
        restored = decode(encode(original))

        entries = restored.documents[0].root.entries
        assert len(entries) == 4
        assert isinstance(entries[0], Comment) and entries[0].text == "Server configuration"
        assert entries[1][0].value == "host"
        assert isinstance(entries[2], Comment) and entries[2].text == "Default port"
        assert entries[3][0].value == "port"

    def test_xml_comments_in_element(self):
        original = Stream(documents=[Document(root=Element(
            name="root",
            children=[
                Comment("section 1"),
                Element(name="a"),
                Comment("section 2"),
                Element(name="b"),
            ],
        ), source_format_hint=SourceFormat.XML)])
        restored = decode(encode(original))

        children = restored.documents[0].root.children
        assert len(children) == 4
        assert isinstance(children[0], Comment) and children[0].text == "section 1"
        assert isinstance(children[1], Element) and children[1].name == "a"
        assert isinstance(children[2], Comment) and children[2].text == "section 2"
        assert isinstance(children[3], Element) and children[3].name == "b"

    def test_comments_in_sequence(self):
        root = Sequence(items=[
            Comment("first item"),
            Scalar(ScalarType.INTEGER, 1),
            Comment("second item"),
            Scalar(ScalarType.INTEGER, 2),
        ])
        original = Stream(documents=[Document(root=root, source_format_hint=SourceFormat.TOML)])
        restored = decode(encode(original))

        items = restored.documents[0].root.items
        assert len(items) == 4
        assert isinstance(items[0], Comment)
        assert items[1].value == 1
        assert isinstance(items[2], Comment)
        assert items[3].value == 2

    def test_document_level_comments(self):
        original = Stream(documents=[Document(
            root=Map(),
            preamble=[Comment("file header")],
            postamble=[Comment("file footer")],
        )])
        restored = decode(encode(original))
        assert restored.documents[0].preamble[0].text == "file header"
        assert restored.documents[0].postamble[0].text == "file footer"


# ===================================================================
# CLAIM 7: DAG-only enforcement (Core) / cycles permitted (Extended)
# ===================================================================

class TestClaimCycleEnforcement:
    """CDXF Core rejects cyclic graphs. CDXF Extended accepts them."""

    def test_dag_accepted_in_core(self):
        """A valid DAG (shared reference, no cycle) must be accepted."""
        shared = Scalar(ScalarType.INTEGER, 42, anchor=Anchor("val"))
        root = Sequence(items=[shared, Alias("val")])
        original = Stream(documents=[Document(root=root, allows_cycles=False)])
        # Should not raise
        data = encode(original)
        restored = decode(data)
        assert isinstance(restored.documents[0].root.items[1], Alias)

    # NOTE: Cycle rejection/acceptance tests will be added when the
    # encoder implements graph validation. For now, we test the flag
    # round-trips correctly.
    def test_allows_cycles_flag_false(self):
        original = Stream(documents=[Document(root=Map(), allows_cycles=False)])
        restored = decode(encode(original))
        assert restored.documents[0].allows_cycles is False

    def test_allows_cycles_flag_true(self):
        original = Stream(documents=[Document(root=Map(), allows_cycles=True)])
        restored = decode(encode(original))
        assert restored.documents[0].allows_cycles is True


# ===================================================================
# CLAIM 8: Canonical determinism
# ===================================================================

class TestClaimCanonicalDeterminism:
    """Same logical data must always produce identical bytes in
    canonical mode, regardless of insertion order or annotation
    differences."""

    def test_map_key_order_irrelevant_in_canonical(self):
        """Two maps with the same entries in different order must
        produce identical canonical bytes."""
        map1 = Map(entries=[
            (Scalar(ScalarType.STRING, "b"), Scalar(ScalarType.INTEGER, 2)),
            (Scalar(ScalarType.STRING, "a"), Scalar(ScalarType.INTEGER, 1)),
        ])
        map2 = Map(entries=[
            (Scalar(ScalarType.STRING, "a"), Scalar(ScalarType.INTEGER, 1)),
            (Scalar(ScalarType.STRING, "b"), Scalar(ScalarType.INTEGER, 2)),
        ])
        encoder = Encoder(canonical=True)
        bytes1 = encoder.encode(Stream(documents=[Document(root=map1)]))
        bytes2 = encoder.encode(Stream(documents=[Document(root=map2)]))
        assert bytes1 == bytes2

    def test_comments_stripped_in_canonical(self):
        """Comments must not affect canonical form."""
        map_with_comments = Map(entries=[
            Comment("a comment"),
            (Scalar(ScalarType.STRING, "key"), Scalar(ScalarType.INTEGER, 1)),
        ])
        map_without_comments = Map(entries=[
            (Scalar(ScalarType.STRING, "key"), Scalar(ScalarType.INTEGER, 1)),
        ])
        encoder = Encoder(canonical=True)
        bytes1 = encoder.encode(Stream(documents=[Document(root=map_with_comments)]))
        bytes2 = encoder.encode(Stream(documents=[Document(root=map_without_comments)]))
        assert bytes1 == bytes2

    def test_source_format_hint_stripped_in_canonical(self):
        """Source format hint must not affect canonical form."""
        encoder = Encoder(canonical=True)
        bytes_json = encoder.encode(Stream(documents=[Document(
            root=Map(), source_format_hint=SourceFormat.JSON,
        )]))
        bytes_yaml = encoder.encode(Stream(documents=[Document(
            root=Map(), source_format_hint=SourceFormat.YAML,
        )]))
        assert bytes_json == bytes_yaml

    def test_canonical_is_deterministic_across_calls(self):
        """Multiple encodes of the same data produce identical bytes."""
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "x"), Scalar(ScalarType.INTEGER, 1)),
            (Scalar(ScalarType.STRING, "y"), Sequence(items=[
                Scalar(ScalarType.FLOAT, 3.14),
                Scalar(ScalarType.BOOLEAN, True),
            ])),
        ])
        encoder = Encoder(canonical=True)
        results = [
            encoder.encode(Stream(documents=[Document(root=root)]))
            for _ in range(10)
        ]
        assert all(r == results[0] for r in results)


# ===================================================================
# CLAIM 9: YAML merge key preservation
# ===================================================================

class TestClaimYamlMergeKeyPreservation:
    """The << merge key with its merge tag and alias value must
    survive round-trip as-is, not expanded into the target map."""

    def test_merge_key_preserved(self):
        defaults = Map(
            entries=[
                (Scalar(ScalarType.STRING, "timeout"), Scalar(ScalarType.INTEGER, 30)),
                (Scalar(ScalarType.STRING, "retries"), Scalar(ScalarType.INTEGER, 3)),
            ],
            anchor=Anchor("defaults"),
        )
        merge_key = Scalar(
            ScalarType.STRING,
            "<<",
            tag=TagAnnotation("tag:yaml.org,2002:merge"),
        )
        production = Map(entries=[
            (merge_key, Alias("defaults")),
            (Scalar(ScalarType.STRING, "timeout"), Scalar(ScalarType.INTEGER, 60)),
        ])
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "defaults"), defaults),
            (Scalar(ScalarType.STRING, "production"), production),
        ])
        original = Stream(documents=[Document(root=root, source_format_hint=SourceFormat.YAML)])
        restored = decode(encode(original))

        prod = restored.documents[0].root.entries[1][1]
        merge_entry = prod.entries[0]
        # Key must have the merge tag
        assert merge_entry[0].tag is not None
        assert merge_entry[0].tag.uri == "tag:yaml.org,2002:merge"
        assert merge_entry[0].value == "<<"
        # Value must be an alias, not an expanded copy
        assert isinstance(merge_entry[1], Alias), (
            "Merge key value was expanded — merge key preservation violated."
        )
        assert merge_entry[1].name == "defaults"


# ===================================================================
# CLAIM: Multi-document YAML stream preservation
# ===================================================================

class TestClaimMultiDocumentStream:
    """YAML multi-document streams with directives must survive
    round-trip with document boundaries and directives preserved."""

    def test_three_document_stream(self):
        docs = [
            Document(
                root=Scalar(ScalarType.STRING, "doc1"),
                preamble=[Directive(name="YAML", parameters=["1.2"])],
                source_format_hint=SourceFormat.YAML,
            ),
            Document(
                root=Map(entries=[
                    (Scalar(ScalarType.STRING, "key"), Scalar(ScalarType.INTEGER, 2)),
                ]),
                source_format_hint=SourceFormat.YAML,
            ),
            Document(
                root=Sequence(items=[
                    Scalar(ScalarType.INTEGER, 1),
                    Scalar(ScalarType.INTEGER, 2),
                ]),
                source_format_hint=SourceFormat.YAML,
            ),
        ]
        original = Stream(documents=docs)
        restored = decode(encode(original))

        assert len(restored.documents) == 3
        assert restored.documents[0].root.value == "doc1"
        assert restored.documents[0].preamble[0].name == "YAML"
        assert isinstance(restored.documents[1].root, Map)
        assert isinstance(restored.documents[2].root, Sequence)


# ===================================================================
# CLAIM: TOML native datetime types preserved
# ===================================================================

class TestClaimTomlDatetimeTypes:
    """TOML's four datetime types must each survive round-trip with
    their specific type preserved, not collapsed into a generic string."""

    def test_all_toml_datetime_types(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "offset_dt"),
             Scalar(ScalarType.TIMESTAMP_OFFSET, "2024-05-27T07:32:00-04:00")),
            (Scalar(ScalarType.STRING, "local_dt"),
             Scalar(ScalarType.TIMESTAMP_LOCAL, "2024-05-27T07:32:00")),
            (Scalar(ScalarType.STRING, "date"),
             Scalar(ScalarType.DATE, "2024-05-27")),
            (Scalar(ScalarType.STRING, "time"),
             Scalar(ScalarType.TIME, "07:32:00.999")),
        ])
        original = Stream(documents=[Document(root=root, source_format_hint=SourceFormat.TOML)])
        restored = decode(encode(original))

        entries = {e[0].value: e[1] for e in restored.documents[0].root.entries}
        assert entries["offset_dt"].scalar_type == ScalarType.TIMESTAMP_OFFSET
        assert entries["local_dt"].scalar_type == ScalarType.TIMESTAMP_LOCAL
        assert entries["date"].scalar_type == ScalarType.DATE
        assert entries["time"].scalar_type == ScalarType.TIME
        assert entries["offset_dt"].value == "2024-05-27T07:32:00-04:00"
        assert entries["local_dt"].value == "2024-05-27T07:32:00"
        assert entries["date"].value == "2024-05-27"
        assert entries["time"].value == "07:32:00.999"
