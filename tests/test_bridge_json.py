"""Tests for the JSON bridge — from_json() and to_json()."""

import json

import pytest

from cdxf.model import (
    Alias,
    Anchor,
    Comment,
    Document,
    Map,
    Scalar,
    ScalarType,
    Sequence,
    SourceFormat,
    Stream,
)
from cdxf.bridges.json_bridge import from_json, to_json


# ===================================================================
# from_json: JSON text → CDXF model
# ===================================================================

class TestFromJson:
    def test_empty_object(self):
        stream = from_json("{}")
        root = stream.documents[0].root
        assert isinstance(root, Map)
        assert len(root.entries) == 0

    def test_empty_array(self):
        stream = from_json("[]")
        root = stream.documents[0].root
        assert isinstance(root, Sequence)
        assert len(root.items) == 0

    def test_string_value(self):
        stream = from_json('{"key": "value"}')
        root = stream.documents[0].root
        entry = root.entries[0]
        assert entry[0].scalar_type == ScalarType.STRING
        assert entry[0].value == "key"
        assert entry[1].scalar_type == ScalarType.STRING
        assert entry[1].value == "value"

    def test_integer_value(self):
        stream = from_json('{"n": 42}')
        val = stream.documents[0].root.entries[0][1]
        assert val.scalar_type == ScalarType.INTEGER
        assert val.value == 42

    def test_negative_integer(self):
        stream = from_json('{"n": -7}')
        val = stream.documents[0].root.entries[0][1]
        assert val.scalar_type == ScalarType.INTEGER
        assert val.value == -7

    def test_zero_integer(self):
        stream = from_json('{"n": 0}')
        val = stream.documents[0].root.entries[0][1]
        assert val.scalar_type == ScalarType.INTEGER
        assert val.value == 0

    def test_float_value(self):
        stream = from_json('{"n": 3.14}')
        val = stream.documents[0].root.entries[0][1]
        assert val.scalar_type == ScalarType.FLOAT
        assert val.value == pytest.approx(3.14)

    def test_float_exponent(self):
        stream = from_json('{"n": 1.5e10}')
        val = stream.documents[0].root.entries[0][1]
        assert val.scalar_type == ScalarType.FLOAT
        assert val.value == pytest.approx(1.5e10)

    def test_boolean_true(self):
        stream = from_json('{"b": true}')
        val = stream.documents[0].root.entries[0][1]
        assert val.scalar_type == ScalarType.BOOLEAN
        assert val.value is True

    def test_boolean_false(self):
        stream = from_json('{"b": false}')
        val = stream.documents[0].root.entries[0][1]
        assert val.scalar_type == ScalarType.BOOLEAN
        assert val.value is False

    def test_null_value(self):
        stream = from_json('{"n": null}')
        val = stream.documents[0].root.entries[0][1]
        assert val.scalar_type == ScalarType.NULL
        assert val.value is None

    def test_nested_object(self):
        stream = from_json('{"outer": {"inner": 1}}')
        outer = stream.documents[0].root.entries[0][1]
        assert isinstance(outer, Map)
        assert outer.entries[0][1].value == 1

    def test_nested_array(self):
        stream = from_json('{"arr": [1, 2, 3]}')
        arr = stream.documents[0].root.entries[0][1]
        assert isinstance(arr, Sequence)
        assert [i.value for i in arr.items] == [1, 2, 3]

    def test_array_of_objects(self):
        stream = from_json('[{"a": 1}, {"b": 2}]')
        root = stream.documents[0].root
        assert isinstance(root, Sequence)
        assert len(root.items) == 2
        assert isinstance(root.items[0], Map)
        assert isinstance(root.items[1], Map)

    def test_deeply_nested(self):
        stream = from_json('{"a": {"b": {"c": {"d": 42}}}}')
        node = stream.documents[0].root
        for _ in range(3):
            node = node.entries[0][1]
        assert node.entries[0][1].value == 42

    def test_mixed_array(self):
        stream = from_json('[1, "two", true, null, 3.14, []]')
        items = stream.documents[0].root.items
        assert items[0].scalar_type == ScalarType.INTEGER
        assert items[1].scalar_type == ScalarType.STRING
        assert items[2].scalar_type == ScalarType.BOOLEAN
        assert items[3].scalar_type == ScalarType.NULL
        assert items[4].scalar_type == ScalarType.FLOAT
        assert isinstance(items[5], Sequence)

    def test_unicode_string(self):
        stream = from_json('{"emoji": "🌍🚀"}')
        val = stream.documents[0].root.entries[0][1]
        assert val.value == "🌍🚀"

    def test_escaped_characters(self):
        stream = from_json(r'{"s": "line1\nline2\ttab"}')
        val = stream.documents[0].root.entries[0][1]
        assert val.value == "line1\nline2\ttab"

    def test_source_format_hint_is_json(self):
        stream = from_json("{}")
        assert stream.documents[0].source_format_hint == SourceFormat.JSON

    def test_single_document_stream(self):
        stream = from_json("{}")
        assert len(stream.documents) == 1

    def test_key_order_preserved(self):
        stream = from_json('{"z": 1, "a": 2, "m": 3}')
        keys = [e[0].value for e in stream.documents[0].root.entries]
        assert keys == ["z", "a", "m"]

    def test_top_level_string(self):
        stream = from_json('"just a string"')
        root = stream.documents[0].root
        assert isinstance(root, Scalar)
        assert root.value == "just a string"

    def test_top_level_number(self):
        stream = from_json("42")
        root = stream.documents[0].root
        assert isinstance(root, Scalar)
        assert root.value == 42

    def test_top_level_null(self):
        stream = from_json("null")
        root = stream.documents[0].root
        assert root.scalar_type == ScalarType.NULL

    def test_top_level_boolean(self):
        stream = from_json("true")
        root = stream.documents[0].root
        assert root.value is True

    def test_large_integer(self):
        stream = from_json(f'{{"n": {2**53}}}')
        val = stream.documents[0].root.entries[0][1]
        assert val.value == 2**53

    def test_empty_string_key(self):
        stream = from_json('{"": "empty key"}')
        key = stream.documents[0].root.entries[0][0]
        assert key.value == ""

    def test_empty_nested_structures(self):
        stream = from_json('{"a": {}, "b": []}')
        entries = stream.documents[0].root.entries
        assert isinstance(entries[0][1], Map) and len(entries[0][1].entries) == 0
        assert isinstance(entries[1][1], Sequence) and len(entries[1][1].items) == 0


# ===================================================================
# to_json: CDXF model → JSON text
# ===================================================================

class TestToJson:
    def test_empty_object(self):
        stream = Stream(documents=[Document(root=Map(), source_format_hint=SourceFormat.JSON)])
        result = json.loads(to_json(stream))
        assert result == {}

    def test_empty_array(self):
        stream = Stream(documents=[Document(root=Sequence(), source_format_hint=SourceFormat.JSON)])
        result = json.loads(to_json(stream))
        assert result == []

    def test_simple_object(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "name"), Scalar(ScalarType.STRING, "Alice")),
            (Scalar(ScalarType.STRING, "age"), Scalar(ScalarType.INTEGER, 30)),
        ])
        stream = Stream(documents=[Document(root=root)])
        result = json.loads(to_json(stream))
        assert result == {"name": "Alice", "age": 30}

    def test_nested_structure(self):
        inner = Map(entries=[
            (Scalar(ScalarType.STRING, "x"), Scalar(ScalarType.INTEGER, 1)),
        ])
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "inner"), inner),
            (Scalar(ScalarType.STRING, "arr"), Sequence(items=[
                Scalar(ScalarType.INTEGER, 2),
                Scalar(ScalarType.INTEGER, 3),
            ])),
        ])
        stream = Stream(documents=[Document(root=root)])
        result = json.loads(to_json(stream))
        assert result == {"inner": {"x": 1}, "arr": [2, 3]}

    def test_all_scalar_types(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "str"), Scalar(ScalarType.STRING, "hello")),
            (Scalar(ScalarType.STRING, "int"), Scalar(ScalarType.INTEGER, 42)),
            (Scalar(ScalarType.STRING, "float"), Scalar(ScalarType.FLOAT, 3.14)),
            (Scalar(ScalarType.STRING, "bool"), Scalar(ScalarType.BOOLEAN, True)),
            (Scalar(ScalarType.STRING, "null"), Scalar(ScalarType.NULL, None)),
        ])
        stream = Stream(documents=[Document(root=root)])
        result = json.loads(to_json(stream))
        assert result["str"] == "hello"
        assert result["int"] == 42
        assert result["float"] == pytest.approx(3.14)
        assert result["bool"] is True
        assert result["null"] is None

    def test_top_level_scalar(self):
        stream = Stream(documents=[Document(root=Scalar(ScalarType.STRING, "solo"))])
        result = json.loads(to_json(stream))
        assert result == "solo"

    def test_comments_stripped(self):
        """JSON has no comments — they should be silently dropped."""
        root = Map(entries=[
            Comment("this will be lost"),
            (Scalar(ScalarType.STRING, "key"), Scalar(ScalarType.INTEGER, 1)),
        ])
        stream = Stream(documents=[Document(root=root)])
        result = json.loads(to_json(stream))
        assert result == {"key": 1}

    def test_indent_option(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "a"), Scalar(ScalarType.INTEGER, 1)),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_json(stream, indent=2)
        assert "\n" in text  # pretty-printed

    def test_uses_first_document(self):
        """Multi-doc stream: to_json uses the first document."""
        stream = Stream(documents=[
            Document(root=Scalar(ScalarType.INTEGER, 1)),
            Document(root=Scalar(ScalarType.INTEGER, 2)),
        ])
        result = json.loads(to_json(stream))
        assert result == 1


# ===================================================================
# Round-trip: JSON text → CDXF → JSON text
# ===================================================================

class TestJsonRoundTrip:
    def _roundtrip(self, json_text: str) -> dict | list:
        stream = from_json(json_text)
        output = to_json(stream)
        return json.loads(output)

    def test_simple_roundtrip(self):
        original = {"name": "Alice", "age": 30, "active": True}
        assert self._roundtrip(json.dumps(original)) == original

    def test_nested_roundtrip(self):
        original = {
            "users": [
                {"name": "Alice", "scores": [95, 87, 92]},
                {"name": "Bob", "scores": [78, 85, 90]},
            ],
            "meta": {"count": 2, "active": True},
        }
        assert self._roundtrip(json.dumps(original)) == original

    def test_all_types_roundtrip(self):
        original = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "true": True,
            "false": False,
            "null": None,
            "array": [1, "two", None],
            "object": {"nested": True},
        }
        assert self._roundtrip(json.dumps(original)) == original

    def test_empty_structures_roundtrip(self):
        original = {"obj": {}, "arr": [], "str": ""}
        assert self._roundtrip(json.dumps(original)) == original

    def test_unicode_roundtrip(self):
        original = {"greeting": "こんにちは", "emoji": "🌍🚀"}
        assert self._roundtrip(json.dumps(original, ensure_ascii=False)) == original

    def test_large_number_roundtrip(self):
        original = {"big": 2**53, "negative": -(2**31)}
        assert self._roundtrip(json.dumps(original)) == original

    def test_key_order_roundtrip(self):
        """Key insertion order must survive round-trip."""
        text = '{"z": 1, "a": 2, "m": 3}'
        stream = from_json(text)
        output = to_json(stream)
        keys = list(json.loads(output).keys())
        assert keys == ["z", "a", "m"]

    def test_deeply_nested_roundtrip(self):
        original = {"a": {"b": {"c": {"d": {"e": 42}}}}}
        assert self._roundtrip(json.dumps(original)) == original


# ===================================================================
# Full pipeline: JSON → CDXF model → CDXF binary → CDXF model → JSON
# ===================================================================

class TestJsonFullPipeline:
    """End-to-end: JSON text → CDXF model → CBOR bytes → CDXF model → JSON text."""

    def test_full_pipeline(self):
        from cdxf.codec import encode, decode

        original_text = '{"name": "Alice", "scores": [95, 87], "active": true}'
        original = json.loads(original_text)

        # JSON → CDXF model
        stream = from_json(original_text)
        # CDXF model → CBOR bytes
        binary = encode(stream)
        # CBOR bytes → CDXF model
        restored_stream = decode(binary)
        # CDXF model → JSON
        output_text = to_json(restored_stream)

        assert json.loads(output_text) == original

    def test_full_pipeline_complex(self):
        from cdxf.codec import encode, decode

        original = {
            "users": [
                {"id": 1, "name": "Alice", "meta": {"role": "admin"}},
                {"id": 2, "name": "Bob", "meta": {"role": "user"}},
            ],
            "pagination": {"page": 1, "total": 42, "has_next": True},
            "empty": None,
        }
        original_text = json.dumps(original)

        stream = from_json(original_text)
        binary = encode(stream)
        restored = decode(binary)
        output = to_json(restored)

        assert json.loads(output) == original
