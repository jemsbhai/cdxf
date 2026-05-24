"""Tests for the TOML bridge — from_toml() and to_toml().

TOML is the simplest bridge: no namespaces, no elements, no mixed content.
The key differentiators are typed scalars (dates, times, datetimes) and
comment preservation via tomlkit.

TDD: This file is written BEFORE the implementation.
"""

import pytest
from datetime import date, time, datetime, timezone, timedelta

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

# The bridge module under test — does not exist yet.
from cdxf.bridges.toml_bridge import from_toml, to_toml


# ===================================================================
# from_toml: basic scalar types
# ===================================================================

class TestFromTomlScalars:
    """Verify that TOML scalar types map to correct ScalarType values."""

    def test_string(self):
        stream = from_toml('name = "Alice"')
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].scalar_type == ScalarType.STRING
        assert entries[0][1].value == "Alice"

    def test_integer(self):
        stream = from_toml("count = 42")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].scalar_type == ScalarType.INTEGER
        assert entries[0][1].value == 42

    def test_float(self):
        stream = from_toml("pi = 3.14")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].scalar_type == ScalarType.FLOAT
        assert entries[0][1].value == pytest.approx(3.14)

    def test_boolean_true(self):
        stream = from_toml("flag = true")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].scalar_type == ScalarType.BOOLEAN
        assert entries[0][1].value is True

    def test_boolean_false(self):
        stream = from_toml("flag = false")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].value is False

    def test_multiline_string(self):
        toml_text = 'desc = """\nline one\nline two"""'
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert "line one" in entries[0][1].value
        assert "line two" in entries[0][1].value

    def test_source_format_hint(self):
        stream = from_toml('x = 1')
        assert stream.documents[0].source_format_hint == SourceFormat.TOML

    def test_single_document(self):
        stream = from_toml('x = 1')
        assert len(stream.documents) == 1


# ===================================================================
# from_toml: date/time types
# ===================================================================

class TestFromTomlDatetime:
    """TOML has four temporal types — a key differentiator from JSON."""

    def test_offset_datetime(self):
        stream = from_toml("ts = 1979-05-27T07:32:00Z")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        scalar = entries[0][1]
        assert scalar.scalar_type == ScalarType.TIMESTAMP_OFFSET
        assert isinstance(scalar.value, datetime)
        assert scalar.value.tzinfo is not None

    def test_offset_datetime_with_offset(self):
        stream = from_toml("ts = 1979-05-27T07:32:00+05:30")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        scalar = entries[0][1]
        assert scalar.scalar_type == ScalarType.TIMESTAMP_OFFSET

    def test_local_datetime(self):
        stream = from_toml("ts = 1979-05-27T07:32:00")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        scalar = entries[0][1]
        assert scalar.scalar_type == ScalarType.TIMESTAMP_LOCAL
        assert isinstance(scalar.value, datetime)

    def test_local_date(self):
        stream = from_toml("d = 1979-05-27")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        scalar = entries[0][1]
        assert scalar.scalar_type == ScalarType.DATE
        assert isinstance(scalar.value, date)
        assert not isinstance(scalar.value, datetime)  # date, not datetime

    def test_local_time(self):
        stream = from_toml("t = 07:32:00")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        scalar = entries[0][1]
        assert scalar.scalar_type == ScalarType.TIME
        assert isinstance(scalar.value, time)


# ===================================================================
# from_toml: tables (maps)
# ===================================================================

class TestFromTomlTables:
    """Verify TOML table → CDXF Map mapping."""

    def test_root_is_map(self):
        stream = from_toml("x = 1\ny = 2")
        root = stream.documents[0].root
        assert isinstance(root, Map)

    def test_table_section(self):
        toml_text = "[server]\nhost = 'localhost'\nport = 8080"
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        # Root has one entry: "server" -> Map
        assert entries[0][0].value == "server"
        server = entries[0][1]
        assert isinstance(server, Map)
        server_entries = [e for e in server.entries if not isinstance(e, Comment)]
        keys = [e[0].value for e in server_entries]
        assert "host" in keys
        assert "port" in keys

    def test_nested_table(self):
        toml_text = "[a.b]\nx = 1"
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        # "a" -> Map containing "b" -> Map containing "x" -> 1
        a_val = entries[0][1]
        assert isinstance(a_val, Map)
        a_entries = [e for e in a_val.entries if not isinstance(e, Comment)]
        b_val = a_entries[0][1]
        assert isinstance(b_val, Map)

    def test_inline_table(self):
        toml_text = 'point = {x = 1, y = 2}'
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        point = entries[0][1]
        assert isinstance(point, Map)
        point_entries = [e for e in point.entries if not isinstance(e, Comment)]
        keys = [e[0].value for e in point_entries]
        assert "x" in keys
        assert "y" in keys

    def test_key_order_preserved(self):
        toml_text = "z = 1\na = 2\nm = 3"
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        keys = [e[0].value for e in entries]
        assert keys == ["z", "a", "m"]

    def test_string_keys(self):
        """TOML keys are always strings."""
        toml_text = 'key = "value"'
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][0].scalar_type == ScalarType.STRING


# ===================================================================
# from_toml: arrays (sequences)
# ===================================================================

class TestFromTomlArrays:
    """Verify TOML array → CDXF Sequence mapping."""

    def test_simple_array(self):
        stream = from_toml("items = [1, 2, 3]")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        arr = entries[0][1]
        assert isinstance(arr, Sequence)
        values = [i.value for i in arr.items if not isinstance(i, Comment)]
        assert values == [1, 2, 3]

    def test_string_array(self):
        stream = from_toml('colors = ["red", "green", "blue"]')
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        arr = entries[0][1]
        values = [i.value for i in arr.items if not isinstance(i, Comment)]
        assert values == ["red", "green", "blue"]

    def test_nested_array(self):
        stream = from_toml("matrix = [[1, 2], [3, 4]]")
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        arr = entries[0][1]
        assert isinstance(arr, Sequence)
        items = [i for i in arr.items if not isinstance(i, Comment)]
        assert isinstance(items[0], Sequence)
        assert isinstance(items[1], Sequence)

    def test_array_of_tables(self):
        toml_text = "[[products]]\nname = 'Hammer'\n\n[[products]]\nname = 'Nail'"
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        products = entries[0][1]
        assert isinstance(products, Sequence)
        items = [i for i in products.items if not isinstance(i, Comment)]
        assert len(items) == 2
        assert isinstance(items[0], Map)

    def test_mixed_type_array(self):
        """TOML 1.0 allows mixed-type arrays."""
        stream = from_toml('mixed = [1, "two", true]')
        root = stream.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        arr = entries[0][1]
        items = [i for i in arr.items if not isinstance(i, Comment)]
        assert items[0].scalar_type == ScalarType.INTEGER
        assert items[1].scalar_type == ScalarType.STRING
        assert items[2].scalar_type == ScalarType.BOOLEAN


# ===================================================================
# from_toml: comments
# ===================================================================

class TestFromTomlComments:
    """Verify comment preservation via tomlkit — a CDXF differentiator."""

    def test_comment_before_key(self):
        toml_text = "# server config\nhost = 'localhost'"
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        has_comment = any(isinstance(e, Comment) for e in root.entries)
        assert has_comment, "Comment before key was lost"

    def test_comment_between_keys(self):
        toml_text = "a = 1\n# between\nb = 2"
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        comments = [e for e in root.entries if isinstance(e, Comment)]
        assert len(comments) >= 1
        assert any("between" in c.text for c in comments)

    def test_inline_comment(self):
        toml_text = "port = 8080 # default port"
        stream = from_toml(toml_text)
        root = stream.documents[0].root
        comments = [e for e in root.entries if isinstance(e, Comment)]
        assert len(comments) >= 1
        assert any("default port" in c.text for c in comments)


# ===================================================================
# to_toml: CDXF model → TOML text
# ===================================================================

class TestToToml:
    """Verify conversion from CDXF model to TOML text."""

    def test_simple_key_value(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "name"), Scalar(ScalarType.STRING, "Alice")),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_toml(stream)
        assert "name" in text
        assert "Alice" in text

    def test_integer_value(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "count"), Scalar(ScalarType.INTEGER, 42)),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_toml(stream)
        assert "42" in text

    def test_float_value(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "pi"), Scalar(ScalarType.FLOAT, 3.14)),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_toml(stream)
        assert "3.14" in text

    def test_boolean_value(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "flag"), Scalar(ScalarType.BOOLEAN, True)),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_toml(stream)
        assert "true" in text

    def test_array_output(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "items"),
             Sequence(items=[
                 Scalar(ScalarType.INTEGER, 1),
                 Scalar(ScalarType.INTEGER, 2),
             ])),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_toml(stream)
        assert "1" in text
        assert "2" in text

    def test_nested_table_output(self):
        inner = Map(entries=[
            (Scalar(ScalarType.STRING, "host"), Scalar(ScalarType.STRING, "localhost")),
        ])
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "server"), inner),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_toml(stream)
        assert "server" in text
        assert "host" in text
        assert "localhost" in text

    def test_date_output(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "d"),
             Scalar(ScalarType.DATE, date(1979, 5, 27))),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_toml(stream)
        assert "1979-05-27" in text

    def test_time_output(self):
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "t"),
             Scalar(ScalarType.TIME, time(7, 32, 0))),
        ])
        stream = Stream(documents=[Document(root=root)])
        text = to_toml(stream)
        assert "07:32:00" in text

    def test_empty_stream(self):
        stream = Stream(documents=[])
        text = to_toml(stream)
        assert text == ""


# ===================================================================
# Round-trip: TOML → CDXF → TOML (semantic equivalence)
# ===================================================================

class TestTomlRoundTrip:
    """Verify TOML → CDXF → TOML preserves semantics."""

    def test_simple_roundtrip(self):
        toml_text = 'name = "Alice"\nage = 30'
        stream = from_toml(toml_text)
        output = to_toml(stream)
        restored = from_toml(output)
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        values = {e[0].value: e[1].value for e in entries}
        assert values["name"] == "Alice"
        assert values["age"] == 30

    def test_nested_roundtrip(self):
        toml_text = "[server]\nhost = 'localhost'\nport = 8080"
        stream = from_toml(toml_text)
        output = to_toml(stream)
        restored = from_toml(output)
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        server = entries[0][1]
        server_entries = [e for e in server.entries if not isinstance(e, Comment)]
        vals = {e[0].value: e[1].value for e in server_entries}
        assert vals["host"] == "localhost"
        assert vals["port"] == 8080

    def test_array_roundtrip(self):
        toml_text = 'items = [1, 2, 3]'
        stream = from_toml(toml_text)
        output = to_toml(stream)
        restored = from_toml(output)
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        arr = entries[0][1]
        values = [i.value for i in arr.items if not isinstance(i, Comment)]
        assert values == [1, 2, 3]

    def test_datetime_roundtrip(self):
        toml_text = "ts = 1979-05-27T07:32:00Z"
        stream = from_toml(toml_text)
        output = to_toml(stream)
        restored = from_toml(output)
        root = restored.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        assert entries[0][1].scalar_type == ScalarType.TIMESTAMP_OFFSET


# ===================================================================
# Full pipeline: TOML → CDXF model → CBOR → CDXF model → TOML
# ===================================================================

class TestTomlFullPipeline:
    """End-to-end: TOML text → CDXF → binary → CDXF → TOML text."""

    def test_pipeline_simple(self):
        from cdxf.codec import encode, decode

        toml_text = 'name = "Alice"\nage = 30'
        stream = from_toml(toml_text)
        binary = encode(stream)
        restored = decode(binary)
        output = to_toml(restored)
        final = from_toml(output)

        root = final.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        vals = {e[0].value: e[1].value for e in entries}
        assert vals["name"] == "Alice"
        assert vals["age"] == 30

    def test_pipeline_nested(self):
        from cdxf.codec import encode, decode

        toml_text = "[database]\nserver = 'localhost'\nport = 5432\nenabled = true"
        stream = from_toml(toml_text)
        binary = encode(stream)
        restored = decode(binary)
        output = to_toml(restored)
        final = from_toml(output)

        root = final.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        db = entries[0][1]
        db_entries = [e for e in db.entries if not isinstance(e, Comment)]
        vals = {e[0].value: e[1].value for e in db_entries}
        assert vals["server"] == "localhost"
        assert vals["port"] == 5432
        assert vals["enabled"] is True

    def test_pipeline_array_of_tables(self):
        from cdxf.codec import encode, decode

        toml_text = '[[items]]\nname = "A"\n\n[[items]]\nname = "B"'
        stream = from_toml(toml_text)
        binary = encode(stream)
        restored = decode(binary)
        output = to_toml(restored)
        final = from_toml(output)

        root = final.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        items = entries[0][1]
        assert isinstance(items, Sequence)
        item_list = [i for i in items.items if not isinstance(i, Comment)]
        assert len(item_list) == 2

    def test_pipeline_complex(self):
        """A realistic TOML document through the full pipeline."""
        from cdxf.codec import encode, decode

        toml_text = (
            'title = "TOML Example"\n'
            '\n'
            '[owner]\n'
            'name = "Tom Preston-Werner"\n'
            '\n'
            '[database]\n'
            'server = "192.168.1.1"\n'
            'ports = [8001, 8001, 8002]\n'
            'enabled = true\n'
        )
        stream = from_toml(toml_text)
        binary = encode(stream)
        restored = decode(binary)
        output = to_toml(restored)
        final = from_toml(output)

        root = final.documents[0].root
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        keys = [e[0].value for e in entries]
        assert "title" in keys
        assert "owner" in keys
        assert "database" in keys
