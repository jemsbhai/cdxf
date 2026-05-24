"""EXP-001b: Feature Preservation Matrix.

For each (construct, baseline) pair, run a targeted test to determine
whether the baseline preserves that construct. Results are binary: ✓ or ✗.

Usage:
    python benchmarks/src/run_feature_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cbor2
import msgpack

try:
    import bson
    HAS_BSON = True
except ImportError:
    HAS_BSON = False

try:
    import amazon.ion.simpleion as ion
    HAS_ION = True
except ImportError:
    HAS_ION = False

from cdxf.codec import encode, decode
from cdxf.model import (
    Alias, Anchor, Attribute, Comment, Document, Element,
    Map, ProcessingInstruction, Scalar, ScalarType, Sequence,
    SourceFormat, Stream, TagAnnotation,
)
from datetime import datetime, timezone, date, time, timedelta

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "exp_001"

# ===================================================================
# Test constructs
# ===================================================================

CONSTRUCTS = [
    "map_key_order",
    "non_string_map_keys",
    "comments",
    "anchors_aliases",
    "merge_keys",
    "multi_document_streams",
    "xml_elements_attributes",
    "xml_namespaces",
    "xml_mixed_content",
    "processing_instructions",
    "typed_timestamps",
    "typed_date_time_local",
]

BASELINES = ["cdxf", "cbor", "msgpack", "bson", "ion"]


def test_cdxf(construct: str) -> bool:
    """Test CDXF preservation via encode/decode round-trip."""
    try:
        stream = _build_test_stream(construct)
        restored = decode(encode(stream))
        return _verify(construct, restored)
    except Exception:
        return False


def test_cbor(construct: str) -> bool:
    """Test naive CBOR preservation."""
    try:
        native = _build_native(construct)
        if native is None:
            return False
        restored = cbor2.loads(cbor2.dumps(native))
        return _verify_native(construct, native, restored)
    except Exception:
        return False


def test_msgpack(construct: str) -> bool:
    try:
        native = _build_native(construct)
        if native is None:
            return False
        restored = msgpack.unpackb(
            msgpack.packb(native, use_bin_type=True, default=str),
            raw=False
        )
        return _verify_native(construct, native, restored)
    except Exception:
        return False


def test_bson(construct: str) -> bool:
    if not HAS_BSON:
        return False
    try:
        native = _build_native(construct)
        if native is None or not isinstance(native, dict):
            return False
        restored = bson.decode(bson.encode(native))
        return _verify_native(construct, native, restored)
    except Exception:
        return False


def test_ion(construct: str) -> bool:
    if not HAS_ION:
        return False
    try:
        native = _build_native(construct)
        if native is None:
            return False
        restored = ion.loads(ion.dumps(native, binary=True))
        return _verify_native(construct, native, restored)
    except Exception:
        return False


# ===================================================================
# Build test data
# ===================================================================

def _build_test_stream(construct: str) -> Stream:
    """Build a CDXF Stream that exercises the given construct."""
    if construct == "map_key_order":
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "z"), Scalar(ScalarType.INTEGER, 3)),
            (Scalar(ScalarType.STRING, "a"), Scalar(ScalarType.INTEGER, 1)),
            (Scalar(ScalarType.STRING, "m"), Scalar(ScalarType.INTEGER, 2)),
        ])
    elif construct == "non_string_map_keys":
        root = Map(entries=[
            (Scalar(ScalarType.INTEGER, 42), Scalar(ScalarType.STRING, "answer")),
            (Scalar(ScalarType.BOOLEAN, True), Scalar(ScalarType.STRING, "yes")),
        ])
    elif construct == "comments":
        root = Map(entries=[
            Comment("important note"),
            (Scalar(ScalarType.STRING, "key"), Scalar(ScalarType.STRING, "val")),
        ])
    elif construct == "anchors_aliases":
        shared = Scalar(ScalarType.STRING, "shared", anchor=Anchor("ref"))
        root = Sequence(items=[shared, Alias("ref")])
    elif construct == "merge_keys":
        defaults = Map(entries=[
            (Scalar(ScalarType.STRING, "x"), Scalar(ScalarType.INTEGER, 1)),
        ], anchor=Anchor("defaults"))
        merge_key = Scalar(ScalarType.STRING, "<<",
                           tag=TagAnnotation("tag:yaml.org,2002:merge"))
        service = Map(entries=[
            (merge_key, Alias("defaults")),
            (Scalar(ScalarType.STRING, "y"), Scalar(ScalarType.INTEGER, 2)),
        ])
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "defaults"), defaults),
            (Scalar(ScalarType.STRING, "service"), service),
        ])
    elif construct == "multi_document_streams":
        return Stream(documents=[
            Document(root=Scalar(ScalarType.STRING, "doc1")),
            Document(root=Scalar(ScalarType.STRING, "doc2")),
        ])
    elif construct == "xml_elements_attributes":
        root = Element(name="item",
                       attributes=[Attribute(name="id", value="42")],
                       children=[Element(name="child")])
    elif construct == "xml_namespaces":
        root = Element(name="root",
                       namespace_uri="http://example.com/ns",
                       prefix="ex",
                       namespace_declarations={"ex": "http://example.com/ns"})
    elif construct == "xml_mixed_content":
        root = Element(name="p", children=[
            Scalar(ScalarType.STRING, "Hello "),
            Element(name="b", children=[Scalar(ScalarType.STRING, "world")]),
            Scalar(ScalarType.STRING, "!"),
        ])
    elif construct == "processing_instructions":
        root = Element(name="root", children=[
            ProcessingInstruction(target="app", data="action"),
        ])
    elif construct == "typed_timestamps":
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "ts"),
             Scalar(ScalarType.TIMESTAMP_OFFSET,
                    datetime(2024, 1, 1, tzinfo=timezone.utc))),
        ])
    elif construct == "typed_date_time_local":
        root = Map(entries=[
            (Scalar(ScalarType.STRING, "d"),
             Scalar(ScalarType.DATE, date(2024, 5, 27))),
            (Scalar(ScalarType.STRING, "t"),
             Scalar(ScalarType.TIME, time(7, 32))),
            (Scalar(ScalarType.STRING, "local"),
             Scalar(ScalarType.TIMESTAMP_LOCAL,
                    datetime(2024, 5, 27, 7, 32))),
        ])
    else:
        raise ValueError(f"Unknown construct: {construct}")

    return Stream(documents=[Document(root=root)])


def _build_native(construct: str):
    """Build Python-native test data for baseline formats."""
    if construct == "map_key_order":
        return {"z": 3, "a": 1, "m": 2}
    elif construct == "non_string_map_keys":
        return {42: "answer", True: "yes"}
    elif construct == "comments":
        return None  # No native representation
    elif construct == "anchors_aliases":
        return None
    elif construct == "merge_keys":
        return None
    elif construct == "multi_document_streams":
        return None
    elif construct == "xml_elements_attributes":
        return None
    elif construct == "xml_namespaces":
        return None
    elif construct == "xml_mixed_content":
        return None
    elif construct == "processing_instructions":
        return None
    elif construct == "typed_timestamps":
        return {"ts": datetime(2024, 1, 1, tzinfo=timezone.utc)}
    elif construct == "typed_date_time_local":
        return {"d": date(2024, 5, 27), "t": time(7, 32),
                "local": datetime(2024, 5, 27, 7, 32)}
    return None


# ===================================================================
# Verification
# ===================================================================

def _verify(construct: str, restored: Stream) -> bool:
    """Verify CDXF round-trip preserved the construct."""
    root = restored.documents[0].root

    if construct == "map_key_order":
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        keys = [e[0].value for e in entries]
        return keys == ["z", "a", "m"]

    elif construct == "non_string_map_keys":
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        types = {e[0].scalar_type for e in entries}
        return ScalarType.INTEGER in types and ScalarType.BOOLEAN in types

    elif construct == "comments":
        return any(isinstance(e, Comment) for e in root.entries)

    elif construct == "anchors_aliases":
        items = root.items
        return (items[0].anchor is not None
                and isinstance(items[1], Alias)
                and items[1].name == "ref")

    elif construct == "merge_keys":
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        service = entries[1][1]
        svc_entries = [e for e in service.entries if not isinstance(e, Comment)]
        merge_entry = svc_entries[0]
        return (merge_entry[0].value == "<<"
                and merge_entry[0].tag is not None
                and isinstance(merge_entry[1], Alias))

    elif construct == "multi_document_streams":
        return len(restored.documents) == 2

    elif construct == "xml_elements_attributes":
        return (isinstance(root, Element)
                and root.name == "item"
                and len(root.attributes) == 1
                and root.attributes[0].value == "42"
                and len(root.children) == 1
                and isinstance(root.children[0], Element))

    elif construct == "xml_namespaces":
        return (root.namespace_uri == "http://example.com/ns"
                and root.prefix == "ex")

    elif construct == "xml_mixed_content":
        return (len(root.children) == 3
                and isinstance(root.children[0], Scalar)
                and isinstance(root.children[1], Element)
                and isinstance(root.children[2], Scalar))

    elif construct == "processing_instructions":
        pis = [c for c in root.children
               if isinstance(c, ProcessingInstruction)]
        return len(pis) == 1 and pis[0].target == "app"

    elif construct == "typed_timestamps":
        entries = [e for e in root.entries if not isinstance(e, Comment)]
        return entries[0][1].scalar_type == ScalarType.TIMESTAMP_OFFSET

    elif construct == "typed_date_time_local":
        entries = {
            e[0].value: e[1]
            for e in root.entries if not isinstance(e, Comment)
        }
        return (entries["d"].scalar_type == ScalarType.DATE
                and entries["t"].scalar_type == ScalarType.TIME
                and entries["local"].scalar_type == ScalarType.TIMESTAMP_LOCAL)

    return False


def _verify_native(construct: str, original, restored) -> bool:
    """Verify a baseline round-trip preserved the construct."""
    if construct == "map_key_order":
        if isinstance(restored, dict):
            return list(restored.keys()) == list(original.keys())
        return False

    elif construct == "non_string_map_keys":
        if isinstance(restored, dict):
            return any(not isinstance(k, str) for k in restored.keys())
        return False

    elif construct == "typed_timestamps":
        if isinstance(restored, dict) and "ts" in restored:
            return isinstance(restored["ts"], datetime)
        return False

    elif construct == "typed_date_time_local":
        if not isinstance(restored, dict):
            return False
        # Check each type is preserved distinctly
        d = restored.get("d")
        t = restored.get("t")
        local = restored.get("local")
        return (isinstance(d, date) and not isinstance(d, datetime)
                and isinstance(t, time)
                and isinstance(local, datetime)
                and local.tzinfo is None)

    return False


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("EXP-001b: Feature Preservation Matrix")
    print("=" * 70)

    test_funcs = {
        "cdxf": test_cdxf,
        "cbor": test_cbor,
        "msgpack": test_msgpack,
        "bson": test_bson,
        "ion": test_ion,
    }

    matrix: dict[str, dict[str, bool]] = {}
    for construct in CONSTRUCTS:
        matrix[construct] = {}
        for baseline in BASELINES:
            result = test_funcs[baseline](construct)
            matrix[construct][baseline] = result

    # Print matrix
    header = f"{'Construct':<30} " + " ".join(f"{b:>8}" for b in BASELINES)
    print(f"\n{header}")
    print("-" * len(header))

    for construct in CONSTRUCTS:
        row = f"{construct:<30} "
        for baseline in BASELINES:
            val = matrix[construct][baseline]
            sym = "  YES" if val else "   NO"
            row += f"{sym:>8} "
        print(row)

    # Count totals
    print("-" * len(header))
    totals = f"{'TOTAL':<30} "
    for baseline in BASELINES:
        total = sum(1 for c in CONSTRUCTS if matrix[c][baseline])
        totals += f"{total:>5}/12 "
    print(totals)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_matrix.json"
    out_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
