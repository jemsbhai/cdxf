"""EXP-003: Cross-Format Interchange Fidelity.

Tests all 6 conversion pairs between JSON, YAML, and TOML through CDXF,
plus format-specific construct handling (graceful degradation).

Usage:
    python benchmarks/src/run_exp003.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, date, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cdxf.bridges.json_bridge import from_json, to_json
from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
from cdxf.bridges.toml_bridge import from_toml, to_toml
from cdxf.bridges.xml_bridge import from_xml, to_xml

import tomlkit
from ruamel.yaml import YAML

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "exp_003"

# ===================================================================
# Test documents — shared data model
# ===================================================================

# Each document is defined in all 3 text formats (JSON, YAML, TOML)
# to enable cross-checking. The data model is restricted to what all
# 3 formats can represent natively.

TEST_DOCUMENTS = {
    "flat_map": {
        "json": '{"name": "Alice", "age": 30, "active": true}',
        "yaml": "name: Alice\nage: 30\nactive: true\n",
        "toml": 'name = "Alice"\nage = 30\nactive = true\n',
        "expected": {"name": "Alice", "age": 30, "active": True},
    },
    "nested_map": {
        "json": '{"server": {"host": "localhost", "port": 8080}, "debug": false}',
        "yaml": "server:\n  host: localhost\n  port: 8080\ndebug: false\n",
        "toml": 'debug = false\n\n[server]\nhost = "localhost"\nport = 8080\n',
        "expected": {"server": {"host": "localhost", "port": 8080}, "debug": False},
    },
    "arrays": {
        "json": '{"items": [1, 2, 3], "tags": ["a", "b"]}',
        "yaml": "items:\n  - 1\n  - 2\n  - 3\ntags:\n  - a\n  - b\n",
        "toml": 'items = [1, 2, 3]\ntags = ["a", "b"]\n',
        "expected": {"items": [1, 2, 3], "tags": ["a", "b"]},
    },
    "mixed_types": {
        "json": '{"s": "hello", "i": 42, "f": 3.14, "b": true, "n": null}',
        "yaml": "s: hello\ni: 42\nf: 3.14\nb: true\nn: null\n",
        # TOML has no null — omit n
        "toml": 's = "hello"\ni = 42\nf = 3.14\nb = true\n',
        "expected": {"s": "hello", "i": 42, "f": 3.14, "b": True, "n": None},
        "expected_toml": {"s": "hello", "i": 42, "f": 3.14, "b": True},
    },
    "nested_arrays": {
        "json": '{"matrix": [[1, 2], [3, 4]]}',
        "yaml": "matrix:\n  - - 1\n    - 2\n  - - 3\n    - 4\n",
        "toml": "matrix = [[1, 2], [3, 4]]\n",
        "expected": {"matrix": [[1, 2], [3, 4]]},
    },
    "array_of_tables": {
        "json": '{"products": [{"name": "Hammer", "price": 10}, {"name": "Nail", "price": 1}]}',
        "yaml": "products:\n  - name: Hammer\n    price: 10\n  - name: Nail\n    price: 1\n",
        "toml": '[[products]]\nname = "Hammer"\nprice = 10\n\n[[products]]\nname = "Nail"\nprice = 1\n',
        "expected": {"products": [{"name": "Hammer", "price": 10}, {"name": "Nail", "price": 1}]},
    },
    "string_escapes": {
        "json": '{"msg": "line1\\nline2", "path": "C:\\\\dir"}',
        "yaml": 'msg: "line1\\nline2"\npath: "C:\\\\dir"\n',
        "toml": 'msg = "line1\\nline2"\npath = "C:\\\\dir"\n',
        "expected": {"msg": "line1\nline2", "path": "C:\\dir"},
    },
    "empty_structures": {
        "json": '{"empty_map": {}, "empty_arr": []}',
        "yaml": "empty_map: {}\nempty_arr: []\n",
        # TOML: empty table, empty array
        "toml": 'empty_arr = []\n\n[empty_map]\n',
        "expected": {"empty_map": {}, "empty_arr": []},
    },
    "deeply_nested": {
        "json": '{"a": {"b": {"c": {"d": "deep"}}}}',
        "yaml": "a:\n  b:\n    c:\n      d: deep\n",
        "toml": '[a.b.c]\nd = "deep"\n',
        "expected": {"a": {"b": {"c": {"d": "deep"}}}},
    },
    "large_array": {
        "json": json.dumps({"numbers": list(range(100))}),
        "yaml": "numbers:\n" + "".join(f"  - {i}\n" for i in range(100)),
        "toml": "numbers = [" + ", ".join(str(i) for i in range(100)) + "]\n",
        "expected": {"numbers": list(range(100))},
    },
}


# ===================================================================
# Native parsers
# ===================================================================

def parse_native(text: str, fmt: str):
    """Parse text to Python native types using the standard parser."""
    if fmt == "json":
        return json.loads(text)
    if fmt == "yaml":
        y = YAML(typ="safe")
        return y.load(text)
    if fmt == "toml":
        return tomlkit.parse(text).unwrap()
    return None


def normalize(obj):
    """Normalize Python types for cross-format comparison.

    Handles: tomlkit wrappers, OrderedDict→dict, datetime→str for JSON.
    """
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, float):
        return round(obj, 10)  # avoid float precision noise
    return obj


# ===================================================================
# Cross-conversion test
# ===================================================================

FORMATS = ["json", "yaml", "toml"]

BRIDGES_FROM = {"json": from_json, "yaml": from_yaml, "toml": from_toml}
BRIDGES_TO = {
    "json": lambda s: to_json(s, indent=2),
    "yaml": to_yaml,
    "toml": to_toml,
}


def convert(text: str, src_fmt: str, dst_fmt: str) -> str:
    """Convert text from src_fmt to dst_fmt via CDXF."""
    stream = BRIDGES_FROM[src_fmt](text)
    return BRIDGES_TO[dst_fmt](stream)


def test_conversion(doc_name: str, doc: dict, src: str, dst: str) -> dict:
    """Test one (document, source, target) triple."""
    src_text = doc.get(src)
    if src_text is None:
        return {"status": "skip", "reason": f"no {src} source text"}

    # Expected data for comparison
    expected_key = f"expected_{dst}" if f"expected_{dst}" in doc else "expected"
    expected = doc.get(expected_key, doc["expected"])

    try:
        # Convert src → dst via CDXF
        dst_text = convert(src_text, src, dst)
    except Exception as e:
        return {"status": "error", "reason": f"conversion failed: {type(e).__name__}: {e}"}

    try:
        # Parse the result with the target format's native parser
        restored = parse_native(dst_text, dst)
    except Exception as e:
        return {
            "status": "error",
            "reason": f"target parse failed: {type(e).__name__}: {e}",
            "output": dst_text[:200],
        }

    # Normalize and compare
    norm_expected = normalize(expected)
    norm_restored = normalize(restored)

    if norm_expected == norm_restored:
        return {"status": "pass"}
    else:
        # Find first difference
        diff = _find_diff(norm_expected, norm_restored)
        return {
            "status": "fail",
            "reason": f"data mismatch: {diff}",
        }


def _find_diff(a, b, path="root") -> str:
    """Find the first difference between two structures."""
    if type(a) != type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        if a_keys != b_keys:
            missing = a_keys - b_keys
            extra = b_keys - a_keys
            parts = []
            if missing:
                parts.append(f"missing keys {missing}")
            if extra:
                parts.append(f"extra keys {extra}")
            return f"{path}: {', '.join(parts)}"
        for k in a:
            if a[k] != b[k]:
                return _find_diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: list len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _find_diff(x, y, f"{path}[{i}]")
    else:
        return f"{path}: {a!r} vs {b!r}"
    return f"{path}: unknown diff"


# ===================================================================
# Format-specific construct tests
# ===================================================================

def test_format_specific() -> list[dict]:
    """Test graceful handling of format-specific constructs in cross-conversion."""
    results = []

    # --- YAML comments → JSON (comments lost, data preserved) ---
    yaml_with_comments = "# important\nkey: value\n# trailing\n"
    try:
        json_text = convert(yaml_with_comments, "yaml", "json")
        restored = json.loads(json_text)
        ok = restored == {"key": "value"}
        results.append({
            "test": "yaml_comments_to_json",
            "description": "YAML comments gracefully dropped in JSON output",
            "status": "pass" if ok else "fail",
        })
    except Exception as e:
        results.append({
            "test": "yaml_comments_to_json",
            "status": "error", "reason": str(e),
        })

    # --- YAML anchors → JSON (anchors expanded, data preserved) ---
    yaml_with_anchors = "defaults: &def\n  x: 1\noverride:\n  <<: *def\n  y: 2\n"
    try:
        json_text = convert(yaml_with_anchors, "yaml", "json")
        restored = json.loads(json_text)
        # After merge expansion, override should have x and y
        ok = (restored.get("defaults") == {"x": 1}
              and "y" in restored.get("override", {}))
        results.append({
            "test": "yaml_anchors_to_json",
            "description": "YAML anchors/merge keys expanded in JSON output",
            "status": "pass" if ok else "fail",
            "detail": f"restored={restored}",
        })
    except Exception as e:
        results.append({
            "test": "yaml_anchors_to_json",
            "status": "error", "reason": str(e),
        })

    # --- YAML multi-doc → JSON (first doc only, or error) ---
    yaml_multi = "---\na: 1\n---\nb: 2\n"
    try:
        json_text = convert(yaml_multi, "yaml", "json")
        # Should produce valid JSON for at least the first doc
        restored = json.loads(json_text)
        ok = isinstance(restored, dict)
        results.append({
            "test": "yaml_multidoc_to_json",
            "description": "YAML multi-document stream produces valid JSON",
            "status": "pass" if ok else "fail",
        })
    except Exception as e:
        results.append({
            "test": "yaml_multidoc_to_json",
            "status": "error", "reason": str(e),
        })

    # --- TOML dates → JSON (dates become strings) ---
    toml_with_dates = 'ts = 1979-05-27T07:32:00Z\nd = 1979-05-27\n'
    try:
        json_text = convert(toml_with_dates, "toml", "json")
        restored = json.loads(json_text)
        # Dates should appear as ISO strings in JSON
        ok = (isinstance(restored.get("ts"), str)
              and isinstance(restored.get("d"), str)
              and "1979" in restored["ts"]
              and "1979" in restored["d"])
        results.append({
            "test": "toml_dates_to_json",
            "description": "TOML datetime/date become ISO strings in JSON",
            "status": "pass" if ok else "fail",
            "detail": f"ts={restored.get('ts')}, d={restored.get('d')}",
        })
    except Exception as e:
        results.append({
            "test": "toml_dates_to_json",
            "status": "error", "reason": str(e),
        })

    # --- TOML dates → YAML (dates preserved as datetime objects) ---
    try:
        yaml_text = convert(toml_with_dates, "toml", "yaml")
        y = YAML(typ="safe")
        restored = y.load(yaml_text)
        # YAML should preserve datetime objects
        ts_val = restored.get("ts")
        ok = isinstance(ts_val, datetime)
        results.append({
            "test": "toml_dates_to_yaml",
            "description": "TOML datetime preserved as YAML datetime",
            "status": "pass" if ok else "fail",
            "detail": f"ts type={type(ts_val).__name__}",
        })
    except Exception as e:
        results.append({
            "test": "toml_dates_to_yaml",
            "status": "error", "reason": str(e),
        })

    # --- JSON null → TOML (no null in TOML — key should be absent or error) ---
    json_with_null = '{"key": "value", "nothing": null}'
    try:
        toml_text = convert(json_with_null, "json", "toml")
        restored = tomlkit.parse(toml_text).unwrap()
        # key should survive, nothing may be absent or empty string
        ok = restored.get("key") == "value"
        results.append({
            "test": "json_null_to_toml",
            "description": "JSON null handled gracefully in TOML conversion",
            "status": "pass" if ok else "fail",
            "detail": f"nothing={restored.get('nothing')!r}",
        })
    except Exception as e:
        results.append({
            "test": "json_null_to_toml",
            "status": "error", "reason": str(e),
        })

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("EXP-003: Cross-Format Interchange Fidelity")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    pass_count = 0
    fail_count = 0
    error_count = 0
    skip_count = 0

    # --- Part 1: All 6 conversion pairs × all test documents ---
    print("\n--- Part 1: Shared data model (6 conversion pairs) ---\n")

    pairs = [(s, d) for s in FORMATS for d in FORMATS if s != d]

    for src, dst in pairs:
        print(f"  {src.upper()} → {dst.upper()}:")
        for doc_name, doc in TEST_DOCUMENTS.items():
            result = test_conversion(doc_name, doc, src, dst)
            result["source"] = src
            result["target"] = dst
            result["document"] = doc_name
            all_results.append(result)

            status = result["status"]
            if status == "pass":
                pass_count += 1
                sym = "✓"
            elif status == "fail":
                fail_count += 1
                sym = "✗"
            elif status == "error":
                error_count += 1
                sym = "!"
            else:
                skip_count += 1
                sym = "-"

            reason = f" ({result.get('reason', '')})" if status != "pass" else ""
            print(f"    [{sym}] {doc_name}{reason}")

    # --- Part 2: Format-specific construct handling ---
    print(f"\n--- Part 2: Format-specific constructs (graceful degradation) ---\n")

    specific_results = test_format_specific()
    for r in specific_results:
        status = r["status"]
        if status == "pass":
            pass_count += 1
            sym = "✓"
        elif status == "fail":
            fail_count += 1
            sym = "✗"
        else:
            error_count += 1
            sym = "!"

        desc = r.get("description", r["test"])
        reason = f" ({r.get('reason', r.get('detail', ''))})" if status != "pass" else ""
        print(f"  [{sym}] {desc}{reason}")
        all_results.append(r)

    # --- Summary ---
    total = pass_count + fail_count + error_count + skip_count
    print(f"\n{'='*70}")
    print(f"Total: {total} tests — {pass_count} passed, {fail_count} failed, "
          f"{error_count} errors, {skip_count} skipped")

    # Per-pair summary
    print(f"\nPer-pair summary:")
    for src, dst in pairs:
        pair_results = [r for r in all_results
                        if r.get("source") == src and r.get("target") == dst]
        pair_pass = sum(1 for r in pair_results if r["status"] == "pass")
        pair_total = len(pair_results)
        print(f"  {src.upper()} → {dst.upper()}: {pair_pass}/{pair_total}")

    # --- Save ---
    output = {
        "experiment": "EXP-003",
        "total": total,
        "passed": pass_count,
        "failed": fail_count,
        "errors": error_count,
        "skipped": skip_count,
        "results": all_results,
    }
    results_path = RESULTS_DIR / "exp_003_results.json"
    results_path.write_text(json.dumps(output, indent=2, default=str),
                            encoding="utf-8")
    print(f"\nResults saved: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
