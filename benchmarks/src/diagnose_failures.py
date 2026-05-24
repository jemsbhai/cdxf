"""Diagnostic script — inspect each failing round-trip to find root cause."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cdxf.codec import encode, decode
from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
from cdxf.bridges.xml_bridge import from_xml, to_xml
from cdxf.bridges.toml_bridge import from_toml, to_toml
from cdxf.model import Comment, Map, Scalar, Sequence

DATA_RAW = PROJECT_ROOT / "data" / "raw"

FAILURES = [
    ("synthetic/yaml_anchor_heavy.yaml", "yaml"),
    ("toml/tier2_canonical/serde-cargo.toml", "toml"),
    ("toml/tier2_canonical/tokio-cargo.toml", "toml"),
    ("toml/tier2_canonical/ruff-pyproject.toml", "toml"),
    ("xml/tier2_canonical/atom-namespace.xml", "xml"),
]


def diagnose(rel_path, fmt):
    fpath = DATA_RAW / rel_path
    text = fpath.read_text(encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {rel_path} ({fmt})")
    print(f"{'='*70}")

    # Step 1: Parse to CDXF
    try:
        if fmt == "yaml":
            stream = from_yaml(text)
        elif fmt == "xml":
            stream = from_xml(text)
        elif fmt == "toml":
            stream = from_toml(text)
        print("  [OK] Parse to CDXF model")
    except Exception as e:
        print(f"  [FAIL] Parse to CDXF: {e}")
        return

    # Step 2: Encode to CBOR
    try:
        binary = encode(stream)
        print(f"  [OK] Encode to CBOR ({len(binary)} bytes)")
    except Exception as e:
        print(f"  [FAIL] Encode to CBOR: {e}")
        return

    # Step 3: Decode from CBOR
    try:
        restored = decode(binary)
        print("  [OK] Decode from CBOR")
    except Exception as e:
        print(f"  [FAIL] Decode from CBOR: {e}")
        return

    # Step 4: Convert back to text
    try:
        if fmt == "yaml":
            output = to_yaml(restored)
        elif fmt == "xml":
            output = to_xml(restored)
        elif fmt == "toml":
            output = to_toml(restored)
        print(f"  [OK] Convert to text ({len(output)} chars)")
    except Exception as e:
        print(f"  [FAIL] Convert to text: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Compare
    if fmt == "yaml":
        from ruamel.yaml import YAML
        yaml = YAML(typ="safe")
        try:
            orig_docs = list(yaml.load_all(text))
            rest_docs = list(yaml.load_all(output))
            if orig_docs == rest_docs:
                print("  [OK] Semantic equality — THIS SHOULD NOT BE A FAILURE?")
            else:
                print(f"  [MISMATCH] YAML semantic diff:")
                print(f"    Original docs: {len(orig_docs)}")
                print(f"    Restored docs: {len(rest_docs)}")
                for i, (o, r) in enumerate(zip(orig_docs, rest_docs)):
                    if o != r:
                        print(f"    Doc {i} differs:")
                        print(f"      Original type: {type(o).__name__}")
                        print(f"      Restored type: {type(r).__name__}")
                        if isinstance(o, dict) and isinstance(r, dict):
                            for k in set(list(o.keys()) + list(r.keys())):
                                if o.get(k) != r.get(k):
                                    print(f"      Key '{k}': orig={o.get(k)!r} != rest={r.get(k)!r}")
                                    break  # show first diff only
        except Exception as e:
            print(f"  [ERROR] Comparison: {e}")

    elif fmt == "toml":
        import tomlkit
        try:
            orig = tomlkit.parse(text).unwrap()
            rest = tomlkit.parse(output).unwrap()
            if orig == rest:
                print("  [OK] Semantic equality — THIS SHOULD NOT BE A FAILURE?")
            else:
                print(f"  [MISMATCH] TOML semantic diff:")
                _diff_dicts(orig, rest, "  ")
        except Exception as e:
            print(f"  [ERROR] Comparison: {type(e).__name__}: {e}")
            print(f"  Output first 500 chars:")
            print(f"  {output[:500]}")

    elif fmt == "xml":
        # Print first/last of original vs output
        print(f"  Original first 300 chars:")
        print(f"    {text[:300]}")
        print(f"  Output first 300 chars:")
        print(f"    {output[:300]}")


def _diff_dicts(orig, rest, indent=""):
    """Recursively find first difference between two dicts."""
    if type(orig) != type(rest):
        print(f"{indent}Type mismatch: {type(orig).__name__} vs {type(rest).__name__}")
        return
    if isinstance(orig, dict):
        orig_keys = set(orig.keys())
        rest_keys = set(rest.keys())
        if orig_keys != rest_keys:
            missing = orig_keys - rest_keys
            extra = rest_keys - orig_keys
            if missing:
                print(f"{indent}Missing keys in restored: {missing}")
            if extra:
                print(f"{indent}Extra keys in restored: {extra}")
            return
        for k in orig:
            if orig[k] != rest[k]:
                print(f"{indent}Key '{k}' differs:")
                _diff_dicts(orig[k], rest[k], indent + "  ")
                return
    elif isinstance(orig, list):
        if len(orig) != len(rest):
            print(f"{indent}List length: {len(orig)} vs {len(rest)}")
            return
        for i, (o, r) in enumerate(zip(orig, rest)):
            if o != r:
                print(f"{indent}  [{i}] differs:")
                _diff_dicts(o, r, indent + "    ")
                return
    else:
        print(f"{indent}Values: {orig!r} vs {rest!r}")
        print(f"{indent}Types: {type(orig).__name__} vs {type(rest).__name__}")


if __name__ == "__main__":
    for rel, fmt in FAILURES:
        diagnose(rel, fmt)
