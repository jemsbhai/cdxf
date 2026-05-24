"""Detailed TOML diagnostic — inspect the exact structure differences."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tomlkit
from tomlkit import items as toml_items
from cdxf.bridges.toml_bridge import from_toml, to_toml, _get_body, _unwrap_key
from cdxf.codec import encode, decode
from cdxf.model import Comment, Map, Scalar

DATA_RAW = PROJECT_ROOT / "data" / "raw"

FILES = [
    "toml/tier2_canonical/serde-cargo.toml",
    "toml/tier2_canonical/tokio-cargo.toml",
    "toml/tier2_canonical/ruff-pyproject.toml",
]

for rel in FILES:
    fpath = DATA_RAW / rel
    text = fpath.read_text(encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"FILE: {rel}")
    print(f"{'='*70}")

    # Parse with tomlkit directly to inspect structure
    doc = tomlkit.parse(text)
    orig = doc.unwrap()

    # Round-trip through CDXF
    stream = from_toml(text)
    binary = encode(stream)
    restored = decode(binary)
    output = to_toml(restored)

    try:
        rest = tomlkit.parse(output).unwrap()
    except Exception as e:
        print(f"  PARSE ERROR on restored output: {e}")
        print(f"  Output first 500 chars:\n{output[:500]}")
        continue

    # Deep comparison
    def diff(o, r, path=""):
        if type(o) != type(r):
            print(f"  {path}: type {type(o).__name__} vs {type(r).__name__}")
            return
        if isinstance(o, dict):
            o_keys = set(o.keys())
            r_keys = set(r.keys())
            if o_keys != r_keys:
                missing = o_keys - r_keys
                extra = r_keys - o_keys
                if missing:
                    print(f"  {path}: MISSING keys: {missing}")
                if extra:
                    print(f"  {path}: EXTRA keys: {extra}")
                # Show what the extra/missing keys look like
                for k in missing:
                    print(f"    orig[{k!r}] = {repr(o[k])[:100]}")
                for k in extra:
                    print(f"    rest[{k!r}] = {repr(r[k])[:100]}")
                return
            for k in o:
                if o[k] != r[k]:
                    diff(o[k], r[k], f"{path}.{k}")
                    return  # show first diff only per level
        elif isinstance(o, list):
            if len(o) != len(r):
                print(f"  {path}: list len {len(o)} vs {len(r)}")
                return
            for i, (a, b) in enumerate(zip(o, r)):
                if a != b:
                    diff(a, b, f"{path}[{i}]")
                    return
        else:
            print(f"  {path}: {o!r} vs {r!r}")

    if orig == rest:
        print("  OK (this should not be failing)")
    else:
        diff(orig, rest, "root")

    # Also inspect tomlkit body structure for the problematic table
    print(f"\n  --- tomlkit body structure (top-level) ---")
    body = doc.body
    for key, item in body:
        if key is not None:
            kstr = _unwrap_key(key)
            print(f"    key={kstr!r} type={type(item).__name__}", end="")
            if isinstance(item, toml_items.Table):
                inner_body = _get_body(item)
                inner_keys = [_unwrap_key(k) for k, v in inner_body if k is not None]
                # Also check dict-like access
                dict_keys = list(item.keys()) if hasattr(item, 'keys') else []
                print(f"  body_keys={inner_keys}  dict_keys={dict_keys}")
            else:
                print()
