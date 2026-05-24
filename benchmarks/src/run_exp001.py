"""EXP-001 Benchmark Runner — Size Efficiency & Round-Trip Fidelity.

Measures encoded size across all baselines and tests CDXF round-trip
fidelity for every corpus file. Outputs per-file JSON results.

Usage:
    python benchmarks/src/run_exp001.py
"""

from __future__ import annotations

import gzip
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- Setup paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cbor2
import msgpack
import zstandard

from cdxf.codec import encode, decode, Encoder
from cdxf.bridges.json_bridge import from_json, to_json
from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
from cdxf.bridges.xml_bridge import from_xml, to_xml
from cdxf.bridges.toml_bridge import from_toml, to_toml
from cdxf.model import Comment, Map, Sequence, Scalar, Element

# Optional baselines — fail gracefully if unavailable
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

DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "exp_001"


# ===================================================================
# Format detection
# ===================================================================

def detect_format(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext == ".json":
        return "json"
    if ext in (".yaml", ".yml"):
        return "yaml"
    if ext == ".xml":
        return "xml"
    if ext == ".toml":
        return "toml"
    return None


# ===================================================================
# Native parsing (for baselines that need Python-native types)
# ===================================================================

def to_native(text: str, fmt: str):
    """Parse text to Python native types for baseline encoding.

    Returns None if parsing fails or the format has no natural
    dict/list representation for baselines.
    """
    try:
        if fmt == "json":
            return json.loads(text)
        if fmt == "yaml":
            from ruamel.yaml import YAML
            yaml = YAML(typ="safe")
            docs = list(yaml.load_all(text))
            if len(docs) == 1:
                return docs[0]
            return docs  # multi-document → list for baselines
        if fmt == "toml":
            import tomlkit
            doc = tomlkit.parse(text)
            return _tomlkit_unwrap(doc.unwrap())
    except Exception:
        return None
    return None  # XML has no natural dict representation


def _tomlkit_unwrap(obj):
    """Recursively unwrap tomlkit types to plain Python types."""
    if isinstance(obj, dict):
        return {k: _tomlkit_unwrap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tomlkit_unwrap(v) for v in obj]
    return obj


# ===================================================================
# Size measurement
# ===================================================================

def safe_encode(func, data) -> int | None:
    """Try encoding; return byte count or None on failure."""
    try:
        result = func(data)
        if isinstance(result, (bytes, bytearray)):
            return len(result)
        return None
    except Exception:
        return None


def measure_sizes(text_bytes: bytes, text: str, fmt: str) -> dict:
    """Measure encoded sizes across all baselines for one file."""
    sizes: dict[str, int | None] = {}
    sizes["raw_text"] = len(text_bytes)

    # --- CDXF encoding ---
    try:
        if fmt == "json":
            stream = from_json(text)
        elif fmt == "yaml":
            stream = from_yaml(text)
        elif fmt == "xml":
            stream = from_xml(text)
        elif fmt == "toml":
            stream = from_toml(text)
        else:
            stream = None

        if stream:
            cdxf_bytes = encode(stream)
            sizes["cdxf_full"] = len(cdxf_bytes)

            # Shorthand mode (only meaningful for JSON-model data)
            if fmt == "json":
                encoder = Encoder(shorthand=True)
                cdxf_short = encoder.encode(stream)
                sizes["cdxf_shorthand"] = len(cdxf_short)
            else:
                sizes["cdxf_shorthand"] = None

            # Compressed CDXF
            sizes["gzip_cdxf"] = len(gzip.compress(cdxf_bytes, compresslevel=9))
            zctx = zstandard.ZstdCompressor(level=3)
            sizes["zstd_cdxf"] = len(zctx.compress(cdxf_bytes))
        else:
            sizes["cdxf_full"] = None
            sizes["cdxf_shorthand"] = None
            sizes["gzip_cdxf"] = None
            sizes["zstd_cdxf"] = None
    except Exception:
        sizes["cdxf_full"] = None
        sizes["cdxf_shorthand"] = None
        sizes["gzip_cdxf"] = None
        sizes["zstd_cdxf"] = None

    # --- Compression baselines ---
    sizes["gzip_text"] = len(gzip.compress(text_bytes, compresslevel=9))
    zctx = zstandard.ZstdCompressor(level=3)
    sizes["zstd_text"] = len(zctx.compress(text_bytes))

    # --- Binary format baselines (need native Python types) ---
    native = to_native(text, fmt) if fmt != "xml" else None

    # CBOR naive
    sizes["cbor_naive"] = safe_encode(cbor2.dumps, native) if native is not None else None

    # MessagePack
    sizes["msgpack"] = safe_encode(
        lambda d: msgpack.packb(d, use_bin_type=True, default=str), native
    ) if native is not None else None

    # BSON (requires top-level dict)
    if HAS_BSON and native is not None and isinstance(native, dict):
        sizes["bson"] = safe_encode(bson.encode, native)
    else:
        sizes["bson"] = None

    # Amazon Ion binary
    if HAS_ION and native is not None:
        sizes["ion_binary"] = safe_encode(
            lambda d: ion.dumps(d, binary=True), native
        )
    else:
        sizes["ion_binary"] = None

    return sizes


# ===================================================================
# Round-trip fidelity testing
# ===================================================================

def _semantic_equal_json(original_text: str, restored_text: str) -> bool:
    try:
        return json.loads(original_text) == json.loads(restored_text)
    except Exception:
        return False


def _semantic_equal_yaml(original_text: str, restored_text: str) -> bool:
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    try:
        orig_docs = list(yaml.load_all(original_text))
        rest_docs = list(yaml.load_all(restored_text))
        return orig_docs == rest_docs
    except Exception:
        return False


def _semantic_equal_xml(original_text: str, restored_text: str) -> bool:
    try:
        orig_stream = from_xml(original_text)
        rest_stream = from_xml(restored_text)
        return _compare_elements(orig_stream.documents[0].root,
                                 rest_stream.documents[0].root)
    except Exception:
        return False


def _compare_elements(a, b) -> bool:
    """Recursively compare two CDXF Element trees."""
    if type(a) != type(b):
        return False
    if isinstance(a, Element):
        if a.name != b.name or a.namespace_uri != b.namespace_uri:
            return False
        a_attrs = {(at.name, at.namespace_uri): at.value for at in a.attributes}
        b_attrs = {(at.name, at.namespace_uri): at.value for at in b.attributes}
        if a_attrs != b_attrs:
            return False
        a_children = [c for c in a.children if not isinstance(c, Comment)]
        b_children = [c for c in b.children if not isinstance(c, Comment)]
        if len(a_children) != len(b_children):
            return False
        return all(_compare_elements(ac, bc) for ac, bc in zip(a_children, b_children))
    if isinstance(a, Scalar):
        return a.value == b.value
    return True


def _semantic_equal_toml(original_text: str, restored_text: str) -> bool:
    import tomlkit
    try:
        orig = tomlkit.parse(original_text).unwrap()
        rest = tomlkit.parse(restored_text).unwrap()
        return orig == rest
    except Exception:
        return False


def test_fidelity(text: str, fmt: str) -> tuple[bool, str]:
    """Test full pipeline: text → CDXF → CBOR → CDXF → text → semantic eq.

    Returns (passed, detail) where detail explains failures.
    """
    try:
        # Parse to CDXF
        if fmt == "json":
            stream = from_json(text)
        elif fmt == "yaml":
            stream = from_yaml(text)
        elif fmt == "xml":
            stream = from_xml(text)
        elif fmt == "toml":
            stream = from_toml(text)
        else:
            return False, "unknown format"

        # Encode to binary
        binary = encode(stream)

        # Decode back to CDXF
        restored = decode(binary)

        # Convert back to text
        if fmt == "json":
            output = to_json(restored)
            ok = _semantic_equal_json(text, output)
        elif fmt == "yaml":
            output = to_yaml(restored)
            ok = _semantic_equal_yaml(text, output)
        elif fmt == "xml":
            output = to_xml(restored)
            ok = _semantic_equal_xml(text, output)
        elif fmt == "toml":
            output = to_toml(restored)
            ok = _semantic_equal_toml(text, output)
        else:
            return False, "unknown format"

        if ok:
            return True, "ok"
        else:
            return False, "semantic mismatch after round-trip"

    except Exception as e:
        return False, f"exception: {type(e).__name__}: {e}"


# ===================================================================
# Environment snapshot
# ===================================================================

def capture_environment() -> dict:
    import importlib.metadata
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    for pkg in ["cbor2", "msgpack", "ruamel.yaml", "tomlkit",
                 "zstandard", "pymongo", "amazon.ion"]:
        try:
            env[f"pkg_{pkg}"] = importlib.metadata.version(pkg)
        except Exception:
            env[f"pkg_{pkg}"] = "not installed"
    return env


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("EXP-001: Size Efficiency & Round-Trip Fidelity Benchmark")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Capture environment
    env = capture_environment()
    env_path = RESULTS_DIR / "environment.json"
    env_path.write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"\nEnvironment snapshot: {env_path}")

    # Collect all corpus files
    corpus_files: list[tuple[Path, str]] = []
    for root, _dirs, files in os.walk(DATA_RAW):
        for fname in sorted(files):
            fpath = Path(root) / fname
            fmt = detect_format(fpath)
            if fmt:
                corpus_files.append((fpath, fmt))

    print(f"Corpus: {len(corpus_files)} files\n")

    # Run measurements
    results: list[dict] = []
    passed = 0
    failed = 0
    fail_details: list[str] = []

    for fpath, fmt in corpus_files:
        rel = fpath.relative_to(DATA_RAW)
        text_bytes = fpath.read_bytes()
        text = text_bytes.decode("utf-8", errors="replace")

        # Sizes
        sizes = measure_sizes(text_bytes, text, fmt)

        # Fidelity
        fidelity, detail = test_fidelity(text, fmt)

        if fidelity:
            passed += 1
            fidelity_str = "PASS"
        else:
            failed += 1
            fidelity_str = "FAIL"
            fail_details.append(f"    {rel}: {detail}")

        # Compute ratios
        raw = sizes["raw_text"]
        ratios = {}
        for key, val in sizes.items():
            if key != "raw_text" and val is not None:
                ratios[f"ratio_{key}"] = round(val / raw, 4)

        cbor_naive = sizes.get("cbor_naive")
        if cbor_naive and sizes.get("cdxf_full"):
            ratios["cdxf_overhead_vs_cbor"] = round(
                sizes["cdxf_full"] / cbor_naive, 4
            )

        record = {
            "file": str(rel),
            "format": fmt,
            "sizes": sizes,
            "ratios": ratios,
            "fidelity": fidelity,
            "fidelity_detail": detail,
        }
        results.append(record)

        # Print progress
        cdxf_ratio = ratios.get("ratio_cdxf_full", "N/A")
        print(f"  [{fidelity_str}] {rel}  "
              f"(text={raw:,}B, cdxf_ratio={cdxf_ratio})")

    # --- Aggregate statistics ---
    print(f"\n{'=' * 70}")
    print(f"Round-trip fidelity: {passed}/{passed+failed} passed")

    if fail_details:
        print(f"\n  Failures ({len(fail_details)}):")
        for d in fail_details:
            print(d)

    # Per-format size ratio summaries
    for fmt_name in ["json", "yaml", "xml", "toml"]:
        fmt_results = [r for r in results if r["format"] == fmt_name]
        if not fmt_results:
            continue
        cdxf_ratios = [
            r["ratios"]["ratio_cdxf_full"]
            for r in fmt_results
            if "ratio_cdxf_full" in r["ratios"]
        ]
        if cdxf_ratios:
            med = statistics.median(cdxf_ratios)
            avg = statistics.mean(cdxf_ratios)
            std = statistics.stdev(cdxf_ratios) if len(cdxf_ratios) > 1 else 0
            print(f"\n  {fmt_name.upper()} CDXF/text ratio "
                  f"(n={len(cdxf_ratios)}): "
                  f"median={med:.3f}, mean={avg:.3f}, std={std:.3f}")

    # --- Save results ---
    results_path = RESULTS_DIR / "exp_001_results.json"
    output = {
        "experiment": "EXP-001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "corpus_size": len(results),
        "fidelity_passed": passed,
        "fidelity_failed": failed,
        "results": results,
    }
    results_path.write_text(json.dumps(output, indent=2, default=str),
                            encoding="utf-8")
    print(f"\nResults saved: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
