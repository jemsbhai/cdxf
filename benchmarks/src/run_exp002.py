"""EXP-002: Encode/Decode Throughput Benchmark.

Measures wall-clock time for encoding and decoding across all baselines
with statistical rigor: 100 iterations, warm-up, GC control, 95% CI.

Usage:
    python benchmarks/src/run_exp002.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
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

from cdxf.codec import encode as cdxf_encode, decode as cdxf_decode
from cdxf.bridges.json_bridge import from_json, to_json
from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
from cdxf.bridges.xml_bridge import from_xml, to_xml
from cdxf.bridges.toml_bridge import from_toml, to_toml

DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "exp_002"

WARMUP = 5
ITERATIONS = 100


# ===================================================================
# Timing
# ===================================================================

def bench(func, iterations: int = ITERATIONS, warmup: int = WARMUP) -> list[float]:
    """Time a function over multiple iterations. Returns list of seconds."""
    # Warm-up
    for _ in range(warmup):
        func()

    # Measure
    gc.disable()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e9)
    gc.enable()
    return times


def summarize(times: list[float]) -> dict:
    """Compute summary statistics."""
    n = len(times)
    med = statistics.median(times)
    avg = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0.0
    # 95% CI using t-distribution approximation
    ci_half = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "median_s": round(med, 9),
        "mean_s": round(avg, 9),
        "std_s": round(std, 9),
        "ci95_lower_s": round(avg - ci_half, 9),
        "ci95_upper_s": round(avg + ci_half, 9),
        "min_s": round(min(times), 9),
        "max_s": round(max(times), 9),
    }


# ===================================================================
# Format detection and native parsing
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


def to_native(text: str, fmt: str):
    """Parse to Python-native types for baseline encoding."""
    try:
        if fmt == "json":
            return json.loads(text)
        if fmt == "yaml":
            from ruamel.yaml import YAML
            y = YAML(typ="safe")
            docs = list(y.load_all(text))
            return docs[0] if len(docs) == 1 else docs
        if fmt == "toml":
            import tomlkit
            return tomlkit.parse(text).unwrap()
    except Exception:
        return None
    return None


# ===================================================================
# Benchmark runners for each baseline
# ===================================================================

def bench_cdxf_full(text: str, fmt: str) -> dict:
    """Benchmark full pipeline: text -> CDXF -> binary and binary -> CDXF -> text."""
    bridges_from = {"json": from_json, "yaml": from_yaml,
                    "xml": from_xml, "toml": from_toml}
    bridges_to = {"json": to_json, "yaml": to_yaml,
                  "xml": to_xml, "toml": to_toml}

    bridge_from = bridges_from.get(fmt)
    bridge_to = bridges_to.get(fmt)
    if not bridge_from or not bridge_to:
        return {}

    # Pre-parse for decode benchmark
    stream = bridge_from(text)
    binary = cdxf_encode(stream)

    enc_times = bench(lambda: cdxf_encode(bridge_from(text)))
    dec_times = bench(lambda: bridge_to(cdxf_decode(binary)))

    return {
        "encode": summarize(enc_times),
        "decode": summarize(dec_times),
    }


def bench_cdxf_codec(text: str, fmt: str) -> dict:
    """Benchmark codec only: Stream -> binary and binary -> Stream."""
    bridges_from = {"json": from_json, "yaml": from_yaml,
                    "xml": from_xml, "toml": from_toml}
    bridge_from = bridges_from.get(fmt)
    if not bridge_from:
        return {}

    stream = bridge_from(text)
    binary = cdxf_encode(stream)

    enc_times = bench(lambda: cdxf_encode(stream))
    dec_times = bench(lambda: cdxf_decode(binary))

    return {
        "encode": summarize(enc_times),
        "decode": summarize(dec_times),
    }


def bench_json_stdlib(text: str, fmt: str) -> dict:
    """Benchmark stdlib json encode/decode."""
    if fmt != "json":
        return {}
    native = json.loads(text)
    encoded = json.dumps(native).encode()

    enc_times = bench(lambda: json.dumps(native))
    dec_times = bench(lambda: json.loads(encoded))

    return {
        "encode": summarize(enc_times),
        "decode": summarize(dec_times),
    }


def bench_cbor(text: str, fmt: str) -> dict:
    native = to_native(text, fmt) if fmt != "xml" else None
    if native is None:
        return {}

    try:
        encoded = cbor2.dumps(native)
    except Exception:
        return {}

    enc_times = bench(lambda: cbor2.dumps(native))
    dec_times = bench(lambda: cbor2.loads(encoded))

    return {
        "encode": summarize(enc_times),
        "decode": summarize(dec_times),
    }


def bench_msgpack(text: str, fmt: str) -> dict:
    native = to_native(text, fmt) if fmt != "xml" else None
    if native is None:
        return {}

    try:
        encoded = msgpack.packb(native, use_bin_type=True, default=str)
    except Exception:
        return {}

    enc_times = bench(lambda: msgpack.packb(native, use_bin_type=True, default=str))
    dec_times = bench(lambda: msgpack.unpackb(encoded, raw=False))

    return {
        "encode": summarize(enc_times),
        "decode": summarize(dec_times),
    }


def bench_bson_baseline(text: str, fmt: str) -> dict:
    if not HAS_BSON:
        return {}
    native = to_native(text, fmt) if fmt != "xml" else None
    if native is None or not isinstance(native, dict):
        return {}

    try:
        encoded = bson.encode(native)
    except Exception:
        return {}

    enc_times = bench(lambda: bson.encode(native))
    dec_times = bench(lambda: bson.decode(encoded))

    return {
        "encode": summarize(enc_times),
        "decode": summarize(dec_times),
    }


def bench_ion(text: str, fmt: str) -> dict:
    if not HAS_ION:
        return {}
    native = to_native(text, fmt) if fmt != "xml" else None
    if native is None:
        return {}

    try:
        encoded = ion.dumps(native, binary=True)
    except Exception:
        return {}

    enc_times = bench(lambda: ion.dumps(native, binary=True))
    dec_times = bench(lambda: ion.loads(encoded))

    return {
        "encode": summarize(enc_times),
        "decode": summarize(dec_times),
    }


# ===================================================================
# Environment
# ===================================================================

def capture_environment() -> dict:
    import importlib.metadata
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "warmup": WARMUP,
        "iterations": ITERATIONS,
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
    print("EXP-002: Encode/Decode Throughput Benchmark")
    print(f"Warm-up: {WARMUP}, Iterations: {ITERATIONS}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = capture_environment()
    env_path = RESULTS_DIR / "environment.json"
    env_path.write_text(json.dumps(env, indent=2), encoding="utf-8")

    # Collect corpus
    corpus_files: list[tuple[Path, str]] = []
    for root, _dirs, files in os.walk(DATA_RAW):
        for fname in sorted(files):
            fpath = Path(root) / fname
            fmt = detect_format(fpath)
            if fmt:
                corpus_files.append((fpath, fmt))

    print(f"\nCorpus: {len(corpus_files)} files\n")

    baselines = [
        ("cdxf_full", bench_cdxf_full),
        ("cdxf_codec", bench_cdxf_codec),
        ("json_stdlib", bench_json_stdlib),
        ("cbor", bench_cbor),
        ("msgpack", bench_msgpack),
        ("bson", bench_bson_baseline),
        ("ion", bench_ion),
    ]

    results: list[dict] = []

    for fpath, fmt in corpus_files:
        rel = str(fpath.relative_to(DATA_RAW))
        text = fpath.read_text(encoding="utf-8", errors="replace")
        size = len(text.encode())

        print(f"  {rel} ({size:,}B, {fmt})...", end="", flush=True)

        file_result = {
            "file": rel,
            "format": fmt,
            "size_bytes": size,
            "baselines": {},
        }

        for bl_name, bl_func in baselines:
            gc.collect()
            try:
                bl_result = bl_func(text, fmt)
                if bl_result:
                    # Add throughput metrics
                    for op in ("encode", "decode"):
                        if op in bl_result:
                            med = bl_result[op]["median_s"]
                            if med > 0:
                                bl_result[op]["throughput_bytes_per_sec"] = round(
                                    size / med
                                )
                                bl_result[op]["ops_per_sec"] = round(1.0 / med, 1)
                    file_result["baselines"][bl_name] = bl_result
            except Exception as e:
                file_result["baselines"][bl_name] = {"error": str(e)}

        results.append(file_result)
        # Print summary for this file
        cdxf_enc = file_result["baselines"].get("cdxf_codec", {}).get("encode", {})
        cdxf_med = cdxf_enc.get("median_s", 0)
        ops = cdxf_enc.get("ops_per_sec", 0)
        print(f" cdxf_codec_enc={cdxf_med*1e6:.0f}us ({ops:.0f} ops/s)")

    # --- Aggregate summaries ---
    print(f"\n{'='*70}")
    print("Aggregate: median encode ops/s by baseline (JSON files only)")
    print("-" * 70)

    json_results = [r for r in results if r["format"] == "json"]
    for bl_name, _ in baselines:
        ops_list = []
        for r in json_results:
            bl = r["baselines"].get(bl_name, {})
            enc = bl.get("encode", {})
            ops = enc.get("ops_per_sec")
            if ops:
                ops_list.append(ops)
        if ops_list:
            med = statistics.median(ops_list)
            print(f"  {bl_name:<15} median={med:>12,.0f} ops/s  (n={len(ops_list)})")

    # Save
    output = {
        "experiment": "EXP-002",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "corpus_size": len(results),
        "results": results,
    }
    results_path = RESULTS_DIR / "exp_002_results.json"
    results_path.write_text(json.dumps(output, indent=2, default=str),
                            encoding="utf-8")
    print(f"\nResults saved: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
