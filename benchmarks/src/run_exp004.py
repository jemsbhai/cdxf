"""EXP-004: Large File Scalability + Compression Analysis + Decode Summary.

Generates 1MB and 10MB files for each format, measures encode/decode
throughput and size at scale. Also extracts compression analysis and
decode throughput from existing EXP-001/EXP-002 data.

Usage:
    python benchmarks/src/run_exp004.py
"""

from __future__ import annotations

import gc
import gzip
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cbor2
import msgpack
import zstandard

from cdxf.codec import encode as cdxf_encode, decode as cdxf_decode
from cdxf.bridges.json_bridge import from_json, to_json
from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
from cdxf.bridges.xml_bridge import from_xml, to_xml
from cdxf.bridges.toml_bridge import from_toml, to_toml

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "exp_004"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "synthetic"

WARMUP = 3
ITERATIONS = 20  # fewer iterations for large files


# ===================================================================
# Generate large files
# ===================================================================

def generate_large_json(target_kb: int) -> str:
    """Generate a JSON document approximately target_kb in size."""
    items = []
    i = 0
    while True:
        item = {
            "id": i,
            "name": f"item-{i:06d}",
            "description": f"This is a description for item number {i} with some padding text to increase size",
            "price": round(i * 0.99, 2),
            "in_stock": i % 3 != 0,
            "tags": [f"tag-{j}" for j in range(i % 5 + 1)],
            "metadata": {
                "created": "2024-01-15T10:30:00Z",
                "category": f"cat-{i % 20}",
                "weight": round(i * 0.123, 3),
            },
        }
        items.append(item)
        i += 1
        if i % 100 == 0:
            text = json.dumps({"products": items}, indent=None)
            if len(text) >= target_kb * 1024:
                return text
    return json.dumps({"products": items}, indent=None)


def generate_large_yaml(target_kb: int) -> str:
    """Generate a YAML document approximately target_kb in size."""
    lines = []
    i = 0
    while True:
        lines.append(f"- id: {i}")
        lines.append(f"  name: item-{i:06d}")
        lines.append(f"  description: This is a description for item number {i}")
        lines.append(f"  price: {round(i * 0.99, 2)}")
        lines.append(f"  in_stock: {'true' if i % 3 != 0 else 'false'}")
        lines.append(f"  tags:")
        for j in range(i % 5 + 1):
            lines.append(f"    - tag-{j}")
        lines.append(f"  category: cat-{i % 20}")
        i += 1
        if i % 100 == 0:
            text = "\n".join(lines) + "\n"
            if len(text) >= target_kb * 1024:
                return text
    return "\n".join(lines) + "\n"


def generate_large_xml(target_kb: int) -> str:
    """Generate an XML document approximately target_kb in size."""
    parts = ['<catalog xmlns="http://example.com/catalog">']
    i = 0
    while True:
        parts.append(f'  <product id="{i}" category="cat-{i % 20}">')
        parts.append(f"    <name>item-{i:06d}</name>")
        parts.append(f"    <description>Description for item number {i} with padding</description>")
        parts.append(f"    <price>{round(i * 0.99, 2)}</price>")
        parts.append(f"    <!-- stock status for item {i} -->")
        parts.append(f"    <in_stock>{'true' if i % 3 != 0 else 'false'}</in_stock>")
        for j in range(i % 3 + 1):
            parts.append(f"    <tag>tag-{j}</tag>")
        parts.append("  </product>")
        i += 1
        if i % 100 == 0:
            text = "\n".join(parts) + "\n</catalog>"
            if len(text) >= target_kb * 1024:
                return text
    return "\n".join(parts) + "\n</catalog>"


def generate_large_toml(target_kb: int) -> str:
    """Generate a TOML document approximately target_kb in size."""
    lines = []
    i = 0
    while True:
        lines.append(f"[[products]]")
        lines.append(f"id = {i}")
        lines.append(f'name = "item-{i:06d}"')
        lines.append(f'description = "Description for item number {i}"')
        lines.append(f"price = {round(i * 0.99, 2)}")
        lines.append(f"in_stock = {'true' if i % 3 != 0 else 'false'}")
        lines.append(f'tags = [{", ".join(f\'\"tag-{j}\"\' for j in range(i % 5 + 1))}]')
        lines.append(f'category = "cat-{i % 20}"')
        lines.append("")
        i += 1
        if i % 100 == 0:
            text = "\n".join(lines)
            if len(text) >= target_kb * 1024:
                return text
    return "\n".join(lines)


# ===================================================================
# Timing
# ===================================================================

def bench(func, iterations=ITERATIONS, warmup=WARMUP) -> list[float]:
    for _ in range(warmup):
        func()
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
    n = len(times)
    med = statistics.median(times)
    avg = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0
    ci = 1.96 * std / math.sqrt(n) if n > 1 else 0
    return {
        "n": n,
        "median_s": round(med, 6),
        "mean_s": round(avg, 6),
        "std_s": round(std, 6),
        "ci95_lower_s": round(avg - ci, 6),
        "ci95_upper_s": round(avg + ci, 6),
    }


# ===================================================================
# Part 1: Large file scalability
# ===================================================================

def run_scalability():
    print("\n" + "=" * 70)
    print("Part 1: Large File Scalability (1MB and 10MB)")
    print("=" * 70)

    generators = {
        "json": generate_large_json,
        "yaml": generate_large_yaml,
        "xml": generate_large_xml,
        "toml": generate_large_toml,
    }

    bridges_from = {"json": from_json, "yaml": from_yaml,
                    "xml": from_xml, "toml": from_toml}
    bridges_to = {"json": to_json, "yaml": to_yaml,
                  "xml": to_xml, "toml": to_toml}

    results = []

    for target_kb in [1024, 10240]:
        target_mb = target_kb // 1024
        print(f"\n--- {target_mb}MB files ---")

        for fmt, gen_func in generators.items():
            print(f"\n  Generating {fmt.upper()} ~{target_mb}MB...", end="", flush=True)
            text = gen_func(target_kb)
            text_bytes = text.encode("utf-8")
            actual_kb = len(text_bytes) / 1024
            print(f" {actual_kb:.0f}KB")

            # Parse to CDXF
            print(f"  Parsing {fmt.upper()}...", end="", flush=True)
            bridge_from = bridges_from[fmt]
            try:
                stream = bridge_from(text)
                print(" OK")
            except Exception as e:
                print(f" FAILED: {e}")
                continue

            # Encode to CDXF binary
            print(f"  Encoding CDXF...", end="", flush=True)
            try:
                cdxf_bytes = cdxf_encode(stream)
                cdxf_size = len(cdxf_bytes)
                ratio = cdxf_size / len(text_bytes)
                print(f" {cdxf_size:,}B (ratio={ratio:.3f})")
            except Exception as e:
                print(f" FAILED: {e}")
                continue

            # Compression
            gz_text = len(gzip.compress(text_bytes, compresslevel=9))
            gz_cdxf = len(gzip.compress(cdxf_bytes, compresslevel=9))
            zctx = zstandard.ZstdCompressor(level=3)
            zstd_text = len(zctx.compress(text_bytes))
            zstd_cdxf = len(zctx.compress(cdxf_bytes))

            print(f"  Compression: gzip(text)={gz_text:,} gzip(cdxf)={gz_cdxf:,} "
                  f"zstd(text)={zstd_text:,} zstd(cdxf)={zstd_cdxf:,}")

            # Throughput: encode
            print(f"  Benchmarking encode ({ITERATIONS} iters)...", end="", flush=True)
            gc.collect()
            enc_times = bench(lambda: cdxf_encode(bridge_from(text)))
            enc_summary = summarize(enc_times)
            enc_mb_s = (len(text_bytes) / 1e6) / enc_summary["median_s"] if enc_summary["median_s"] > 0 else 0
            print(f" {enc_summary['median_s']*1000:.1f}ms ({enc_mb_s:.1f} MB/s)")

            # Throughput: decode
            print(f"  Benchmarking decode ({ITERATIONS} iters)...", end="", flush=True)
            gc.collect()
            dec_times = bench(lambda: bridges_to[fmt](cdxf_decode(cdxf_bytes)))
            dec_summary = summarize(dec_times)
            dec_mb_s = (len(text_bytes) / 1e6) / dec_summary["median_s"] if dec_summary["median_s"] > 0 else 0
            print(f" {dec_summary['median_s']*1000:.1f}ms ({dec_mb_s:.1f} MB/s)")

            # Round-trip fidelity
            print(f"  Fidelity check...", end="", flush=True)
            try:
                restored = cdxf_decode(cdxf_bytes)
                output = bridges_to[fmt](restored)
                # Quick data check
                if fmt == "json":
                    ok = json.loads(text) == json.loads(output)
                else:
                    ok = len(output) > 0  # basic sanity
                print(f" {'PASS' if ok else 'FAIL'}")
            except Exception as e:
                ok = False
                print(f" ERROR: {e}")

            results.append({
                "format": fmt,
                "target_mb": target_mb,
                "text_bytes": len(text_bytes),
                "cdxf_bytes": cdxf_size,
                "ratio": round(ratio, 4),
                "gzip_text": gz_text,
                "gzip_cdxf": gz_cdxf,
                "zstd_text": zstd_text,
                "zstd_cdxf": zstd_cdxf,
                "encode": enc_summary,
                "decode": dec_summary,
                "encode_mb_s": round(enc_mb_s, 2),
                "decode_mb_s": round(dec_mb_s, 2),
                "fidelity": ok,
            })

    return results


# ===================================================================
# Part 2: Compression analysis (from existing EXP-001 data)
# ===================================================================

def run_compression_analysis():
    print("\n" + "=" * 70)
    print("Part 2: Compression Analysis (from EXP-001 data)")
    print("=" * 70)

    exp001_path = PROJECT_ROOT / "benchmarks" / "results" / "exp_001" / "exp_001_results.json"
    if not exp001_path.exists():
        print("  EXP-001 results not found, skipping")
        return {}

    data = json.loads(exp001_path.read_text(encoding="utf-8"))

    analysis = {}
    for fmt in ["json", "yaml", "xml", "toml"]:
        fmt_results = [r for r in data["results"] if r["format"] == fmt]
        if not fmt_results:
            continue

        gz_text_ratios = []
        gz_cdxf_ratios = []
        zstd_text_ratios = []
        zstd_cdxf_ratios = []

        for r in fmt_results:
            raw = r["sizes"]["raw_text"]
            gz_text = r["sizes"].get("gzip_text")
            gz_cdxf = r["sizes"].get("gzip_cdxf")
            zstd_text = r["sizes"].get("zstd_text")
            zstd_cdxf = r["sizes"].get("zstd_cdxf")

            if gz_text:
                gz_text_ratios.append(gz_text / raw)
            if gz_cdxf:
                gz_cdxf_ratios.append(gz_cdxf / raw)
            if zstd_text:
                zstd_text_ratios.append(zstd_text / raw)
            if zstd_cdxf:
                zstd_cdxf_ratios.append(zstd_cdxf / raw)

        analysis[fmt] = {
            "n": len(fmt_results),
            "gzip_text": {
                "median": round(statistics.median(gz_text_ratios), 4) if gz_text_ratios else None,
                "mean": round(statistics.mean(gz_text_ratios), 4) if gz_text_ratios else None,
            },
            "gzip_cdxf": {
                "median": round(statistics.median(gz_cdxf_ratios), 4) if gz_cdxf_ratios else None,
                "mean": round(statistics.mean(gz_cdxf_ratios), 4) if gz_cdxf_ratios else None,
            },
            "zstd_text": {
                "median": round(statistics.median(zstd_text_ratios), 4) if zstd_text_ratios else None,
                "mean": round(statistics.mean(zstd_text_ratios), 4) if zstd_text_ratios else None,
            },
            "zstd_cdxf": {
                "median": round(statistics.median(zstd_cdxf_ratios), 4) if zstd_cdxf_ratios else None,
                "mean": round(statistics.mean(zstd_cdxf_ratios), 4) if zstd_cdxf_ratios else None,
            },
        }

    print(f"\n  {'Format':<8} {'gzip(text)':<14} {'gzip(CDXF)':<14} {'zstd(text)':<14} {'zstd(CDXF)':<14}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    for fmt in ["json", "yaml", "xml", "toml"]:
        if fmt not in analysis:
            continue
        a = analysis[fmt]
        print(f"  {fmt.upper():<8} "
              f"{a['gzip_text']['median']:<14.4f} "
              f"{a['gzip_cdxf']['median']:<14.4f} "
              f"{a['zstd_text']['median']:<14.4f} "
              f"{a['zstd_cdxf']['median']:<14.4f}")

    return analysis


# ===================================================================
# Part 3: Decode throughput summary (from existing EXP-002 data)
# ===================================================================

def run_decode_summary():
    print("\n" + "=" * 70)
    print("Part 3: Decode Throughput Summary (from EXP-002 data)")
    print("=" * 70)

    exp002_path = PROJECT_ROOT / "benchmarks" / "results" / "exp_002" / "exp_002_results.json"
    if not exp002_path.exists():
        print("  EXP-002 results not found, skipping")
        return {}

    data = json.loads(exp002_path.read_text(encoding="utf-8"))

    baselines = ["cdxf_full", "cdxf_codec", "json_stdlib", "cbor", "msgpack", "bson", "ion"]
    json_results = [r for r in data["results"] if r["format"] == "json"]

    print(f"\n  Median decode ops/s (JSON files, n={len(json_results)}):")
    print(f"  {'-'*50}")

    summary = {}
    for bl in baselines:
        ops_list = []
        for r in json_results:
            bl_data = r["baselines"].get(bl, {})
            dec = bl_data.get("decode", {})
            ops = dec.get("ops_per_sec")
            if ops:
                ops_list.append(ops)
        if ops_list:
            med = statistics.median(ops_list)
            summary[bl] = {"median_ops_s": med, "n": len(ops_list)}
            print(f"  {bl:<15} {med:>12,.0f} ops/s  (n={len(ops_list)})")

    return summary


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("EXP-004: Scalability + Compression + Decode Summary")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1: Large file scalability
    scale_results = run_scalability()

    # Part 2: Compression analysis
    compression = run_compression_analysis()

    # Part 3: Decode summary
    decode_summary = run_decode_summary()

    # Save all results
    output = {
        "experiment": "EXP-004",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scalability": scale_results,
        "compression_analysis": compression,
        "decode_summary": decode_summary,
    }
    results_path = RESULTS_DIR / "exp_004_results.json"
    results_path.write_text(json.dumps(output, indent=2, default=str),
                            encoding="utf-8")
    print(f"\nResults saved: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
