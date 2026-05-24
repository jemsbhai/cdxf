"""Corpus collection script for EXP-001.

Downloads Tier 1 (Viotti SOTA) and Tier 2 (canonical) benchmark documents.
Generates Tier 3 (synthetic) stress-test documents.
Computes SHA-256 checksums for all files.

Usage:
    python benchmarks/src/collect_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
CHECKSUMS_FILE = PROJECT_ROOT / "data" / "checksums.sha256"

# ---------------------------------------------------------------------------
# Tier 1: Viotti SchemaStore benchmark (arXiv:2201.03051)
# Commit hash: 0b6bd2a08005e6f7a65a68acaf3064d6e2670872
# ---------------------------------------------------------------------------

VIOTTI_BASE = (
    "https://raw.githubusercontent.com/jviotti/binary-json-size-benchmark"
    "/main/benchmark"
)

VIOTTI_DOCUMENTS = [
    ("circleciblank",         "CircleCI Definition (Blank)"),
    ("circlecimatrix",        "CircleCI Matrix Definition"),
    ("commitlint",            "CommitLint Configuration"),
    ("commitlintbasic",       "CommitLint Configuration (Basic)"),
    ("epr",                   "Entry Point Regulation Manifest"),
    ("eslintrc",              "ESLint Configuration Document"),
    ("esmrc",                 "ECMAScript Module Loader Definition"),
    ("geojson",               "GeoJSON Example Document"),
    ("githubfundingblank",    "GitHub FUNDING Sponsorship Definition"),
    ("githubworkflow",        "GitHub Workflow Definition"),
    ("gruntcontribclean",     "Grunt.js Clean Task Definition"),
    ("imageoptimizerwebjob",  "ImageOptimizer Azure Webjob Config"),
    ("jsonereversesort",      "JSON-e Templating Engine Reverse Sort"),
    ("jsonesort",             "JSON-e Templating Engine Sort"),
    ("jsonfeed",              "JSON Feed Example Document"),
    ("jsonresume",            "JSON Resume Example"),
    ("netcoreproject",        ".NET Core Project"),
    ("nightwatch",            "Nightwatch.js Test Framework Config"),
    ("openweathermap",        "OpenWeatherMap API Example"),
    ("openweatherroadrisk",   "OpenWeather Road Risk API Example"),
    ("packagejson",           "NPM Package.json Example Manifest"),
    ("packagejsonlintrc",     "NPM Package.json Linter Config"),
    ("sapcloudsdkpipeline",   "SAP Cloud SDK Pipeline Config"),
    ("travisnotifications",   "TravisCI Notifications Configuration"),
    ("tslintbasic",           "TSLint Linter Definition (Basic)"),
    ("tslintextend",          "TSLint Linter Definition (Extends Only)"),
    ("tslintmulti",           "TSLint Linter Definition (Multi-rule)"),
]

# ---------------------------------------------------------------------------
# Tier 2: Canonical real-world documents
# ---------------------------------------------------------------------------

TIER2_DOWNLOADS: list[tuple[str, str, str]] = [
    # (url, relative_path, description)

    # YAML — Kubernetes examples
    (
        "https://raw.githubusercontent.com/kubernetes/website/main/content"
        "/en/examples/controllers/nginx-deployment.yaml",
        "yaml/tier2_kubernetes/nginx-deployment.yaml",
        "Kubernetes nginx Deployment manifest",
    ),
    (
        "https://raw.githubusercontent.com/kubernetes/website/main/content"
        "/en/examples/service/load-balancer-example.yaml",
        "yaml/tier2_kubernetes/load-balancer-service.yaml",
        "Kubernetes LoadBalancer Service manifest",
    ),

    # XML — Maven POM (Spring Boot — a widely used real-world POM)
    (
        "https://raw.githubusercontent.com/spring-projects/spring-boot"
        "/main/spring-boot-project/spring-boot/pom.xml",
        "xml/tier2_canonical/spring-boot-pom.xml",
        "Spring Boot Maven POM",
    ),

    # XML — W3C Atom example
    (
        "https://www.w3.org/2005/Atom",
        "xml/tier2_canonical/atom-namespace.xml",
        "W3C Atom namespace document",
    ),

    # TOML — popular Rust crates
    (
        "https://raw.githubusercontent.com/serde-rs/serde/master/Cargo.toml",
        "toml/tier2_canonical/serde-cargo.toml",
        "serde Cargo.toml (top Rust crate)",
    ),
    (
        "https://raw.githubusercontent.com/tokio-rs/tokio/master/Cargo.toml",
        "toml/tier2_canonical/tokio-cargo.toml",
        "tokio Cargo.toml (top Rust crate)",
    ),
    (
        "https://raw.githubusercontent.com/astral-sh/ruff/main/pyproject.toml",
        "toml/tier2_canonical/ruff-pyproject.toml",
        "ruff pyproject.toml (popular Python tool)",
    ),
]

# ---------------------------------------------------------------------------
# Tier 3: Synthetic stress-test documents (deterministic generation)
# ---------------------------------------------------------------------------


def _generate_synthetics(out_dir: Path) -> list[tuple[str, str]]:
    """Generate synthetic documents. Returns list of (path, description)."""
    files: list[tuple[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON: deeply nested ---
    def _nested_json(depth: int) -> dict:
        if depth == 0:
            return {"leaf": True}
        return {"level": depth, "child": _nested_json(depth - 1)}

    p = out_dir / "json_nested_128.json"
    p.write_text(json.dumps(_nested_json(128), indent=None), encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "JSON: 128 levels deep nesting"))

    # --- JSON: flat 10k-element array ---
    p = out_dir / "json_flat_10k_array.json"
    p.write_text(json.dumps(list(range(10000)), indent=None), encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "JSON: flat array with 10,000 integers"))

    # --- JSON: large object with many keys ---
    p = out_dir / "json_wide_object_1k.json"
    obj = {f"key_{i:04d}": f"value_{i}" for i in range(1000)}
    p.write_text(json.dumps(obj, indent=None), encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "JSON: object with 1,000 string keys"))

    # --- YAML: comment-dense (50%+ lines are comments) ---
    p = out_dir / "yaml_comment_dense.yaml"
    lines = []
    for i in range(200):
        lines.append(f"# Comment line {i}: this is important context")
        lines.append(f"key_{i:03d}: value_{i}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "YAML: 200 keys with 200 comments (50% comment lines)"))

    # --- YAML: anchor-heavy DAG ---
    p = out_dir / "yaml_anchor_heavy.yaml"
    lines = ["defaults: &defaults", "  timeout: 30", "  retries: 3", ""]
    for i in range(50):
        lines.append(f"service_{i:02d}:")
        lines.append(f"  <<: *defaults")
        lines.append(f"  name: service-{i}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "YAML: 50 services sharing one anchor via merge keys"))

    # --- YAML: multi-document stream ---
    p = out_dir / "yaml_multi_document.yaml"
    docs = []
    for i in range(20):
        docs.append(f"---\nid: {i}\nname: document-{i}\ndata:\n  - {i*10}\n  - {i*10+1}")
    p.write_text("\n".join(docs) + "\n", encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "YAML: 20-document stream"))

    # --- XML: namespace-heavy ---
    p = out_dir / "xml_namespace_heavy.xml"
    ns_decls = " ".join(f'xmlns:ns{i}="http://example.com/ns{i}"' for i in range(10))
    children = "\n".join(
        f'  <ns{i % 10}:item ns{(i+1) % 10}:id="{i}">content-{i}</ns{i % 10}:item>'
        for i in range(100)
    )
    xml_text = f'<root {ns_decls}>\n{children}\n</root>'
    p.write_text(xml_text, encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "XML: 10 namespace prefixes, 100 elements"))

    # --- XML: mixed content intensive ---
    p = out_dir / "xml_mixed_content.xml"
    paragraphs = "\n".join(
        f'  <p>Text before <em>emphasis {i}</em> and <strong>bold {i}</strong> after.</p>'
        for i in range(50)
    )
    xml_text = f'<article>\n{paragraphs}\n</article>'
    p.write_text(xml_text, encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "XML: 50 paragraphs with mixed content"))

    # --- XML: comment-heavy ---
    p = out_dir / "xml_comment_heavy.xml"
    children = "\n".join(
        f'  <!-- Configuration for item {i} -->\n  <item id="{i}">value-{i}</item>'
        for i in range(100)
    )
    xml_text = f'<!-- Master config file -->\n<config>\n{children}\n</config>'
    p.write_text(xml_text, encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "XML: 101 comments, 100 elements"))

    # --- TOML: all temporal types ---
    p = out_dir / "toml_temporal_types.toml"
    lines = [
        'title = "Temporal type stress test"',
        "",
        "[[events]]",
        'name = "conference"',
        "start_utc = 2026-06-15T09:00:00Z",
        "start_local = 2026-06-15T09:00:00",
        "date_only = 2026-06-15",
        "time_only = 09:00:00",
        "",
    ]
    for i in range(50):
        h = i % 24
        lines.append("[[events]]")
        lines.append(f'name = "event-{i}"')
        lines.append(f"start_utc = 2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T{h:02d}:00:00+05:30")
        lines.append(f"start_local = 2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T{h:02d}:00:00")
        lines.append(f"date_only = 2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}")
        lines.append(f"time_only = {h:02d}:00:00")
        lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
    files.append((str(p.relative_to(DATA_RAW)), "TOML: 51 events with all 4 temporal types"))

    return files


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> bool:
    """Download a URL to a local path. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CDXF-Benchmark/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"  WARN: failed to download {url}: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CDXF Benchmark Corpus Collection — EXP-001")
    print("=" * 60)

    checksums: list[str] = []
    manifest_rows: list[tuple[str, str, str]] = []  # (path, source, description)

    # --- Tier 1: Viotti SchemaStore ---
    print("\n--- Tier 1: Viotti SchemaStore benchmark (27 documents) ---")
    t1_dir = DATA_RAW / "json" / "tier1_schemastore"
    t1_dir.mkdir(parents=True, exist_ok=True)

    for slug, desc in VIOTTI_DOCUMENTS:
        url = f"{VIOTTI_BASE}/{slug}/document.json"
        dest = t1_dir / f"{slug}.json"
        print(f"  Downloading {slug}...", end=" ")
        if download(url, dest):
            cs = sha256_file(dest)
            checksums.append(f"{cs}  {dest.relative_to(PROJECT_ROOT)}")
            manifest_rows.append((str(dest.relative_to(DATA_RAW)), url, desc))
            size = dest.stat().st_size
            print(f"OK ({size:,} bytes)")
        else:
            print("FAILED")

    # --- Tier 2: Canonical real-world ---
    print("\n--- Tier 2: Canonical real-world documents ---")
    for url, rel_path, desc in TIER2_DOWNLOADS:
        dest = DATA_RAW / rel_path
        print(f"  Downloading {rel_path}...", end=" ")
        if download(url, dest):
            cs = sha256_file(dest)
            checksums.append(f"{cs}  {dest.relative_to(PROJECT_ROOT)}")
            manifest_rows.append((rel_path, url, desc))
            size = dest.stat().st_size
            print(f"OK ({size:,} bytes)")
        else:
            print("FAILED")

    # --- Tier 3: Synthetic ---
    print("\n--- Tier 3: Synthetic stress-test documents ---")
    synth_dir = DATA_RAW / "synthetic"
    synth_files = _generate_synthetics(synth_dir)
    for rel_path, desc in synth_files:
        full = DATA_RAW / rel_path
        cs = sha256_file(full)
        checksums.append(f"{cs}  {full.relative_to(PROJECT_ROOT)}")
        manifest_rows.append((rel_path, "generated", desc))
        size = full.stat().st_size
        print(f"  Generated {rel_path} ({size:,} bytes)")

    # --- Write checksums ---
    print(f"\nWriting checksums to {CHECKSUMS_FILE}...")
    CHECKSUMS_FILE.write_text("\n".join(sorted(checksums)) + "\n", encoding="utf-8")

    # --- Summary ---
    total = len(manifest_rows)
    total_bytes = sum(
        (DATA_RAW / row[0]).stat().st_size
        for row in manifest_rows
        if (DATA_RAW / row[0]).exists()
    )
    print(f"\n{'=' * 60}")
    print(f"Corpus collection complete: {total} files, {total_bytes:,} bytes total")
    print(f"Checksums: {CHECKSUMS_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
