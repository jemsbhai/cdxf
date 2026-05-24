# CDXF Experimental Logbook

> **Append-only.** Never delete or modify past entries. If a correction is
> needed, add a dated addendum referencing the original entry.
>
> Every experiment gets a plan entry BEFORE execution and a results entry
> AFTER completion. Failed experiments are data — always logged.

---

## EXP-001: Size Efficiency & Round-Trip Fidelity Across Four Format Families

**Date:** 2026-05-24 (UTC-4, Melbourne FL)
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

### Hypothesis

CDXF's binary encoding adds minimal overhead (≤5% for JSON-model data via
shorthand mode, ≤15% for format-rich documents with CDXF semantic tags)
compared to source text size, while achieving 100% round-trip fidelity across
all four format families (JSON, YAML, XML, TOML) — a capability no existing
binary format provides.

Secondary hypothesis: CDXF compresses (gzip/zstd) comparably to or better than
the original text formats, because binary encodings eliminate redundant syntax
(braces, quotes, indentation).

### Independent Variables

- **Source format:** JSON, YAML, XML, TOML
- **Document complexity:** small (~100B), medium (~1KB), large (~100KB+)
- **Document richness:**
  - Plain data (maps, sequences, scalars only)
  - Format-rich (comments, anchors/aliases, namespaces, PIs, mixed content,
    typed scalars including dates/times)
- **Encoding mode:** CDXF full (all semantic tags) vs CDXF shorthand
  (byte-identical to CBOR for JSON-model data)

### Dependent Variables / Metrics

1. `size_bytes` — encoded size in bytes for each format/baseline
2. `size_ratio_vs_text` — encoded bytes / original text bytes
3. `size_ratio_vs_cbor` — CDXF bytes / naive CBOR bytes (JSON-model data only;
   measures CDXF semantic tag overhead)
4. `fidelity` — binary pass/fail: does
   `text → CDXF → CBOR → CDXF → text` produce semantically equivalent output?
5. `feature_preserved` — per-construct binary pass/fail across all baselines
   (see Feature Preservation Matrix)

### Control Conditions

**Baselines (all with Python implementations):**

| Baseline              | Library       | Measures                               |
|-----------------------|---------------|----------------------------------------|
| Raw text              | (identity)    | Original file size                     |
| CBOR (naive)          | `cbor2`       | Binary JSON sans CDXF info model       |
| MessagePack           | `msgpack`     | Most popular binary JSON format        |
| BSON                  | `bson`/pymongo| MongoDB's binary format                |
| Amazon Ion (binary)   | `amazon.ion`  | Closest expressiveness competitor      |
| gzip(text)            | stdlib `gzip` | Compression baseline                   |
| zstd(text)            | `zstandard`   | Modern compression baseline            |
| CDXF (full tags)      | ours          | Full information model                 |
| CDXF (shorthand)      | ours          | JSON-model data, minimal tags          |
| gzip(CDXF)            | gzip + ours   | Combined pipeline                      |
| zstd(CDXF)            | zstd + ours   | Combined pipeline                      |

**Feature Preservation Matrix — each cell is an automated test:**

| Construct                  | CDXF | CBOR | MsgPack | BSON | Ion |
|----------------------------|------|------|---------|------|-----|
| Map key order              |  ?   |  ?   |    ?    |  ?   |  ?  |
| Non-string map keys        |  ?   |  ?   |    ?    |  ?   |  ?  |
| Comments                   |  ?   |  ?   |    ?    |  ?   |  ?  |
| Anchors/Aliases (graph)    |  ?   |  ?   |    ?    |  ?   |  ?  |
| Merge keys                 |  ?   |  ?   |    ?    |  ?   |  ?  |
| Multi-document streams     |  ?   |  ?   |    ?    |  ?   |  ?  |
| XML elements/attributes    |  ?   |  ?   |    ?    |  ?   |  ?  |
| XML namespaces             |  ?   |  ?   |    ?    |  ?   |  ?  |
| XML mixed content          |  ?   |  ?   |    ?    |  ?   |  ?  |
| Processing instructions    |  ?   |  ?   |    ?    |  ?   |  ?  |
| Typed timestamps           |  ?   |  ?   |    ?    |  ?   |  ?  |
| Typed date/time (local)    |  ?   |  ?   |    ?    |  ?   |  ?  |

### Corpus

Three tiers with documented provenance:

**Tier 1 — SOTA benchmark corpus (enables comparison with prior work):**
- Viotti SchemaStore selection (arXiv:2201.03051): package.json, tsconfig.json,
  GitHub Actions workflow, ESLint config, and others as identified in the paper.

**Tier 2 — Canonical real-world documents:**
- JSON: GeoJSON features (geojson.org), CityLots dataset (SF OpenData)
- YAML: Kubernetes manifests (k8s.io/examples: Deployment, Service, ConfigMap),
  GitHub Actions workflows from major repos
- XML: Maven POM (Maven Central), Atom feed (W3C examples), SVG (W3C test
  suite), XHTML page
- TOML: Cargo.toml (popular Rust crates: serde, tokio, clap), pyproject.toml
  (popular Python packages: black, ruff, pytest)

**Tier 3 — Stress-test synthetic documents:**
- Pathological nesting depth (32, 64, 128 levels)
- Flat 10,000-element arrays
- Comment-dense YAML (50%+ lines are comments)
- Anchor-heavy YAML DAGs (many shared references)
- Namespace-heavy XML (10+ namespace prefixes)
- Maximum mixed content XML (alternating text/element children)
- Large TOML with all temporal types

### Protocol

1. Collect corpus files. Compute SHA-256 checksums. Write `data/DATA_README.md`
   with full provenance for every file.
2. Install baseline libraries: `msgpack`, `bson` (via pymongo), `amazon.ion`,
   `zstandard`. Record exact versions.
3. For each corpus file:
   a. Record original text size.
   b. Parse with the appropriate CDXF bridge (`from_json`, `from_yaml`,
      `from_xml`, `from_toml`) and encode to CDXF binary via `encode()`.
      Record CDXF size.
   c. Parse to Python native types and encode with each baseline
      (cbor2, msgpack, bson, ion). Record sizes.
   d. Compress original text and CDXF binary with gzip (level 9) and zstd
      (level 3, default). Record compressed sizes.
   e. Test CDXF round-trip fidelity: `text → CDXF → CBOR → CDXF → text`.
      Compare output against original for semantic equivalence. Record pass/fail.
   f. Also test CDXF shorthand mode for JSON-model documents.
4. For the Feature Preservation Matrix, run a dedicated test for each
   (construct, baseline) cell using a minimal document that exercises that
   construct. Record ✓/✗.
5. Aggregate results: compute median, mean, std, and 95% CI for size ratios
   across the corpus. Report per-file results in supplementary tables.
6. Generate figures: box plots of size distributions by format, feature
   preservation matrix heatmap.

### Environment

- **Hardware:** Windows laptop, 64GB RAM, NVIDIA RTX 4090
- **Software:** Python 3.12, Windows 11
- **Key packages:** cbor2, ruamel.yaml, tomlkit (exact versions TBD at runtime)
- **Git commit:** TBD (will be recorded at execution time)
- **Config file:** benchmarks/configs/exp_001_size_fidelity.yaml
- **Seeds:** N/A (deterministic — no stochastic component)

### Results

*To be filled after experiment completes.*

### Observations

*To be filled after experiment completes.*

### Interpretation

*To be filled after experiment completes.*

### Artifacts

- Config: benchmarks/configs/exp_001_size_fidelity.yaml
- Raw results: benchmarks/results/exp_001/
- Figures: benchmarks/results/exp_001/figures/
- Corpus: data/raw/
- Checksums: data/checksums.sha256
