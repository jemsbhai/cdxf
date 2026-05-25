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

**Executed:** 2026-05-24T20:02 UTC
**Git commit:** post feat(bridges) + fix(codec+bridges) commits
**Corpus:** 43 files (27 Tier 1 SchemaStore, 6 Tier 2 canonical, 10 Tier 3 synthetic)
**Total corpus size:** 155,213 bytes

**Environment (exact versions):**
- Python 3.12.2, Windows 11, AMD64
- cbor2 5.8.0, msgpack 1.1.2, ruamel.yaml 0.18.14, tomlkit 0.14.0
- zstandard 0.25.0, pymongo 4.16.0, amazon.ion 0.13.0

#### Round-Trip Fidelity

**43/43 files pass** full pipeline: text → CDXF → CBOR → CDXF → text.
100% data-level semantic equivalence across all 4 format families.

Additionally, 31 structural fidelity tests verify that format-specific
constructs (comments, anchors, namespaces, PIs, mixed content, typed
scalars) survive the binary round-trip at the information model level.

#### Size Efficiency (CDXF full / original text)

| Format | n  | Median | Mean  | Std   |
|--------|----|--------|-------|-------|
| JSON   | 30 | 0.663  | 0.673 | 0.191 |
| YAML   |  5 | 0.822  | 0.947 | 0.313 |
| XML    |  4 | 1.262  | 1.324 | 0.399 |
| TOML   |  4 | 0.754  | 0.767 | 0.083 |

JSON shorthand mode is byte-identical to naive CBOR (overhead = 0%).
CDXF full mode adds median 1.5% overhead vs naive CBOR for JSON documents
(driven by Stream/Document wrapper tags; negligible for files > 100 bytes).

XML CDXF encoding is larger than source text (median 1.26x) because the
binary encoding must store element names, namespace URIs, and structural
tags that are implicit in XML’s angle-bracket syntax. However, after
compression (gzip/zstd), CDXF and raw XML converge.

#### Feature Preservation Matrix

| Construct                  | CDXF | CBOR | MsgPack | BSON | Ion |
|----------------------------|------|------|---------|------|-----|
| Map key order              |  ✓   |  ✓   |    ✓    |  ✓   |  ✗  |
| Non-string map keys        |  ✓   |  ✓   |    ✗    |  ✗   |  ✗  |
| Comments                   |  ✓   |  ✗   |    ✗    |  ✗   |  ✗  |
| Anchors/Aliases (graph)    |  ✓   |  ✗   |    ✗    |  ✗   |  ✗  |
| Merge keys                 |  ✓   |  ✗   |    ✗    |  ✗   |  ✗  |
| Multi-document streams     |  ✓   |  ✗   |    ✗    |  ✗   |  ✗  |
| XML elements/attributes    |  ✓   |  ✗   |    ✗    |  ✗   |  ✗  |
| XML namespaces             |  ✓   |  ✗   |    ✗    |  ✗   |  ✗  |
| XML mixed content          |  ✓   |  ✗   |    ✗    |  ✗   |  ✗  |
| Processing instructions    |  ✓   |  ✗   |    ✗    |  ✗   |  ✗  |
| Typed timestamps           |  ✓   |  ✓   |    ✗    |  ✓   |  ✗  |
| Typed date/time (local)    |  ✓   |  ✗   |    ✗    |  ✗   |  ✗  |
| **Total**                  |**12**| **3**|  **1**  | **2**|**0**|

CDXF is the only format that preserves all 12 constructs.

### Observations

1. CDXF shorthand mode achieves the zero-overhead claim: byte-identical to
   naive CBOR for JSON-model data.
2. For format-rich documents (comments, anchors, namespaces), CDXF is larger
   than naive CBOR — but naive CBOR destroys the information those bytes
   encode. The overhead is the cost of fidelity, not waste.
3. Comment-dense YAML is the worst case for CDXF vs CBOR overhead (3.7x)
   because CBOR discards all comments while CDXF preserves every one.
4. XML namespace-heavy documents inflate in CDXF (1.84x vs text) because
   each element stores its full namespace URI. This is by design — namespace
   URIs are semantically significant — and compresses well (0.11x with zstd).
5. Amazon Ion scored 0/12 despite being the closest expressiveness competitor
   in the literature. Its binary format preserves timestamps but not key order,
   local datetimes, or any XML/YAML-specific constructs.
6. Compression largely equalizes size differences: gzip(CDXF) and gzip(text)
   are within 5% for most documents.

### Interpretation

The hypothesis is **partially supported with caveats:**

- **JSON overhead hypothesis (≤5%) — SUPPORTED.** Shorthand mode adds 0%.
  Full mode adds median 1.5% for documents > 100 bytes.
- **Format-rich overhead hypothesis (≤15%) — NOT SUPPORTED as stated.**
  Comment-dense and namespace-heavy documents exceed 15%. However, the
  overhead scales with the amount of metadata preserved, not with data size.
  The hypothesis should be restated: CDXF adds overhead proportional to the
  metadata it preserves, with negligible overhead for metadata-free documents.
- **100% round-trip fidelity — SUPPORTED.** 43/43 corpus files, 31 structural
  fidelity tests, 12/12 feature preservation constructs.
- **Secondary hypothesis (compression) — PARTIALLY SUPPORTED.** gzip(CDXF)
  is comparable to gzip(text) for most documents. zstd(CDXF) sometimes
  outperforms zstd(text) due to reduced entropy in binary encoding.
- **Uniqueness claim — STRONGLY SUPPORTED.** No existing binary format
  preserves more than 3/12 constructs. CDXF preserves 12/12.

### Artifacts

- Config: benchmarks/configs/exp_001_size_fidelity.yaml
- Raw results: benchmarks/results/exp_001/
- Figures: benchmarks/results/exp_001/figures/
- Corpus: data/raw/
- Checksums: data/checksums.sha256

---

## EXP-001 Addendum: Compact Element Encoding + Namespace Interning

**Date:** 2026-05-24
**Status:** completed

Optimization applied to codec between EXP-001 and EXP-002. Two changes:

1. **Compact element encoding** — elements without namespaces use shorter
   arrays: `[name, children]` or `[name, attrs, children]` instead of the
   full 5-6 element form. Saves ~3 bytes per element.
2. **Namespace URI interning** — unique URIs stored once in a document-level
   table (`DOC_OPT_NS_TABLE`). Elements reference by integer index.

Both are backward-compatible. Decoder handles old and new forms.

**Impact on XML size ratios:**

| File                    | Before | After  |
|-------------------------|--------|--------|
| xml_namespace_heavy.xml | 1.877x | 1.034x |
| atom-namespace.xml      | 1.420x | 0.998x |
| xml_mixed_content.xml   | 1.118x | 1.002x |
| xml_comment_heavy.xml   | 0.958x | 0.928x |
| **XML median**          | **1.262x** | **0.994x** |

All four format families now have median CDXF/text ratio ≤ 1.0.
43/43 round-trip fidelity maintained. 408 tests pass.

---

## EXP-002: Encode/Decode Throughput

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational (performance)
**Status:** planned

### Hypothesis

CDXF encode/decode throughput is within the same order of magnitude as
native CBOR, MessagePack, and JSON serialization. The overhead of the
richer information model does not impose a prohibitive performance penalty
for practical use.

### Independent Variables

- **Operation:** encode (text/model → binary) vs decode (binary → model)
- **Source format:** JSON, YAML, XML, TOML
- **Document size:** small (~100B), medium (~1KB), large (~10KB+)
- **Baseline format:** CBOR, MessagePack, BSON, Ion, stdlib JSON

### Dependent Variables

1. `time_seconds` — wall-clock time per operation
2. `throughput_bytes_per_sec` — input bytes / time
3. `ops_per_sec` — operations per second

### Statistical Protocol

- **Warm-up:** 5 iterations discarded before measurement
- **Measurement:** 100 iterations per (file, operation, baseline) triple
- **Reporting:** median, mean, std, 95% CI
- **GC control:** `gc.disable()` during measurement, `gc.collect()` between files
- **Timer:** `time.perf_counter_ns()` for nanosecond precision
- **Outlier handling:** report all data, do NOT remove outliers

### Baselines

| Baseline      | Encode                        | Decode                         |
|---------------|-------------------------------|--------------------------------|
| stdlib JSON   | json.dumps(native)            | json.loads(text)               |
| CBOR          | cbor2.dumps(native)           | cbor2.loads(binary)            |
| MessagePack   | msgpack.packb(native)         | msgpack.unpackb(binary)        |
| BSON          | bson.encode(native)           | bson.decode(binary)            |
| Ion           | ion.dumps(native, binary=True)| ion.loads(binary)              |
| CDXF full     | bridge + encode               | decode + bridge                |
| CDXF codec    | encode(stream) only           | decode(binary) only            |

CDXF is measured two ways: full pipeline (including text parsing/emitting)
and codec-only (CDXF model ↔ CBOR binary). This separates the cost of the
information model from the cost of text parsing.

### Corpus

Same 43-file corpus from EXP-001.

### Environment

Same as EXP-001. Python 3.12.2, Windows 11, 64GB RAM.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp002.py
- Config: benchmarks/configs/exp_002_throughput.yaml
- Results: benchmarks/results/exp_002/

---

## EXP-003: Cross-Format Interchange Fidelity

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

### Hypothesis

CDXF enables lossless data-level interchange between JSON, YAML, and TOML
for their shared data model (maps, sequences, typed scalars). XML is
excluded from cross-conversion targets because its element/attribute model
has no natural mapping to key-value formats. Within the shared data model,
all 6 conversion pairs (JSON↔YAML, JSON↔TOML, YAML↔TOML) preserve
semantic equivalence.

### Independent Variables

- **Source format:** JSON, YAML, TOML
- **Target format:** JSON, YAML, TOML (all pairs where source ≠ target)
- **Document complexity:** simple (flat map), nested (multi-level), typed
  (integers, floats, booleans, strings, arrays, nested tables)

### Dependent Variables

1. `data_equivalent` — binary pass/fail: does the converted document
   contain the same data when parsed back to native Python types?
2. `type_preserved` — per-scalar: are integer/float/boolean/string types
   preserved (not collapsed to strings)?

### Conversion Pairs

| # | Source | Target | Key challenge |
|---|--------|--------|---------------|
| 1 | JSON   | YAML   | Straightforward |
| 2 | JSON   | TOML   | Top-level must be table |
| 3 | YAML   | JSON   | YAML anchors/comments lost |
| 4 | YAML   | TOML   | YAML types → TOML types |
| 5 | TOML   | JSON   | Dates → strings |
| 6 | TOML   | YAML   | Dates → strings or tagged |

Additionally, XML→XML round-trip is tested but XML is not a valid target
for key-value source formats, and key-value formats are not valid targets
for XML (lossy by design — no elements/attributes in JSON/YAML/TOML).

### Protocol

1. Build a set of test documents in each source format that exercise the
   shared data model: flat maps, nested maps, arrays, mixed-type arrays,
   all scalar types representable in both source and target.
2. For each (source, target) pair:
   a. Parse source text with source bridge → CDXF Stream
   b. Emit target text with target bridge
   c. Parse target text with target’s native parser (json.loads,
      ruamel.yaml safe load, tomlkit parse)
   d. Parse source text with source’s native parser
   e. Compare native Python data structures for equality
   f. Record pass/fail and any type differences
3. Separately test that format-specific constructs (YAML comments/anchors,
   TOML dates) are gracefully handled when converting to formats that
   cannot represent them.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp003.py
- Results: benchmarks/results/exp_003/
