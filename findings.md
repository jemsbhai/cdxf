# Findings

## Curated Summary

### F1: CDXF achieves 100% round-trip fidelity across all four format families

CDXF losslessly round-trips documents from JSON, YAML, XML, and TOML through
binary (CBOR) encoding and back to text with full semantic equivalence. This was
verified on a 43-file corpus spanning the Viotti SchemaStore benchmark (Tier 1),
canonical real-world documents from Kubernetes, Maven, and Rust/Python package
registries (Tier 2), and synthetic stress tests including 128-level nesting,
10K-element arrays, comment-dense YAML, anchor-heavy DAGs, and namespace-heavy
XML (Tier 3). Additionally, 31 structural fidelity tests verify that
format-specific constructs — not just data values — survive the binary
round-trip at the information model level. (EXP-001)

### F2: No existing binary format preserves more than 25% of cross-format constructs

The Feature Preservation Matrix (EXP-001) empirically tested 12 constructs
across 5 binary formats. Results:

- **CDXF: 12/12** (100%)
- CBOR: 3/12 (map key order, non-string keys, offset timestamps)
- BSON: 2/12 (map key order, offset timestamps)
- MessagePack: 1/12 (map key order)
- Amazon Ion: 0/12

This confirms the gap identified in the literature survey: no binary format
occupies the design space position that CDXF targets. The closest competitor
(CBOR) preserves only basic data-model constructs and loses all format-specific
metadata (comments, anchors, namespaces, PIs, mixed content, local datetimes).

### F3: CDXF adds zero overhead for JSON-model data in shorthand mode

CDXF shorthand encoding produces byte-identical output to standard CBOR for
JSON-model documents (maps, sequences, scalars with no annotations). This means
existing CBOR tooling can read CDXF-shorthand documents without modification.
For full-mode encoding of JSON documents, median overhead vs naive CBOR is 1.5%,
driven entirely by the Stream/Document wrapper tags. (EXP-001)

### F4: Size efficiency varies by format family and metadata density

Median CDXF/text size ratios (EXP-001, n=43):

- JSON: 0.663 (34% smaller than text)
- TOML: 0.754 (25% smaller)
- YAML: 0.822 (18% smaller)
- XML: 1.262 (26% larger)

XML documents are larger in CDXF because the binary encoding must store element
names, namespace URIs, and structural tags as explicit data, whereas XML's
angle-bracket syntax encodes structure implicitly. Compression (gzip/zstd)
largely neutralizes this: gzip(CDXF) and gzip(text) are within 5% for most
documents.

### F5: Overhead is proportional to preserved metadata, not data size

The highest CDXF/CBOR overhead ratios correlate with metadata density, not
document size. Comment-dense YAML (3.7x vs CBOR) and anchor-heavy YAML (1.58x)
have high overhead because CBOR discards comments and aliases entirely. For
plain-data documents (no comments, no anchors), overhead approaches 1.0x. This
is not waste — it is the cost of preserving information that all other formats
destroy. (EXP-001)

---

## Raw Findings Log

### 2026-05-24 — EXP-001 execution

**Run 1 (initial):** 36/43 fidelity, 7 failures. Root causes identified:
- TOML temporal types: codec crashed on naive datetimes (cbor2 limitation)
- XML namespace-heavy: Element.prefix not preserved through codec
- YAML anchor-heavy: merge keys not reconstructed via ruamel.yaml merge API
- TOML real-world files: split table definitions not merged; quoted keys
  included quote characters in key strings
- XML atom-namespace: xml: prefix not pre-seeded in namespace scope

**Run 2 (after codec temporal fix):** 38/43. Temporal types and XML namespace
fixed. TOML and YAML issues remained.

**Run 3 (after YAML merge + TOML key/table + XML xml:ns fixes):** 40/43.
YAML anchor-heavy and XML atom fixed. 3 TOML real-world failures remained.

**Run 4 (after TOML split table merge + inline table in mixed arrays):** 43/43.
All failures resolved. Feature preservation matrix executed: 12/12 for CDXF.

**Key implementation bugs found and fixed during EXP-001:**
1. Codec temporal encoding: datetime objects must be serialized as ISO strings
   in CBOR tags, not passed directly to cbor2 (which rejects naive datetimes).
2. Codec Element encoding: prefix field must be stored for round-trip fidelity.
3. YAML bridge to_yaml: merge keys must use ruamel.yaml's add_yaml_merge() API,
   not plain dict assignment, for proper anchor/alias reconstruction.
4. TOML bridge: tomlkit Key objects must be unwrapped via .key property (str()
   includes quotes). Split table definitions must be merged when the same key
   appears multiple times in the body.
5. XML bridge: the xml namespace (http://www.w3.org/XML/1998/namespace) must be
   pre-seeded in the parser's namespace scope (it is always implicitly bound per
   the XML specification).

All bugs were implementation defects, not information model design issues. The
model specification was correct throughout.
