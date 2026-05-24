# CDXF — Compact Data Exchange Format

A universal binary interchange format whose information model is a provable superset of JSON, YAML, XML, and TOML, enabling lossless round-trip encoding of documents from any of these format families.

## Status

**Alpha.** The specification and reference implementation are functional with 408 tests passing. All four format bridges (JSON, YAML, XML, TOML) are complete with full round-trip fidelity verified on a 43-file benchmark corpus.

## Motivation

Every existing binary serialization format is anchored to a single text format's information model. CBOR and MessagePack encode the JSON data model. EXI and Fast Infoset encode the XML Information Set. Nothing encodes YAML's graph-structured representation or TOML's typed configuration model. And critically, no binary format preserves all of them through a single unified information model.

CDXF fills that gap.

## What CDXF preserves that others lose

Empirically verified via automated tests (EXP-001 Feature Preservation Matrix):

| Construct | CDXF | CBOR | MsgPack | BSON | Ion |
|---|:---:|:---:|:---:|:---:|:---:|
| Map key order | ✓ | ✓ | ✓ | ✓ | ✗ |
| Non-string map keys | ✓ | ✓ | ✗ | ✗ | ✗ |
| Comments | ✓ | ✗ | ✗ | ✗ | ✗ |
| Anchors/Aliases (graph) | ✓ | ✗ | ✗ | ✗ | ✗ |
| Merge keys | ✓ | ✗ | ✗ | ✗ | ✗ |
| Multi-document streams | ✓ | ✗ | ✗ | ✗ | ✗ |
| XML elements/attributes | ✓ | ✗ | ✗ | ✗ | ✗ |
| XML namespaces | ✓ | ✗ | ✗ | ✗ | ✗ |
| XML mixed content | ✓ | ✗ | ✗ | ✗ | ✗ |
| Processing instructions | ✓ | ✗ | ✗ | ✗ | ✗ |
| Typed timestamps | ✓ | ✓ | ✗ | ✓ | ✗ |
| Typed date/time (local) | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Total** | **12/12** | **3/12** | **1/12** | **2/12** | **0/12** |

## Installation

```bash
pip install cdxf
```

## CLI Usage

```bash
# Encode a JSON file to CDXF binary
cdxf encode config.json                   # -> config.cdxf

# Decode back to text (auto-detects source format)
cdxf decode config.cdxf                   # -> config.json

# Convert between formats via CDXF
cdxf convert data.yaml --to json          # -> data.json
cdxf convert config.json --to toml        # -> config.toml

# Inspect a file
cdxf info config.json
# File:          config.json
# Type:          JSON text
# Text size:     2,946 bytes
# CDXF size:     2,001 bytes
# Ratio:         0.679
# Node counts:   {'Map': 9, 'Scalar': 114, 'Sequence': 2}
```

## Python API

```python
from cdxf.bridges import from_json, to_json, from_yaml, to_yaml
from cdxf.bridges import from_xml, to_xml, from_toml, to_toml
from cdxf.codec import encode, decode

# JSON round-trip through binary
stream = from_json('{"name": "Alice", "age": 30}')
binary = encode(stream)          # compact CBOR-based binary
restored = decode(binary)
print(to_json(restored))         # '{"name": "Alice", "age": 30}'

# Cross-format conversion: YAML → CDXF → JSON
stream = from_yaml("name: Alice\nage: 30\n")
print(to_json(stream))           # '{"name": "Alice", "age": 30}'

# YAML with anchors and comments — preserved through binary
yaml_doc = """
# Server defaults
defaults: &defaults
  timeout: 30
  retries: 3
production:
  <<: *defaults
  timeout: 60
"""
stream = from_yaml(yaml_doc)
binary = encode(stream)          # anchors, comments, merge keys preserved
restored = decode(binary)
print(to_yaml(restored))         # anchors and comments survive

# XML with namespaces and mixed content
xml_doc = '<p xmlns="http://www.w3.org/1999/xhtml">Hello <b>world</b>!</p>'
stream = from_xml(xml_doc)
binary = encode(stream)          # namespaces, mixed content preserved
restored = decode(binary)
print(to_xml(restored))          # faithful reconstruction
```

## Size Efficiency

Median CDXF/text size ratios on a 43-file benchmark corpus:

| Format | Median | Interpretation |
|---|---|---|
| JSON | 0.66 | 34% smaller than text |
| TOML | 0.75 | 25% smaller |
| YAML | 0.82 | 18% smaller |
| XML | 1.26 | 26% larger (namespace URIs stored explicitly) |

For JSON data, CDXF shorthand mode produces byte-identical output to standard CBOR — zero overhead.

## Documentation

- [`docs/information_model.md`](docs/information_model.md) — Formal specification: 9 node kinds, 2 conformance levels, format mapping proofs
- [`docs/binary_encoding.md`](docs/binary_encoding.md) — CBOR-based wire format with 16 semantic tags
- [`docs/literature_survey_universal_binary_interchange.md`](docs/literature_survey_universal_binary_interchange.md) — Comprehensive gap analysis of existing binary formats

## License

MIT. See [LICENSE](LICENSE).

## Author

Muntaser Syed ([@jemsbhai](https://github.com/jemsbhai)) — Florida Institute of Technology
