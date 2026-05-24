# CDXF -- Compact Data Exchange Format

A universal binary interchange format whose information model is a provable superset of JSON, YAML, XML, and TOML, enabling lossless round-trip encoding of documents from any of these format families.

## Status

**Research / Pre-alpha.** The specification and reference implementation are under active development.

## Motivation

Every existing binary serialization format is anchored to a single text format's information model. CBOR and MessagePack encode the JSON data model. EXI and Fast Infoset encode the XML Information Set. Nothing encodes YAML's graph-structured representation or TOML's typed configuration model. And critically, no binary format preserves all of them through a single unified information model.

CDXF fills that gap.

## What CDXF preserves that others lose

| Feature | CBOR | EXI | Ion | CDXF |
|---|---|---|---|---|
| JSON trees | Yes | Via XML mapping | Yes | Yes |
| YAML anchors/aliases (DAG) | No | No | No | Yes |
| YAML tags | Partial (CBOR tags) | No | Partial (annotations) | Yes |
| XML attributes vs. elements | No | Yes | No | Yes |
| XML namespaces | No | Yes | No | Yes |
| XML mixed content | No | Yes | No | Yes |
| TOML native datetimes | Via tags | No | Yes | Yes |
| Comments | No | Yes | No | Yes |
| Multi-document streams | No | No | No | Yes |

## Installation

```bash
pip install cdxf
```

## Quick Start

```python
import cdxf

# Encode any supported format to CDXF binary
binary = cdxf.encode({"hello": "world"}, source_format="json")

# Decode back
data = cdxf.decode(binary)

# Round-trip from YAML with anchors preserved
yaml_doc = """
defaults: &defaults
  timeout: 30
  retries: 3
production:
  <<: *defaults
  timeout: 60
"""
binary = cdxf.encode(yaml_doc, source_format="yaml")
restored = cdxf.decode(binary, target_format="yaml")
# Anchors and aliases are preserved
```

## Documentation

See [`docs/`](docs/) for the literature survey and specification drafts.

## License

MIT. See [LICENSE](LICENSE).

## Author

Muntaser Syed ([@jemsbhai](https://github.com/jemsbhai)) -- Florida Institute of Technology
