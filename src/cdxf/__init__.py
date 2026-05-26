"""CDXF -- Compact Data Exchange Format.

A universal binary interchange format whose information model is a provable
superset of JSON, YAML, XML, and TOML.
"""

__version__ = "0.1.3"

from cdxf.model import (
    Stream,
    Document,
    Map,
    Sequence,
    Scalar,
    Element,
    Attribute,
    Comment,
    ProcessingInstruction,
    Directive,
    TagAnnotation,
    Anchor,
    Alias,
    ScalarType,
    SourceFormat,
)

from cdxf.codec import encode, decode, Encoder, Decoder

__all__ = [
    # Model
    "Stream",
    "Document",
    "Map",
    "Sequence",
    "Scalar",
    "Element",
    "Attribute",
    "Comment",
    "ProcessingInstruction",
    "Directive",
    "TagAnnotation",
    "Anchor",
    "Alias",
    "ScalarType",
    "SourceFormat",
    # Codec
    "encode",
    "decode",
    "Encoder",
    "Decoder",
]
