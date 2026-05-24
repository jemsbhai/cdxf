"""CDXF core data model — the 9 node kinds and supporting types.

This module defines the abstract information model for CDXF as Python
dataclasses. These classes represent the in-memory form of a CDXF document,
independent of any particular encoding (binary or text).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScalarType(Enum):
    """The set of scalar subtypes supported by CDXF."""
    NULL = 0
    BOOLEAN = 1
    INTEGER = 2
    FLOAT = 3
    DECIMAL = 4
    STRING = 5
    BYTE_STRING = 6
    TIMESTAMP_OFFSET = 7
    TIMESTAMP_LOCAL = 8
    DATE = 9
    TIME = 10


class SourceFormat(Enum):
    """Advisory hint indicating the original source format."""
    UNSPECIFIED = 0
    JSON = 1
    YAML = 2
    XML = 3
    TOML = 4


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TagAnnotation:
    """A URI-based type tag attached to a node."""
    uri: str


@dataclass(frozen=True)
class Anchor:
    """An identity label establishing a node as a reference target."""
    name: str


@dataclass(frozen=True)
class Alias:
    """A reference to a previously anchored node by name."""
    name: str


# ---------------------------------------------------------------------------
# Node kinds
# ---------------------------------------------------------------------------

# Type alias for all node types (forward-declared, populated after classes).
Node = Union[
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
    "Alias",
]

# Type alias for items that can appear in a Map's entry list.
MapEntry = Union[tuple, "Comment"]

# Type alias for items that can appear in a Sequence's item list.
SequenceItem = Union["Scalar", "Map", "Sequence", "Element", "Comment", "Alias"]


@dataclass
class Scalar:
    """A leaf node carrying a typed atomic value."""
    scalar_type: ScalarType
    value: object
    tag: TagAnnotation | None = None
    anchor: Anchor | None = None


@dataclass
class Comment:
    """A text annotation preserved for round-trip fidelity."""
    text: str


@dataclass
class Map:
    """An ordered collection of key-value entries, optionally interleaved
    with Comment nodes."""
    entries: list[MapEntry] = field(default_factory=list)
    tag: TagAnnotation | None = None
    anchor: Anchor | None = None


@dataclass
class Sequence:
    """An ordered collection of nodes, optionally interleaved with
    Comment nodes."""
    items: list = field(default_factory=list)
    tag: TagAnnotation | None = None
    anchor: Anchor | None = None


@dataclass
class Attribute:
    """A name-value pair attached to an Element, distinct from child nodes."""
    name: str
    value: str
    namespace_uri: str | None = None
    prefix: str | None = None


@dataclass
class Element:
    """An XML-style named node with attributes and ordered children."""
    name: str
    namespace_uri: str | None = None
    prefix: str | None = None
    attributes: list[Attribute] = field(default_factory=list)
    children: list = field(default_factory=list)
    namespace_declarations: dict[str, str] = field(default_factory=dict)
    tag: TagAnnotation | None = None
    anchor: Anchor | None = None


@dataclass
class ProcessingInstruction:
    """A target-data pair providing application instructions."""
    target: str
    data: str | None = None


@dataclass
class Directive:
    """A document-level instruction (YAML %YAML/%TAG, XML DOCTYPE, etc.)."""
    name: str
    parameters: list[str] = field(default_factory=list)


@dataclass
class Document:
    """A single logical document within a Stream."""
    root: Node
    source_format_hint: SourceFormat = SourceFormat.UNSPECIFIED
    allows_cycles: bool = False
    preamble: list = field(default_factory=list)
    postamble: list = field(default_factory=list)


@dataclass
class Stream:
    """Top-level container: a sequence of Documents."""
    documents: list[Document] = field(default_factory=list)
