# CDXF Information Model Specification

**Version:** 0.1.0-draft
**Status:** Working Draft
**Author:** Muntaser Syed
**Date:** May 2026

---

## 1. Overview

This document defines the CDXF (Compact Data Exchange Format) abstract
information model. The information model is the set of constructs that a CDXF
document can express, independent of any particular encoding (binary or text).

**Design goal.** The CDXF information model is a *superset* of the information
models defined by:

- JSON (RFC 8259)
- YAML 1.2 representation graph (Spec 1.2.2, Section 3.2)
- XML Information Set (W3C Recommendation, Second Edition)
- TOML (v1.0.0)

**Superset property.** For each of these source formats, there exists a
well-defined, injective (one-to-one) mapping from every construct in the
source format's information model into the CDXF information model. This
guarantees that encoding a source document into CDXF and decoding it back to
the source format produces a semantically equivalent document (lossless
round-trip), modulo insignificant whitespace and formatting choices.

---

## 2. Definitions

**Node.** The fundamental unit of the CDXF information model. Every CDXF
document is a tree or directed acyclic graph (DAG) of nodes (or a directed
graph with cycles in Extended conformance; see Section 11).

**Annotation.** Metadata attached to a node that is not part of the node's
content. Annotations include tags, anchors, namespace bindings, and comments.

**Identity.** A node's identity is its anchor label (if any). Two references
to the same anchor denote the same node, enabling directed graph structures.

**Tag.** A URI that specifies the intended data type or semantic meaning of a
node. Tags generalize YAML's tag system, CBOR's tag numbers, Ion's
annotations, and XML Schema types.

---

## 3. Node Kinds

The CDXF information model defines exactly nine node kinds. Every construct in
JSON, YAML, XML, and TOML maps to one of these kinds.

### 3.1 Stream

A **Stream** is the top-level container. It represents a complete CDXF
transmission unit containing zero or more Documents.

Properties:
- `documents`: ordered list of Document nodes

Rationale: YAML supports multi-document streams delimited by `---` and `...`.
JSON, XML, and TOML are single-document formats. A Stream containing exactly
one Document represents a JSON, XML, or TOML document; a Stream containing
multiple Documents represents a YAML multi-document stream.

### 3.2 Document

A **Document** is a single logical document within a Stream.

Properties:
- `root`: exactly one child node (the document's root content)
- `directives`: ordered list of Directive nodes (may be empty)
- `preamble`: ordered list of interleaved Comment and Directive nodes
  appearing before the root (may be empty)
- `postamble`: ordered list of Comment nodes appearing after the root
  (may be empty)
- `source_format_hint`: optional enum {json, yaml, xml, toml, unspecified}
- `allows_cycles`: boolean (default false; see Section 11)

Rationale: The source format hint is advisory metadata enabling decoders to
reconstruct the original format when desired. It does not affect the
information content.

### 3.3 Map

A **Map** is a collection of key-value entries with preserved insertion order.
Each key is itself a node (not restricted to strings, following YAML's model
where mapping keys can be any node type). Values are nodes.

Properties:
- `entries`: ordered list of (key_node, value_node | Comment) items

The entry list preserves insertion order and interleaves Comment nodes at the
positions where they appeared in the source document. Semantic equality of
Maps considers only the set of (key, value) entries, ignoring order and
comments.

Semantic rules:
- Keys MUST be unique within a Map by deep equality.
- JSON objects, YAML mappings, and TOML tables all map to Map nodes.
- When encoding for JSON or TOML output, non-string keys are an error
  (those formats require string keys).

### 3.4 Sequence

A **Sequence** is an ordered collection of zero or more items, interleaved
with Comment nodes.

Properties:
- `items`: ordered list of (node | Comment) items

Rationale: JSON arrays, YAML sequences, TOML arrays, and XML element children
(when not using mixed content) all map to Sequence nodes. Comment nodes are
interleaved at the positions where they appeared in the source document.

### 3.5 Scalar

A **Scalar** is a leaf node carrying a typed atomic value.

Properties:
- `value`: the scalar's content (type determined by scalar_type)
- `scalar_type`: one of the following:

| scalar_type        | Description                          | Source formats        |
|--------------------|--------------------------------------|-----------------------|
| `null`             | Absence of value                     | JSON, YAML, TOML*    |
| `boolean`          | true or false                        | JSON, YAML, TOML     |
| `integer`          | Arbitrary-precision signed integer   | YAML, TOML, XML**    |
| `float`            | IEEE 754 binary64                    | JSON, YAML, TOML     |
| `decimal`          | Arbitrary-precision decimal          | Ion, XML Schema       |
| `string`           | Unicode character sequence           | All                   |
| `byte_string`      | Raw byte sequence                    | YAML !!binary, CBOR  |
| `timestamp_offset` | Date-time with UTC offset (RFC 3339) | YAML, TOML, XML      |
| `timestamp_local`  | Date-time without offset             | TOML                  |
| `date`             | Calendar date (year-month-day)       | TOML                  |
| `time`             | Time of day (hour:min:sec.frac)      | TOML                  |

*TOML does not have an explicit null, but CDXF can represent absent keys.
**JSON numbers without a decimal point or exponent MAY be stored as integer.

Rationale: The scalar type system is the union of all scalar types across
JSON, YAML, TOML, and XML Schema simple types. The `decimal` type is included
for forward compatibility with Ion and XML Schema's xsd:decimal, though the
initial implementation may defer it.

### 3.6 Element

An **Element** is a named node with an ordered sequence of children and an
unordered set of attributes. This is the construct that makes XML's
information model representable without loss.

Properties:
- `name`: local name (string)
- `namespace_uri`: optional URI string
- `prefix`: optional namespace prefix (advisory, for round-trip fidelity)
- `attributes`: set of Attribute nodes (unordered)
- `children`: ordered list of (Element | Scalar | Comment |
  ProcessingInstruction) nodes
- `namespace_declarations`: set of (prefix, URI) bindings in scope at this
  element

Semantic rules:
- Element children may freely interleave Scalar(string) nodes with other
  Element nodes. This enables XML mixed content:
  `<p>Hello <b>world</b>!</p>` becomes an Element "p" with children
  [Scalar("Hello "), Element("b", children=[Scalar("world")]), Scalar("!")].
- Attribute names are (namespace_uri, local_name) pairs. Attribute names MUST
  be unique within an Element.
- Element is used ONLY for XML-origin data. JSON, YAML, and TOML documents
  never produce Element nodes.

### 3.7 Attribute

An **Attribute** is a name-value pair attached to an Element, structurally
distinct from child nodes.

Properties:
- `name`: local name (string)
- `namespace_uri`: optional URI string
- `prefix`: optional namespace prefix (advisory)
- `value`: string

Rationale: The attribute-vs-child-element distinction is fundamental to XML's
information model. EXI preserves it; CBOR and Ion lose it. CDXF preserves it.

### 3.8 Comment

A **Comment** is a text annotation that is not part of the document's data
content but is preserved for round-trip fidelity.

Properties:
- `text`: string (the comment body)

Comments are interleaved in their parent's child/entry list at the position
where they appeared in the source document. This preserves positional
association without requiring explicit node-to-comment binding.

Rationale: YAML, XML, and TOML all support comments. JSON does not. Preserving
comments enables lossless round-trip for configuration files (a major use case
for YAML and TOML).

### 3.9 Processing Instruction

A **ProcessingInstruction** is a target-data pair providing instructions to
the application processing the document.

Properties:
- `target`: string (the PI target name)
- `data`: optional string (the PI data content)

Rationale: XML processing instructions (e.g., `<?xml-stylesheet ... ?>`) have
no equivalent in JSON, YAML, or TOML. CDXF preserves them for XML round-trip.

---

## 4. Cross-Cutting Annotations

Any node (of any kind) MAY carry the following annotations:

### 4.1 Tag

A **Tag** is a URI that conveys the intended type or semantic meaning of a
node, independent of the node's structural kind.

- Representation: URI string (e.g., `tag:yaml.org,2002:int`,
  `urn:ietf:params:cbor:tag:0`, or a custom application URI)
- YAML tags map directly.
- CBOR tag numbers are mapped to URIs via a defined CBOR-tag-to-URI scheme:
  `urn:ietf:params:cbor:tag:{number}`
- Ion annotations are mapped as application-defined URIs.
- JSON and TOML have no tags; the tag annotation is absent.

### 4.2 Anchor

An **Anchor** is a string label that establishes a node's identity within a
document. A node with an anchor can be referenced by an Alias (see 4.3) from
elsewhere in the document, enabling directed graph structures.

- Only one anchor per node.
- Anchor names are scoped to the Document they appear in.
- Maps directly from YAML anchors (`&name`).
- JSON, XML, and TOML do not have anchors; the annotation is absent.
  (XML's `xml:id` attribute could optionally be interpreted as an anchor by a
  format-aware encoder, but this is not required.)

### 4.3 Alias

An **Alias** is a reference to a previously anchored node within the same
Document.

- Representation: the anchor name being referenced.
- On encounter during decoding, the Alias resolves to the anchored node,
  producing a shared reference (DAG edge). In CDXF Core conformance, the
  resulting graph MUST be acyclic. In CDXF Extended conformance, cycles are
  permitted (see Section 11).
- Maps directly from YAML aliases (`*name`).
- An Alias is not a node kind; it is a reference that resolves to an existing
  node. In the information model graph, an Alias edge and a direct child edge
  are semantically identical -- they point to the same node.

### 4.4 Namespace Bindings

**Namespace bindings** are a set of (prefix, URI) pairs that establish XML
namespace context.

- Present only on Element and Document nodes.
- Prefixes are strings (empty string for the default namespace).
- URIs are strings.
- Not used by JSON, YAML, or TOML.

---

## 5. Directives

A **Directive** is a document-level instruction that affects parsing or
interpretation but is not part of the document's data content.

Properties:
- `name`: string (e.g., "YAML", "TAG", "DOCTYPE")
- `parameters`: ordered list of strings

Known directive types:

| Directive name | Origin | Parameters                                  |
|----------------|--------|---------------------------------------------|
| `YAML`         | YAML   | [version_string]                            |
| `TAG`          | YAML   | [handle, prefix]                            |
| `DOCTYPE`      | XML    | [root_element, system_id?, public_id?]      |

Rationale: YAML has %YAML and %TAG directives. XML has the DOCTYPE
declaration. CDXF unifies these under a single Directive construct. Only the
DOCTYPE identifiers (system/public) are preserved; the internal subset
(entity declarations, notation declarations, attribute defaults) is processed
before encoding, matching the behavior of EXI and Fast Infoset.

---

## 6. YAML Merge Keys

YAML's merge key convention (`<<`) is preserved as a regular map entry. The
key is a Scalar(string) with value `<<` and the YAML merge tag
(`tag:yaml.org,2002:merge`). The value is an Alias referencing the source
mapping, or a Sequence of Aliases for multiple merges.

This requires no special handling in the information model because:
1. The `<<` key is just a string key in a Map.
2. The Tag annotation carries the merge semantics.
3. The Alias reference preserves the graph relationship.
4. A YAML-aware decoder can reconstruct the merge syntax on output.

Merge expansion is the responsibility of the application layer, not the
interchange format. CDXF preserves the unexpanded form to enable lossless
round-trip of configuration files that use merges.

---

## 7. XML Entity References

XML entity references (e.g., `&amp;`, `&copy;`, `&company-name;`) are
expanded to their character content before encoding into CDXF. The entity
reference names are not preserved.

Rationale: This matches the XML Information Set's treatment of entity
references and aligns with how XML parsers behave in practice. The five
predefined XML entities (`&lt;`, `&gt;`, `&amp;`, `&quot;`, `&apos;`) are
reconstructed during XML output as needed for well-formedness.

---

## 8. Format Mapping Proofs (Sketch)

### 8.1 JSON to CDXF (Injective Mapping)

| JSON construct | CDXF construct                          |
|----------------|-----------------------------------------|
| document       | Stream(Document(root=...))              |
| object         | Map (string keys only)                  |
| array          | Sequence                                |
| string         | Scalar(string)                          |
| number         | Scalar(integer) or Scalar(float)        |
| true/false     | Scalar(boolean)                         |
| null           | Scalar(null)                            |

JSON produces no Tags, Anchors, Aliases, Elements, Attributes, Comments,
ProcessingInstructions, Directives, or Namespace Bindings. The mapping is
trivially injective.

### 8.2 YAML to CDXF (Injective Mapping)

| YAML construct          | CDXF construct                           |
|-------------------------|------------------------------------------|
| stream                  | Stream                                   |
| document                | Document                                 |
| mapping                 | Map                                      |
| sequence                | Sequence                                 |
| scalar                  | Scalar (type per tag resolution)         |
| tag                     | Tag annotation                           |
| anchor                  | Anchor annotation                        |
| alias                   | Alias reference                          |
| comment                 | Comment node (interleaved in parent)     |
| merge key (`<<`)        | Map entry with merge Tag + Alias value   |
| %YAML directive         | Directive(name="YAML", ...)              |
| %TAG directive          | Directive(name="TAG", ...)               |

YAML's representation graph maps naturally because CDXF's graph model
(anchors + aliases) was designed to subsume it. The key preservation
guarantee: YAML anchors and aliases survive round-trip as graph structure,
not flattened into duplicate subtrees.

Note: Cyclic YAML graphs require CDXF Extended conformance (Section 11).
CDXF Core rejects cyclic graphs at encoding time.

### 8.3 XML to CDXF (Injective Mapping)

| XML Infoset item              | CDXF construct                       |
|-------------------------------|--------------------------------------|
| document information item     | Stream(Document(root=...))           |
| element information item      | Element                              |
| attribute information item    | Attribute (on parent Element)        |
| namespace declaration         | Namespace binding (on Element)       |
| character information item    | Scalar(string) as Element child      |
| comment information item      | Comment (interleaved in parent)      |
| processing instruction item   | ProcessingInstruction                |
| document type declaration     | Directive(name="DOCTYPE", ...)       |
| unexpanded entity reference   | Expanded to character content        |
| CDATA section                 | Scalar(string) with advisory flag    |

The attribute-vs-element distinction is preserved because CDXF has both
Element (with children) and Attribute (name-value on Element) as separate
constructs. Mixed content is preserved because Element children are an
ordered list that can interleave Scalar(string) and Element nodes.

CDATA sections are semantically equivalent to character data in the XML
Infoset. CDXF preserves them via an optional advisory annotation
(`is_cdata: bool`) on the Scalar node for round-trip fidelity, not as a
distinct node kind.

### 8.4 TOML to CDXF (Injective Mapping)

| TOML construct         | CDXF construct                          |
|------------------------|-----------------------------------------|
| table                  | Map                                     |
| array                  | Sequence                                |
| string                 | Scalar(string)                          |
| integer                | Scalar(integer)                         |
| float                  | Scalar(float)                           |
| boolean                | Scalar(boolean)                         |
| offset datetime        | Scalar(timestamp_offset)                |
| local datetime         | Scalar(timestamp_local)                 |
| local date             | Scalar(date)                            |
| local time             | Scalar(time)                            |
| comment                | Comment (interleaved in parent)         |
| inline table           | Map (with advisory `is_inline` flag)    |
| array of tables        | Sequence of Maps                        |

TOML's explicit datetime types map to CDXF's scalar subtypes. The
inline-table vs. standard-table distinction is advisory (affects
presentation, not semantics).

---

## 9. Semantic Equality

Two CDXF nodes are **semantically equal** if and only if:

1. They have the same node kind.
2. Their tags are identical (or both absent).
3. Their content is recursively equal:
   - Scalars: same scalar_type and same value.
   - Maps: same set of (key, value) entries by recursive equality.
     Entry order and interleaved comments are NOT considered.
   - Sequences: same non-comment items in the same order.
     Interleaved comments are NOT considered.
   - Elements: same name, namespace_uri, attributes (as a set), and
     children (as an ordered list, excluding comments), all by recursive
     equality.
   - Comments: compared by text only when comparing comment-preserving
     equality (see below).
   - ProcessingInstructions: same target and data.
4. Anchor names, alias resolution targets, advisory annotations
   (source_format_hint, is_inline, is_cdata), and namespace prefixes are
   NOT considered for semantic equality.

**Graph-aware equality.** Graph structure (shared references via
anchors/aliases) IS considered: if node A's child at position 0 and
position 2 are the same node (shared reference), then a semantically equal
graph must also share those references.

**Comment-preserving equality.** An optional stricter equality mode that also
compares interleaved comments by text and position. This is useful for tools
that treat comment changes as meaningful (e.g., configuration management
systems).

---

## 10. Canonical Form

CDXF defines a **canonical form** for use in hashing, signing, and
deterministic comparison:

1. All advisory annotations are stripped (source_format_hint, is_inline,
   is_cdata, namespace prefixes).
2. Map entries are sorted by canonical comparison of their serialized keys.
3. Namespace URIs are preserved; prefixes are dropped (or normalized to
   a deterministic assignment).
4. Anchors are renamed to sequential integers ("1", "2", "3", ...) in
   depth-first traversal order.
5. Comments and ProcessingInstructions are stripped (they are not data
   content).
6. Tags are normalized to their full URI form.
7. Scalar values are normalized: integers to canonical decimal, floats to
   IEEE 754 canonical form, strings to NFC normalization.

Two CDXF documents are **canonically equal** if their canonical forms
produce identical byte sequences under the CDXF binary encoding.

---

## 11. Conformance Levels

### 11.1 CDXF Core

CDXF Core is the default conformance level. All implementations MUST support
CDXF Core.

Requirements:
- All nine node kinds MUST be supported.
- All cross-cutting annotations (Tag, Anchor, Alias, Namespace Bindings)
  MUST be supported.
- The graph formed by Anchor/Alias resolution MUST be a directed acyclic
  graph (DAG). An encoder MUST reject documents containing cyclic references
  with a clear error.
- `allows_cycles` on Document MUST be false (or absent).

### 11.2 CDXF Extended

CDXF Extended permits cyclic graphs. Implementations MAY support CDXF
Extended.

Requirements:
- All CDXF Core requirements apply, except:
- The graph formed by Anchor/Alias resolution MAY contain cycles.
- `allows_cycles` on Document MUST be true for documents containing cycles.
- Canonical form computation for cyclic documents uses back-edge markers
  (a cycle-aware hash algorithm; details in the binary encoding spec).

Rationale: YAML's specification permits cyclic graphs via anchors and aliases,
though they are rarely used in practice. CDXF Extended ensures the formal
superset property holds for the complete YAML representation graph. CDXF Core
covers all acyclic YAML documents, which represents the overwhelming majority
of real-world YAML usage. The YAML-LD specification (W3C Community Group
Final Report, 2023) similarly defines Basic (no aliases) and Extended (aliases
allowed) profiles.

---

## 12. Extensibility

The CDXF information model is designed to accommodate future format families
(e.g., S-expressions, EDN, HCL, INI) through:

1. **New scalar subtypes** can be added via the Tag annotation without
   changing the core node kinds. A tagged Scalar with an unrecognized tag
   is preserved as-is by conforming implementations.
2. **New advisory annotations** can be defined for format-specific
   presentation details without affecting semantic equality.
3. **New Directive types** can be defined for format-specific
   document-level metadata.
4. The nine node kinds are intended to be stable. Adding a new node kind
   would constitute a major version change to the information model.

---

## 13. Design Decisions Log

The following decisions were made during the design of this information model
and are recorded here for traceability:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | XML entity references: preserve or expand? | Expand always | Matches XML Infoset behavior and parser practice. Predefined entities reconstructed on XML output. |
| 2 | YAML merge keys (`<<`): preserve or expand? | Preserve as-is | Anchor/alias machinery handles it naturally. Preserves authoring intent for config files. |
| 3 | Cyclic graphs: permit or reject? | DAG-only in Core, cycles in Extended | Simplifies canonical form and hashing. Extended level preserves full YAML superset claim. |
| 4 | Comment association: explicit binding or positional? | Ordered interleaving | Comments interleaved in parent's child list at source position. No heuristic association needed. |
| 5 | XML DTD internals: preserve or discard? | Preserve DOCTYPE identifiers only | Internal subset processed before encoding. Matches EXI/Fast Infoset behavior. |
