# Literature Survey: Toward a Universal Binary Interchange Format for Human-Readable Data Serialization Languages

**Muntaser Syed**
Florida Institute of Technology, Melbourne, FL

**Date:** May 2026

---

## 1. Introduction

Human-readable data serialization formats --- JSON, YAML, XML, TOML --- are foundational to modern computing. Each defines its own *information model*: the abstract set of constructs (nodes, attributes, references, comments, type annotations, etc.) that a document in that format can express. Over decades, binary counterparts have emerged for individual formats, optimizing wire size and parse speed. However, no existing binary format preserves a *superset* information model spanning multiple human-readable format families, enabling lossless round-trip interchange across them. This survey systematically catalogs the state of the art to delineate that gap.

The scope of this survey covers schema-less and schema-driven binary serialization formats, binary XML standards, text-binary dual formats, and adjacent proposals. We explicitly exclude domain-specific binary formats (HDF5, Parquet, Arrow) that target columnar/scientific data rather than general-purpose document interchange.

---

## 2. Background: Information Models of Major Text Formats

### 2.1 JSON (RFC 8259, ECMA-404)

JSON's data model is minimal: objects (unordered key-value maps with string keys), arrays (ordered sequences), strings, numbers, booleans, and null. It has no comment syntax, no type annotation mechanism, no reference/alias system, and no multi-document streaming. Its simplicity is both its greatest strength and its fundamental limitation.

### 2.2 YAML (Spec 1.2.2, 2021)

YAML defines a three-layer processing model: presentation stream, serialization tree, and representation graph. The representation graph is a directed graph (not merely a tree) consisting of three node kinds --- mappings, sequences, and scalars --- each annotated with a *tag* (a URI-based type identifier). YAML's information model includes features absent from JSON:

- **Anchors and aliases** enabling DAG (directed acyclic graph) and cyclic graph structures via shared node references
- **Tags** (e.g., `!!int`, `!!binary`, or custom URIs) providing explicit type annotations
- **Multi-document streams** (delimited by `---` and `...`)
- **Comments** (discarded during composition but present in the serialization layer)
- **Directives** (`%YAML`, `%TAG`) controlling parser behavior

### 2.3 XML (W3C XML 1.0, XML Information Set)

The W3C XML Information Set (Infoset) defines eleven information item types: document, element, attribute, processing instruction, unexpanded entity reference, character, comment, document type declaration, unparsed entity, notation, and namespace. Key features beyond JSON/YAML include:

- **Attribute-vs-element distinction** (structurally significant; attributes are unordered, elements are ordered)
- **Namespaces** (URI-based, with prefix bindings)
- **Mixed content** (character data interleaved with child elements)
- **Processing instructions** (application-targeted metadata)
- **CDATA sections** (raw character data blocks)
- **Document type declarations** (DTDs) with entity and notation definitions

### 2.4 TOML (v1.0.0 / v1.1.0)

TOML maps unambiguously to a hash table (nested key-value structure). Its data model includes strings, integers, floats, booleans, offset/local datetimes, local dates, local times, arrays, and tables (including inline tables and arrays of tables). TOML supports comments but has no reference system, no namespace mechanism, no mixed content, and no multi-document streams.

### 2.5 Summary of Information Model Divergences

| Feature                     | JSON | YAML | XML  | TOML |
|-----------------------------|------|------|------|------|
| Key-value maps              | Yes  | Yes  | Yes* | Yes  |
| Ordered sequences           | Yes  | Yes  | Yes  | Yes  |
| Scalars with type tags      | No   | Yes  | No** | Yes*** |
| Graph references (DAG)      | No   | Yes  | No   | No   |
| Attributes vs. elements     | No   | No   | Yes  | No   |
| Namespaces                  | No   | No   | Yes  | No   |
| Mixed content               | No   | No   | Yes  | No   |
| Comments                    | No   | Yes  | Yes  | Yes  |
| Multi-document streams      | No   | Yes  | No   | No   |
| Processing instructions     | No   | No   | Yes  | No   |
| Native datetime types       | No   | Yes  | No   | Yes  |
| Native binary (byte string) | No   | Yes† | No   | No   |

\* XML attributes are unordered; element children are ordered.
\** XML Schema provides type information external to the document.
\*** TOML's type system is implicit in syntax rather than explicit tags.
† Via `!!binary` tag with Base64 encoding in the text representation.

---

## 3. Existing Binary Serialization Formats

### 3.1 JSON-Compatible Binary Formats (Schema-Less)

The most comprehensive academic treatment of this category is Viotti and Kinderkhedia's 2022 survey and benchmark (arXiv:2201.02089, arXiv:2201.03051), which systematically analyzed thirteen specifications across schema-driven and schema-less categories. We summarize the schema-less formats:

**CBOR (RFC 8949, Concise Binary Object Representation).** Standardized by the IETF, CBOR uses a major-type/additional-info byte encoding scheme. Its data model extends JSON with byte strings, semantic tags (64-bit tag namespace registered via IANA), and indefinite-length encoding for streaming. CBOR tags provide extensible type annotation (e.g., tag 0 for RFC 3339 datetimes, tag 2 for bignums). CBOR has a defined canonical/deterministic encoding profile (Section 4.2), refined further by dCBOR (draft-mcnally-deterministic-cbor) and CBOR Common Deterministic Encoding (CDE, draft-ietf-cbor-cde). Schema validation is available via CDDL (RFC 8610). CBOR is mandated by FIDO2/WebAuthn, CoAP, COSE, and CWT. Despite its tag extensibility, CBOR's core data model remains JSON-isomorphic: maps, arrays, and scalars. It cannot natively represent YAML anchors/aliases, XML attributes-vs-elements, namespaces, or mixed content.

**MessagePack.** A compact, self-describing binary format with types closely mirroring JSON (nil, bool, integers, floats, strings, binary data, arrays, maps) plus an Extension type mechanism allowing application-defined types via signed 8-bit type codes. MessagePack is widely used in Redis, caching, and game networking. Its Extension type system is less expressive than CBOR's 64-bit tag space and lacks IETF standardization. Like CBOR, the information model is JSON-isomorphic.

**BSON (Binary JSON).** Developed by MongoDB for internal storage and wire format. BSON extends JSON with types including binary data, ObjectId, datetime, regex, JavaScript code, Decimal128, and others MongoDB-specific. BSON is traversable (length-prefixed values enable skipping) but often larger than JSON for small documents due to overhead. Its type extensions are MongoDB-centric rather than general-purpose.

**Other schema-less formats.** FlexBuffers (Google, part of the FlatBuffers project), Smile (Jackson binary format for Java), and UBJSON (Universal Binary JSON) all encode JSON-model data in binary with minor type extensions. None extends the information model beyond the JSON tree.

**SuperPack (Shape Security, 2016).** An extensible schema-less binary format optimized for repeated structure (arrays of homogeneous objects, string deduplication). SuperPack uses 256 type tags and an extension mechanism allowing custom serializers. It achieved the smallest payload in Shape Security's benchmarks against YAML, JSON, BSON, and MessagePack for structured API data. However, its extensions are application-defined at encode time, not standardized, and the core model remains JSON-isomorphic.

### 3.2 JSON-Compatible Binary Formats (Schema-Driven)

**Protocol Buffers (Google).** Schema-required (.proto files), field-number-based encoding. Extremely compact and fast, the dominant format for gRPC. No self-description; the schema is required for decoding.

**Apache Avro.** Schema-driven, with the schema embedded or exchanged via a registry. Designed for schema evolution in Hadoop/Kafka ecosystems. Uses a row-oriented binary encoding.

**Cap'n Proto / FlatBuffers.** Zero-copy, in-place-readable binary formats. Schema-required, optimized for deserialization speed by avoiding parsing entirely.

**ASN.1 (ITU-T X.680 family).** The oldest and most general schema-driven serialization framework, with multiple encoding rules: BER/DER (tag-length-value), PER (packed, extremely compact), XER (XML encoding), and JER (JSON encoding). ASN.1's type system is far richer than JSON's (includes ENUMERATED, BIT STRING, OBJECT IDENTIFIER, CHOICE, SEQUENCE OF, SET OF, etc.). ASN.1/PER forms the basis of Fast Infoset (see 3.3).

**Microsoft Bond, Apache Thrift.** Additional schema-driven formats with similar trade-offs: compact binary, schema evolution support, cross-language codegen. Both encode a JSON-compatible data model.

All schema-driven formats share a fundamental characteristic: the information model is defined by the schema, not by the format. They cannot encode arbitrary YAML or XML documents without a predefined schema describing their structure.

### 3.3 Binary XML Formats

**EXI (W3C Efficient XML Interchange, Rec. 2011; 2nd Ed. 2014).** The most mature binary XML standard. EXI uses a grammar-driven approach: it builds a state machine from the XML document's structure (optionally informed by an XML Schema) and encodes events (start-element, attribute, characters, etc.) as compact event codes. EXI achieves compression competitive with gzip while maintaining streaming, random-access, and fast-parse properties. EXI preserves the full XML Information Set losslessly. In November 2016, the W3C working group was renamed from "Efficient XML Interchange" to "Efficient Extensible Interchange" to reflect broader applicability ambitions. A working draft for EXI4JSON (EXI for JSON) was published in 2016, demonstrating that the EXI encoding can represent JSON data. However, EXI4JSON maps JSON into XML's information model (elements and attributes), losing JSON's structural semantics. No EXI for YAML or EXI for TOML work has been published.

**Fast Infoset (ITU-T X.891 / ISO/IEC 24824-1, 2005/2007).** An ASN.1-based binary encoding of the XML Information Set. Fast Infoset uses dynamic string tables to index repeated element/attribute names and values, providing both size reduction and parse-speed improvement. Like EXI, it is lossless for the XML Infoset. Unlike EXI, it does not use schema-informed grammar compression, making it simpler but less compact. Fast Infoset was adopted by the Web3D community for X3D document encoding.

**BiM (Binary MPEG format for XML, ISO/IEC 23001-1).** An MPEG-standardized binary XML format optimized for multimedia metadata and MPEG-7 descriptions. Schema-required. Limited adoption outside MPEG ecosystems.

All three binary XML formats are tightly coupled to the XML Information Set. None can represent YAML-specific constructs (anchors/aliases, tags, multi-document streams) or TOML-specific constructs (inline tables, local datetimes) without lossy transformation.

### 3.4 Text-Binary Dual Formats

**Amazon Ion.** Ion is the closest existing format to the idea of a text-binary dual encoding with a rich type system. Ion's text form is a superset of JSON, adding: annotations (arbitrary symbols attached to values, similar to YAML tags), timestamps with arbitrary precision, blob/clob types, s-expressions (symbolic expressions), typed null values, and comments. Ion's binary encoding preserves the same data model as the text form, enabling lossless bidirectional conversion. Ion 1.1 adds macros and modules for further efficiency. Ion supports skip-scan parsing in binary mode (length-prefixed values allow jumping over unneeded data) and symbol tables for string deduplication. Ion is used extensively within Amazon's infrastructure.

**Ion's limitations relative to a universal format:**
- No graph references (anchors/aliases) --- tree-only
- No attribute-vs-element distinction --- everything is struct fields or list elements
- No namespace mechanism
- No mixed content model
- No multi-document streams (Ion does support multiple top-level values in a stream, but these lack YAML's explicit document delimiters and directives)
- Annotations are simpler than YAML's full tag URI system

Ion is the strongest prior art and the most important comparator for any proposed universal format.

---

## 4. Adjacent Work

### 4.1 YAML-LD (W3C Community Group Final Report, 2023)

YAML-LD defines how to use YAML as a serialization syntax for JSON-LD (Linked Data). It specifies an extended internal representation that preserves YAML-native types (numbers, booleans, null) and YAML tags (mapped to RDF datatypes). The Basic profile prohibits alias nodes; the Extended profile allows them. YAML-LD demonstrates the feasibility of bridging YAML's type/tag system with RDF/JSON-LD's semantic data model, but it is a *mapping specification*, not a binary format.

### 4.2 JSON BinPack (Viotti, 2022)

Introduced alongside the Viotti-Kinderkhedia benchmark suite, JSON BinPack explores schema-driven and schema-less binary encoding strategies specifically optimized for JSON, achieving space efficiency competitive with Protobuf in schema-driven mode. It remains JSON-model-only.

### 4.3 Gordian Envelope (Blockchain Commons)

Gordian Envelope is a "smart document" format built on dCBOR, using CBOR tag 200. It provides a Merkle-tree-based structure enabling selective disclosure, encryption, and signing of document subsets. While not a general interchange format, Envelope demonstrates that CBOR's tag system can support complex document structures with cryptographic properties. Its hash-based design requires deterministic encoding (dCBOR).

### 4.4 Cross-Format Conversion Research

Academic work on JSON-XML conversion is extensive but consistently acknowledges information loss. XML attributes, namespaces, mixed content, and processing instructions have no natural JSON representation. Conversely, JSON's distinction between null, absent keys, and empty arrays has no clean XML mapping. YAML-to-JSON conversion loses anchors/aliases (flattened to duplicated subtrees), comments, tags, and directives. These losses are well-documented but no format has been designed to avoid them.

---

## 5. Identified Gap

The literature reveals a clear, unoccupied position in the design space:

**No existing binary serialization format defines an information model that is a superset of JSON, YAML, XML, and TOML, enabling lossless round-trip encoding of documents from any of these format families.**

Specifically:

1. **All JSON-compatible binary formats** (CBOR, MessagePack, BSON, etc.) encode a tree of maps, arrays, and scalars. They cannot represent YAML's graph structure (anchors/aliases), XML's attribute-element distinction, XML namespaces, or XML mixed content without lossy transformation.

2. **All binary XML formats** (EXI, Fast Infoset, BiM) encode the XML Information Set. They cannot represent YAML anchors/aliases, YAML tags as first-class constructs, or TOML-native types without routing through XML's element/attribute model --- a lossy and semantically awkward transformation.

3. **Amazon Ion** comes closest to a universal model but remains a JSON superset without graph references, namespaces, attribute-element duality, or mixed content.

4. **The W3C's own renaming of EXI** from "Efficient XML Interchange" to "Efficient Extensible Interchange" in 2016, and the creation of EXI4JSON, signal institutional recognition that a single binary encoding *could* serve multiple text formats. However, EXI4JSON maps JSON into XML's model rather than defining a true superset, and no EXI-for-YAML work exists.

5. **No academic paper** in the surveyed literature proposes or evaluates a binary format whose information model is formally proven to be a superset of multiple text format information models, with lossless round-trip guarantees.

---

## 6. Requirements for a Novel Contribution

Based on the gap analysis, a publishable contribution would need to:

### 6.1 Core Requirements

R1. **Define a formal information model** (analogous to the W3C XML Information Set) that is a provable superset of JSON's data model (RFC 8259), YAML's representation graph (Spec 1.2.2), the XML Information Set (W3C Rec.), and TOML's data model (v1.0.0).

R2. **Specify a binary encoding** of this information model that is self-describing (decodable without an external schema), compact (competitive with CBOR/MessagePack on JSON-model data), and streamable.

R3. **Prove lossless round-trip** for each source format: for any well-formed JSON/YAML/XML/TOML document D, encoding D into the universal binary format and decoding it back to the original format family produces a document semantically equivalent to D (modulo whitespace/formatting, as is standard for binary interchange).

R4. **Provide a reference implementation** with benchmarks against CBOR, MessagePack, EXI, and Ion on representative corpora from each format family.

### 6.2 Specific Information Model Features Required

The superset information model must include at minimum:

- Ordered sequences (JSON arrays, YAML sequences, XML element children)
- Key-value maps with string keys (JSON objects, YAML mappings, TOML tables)
- Scalar types: null, boolean, integer, float, string, byte string, datetime
- Semantic type tags (YAML tags, CBOR tags, Ion annotations)
- Graph references / node identity (YAML anchors and aliases)
- Named attributes distinct from child elements (XML attributes)
- Namespace bindings (XML namespaces)
- Mixed content (XML character data interleaved with elements)
- Comments (YAML, XML, TOML)
- Processing instructions (XML)
- Multi-document streams (YAML)
- Document-level directives and metadata

### 6.3 Desirable Properties

- Deterministic/canonical encoding mode (for hashing, signing; cf. dCBOR)
- Extensible tag namespace (for future format families; cf. CBOR IANA tag registry)
- Incremental/streaming decode
- Skip-scan capability (length-prefixed values; cf. Ion, BSON)

---

## 7. Candidate Venues

Given the format specification + systems evaluation nature of this work:

**Tier 1 (highest impact):**
- SIGMOD / VLDB (if framed around data management and query-over-binary-interchange)
- USENIX ATC / OSDI (if framed around systems performance)
- WWW (The Web Conference) (natural home for interchange formats; EXI, JSON-LD, YAML-LD all have W3C lineage)

**Tier 1 Standards Track (parallel path, high legitimacy):**
- IETF Internet-Draft (the path CBOR, dCBOR, and Ion have followed; RFC publication is the gold standard for interchange format specifications)
- W3C Community Group or Working Group (following the EXI model)

**Tier 2 (strong, more accessible):**
- IEEE ICDE (data engineering; serialization is core infrastructure)
- ACM Middleware (format-level interoperability fits middleware concerns)
- IEEE INFOCOM (if network efficiency is a primary evaluation axis)
- SoftwareX / JOSS (if the primary contribution is the reference implementation)

**Workshops:**
- W3C Workshop on Binary Interchange (if convened; precedent: 2003 W3C Workshop on Binary Interchange)
- IETF CBOR Working Group (for community feedback pre-publication)

---

## 8. Key References

### Standards and Specifications

1. C. Bormann and P. Hoffman, "Concise Binary Object Representation (CBOR)," RFC 8949, IETF, Dec. 2020.
2. H. Birkholz et al., "Concise Data Definition Language (CDDL)," RFC 8610, IETF, Jun. 2019.
3. W. McNally and C. Allen, "Deterministic CBOR (dCBOR)," Internet-Draft draft-mcnally-deterministic-cbor, IETF, 2023.
4. C. Bormann, "CBOR Common Deterministic Encoding (CDE)," Internet-Draft draft-ietf-cbor-cde, IETF, 2023.
5. J. Schneider et al., "Efficient XML Interchange (EXI) Format 1.0 (Second Edition)," W3C Recommendation, Feb. 2014.
6. D. Peintner and D. Brutzman, "EXI for JSON (EXI4JSON)," W3C Working Draft, Aug. 2016.
7. ITU-T, "Information technology -- Generic applications of ASN.1: Fast Infoset," Rec. X.891 / ISO/IEC 24824-1, 2005.
8. ISO/IEC, "Binary MPEG format for XML (BiM)," ISO/IEC 23001-1, 2006.
9. O. Ben-Kiki, C. Evans, and I. dot Net, "YAML Ain't Markup Language (YAML) Version 1.2," Revision 1.2.2, Oct. 2021.
10. T. Bray, Ed., "The JavaScript Object Notation (JSON) Data Exchange Format," RFC 8259, IETF, Dec. 2017.
11. W3C, "XML Information Set (Second Edition)," W3C Recommendation, Feb. 2004.
12. T. Preston-Werner et al., "TOML: Tom's Obvious, Minimal Language," v1.0.0, Jan. 2021.
13. Amazon, "Amazon Ion Specification," https://amazon-ion.github.io/ion-docs/, 2016--present.
14. G. Kellogg et al., "YAML-LD," W3C Community Group Final Report, Dec. 2023.
15. F. Furukawa, "MessagePack specification," https://msgpack.org/, 2008--present.
16. MongoDB Inc., "BSON Specification," https://bsonspec.org/, 2009--present.

### Survey and Benchmark Papers

17. J. C. Viotti and M. Kinderkhedia, "A Survey of JSON-compatible Binary Serialization Specifications," arXiv:2201.02089, Jan. 2022.
18. J. C. Viotti and M. Kinderkhedia, "A Benchmark of JSON-compatible Binary Serialization Specifications," arXiv:2201.03051, Jan. 2022.
19. S. L. Snyder, "Efficient XML Interchange (EXI) Compression and Performance Benefits," M.S. Thesis, Naval Postgraduate School, Mar. 2010.
20. B. Hill, "Evaluation of Efficient XML Interchange (EXI) for Large Datasets and as an Alternative to Binary JSON Encodings," M.S. Thesis, Naval Postgraduate School, Mar. 2015.
21. W3C EXI Working Group, "Efficient XML Interchange Evaluation," W3C Working Draft, Apr. 2009.

### Related Systems and Formats

22. Shape Security, "SuperPack: Extensible Schemaless Binary Encoding Format," GitHub: shapesecurity/superpack-spec, 2016.
23. Blockchain Commons, "Gordian Envelope Structured Data Format," Internet-Draft, 2023.
24. Google, "FlatBuffers / FlexBuffers," https://google.github.io/flatbuffers/.
25. Apache Software Foundation, "Apache Avro Specification," https://avro.apache.org/.
26. Google, "Protocol Buffers Language Guide," https://protobuf.dev/.

---

## 9. Conclusion

The binary serialization landscape is mature for individual format families but fragmented across them. CBOR and MessagePack serve JSON. EXI and Fast Infoset serve XML. Nothing serves YAML or TOML. And critically, nothing serves *all of them* through a single unified information model. The W3C's own trajectory with EXI --- renaming it "Efficient Extensible Interchange" and exploring EXI4JSON --- signals recognition that unification is desirable, but the approach of mapping everything into XML's model is semantically lossy for non-XML formats.

A format that defines a superset information model encompassing the constructs of JSON, YAML, XML, and TOML, with a compact self-describing binary encoding, would fill a genuine gap. The contribution would be novel (no prior work proposes this), practically useful (eliminating lossy format conversion in polyglot systems), and theoretically grounded (requiring formal proof of information model containment). This positions it well for a top-tier systems or data engineering venue, or for IETF standardization.
