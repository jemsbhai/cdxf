"""CDXF CBOR codec — encode CDXF model objects to CBOR bytes and back.

Public API:
    encode(stream) -> bytes
    decode(data) -> Stream
    Encoder(canonical=False, shorthand=False)
    Decoder()
"""

from __future__ import annotations

import datetime
from collections import OrderedDict

import cbor2

from cdxf.model import (
    Alias,
    Anchor,
    Attribute,
    Comment,
    Directive,
    Document,
    Element,
    Map,
    ProcessingInstruction,
    Scalar,
    ScalarType,
    Sequence,
    SourceFormat,
    Stream,
    TagAnnotation,
)
from cdxf import tags


# ===================================================================
# Public API
# ===================================================================

def encode(stream: Stream) -> bytes:
    """Encode a CDXF Stream to CBOR bytes (default settings)."""
    return Encoder().encode(stream)


def decode(data: bytes) -> Stream:
    """Decode CBOR bytes to a CDXF Stream."""
    return Decoder().decode(data)


# ===================================================================
# Helpers
# ===================================================================

def _collect_ns_uris(node) -> set[str]:
    """Walk a tree and collect all namespace URIs."""
    uris: set[str] = set()
    if isinstance(node, Element):
        if node.namespace_uri:
            uris.add(node.namespace_uri)
        for attr in node.attributes:
            if attr.namespace_uri:
                uris.add(attr.namespace_uri)
        for child in node.children:
            uris.update(_collect_ns_uris(child))
    elif isinstance(node, Map):
        for entry in node.entries:
            if not isinstance(entry, Comment):
                uris.update(_collect_ns_uris(entry[0]))
                uris.update(_collect_ns_uris(entry[1]))
    elif isinstance(node, Sequence):
        for item in node.items:
            uris.update(_collect_ns_uris(item))
    return uris


# ===================================================================
# Encoder
# ===================================================================

class Encoder:
    """Configurable CDXF-to-CBOR encoder.

    Parameters
    ----------
    canonical : bool
        If True, produce canonical (deterministic) encoding.
    shorthand : bool
        If True and the stream is JSON-model only, emit plain CBOR.
    """

    def __init__(self, *, canonical: bool = False, shorthand: bool = False):
        self.canonical = canonical
        self.shorthand = shorthand
        self._ns_table: list[str] | None = None
        self._ns_index: dict[str, int] | None = None

    def encode(self, stream: Stream) -> bytes:
        if self.shorthand and self._is_shorthand_eligible(stream):
            return self._encode_shorthand(stream)
        return cbor2.dumps(self._encode_stream(stream))

    # ---------------------------------------------------------------
    # Shorthand: plain CBOR, no CDXF tags
    # ---------------------------------------------------------------

    def _is_shorthand_eligible(self, stream: Stream) -> bool:
        if len(stream.documents) != 1:
            return False
        doc = stream.documents[0]
        if doc.preamble or doc.postamble:
            return False
        if doc.allows_cycles:
            return False
        return self._is_json_model_node(doc.root)

    def _is_json_model_node(self, node) -> bool:
        if isinstance(node, Scalar):
            if node.tag is not None or node.anchor is not None:
                return False
            return node.scalar_type in (
                ScalarType.NULL, ScalarType.BOOLEAN, ScalarType.INTEGER,
                ScalarType.FLOAT, ScalarType.STRING,
            )
        if isinstance(node, Map):
            if node.tag is not None or node.anchor is not None:
                return False
            for entry in node.entries:
                if isinstance(entry, Comment):
                    return False
                key, value = entry
                if not self._is_json_model_node(key):
                    return False
                if not self._is_json_model_node(value):
                    return False
            return True
        if isinstance(node, Sequence):
            if node.tag is not None or node.anchor is not None:
                return False
            for item in node.items:
                if isinstance(item, Comment):
                    return False
                if not self._is_json_model_node(item):
                    return False
            return True
        return False

    def _encode_shorthand(self, stream: Stream) -> bytes:
        native = self._to_native(stream.documents[0].root)
        return cbor2.dumps(native, canonical=self.canonical)

    def _to_native(self, node) -> object:
        if isinstance(node, Scalar):
            return node.value
        if isinstance(node, Map):
            result = {}
            for entry in node.entries:
                key, value = entry
                result[self._to_native(key)] = self._to_native(value)
            return result
        if isinstance(node, Sequence):
            return [self._to_native(item) for item in node.items]
        raise ValueError(f"Cannot convert {type(node).__name__} to native")

    # ---------------------------------------------------------------
    # Full encoding: CDXF tags on CBOR
    # ---------------------------------------------------------------

    def _encode_stream(self, stream: Stream) -> cbor2.CBORTag:
        docs = [self._encode_document(doc) for doc in stream.documents]
        return cbor2.CBORTag(tags.CDXF_STREAM, docs)

    def _encode_document(self, doc: Document) -> cbor2.CBORTag:
        # Build namespace URI table if beneficial
        ns_uris = _collect_ns_uris(doc.root)
        if ns_uris:
            self._ns_table = sorted(ns_uris)
            self._ns_index = {uri: i for i, uri in enumerate(self._ns_table)}
        else:
            self._ns_table = None
            self._ns_index = None

        root_encoded = self._encode_node(doc.root)

        options = {}
        if doc.source_format_hint != SourceFormat.UNSPECIFIED:
            if not self.canonical:
                options[tags.DOC_OPT_SOURCE_FORMAT] = doc.source_format_hint.value
        if doc.allows_cycles:
            options[tags.DOC_OPT_ALLOWS_CYCLES] = True
        if doc.preamble:
            if not self.canonical:
                options[tags.DOC_OPT_PREAMBLE] = [
                    self._encode_node(n) for n in doc.preamble
                ]
        if doc.postamble:
            if not self.canonical:
                options[tags.DOC_OPT_POSTAMBLE] = [
                    self._encode_node(n) for n in doc.postamble
                ]
        if self._ns_table:
            options[tags.DOC_OPT_NS_TABLE] = self._ns_table

        # Clean up
        self._ns_table = None
        self._ns_index = None

        if options:
            return cbor2.CBORTag(tags.CDXF_DOCUMENT, [root_encoded, options])
        return cbor2.CBORTag(tags.CDXF_DOCUMENT, root_encoded)

    def _encode_node(self, node) -> object:
        result = self._encode_node_inner(node)
        if hasattr(node, "tag") and node.tag is not None:
            result = cbor2.CBORTag(tags.CDXF_TAG, [node.tag.uri, result])
        if hasattr(node, "anchor") and node.anchor is not None:
            result = cbor2.CBORTag(tags.CDXF_ANCHOR, [node.anchor.name, result])
        return result

    def _encode_node_inner(self, node) -> object:
        if isinstance(node, Scalar):
            return self._encode_scalar(node)
        if isinstance(node, Map):
            return self._encode_map(node)
        if isinstance(node, Sequence):
            return self._encode_sequence(node)
        if isinstance(node, Element):
            return self._encode_element(node)
        if isinstance(node, Alias):
            return cbor2.CBORTag(tags.CDXF_ALIAS, node.name)
        if isinstance(node, Comment):
            if self.canonical:
                return None
            return cbor2.CBORTag(tags.CDXF_COMMENT, node.text)
        if isinstance(node, ProcessingInstruction):
            if self.canonical:
                return None
            return self._encode_pi(node)
        if isinstance(node, Directive):
            return self._encode_directive(node)
        raise ValueError(f"Unknown node type: {type(node).__name__}")

    def _encode_scalar(self, scalar: Scalar) -> object:
        if scalar.scalar_type == ScalarType.NULL:
            return None
        if scalar.scalar_type == ScalarType.BOOLEAN:
            return scalar.value
        if scalar.scalar_type == ScalarType.INTEGER:
            return scalar.value
        if scalar.scalar_type == ScalarType.FLOAT:
            return scalar.value
        if scalar.scalar_type == ScalarType.STRING:
            return scalar.value
        if scalar.scalar_type == ScalarType.BYTE_STRING:
            return scalar.value

        if scalar.scalar_type == ScalarType.TIMESTAMP_OFFSET:
            iso = (scalar.value.isoformat()
                   if hasattr(scalar.value, "isoformat")
                   else str(scalar.value))
            return cbor2.CBORTag(tags.CBOR_DATETIME_RFC3339, iso)

        if scalar.scalar_type == ScalarType.TIMESTAMP_LOCAL:
            iso = (scalar.value.isoformat()
                   if hasattr(scalar.value, "isoformat")
                   else str(scalar.value))
            return cbor2.CBORTag(tags.CDXF_TIMESTAMP_LOCAL, iso)

        if scalar.scalar_type == ScalarType.DATE:
            iso = (scalar.value.isoformat()
                   if hasattr(scalar.value, "isoformat")
                   else str(scalar.value))
            return cbor2.CBORTag(tags.CDXF_DATE, iso)

        if scalar.scalar_type == ScalarType.TIME:
            iso = (scalar.value.isoformat()
                   if hasattr(scalar.value, "isoformat")
                   else str(scalar.value))
            return cbor2.CBORTag(tags.CDXF_TIME, iso)

        if scalar.scalar_type == ScalarType.DECIMAL:
            raise NotImplementedError("Decimal scalar encoding not yet implemented")
        raise ValueError(f"Unknown scalar type: {scalar.scalar_type}")

    def _has_complex_keys(self, map_node: Map) -> bool:
        for entry in map_node.entries:
            if isinstance(entry, Comment):
                continue
            key, _ = entry
            if hasattr(key, "tag") and key.tag is not None:
                return True
            if hasattr(key, "anchor") and key.anchor is not None:
                return True
        return False

    def _encode_map(self, map_node: Map) -> object:
        has_comments = any(isinstance(e, Comment) for e in map_node.entries)
        complex_keys = self._has_complex_keys(map_node)
        use_array = (has_comments and not self.canonical) or complex_keys

        if use_array:
            items = []
            for entry in map_node.entries:
                if isinstance(entry, Comment):
                    if not self.canonical:
                        items.append(cbor2.CBORTag(tags.CDXF_COMMENT, entry.text))
                else:
                    key, value = entry
                    items.append(self._encode_node(key))
                    items.append(self._encode_node(value))
            return cbor2.CBORTag(tags.CDXF_COMMENTED_MAP, items)

        entries_only = [e for e in map_node.entries if not isinstance(e, Comment)]
        pairs = []
        for key, value in entries_only:
            pairs.append((self._encode_node(key), self._encode_node(value)))

        if self.canonical:
            pairs.sort(key=lambda p: cbor2.dumps(p[0], canonical=True))

        result = OrderedDict()
        for k, v in pairs:
            result[k] = v
        return result

    def _encode_sequence(self, seq: Sequence) -> object:
        items = []
        for item in seq.items:
            if isinstance(item, Comment) and self.canonical:
                continue
            items.append(self._encode_node(item))
        return items

    def _ns_ref(self, uri: str | None) -> int | str | None:
        """Return integer index if NS table is active, else the URI."""
        if uri is None:
            return None
        if self._ns_index is not None and uri in self._ns_index:
            return self._ns_index[uri]
        return uri

    def _encode_element(self, elem: Element) -> cbor2.CBORTag:
        """Encode an Element using compact forms when possible.

        Compact forms (no namespace, saves 2-3 bytes per element):
          [name, children]                     — no NS, no attrs
          [name, attrs, children]              — attrs, no NS

        Full form (with namespace):
          [name, ns_ref, prefix, attrs, children, opt(ns_decls)]

        ns_ref is an integer index into the document's NS table when
        the table is active, otherwise the full URI string.
        """
        # Encode attributes
        has_ns_attrs = any(a.namespace_uri for a in elem.attributes)
        if has_ns_attrs:
            attrs = [self._encode_attribute(a) for a in elem.attributes]
        elif elem.attributes:
            attrs = {a.name: a.value for a in elem.attributes}
        else:
            attrs = None  # distinguish "no attrs" from "empty dict"

        # Encode children
        children = []
        for child in elem.children:
            if isinstance(child, Comment) and self.canonical:
                continue
            children.append(self._encode_node(child))

        has_ns = bool(elem.namespace_uri or elem.namespace_declarations
                      or elem.prefix)

        if not has_ns:
            # Compact forms
            if attrs is None or attrs == {}:
                # [name, children]
                content = [elem.name, children]
            else:
                # [name, attrs, children]
                content = [elem.name, attrs, children]
        else:
            # Full form with NS interning
            ns_ref = self._ns_ref(elem.namespace_uri)
            actual_attrs = attrs if attrs is not None else {}
            content = [
                elem.name,
                ns_ref,
                elem.prefix,
                actual_attrs,
                children,
            ]
            if elem.namespace_declarations:
                content.append(
                    cbor2.CBORTag(tags.CDXF_NAMESPACE, elem.namespace_declarations)
                )

        return cbor2.CBORTag(tags.CDXF_ELEMENT, content)

    def _encode_attribute(self, attr: Attribute) -> cbor2.CBORTag | list:
        if attr.namespace_uri:
            ns_ref = self._ns_ref(attr.namespace_uri)
            content = [attr.name, ns_ref, attr.value]
            if attr.prefix:
                content.append(attr.prefix)
            return cbor2.CBORTag(tags.CDXF_ATTRIBUTE, content)
        return [attr.name, attr.value]

    def _encode_pi(self, pi: ProcessingInstruction) -> cbor2.CBORTag:
        content = [pi.target]
        if pi.data is not None:
            content.append(pi.data)
        else:
            content.append(None)
        return cbor2.CBORTag(tags.CDXF_PI, content)

    def _encode_directive(self, directive: Directive) -> cbor2.CBORTag:
        return cbor2.CBORTag(tags.CDXF_DIRECTIVE, [directive.name, directive.parameters])


# ===================================================================
# Decoder
# ===================================================================

class Decoder:
    """Decode CBOR bytes to CDXF model objects."""

    def __init__(self):
        self._ns_table: list[str] | None = None

    def decode(self, data: bytes) -> Stream:
        raw = cbor2.loads(data)
        if isinstance(raw, cbor2.CBORTag) and raw.tag == tags.CDXF_STREAM:
            return self._decode_stream(raw)
        root = self._decode_value(raw)
        return Stream(documents=[Document(root=root)])

    def _decode_stream(self, tag: cbor2.CBORTag) -> Stream:
        docs = [self._decode_document(d) for d in tag.value]
        return Stream(documents=docs)

    def _decode_document(self, raw) -> Document:
        if not isinstance(raw, cbor2.CBORTag) or raw.tag != tags.CDXF_DOCUMENT:
            raise ValueError(f"Expected CDXF_DOCUMENT tag, got {type(raw)}")

        content = raw.value

        if not isinstance(content, list):
            self._ns_table = None
            root = self._decode_value(content)
            return Document(root=root)

        if (len(content) == 2
                and isinstance(content[1], dict)
                and all(isinstance(k, int) for k in content[1])):
            options = content[1]

            # Read NS table before decoding root
            self._ns_table = options.get(tags.DOC_OPT_NS_TABLE)

            root = self._decode_value(content[0])

            doc = Document(
                root=root,
                source_format_hint=SourceFormat(
                    options.get(tags.DOC_OPT_SOURCE_FORMAT, 0)
                ),
                allows_cycles=options.get(tags.DOC_OPT_ALLOWS_CYCLES, False),
                preamble=[
                    self._decode_value(n)
                    for n in options.get(tags.DOC_OPT_PREAMBLE, [])
                ],
                postamble=[
                    self._decode_value(n)
                    for n in options.get(tags.DOC_OPT_POSTAMBLE, [])
                ],
            )
            self._ns_table = None
            return doc

        self._ns_table = None
        root = self._decode_value(content)
        return Document(root=root)

    def _resolve_ns(self, ref) -> str | None:
        """Resolve a namespace reference: integer → table lookup, else pass through."""
        if ref is None:
            return None
        if isinstance(ref, int) and self._ns_table is not None:
            return self._ns_table[ref]
        return ref

    def _decode_value(self, raw) -> object:
        if isinstance(raw, cbor2.CBORTag):
            return self._decode_tagged(raw)
        if isinstance(raw, dict):
            return self._decode_map(raw)
        if isinstance(raw, list):
            return self._decode_sequence(raw)
        if raw is None:
            return Scalar(ScalarType.NULL, None)
        if isinstance(raw, bool):
            return Scalar(ScalarType.BOOLEAN, raw)
        if isinstance(raw, int):
            return Scalar(ScalarType.INTEGER, raw)
        if isinstance(raw, float):
            return Scalar(ScalarType.FLOAT, raw)
        if isinstance(raw, str):
            return Scalar(ScalarType.STRING, raw)
        if isinstance(raw, bytes):
            return Scalar(ScalarType.BYTE_STRING, raw)
        if isinstance(raw, datetime.datetime):
            if raw.tzinfo is not None:
                return Scalar(ScalarType.TIMESTAMP_OFFSET, raw)
            return Scalar(ScalarType.TIMESTAMP_LOCAL, raw)
        if isinstance(raw, datetime.date):
            return Scalar(ScalarType.DATE, raw)
        if isinstance(raw, datetime.time):
            return Scalar(ScalarType.TIME, raw)
        raise ValueError(f"Unexpected CBOR value type: {type(raw)}")

    def _decode_tagged(self, tag: cbor2.CBORTag) -> object:
        t = tag.tag
        v = tag.value

        if t == tags.CBOR_DATETIME_RFC3339:
            if isinstance(v, datetime.datetime):
                return Scalar(ScalarType.TIMESTAMP_OFFSET, v)
            if isinstance(v, str):
                return Scalar(ScalarType.TIMESTAMP_OFFSET,
                              datetime.datetime.fromisoformat(v))
            return Scalar(ScalarType.TIMESTAMP_OFFSET, v)

        if t == tags.CDXF_TIMESTAMP_LOCAL:
            if isinstance(v, str):
                return Scalar(ScalarType.TIMESTAMP_LOCAL,
                              datetime.datetime.fromisoformat(v))
            if isinstance(v, datetime.datetime):
                return Scalar(ScalarType.TIMESTAMP_LOCAL, v)
            return Scalar(ScalarType.TIMESTAMP_LOCAL, v)

        if t == tags.CDXF_DATE:
            if isinstance(v, str):
                return Scalar(ScalarType.DATE,
                              datetime.date.fromisoformat(v))
            if isinstance(v, datetime.date):
                return Scalar(ScalarType.DATE, v)
            return Scalar(ScalarType.DATE, v)

        if t == tags.CDXF_TIME:
            if isinstance(v, str):
                return Scalar(ScalarType.TIME,
                              datetime.time.fromisoformat(v))
            if isinstance(v, datetime.time):
                return Scalar(ScalarType.TIME, v)
            return Scalar(ScalarType.TIME, v)

        if t == tags.CDXF_COMMENT:
            return Comment(v)

        if t == tags.CDXF_ALIAS:
            return Alias(v)

        if t == tags.CDXF_ANCHOR:
            anchor_name = v[0]
            inner = self._decode_value(v[1])
            if hasattr(inner, "anchor"):
                inner.anchor = Anchor(anchor_name)
            return inner

        if t == tags.CDXF_TAG:
            tag_uri = v[0]
            inner = self._decode_value(v[1])
            if hasattr(inner, "tag"):
                inner.tag = TagAnnotation(tag_uri)
            return inner

        if t == tags.CDXF_ELEMENT:
            return self._decode_element(v)

        if t == tags.CDXF_PI:
            target = v[0]
            data = v[1] if len(v) > 1 else None
            return ProcessingInstruction(target=target, data=data)

        if t == tags.CDXF_DIRECTIVE:
            return Directive(name=v[0], parameters=v[1])

        if t == tags.CDXF_COMMENTED_MAP:
            return self._decode_commented_map(v)

        if t == tags.CDXF_NAMESPACE:
            return v

        if t == tags.CDXF_STREAM:
            return self._decode_stream(tag)

        if t == tags.CDXF_DOCUMENT:
            return self._decode_document(tag)

        inner = self._decode_value(v)
        return inner

    def _decode_map(self, raw: dict) -> Map:
        entries = []
        for k, v in raw.items():
            key_node = self._decode_value(k)
            value_node = self._decode_value(v)
            entries.append((key_node, value_node))
        return Map(entries=entries)

    def _decode_commented_map(self, items: list) -> Map:
        entries = []
        i = 0
        while i < len(items):
            item = items[i]
            if isinstance(item, cbor2.CBORTag) and item.tag == tags.CDXF_COMMENT:
                entries.append(Comment(item.value))
                i += 1
            else:
                key = self._decode_value(item)
                value = self._decode_value(items[i + 1])
                entries.append((key, value))
                i += 2
        return Map(entries=entries)

    def _decode_sequence(self, raw: list) -> Sequence:
        items = [self._decode_value(item) for item in raw]
        return Sequence(items=items)

    def _decode_element(self, content: list) -> Element:
        """Decode a CDXF_ELEMENT content array.

        Supports four forms:
          Compact:  [name, children]                       (len 2)
          Compact:  [name, attrs, children]                (len 3)
          Full old: [name, ns_uri, attrs, children, ...]   (len 4+, content[2] is dict/list)
          Full new: [name, ns_ref, prefix, attrs, children, ...]  (len 5+, content[2] is str/None/int)
        """
        n = len(content)

        if n == 2:
            # Compact: [name, children]
            name = content[0]
            children = [self._decode_value(c) for c in content[1]]
            return Element(name=name, children=children)

        if n == 3:
            # Compact: [name, attrs, children]
            name = content[0]
            raw_attrs = content[1]
            children = [self._decode_value(c) for c in content[2]]
            attributes = self._decode_attrs(raw_attrs)
            return Element(name=name, attributes=attributes, children=children)

        # Full form — detect old vs new by type of content[2]
        name = content[0]
        ns_ref = content[1]

        if n >= 3 and not isinstance(content[2], (dict, list)):
            # New format: content[2] is prefix (str, None, or int)
            prefix = content[2]
            raw_attrs = content[3] if n > 3 else {}
            raw_children = content[4] if n > 4 else []
            ns_start = 5
        else:
            # Old format: content[2] is attrs (dict or list)
            prefix = None
            raw_attrs = content[2] if n > 2 else {}
            raw_children = content[3] if n > 3 else []
            ns_start = 4

        namespace_uri = self._resolve_ns(ns_ref)
        attributes = self._decode_attrs(raw_attrs)
        children = [self._decode_value(c) for c in raw_children]

        namespace_declarations = {}
        if n > ns_start:
            ns_raw = content[ns_start]
            if isinstance(ns_raw, cbor2.CBORTag) and ns_raw.tag == tags.CDXF_NAMESPACE:
                namespace_declarations = ns_raw.value
            elif isinstance(ns_raw, dict):
                namespace_declarations = ns_raw

        return Element(
            name=name,
            namespace_uri=namespace_uri,
            prefix=prefix,
            attributes=attributes,
            children=children,
            namespace_declarations=namespace_declarations,
        )

    def _decode_attrs(self, raw_attrs) -> list[Attribute]:
        """Decode attributes from dict or list form."""
        attributes = []
        if isinstance(raw_attrs, dict):
            for k, v in raw_attrs.items():
                attributes.append(Attribute(name=k, value=v))
        elif isinstance(raw_attrs, list):
            for item in raw_attrs:
                if isinstance(item, cbor2.CBORTag) and item.tag == tags.CDXF_ATTRIBUTE:
                    a = item.value
                    ns_uri = self._resolve_ns(a[1])
                    attributes.append(Attribute(
                        name=a[0],
                        namespace_uri=ns_uri,
                        value=a[2],
                        prefix=a[3] if len(a) > 3 else None,
                    ))
                elif isinstance(item, list):
                    attributes.append(Attribute(name=item[0], value=item[1]))
        return attributes
