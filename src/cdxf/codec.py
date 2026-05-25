"""CDXF CBOR codec — encode CDXF model objects to CBOR bytes and back.

Public API:
    encode(stream) -> bytes
    decode(data) -> Stream
    Encoder(canonical=False, shorthand=False)
    Decoder()
"""

from __future__ import annotations

import datetime

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
# Pre-computed lookup tables
# ===================================================================

_PASSTHROUGH_SCALARS = frozenset({
    ScalarType.NULL, ScalarType.BOOLEAN, ScalarType.INTEGER,
    ScalarType.FLOAT, ScalarType.STRING, ScalarType.BYTE_STRING,
})

_TEMPORAL_TAGS = {
    ScalarType.TIMESTAMP_OFFSET: tags.CBOR_DATETIME_RFC3339,
    ScalarType.TIMESTAMP_LOCAL: tags.CDXF_TIMESTAMP_LOCAL,
    ScalarType.DATE: tags.CDXF_DATE,
    ScalarType.TIME: tags.CDXF_TIME,
}

_SENTINEL = object()  # unique marker for _try_native failure


# ===================================================================
# Encoder
# ===================================================================

class Encoder:
    """Configurable CDXF-to-CBOR encoder."""

    def __init__(self, *, canonical: bool = False, shorthand: bool = False):
        self.canonical = canonical
        self.shorthand = shorthand
        self._ns_table: list[str] | None = None
        self._ns_index: dict[str, int] | None = None

    def encode(self, stream: Stream) -> bytes:
        if self.shorthand and self._is_shorthand_eligible(stream):
            return self._encode_shorthand(stream)
        # Fast path: build minimal Python structure, let cbor2 C do the rest
        return cbor2.dumps(self._encode_stream(stream))

    # ---------------------------------------------------------------
    # Shorthand: plain CBOR, no CDXF tags
    # ---------------------------------------------------------------

    def _is_shorthand_eligible(self, stream: Stream) -> bool:
        if len(stream.documents) != 1:
            return False
        doc = stream.documents[0]
        if doc.preamble or doc.postamble or doc.allows_cycles:
            return False
        return self._is_json_model_node(doc.root)

    def _is_json_model_node(self, node) -> bool:
        t = type(node)
        if t is Scalar:
            return (node.tag is None and node.anchor is None
                    and node.scalar_type in _PASSTHROUGH_SCALARS)
        if t is Map:
            if node.tag is not None or node.anchor is not None:
                return False
            for entry in node.entries:
                if type(entry) is Comment:
                    return False
                if not (self._is_json_model_node(entry[0])
                        and self._is_json_model_node(entry[1])):
                    return False
            return True
        if t is Sequence:
            if node.tag is not None or node.anchor is not None:
                return False
            for item in node.items:
                if type(item) is Comment:
                    return False
                if not self._is_json_model_node(item):
                    return False
            return True
        return False

    def _encode_shorthand(self, stream: Stream) -> bytes:
        return cbor2.dumps(
            self._to_native(stream.documents[0].root),
            canonical=self.canonical,
        )

    def _to_native(self, node) -> object:
        """Convert JSON-model CDXF tree to plain Python types (fast)."""
        t = type(node)
        if t is Scalar:
            return node.value
        if t is Map:
            return {self._to_native(e[0]): self._to_native(e[1])
                    for e in node.entries}
        if t is Sequence:
            return [self._to_native(item) for item in node.items]
        raise ValueError(f"Cannot convert {t.__name__} to native")

    def _try_native(self, node) -> object:
        """Single-pass: try converting to native Python types.

        Returns the native value on success, or _SENTINEL if the tree
        contains non-JSON-model nodes (comments, anchors, tags, etc.).
        Avoids the double-walk of check-then-convert.
        """
        t = type(node)
        if t is Scalar:
            if node.tag is not None or node.anchor is not None:
                return _SENTINEL
            if node.scalar_type in _PASSTHROUGH_SCALARS:
                return node.value
            return _SENTINEL
        if t is Map:
            if node.tag is not None or node.anchor is not None:
                return _SENTINEL
            result = {}
            for entry in node.entries:
                if type(entry) is Comment:
                    return _SENTINEL
                k = self._try_native(entry[0])
                if k is _SENTINEL:
                    return _SENTINEL
                v = self._try_native(entry[1])
                if v is _SENTINEL:
                    return _SENTINEL
                result[k] = v
            return result
        if t is Sequence:
            if node.tag is not None or node.anchor is not None:
                return _SENTINEL
            items = []
            for item in node.items:
                if type(item) is Comment:
                    return _SENTINEL
                val = self._try_native(item)
                if val is _SENTINEL:
                    return _SENTINEL
                items.append(val)
            return items
        return _SENTINEL

    # ---------------------------------------------------------------
    # Full encoding
    # ---------------------------------------------------------------

    def _encode_stream(self, stream: Stream) -> cbor2.CBORTag:
        docs = [self._encode_document(doc) for doc in stream.documents]
        return cbor2.CBORTag(tags.CDXF_STREAM, docs)

    def _encode_document(self, doc: Document) -> cbor2.CBORTag:
        # NS table only for Element roots
        if type(doc.root) is Element:
            ns_uris: set[str] = set()
            self._collect_ns_uris_fast(doc.root, ns_uris)
            if ns_uris:
                self._ns_table = sorted(ns_uris)
                self._ns_index = {uri: i for i, uri in enumerate(self._ns_table)}
            else:
                self._ns_table = None
                self._ns_index = None
        else:
            self._ns_table = None
            self._ns_index = None

        # Fast path for JSON-model roots: single-pass try-convert
        root = doc.root
        if (self._ns_table is None and not self.canonical):
            native = self._try_native(root)
            if native is not _SENTINEL:
                root_encoded = native
            else:
                root_encoded = self._encode_node(root)
        else:
            root_encoded = self._encode_node(root)

        options = {}
        if not self.canonical and doc.source_format_hint != SourceFormat.UNSPECIFIED:
            options[tags.DOC_OPT_SOURCE_FORMAT] = doc.source_format_hint.value
        if doc.allows_cycles:
            options[tags.DOC_OPT_ALLOWS_CYCLES] = True
        if not self.canonical:
            if doc.preamble:
                options[tags.DOC_OPT_PREAMBLE] = [
                    self._encode_node(n) for n in doc.preamble
                ]
            if doc.postamble:
                options[tags.DOC_OPT_POSTAMBLE] = [
                    self._encode_node(n) for n in doc.postamble
                ]
        if self._ns_table:
            options[tags.DOC_OPT_NS_TABLE] = self._ns_table

        self._ns_table = None
        self._ns_index = None

        if options:
            return cbor2.CBORTag(tags.CDXF_DOCUMENT, [root_encoded, options])
        return cbor2.CBORTag(tags.CDXF_DOCUMENT, root_encoded)

    @staticmethod
    def _collect_ns_uris_fast(node, uris: set) -> None:
        """Iterative NS URI collection."""
        stack = [node]
        while stack:
            n = stack.pop()
            if type(n) is Element:
                if n.namespace_uri:
                    uris.add(n.namespace_uri)
                for attr in n.attributes:
                    if attr.namespace_uri:
                        uris.add(attr.namespace_uri)
                stack.extend(n.children)

    def _encode_node(self, node) -> object:
        """Encode any CDXF node. Single dispatch, inline annotations."""
        t = type(node)

        # --- Scalar ---
        if t is Scalar:
            st = node.scalar_type
            if st in _PASSTHROUGH_SCALARS:
                result = node.value
            else:
                cbor_tag = _TEMPORAL_TAGS.get(st)
                if cbor_tag is not None:
                    v = node.value
                    iso = v.isoformat() if hasattr(v, "isoformat") else str(v)
                    result = cbor2.CBORTag(cbor_tag, iso)
                elif st == ScalarType.DECIMAL:
                    raise NotImplementedError("Decimal encoding not implemented")
                else:
                    raise ValueError(f"Unknown scalar type: {st}")
            tag = node.tag
            if tag is not None:
                result = cbor2.CBORTag(tags.CDXF_TAG, [tag.uri, result])
            anchor = node.anchor
            if anchor is not None:
                result = cbor2.CBORTag(tags.CDXF_ANCHOR, [anchor.name, result])
            return result

        # --- Map ---
        if t is Map:
            result = self._encode_map(node)
            tag = node.tag
            if tag is not None:
                result = cbor2.CBORTag(tags.CDXF_TAG, [tag.uri, result])
            anchor = node.anchor
            if anchor is not None:
                result = cbor2.CBORTag(tags.CDXF_ANCHOR, [anchor.name, result])
            return result

        # --- Sequence ---
        if t is Sequence:
            items = []
            for item in node.items:
                if type(item) is Comment and self.canonical:
                    continue
                items.append(self._encode_node(item))
            tag = node.tag
            if tag is not None:
                items = cbor2.CBORTag(tags.CDXF_TAG, [tag.uri, items])
            anchor = node.anchor
            if anchor is not None:
                items = cbor2.CBORTag(tags.CDXF_ANCHOR, [anchor.name, items])
            return items

        # --- Element ---
        if t is Element:
            return self._encode_element(node)

        # --- Alias ---
        if t is Alias:
            return cbor2.CBORTag(tags.CDXF_ALIAS, node.name)

        # --- Comment ---
        if t is Comment:
            if self.canonical:
                return None
            return cbor2.CBORTag(tags.CDXF_COMMENT, node.text)

        # --- ProcessingInstruction ---
        if t is ProcessingInstruction:
            if self.canonical:
                return None
            return cbor2.CBORTag(tags.CDXF_PI,
                                 [node.target, node.data if node.data is not None else None])

        # --- Directive ---
        if t is Directive:
            return cbor2.CBORTag(tags.CDXF_DIRECTIVE, [node.name, node.parameters])

        raise ValueError(f"Unknown node type: {t.__name__}")

    def _encode_map(self, map_node: Map) -> object:
        """Encode Map. Fast path for simple maps, slow path for comments/complex keys."""
        entries = map_node.entries
        canonical = self.canonical

        # Fast path: build a simple dict
        result = {}
        for entry in entries:
            if type(entry) is Comment:
                if canonical:
                    continue
                return self._encode_map_commented(entries)
            key = entry[0]
            if key.tag is not None or key.anchor is not None:
                return self._encode_map_commented(entries)
            result[self._encode_node(key)] = self._encode_node(entry[1])

        if canonical and result:
            sorted_items = sorted(result.items(),
                                  key=lambda p: cbor2.dumps(p[0], canonical=True))
            result = dict(sorted_items)

        return result

    def _encode_map_commented(self, entries) -> cbor2.CBORTag:
        items = []
        for entry in entries:
            if type(entry) is Comment:
                if not self.canonical:
                    items.append(cbor2.CBORTag(tags.CDXF_COMMENT, entry.text))
            else:
                items.append(self._encode_node(entry[0]))
                items.append(self._encode_node(entry[1]))
        return cbor2.CBORTag(tags.CDXF_COMMENTED_MAP, items)

    def _ns_ref(self, uri: str | None) -> int | str | None:
        if uri is None:
            return None
        idx = self._ns_index
        if idx is not None:
            i = idx.get(uri)
            if i is not None:
                return i
        return uri

    def _encode_element(self, elem: Element) -> cbor2.CBORTag:
        """Encode Element with compact forms."""
        attributes = elem.attributes
        has_ns = elem.namespace_uri is not None or elem.prefix is not None or elem.namespace_declarations

        if not has_ns:
            if not attributes:
                content = [elem.name,
                           [self._encode_node(c) for c in elem.children
                            if not (type(c) is Comment and self.canonical)]]
            else:
                content = [elem.name,
                           {a.name: a.value for a in attributes},
                           [self._encode_node(c) for c in elem.children
                            if not (type(c) is Comment and self.canonical)]]
        else:
            has_ns_attrs = any(a.namespace_uri for a in attributes)
            if has_ns_attrs:
                attrs = [self._encode_attribute(a) for a in attributes]
            elif attributes:
                attrs = {a.name: a.value for a in attributes}
            else:
                attrs = {}
            children = [self._encode_node(c) for c in elem.children
                        if not (type(c) is Comment and self.canonical)]
            ns_ref = self._ns_ref(elem.namespace_uri)
            content = [elem.name, ns_ref, elem.prefix, attrs, children]
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
            self._ns_table = options.get(tags.DOC_OPT_NS_TABLE)
            root = self._decode_value(content[0])
            doc = Document(
                root=root,
                source_format_hint=SourceFormat(
                    options.get(tags.DOC_OPT_SOURCE_FORMAT, 0)
                ),
                allows_cycles=options.get(tags.DOC_OPT_ALLOWS_CYCLES, False),
                preamble=[self._decode_value(n)
                          for n in options.get(tags.DOC_OPT_PREAMBLE, [])],
                postamble=[self._decode_value(n)
                           for n in options.get(tags.DOC_OPT_POSTAMBLE, [])],
            )
            self._ns_table = None
            return doc

        self._ns_table = None
        root = self._decode_value(content)
        return Document(root=root)

    def _resolve_ns(self, ref) -> str | None:
        if ref is None:
            return None
        if isinstance(ref, int) and self._ns_table is not None:
            return self._ns_table[ref]
        return ref

    def _decode_value(self, raw) -> object:
        if isinstance(raw, cbor2.CBORTag):
            return self._decode_tagged(raw)
        if isinstance(raw, dict):
            return Map(entries=[(self._decode_value(k), self._decode_value(v))
                                for k, v in raw.items()])
        if isinstance(raw, list):
            return Sequence(items=[self._decode_value(item) for item in raw])
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
                return Scalar(ScalarType.DATE, datetime.date.fromisoformat(v))
            if isinstance(v, datetime.date):
                return Scalar(ScalarType.DATE, v)
            return Scalar(ScalarType.DATE, v)

        if t == tags.CDXF_TIME:
            if isinstance(v, str):
                return Scalar(ScalarType.TIME, datetime.time.fromisoformat(v))
            if isinstance(v, datetime.time):
                return Scalar(ScalarType.TIME, v)
            return Scalar(ScalarType.TIME, v)

        if t == tags.CDXF_COMMENT:
            return Comment(v)
        if t == tags.CDXF_ALIAS:
            return Alias(v)

        if t == tags.CDXF_ANCHOR:
            inner = self._decode_value(v[1])
            if hasattr(inner, "anchor"):
                inner.anchor = Anchor(v[0])
            return inner

        if t == tags.CDXF_TAG:
            inner = self._decode_value(v[1])
            if hasattr(inner, "tag"):
                inner.tag = TagAnnotation(v[0])
            return inner

        if t == tags.CDXF_ELEMENT:
            return self._decode_element(v)
        if t == tags.CDXF_PI:
            return ProcessingInstruction(target=v[0],
                                         data=v[1] if len(v) > 1 else None)
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

        return self._decode_value(v)

    def _decode_commented_map(self, items: list) -> Map:
        entries = []
        i = 0
        while i < len(items):
            item = items[i]
            if isinstance(item, cbor2.CBORTag) and item.tag == tags.CDXF_COMMENT:
                entries.append(Comment(item.value))
                i += 1
            else:
                entries.append((self._decode_value(item),
                                self._decode_value(items[i + 1])))
                i += 2
        return Map(entries=entries)

    def _decode_element(self, content: list) -> Element:
        n = len(content)
        if n == 2:
            return Element(name=content[0],
                           children=[self._decode_value(c) for c in content[1]])
        if n == 3:
            return Element(name=content[0],
                           attributes=self._decode_attrs(content[1]),
                           children=[self._decode_value(c) for c in content[2]])

        name = content[0]
        ns_ref = content[1]
        if n >= 3 and not isinstance(content[2], (dict, list)):
            prefix = content[2]
            raw_attrs = content[3] if n > 3 else {}
            raw_children = content[4] if n > 4 else []
            ns_start = 5
        else:
            prefix = None
            raw_attrs = content[2] if n > 2 else {}
            raw_children = content[3] if n > 3 else []
            ns_start = 4

        namespace_declarations = {}
        if n > ns_start:
            ns_raw = content[ns_start]
            if isinstance(ns_raw, cbor2.CBORTag) and ns_raw.tag == tags.CDXF_NAMESPACE:
                namespace_declarations = ns_raw.value
            elif isinstance(ns_raw, dict):
                namespace_declarations = ns_raw

        return Element(
            name=name,
            namespace_uri=self._resolve_ns(ns_ref),
            prefix=prefix,
            attributes=self._decode_attrs(raw_attrs),
            children=[self._decode_value(c) for c in raw_children],
            namespace_declarations=namespace_declarations,
        )

    def _decode_attrs(self, raw_attrs) -> list[Attribute]:
        if isinstance(raw_attrs, dict):
            return [Attribute(name=k, value=v) for k, v in raw_attrs.items()]
        if isinstance(raw_attrs, list):
            attrs = []
            for item in raw_attrs:
                if isinstance(item, cbor2.CBORTag) and item.tag == tags.CDXF_ATTRIBUTE:
                    a = item.value
                    attrs.append(Attribute(
                        name=a[0], namespace_uri=self._resolve_ns(a[1]),
                        value=a[2], prefix=a[3] if len(a) > 3 else None))
                elif isinstance(item, list):
                    attrs.append(Attribute(name=item[0], value=item[1]))
            return attrs
        return []
