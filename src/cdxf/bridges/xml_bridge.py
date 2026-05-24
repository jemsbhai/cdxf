"""XML bridge — convert between XML text and CDXF model.

Uses xml.parsers.expat (stdlib) for full-fidelity parsing that captures
comments, processing instructions, namespace declarations, CDATA sections,
and mixed content — constructs that xml.etree.ElementTree drops.

Functions:
    from_xml(text) -> Stream
    to_xml(stream) -> str
"""

from __future__ import annotations

import xml.parsers.expat

from cdxf.model import (
    Attribute,
    Comment,
    Document,
    Element,
    ProcessingInstruction,
    Scalar,
    ScalarType,
    SourceFormat,
    Stream,
)


# ===================================================================
# Public API
# ===================================================================

def from_xml(text: str) -> Stream:
    """Parse XML text into a CDXF Stream.

    Parameters
    ----------
    text : str
        Well-formed XML text.

    Returns
    -------
    Stream
        A single-document CDXF Stream with source_format_hint=XML.
    """
    handler = _ExpatHandler()
    parser = xml.parsers.expat.ParserCreate(namespace_separator=" ")
    parser.StartElementHandler = handler.start_element
    parser.EndElementHandler = handler.end_element
    parser.CharacterDataHandler = handler.character_data
    parser.CommentHandler = handler.comment
    parser.ProcessingInstructionHandler = handler.processing_instruction
    parser.StartNamespaceDeclHandler = handler.start_namespace_decl
    parser.EndNamespaceDeclHandler = handler.end_namespace_decl
    # Note: XmlDeclHandler is intentionally NOT set — the XML declaration
    # (<?xml ...?>) is not a processing instruction and must not appear as one.

    parser.Parse(text, True)

    doc = Document(
        root=handler.root,
        source_format_hint=SourceFormat.XML,
        preamble=handler.preamble,
        postamble=handler.postamble,
    )
    return Stream(documents=[doc])


def to_xml(stream: Stream) -> str:
    """Convert a CDXF Stream to XML text.

    Parameters
    ----------
    stream : Stream
        A CDXF Stream whose first document's root is an Element.

    Returns
    -------
    str
        Well-formed XML text.
    """
    if not stream.documents:
        return ""

    doc = stream.documents[0]
    parts: list[str] = []

    # Preamble (comments and PIs before the root element)
    for node in doc.preamble:
        if isinstance(node, Comment):
            parts.append(f"<!-- {node.text} -->")
        elif isinstance(node, ProcessingInstruction):
            parts.append(_serialize_pi(node))

    # Root element
    if isinstance(doc.root, Element):
        parts.append(_serialize_element(doc.root))

    # Postamble (comments after the root element)
    for node in doc.postamble:
        if isinstance(node, Comment):
            parts.append(f"<!-- {node.text} -->")
        elif isinstance(node, ProcessingInstruction):
            parts.append(_serialize_pi(node))

    return "".join(parts)


# ===================================================================
# Serialization helpers
# ===================================================================

def _escape_text(text: str) -> str:
    """Escape text content for XML output."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_attr(value: str) -> str:
    """Escape attribute value for XML output."""
    return (
        value
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _serialize_pi(pi: ProcessingInstruction) -> str:
    """Serialize a ProcessingInstruction to XML."""
    if pi.data:
        return f"<?{pi.target} {pi.data}?>"
    return f"<?{pi.target}?>"


def _serialize_element(elem: Element) -> str:
    """Recursively serialize an Element and its children to XML."""
    parts: list[str] = []

    # Tag name (with prefix if present)
    tag_name = f"{elem.prefix}:{elem.name}" if elem.prefix else elem.name
    parts.append(f"<{tag_name}")

    # Namespace declarations (emit as xmlns attributes)
    for prefix, uri in elem.namespace_declarations.items():
        if prefix:
            parts.append(f' xmlns:{prefix}="{_escape_attr(uri)}"')
        else:
            parts.append(f' xmlns="{_escape_attr(uri)}"')

    # Attributes
    for attr in elem.attributes:
        attr_name = f"{attr.prefix}:{attr.name}" if attr.prefix else attr.name
        parts.append(f' {attr_name}="{_escape_attr(attr.value)}"')

    # Children or self-closing
    if not elem.children:
        parts.append("/>")
    else:
        parts.append(">")

        for child in elem.children:
            if isinstance(child, Element):
                parts.append(_serialize_element(child))
            elif isinstance(child, Scalar):
                parts.append(_escape_text(str(child.value)))
            elif isinstance(child, Comment):
                parts.append(f"<!-- {child.text} -->")
            elif isinstance(child, ProcessingInstruction):
                parts.append(_serialize_pi(child))

        parts.append(f"</{tag_name}>")

    return "".join(parts)


# ===================================================================
# Expat-based XML parser handler
# ===================================================================

class _ExpatHandler:
    """Stateful handler that builds a CDXF tree from expat events.

    The handler tracks:
    - A stack of Element nodes being constructed
    - Namespace scope (prefix→URI mappings) via a stack of dicts
    - Pending namespace declarations for the next element
    - Preamble/postamble nodes (comments/PIs before/after root)
    """

    def __init__(self):
        self._stack: list[Element] = []
        self.root: Element | None = None
        self.preamble: list = []
        self.postamble: list = []
        self._root_closed: bool = False
        self._ns_pending: dict[str, str] = {}   # prefix → uri for next element
        self._ns_scopes: list[dict[str, str]] = [{}]  # stack of cumulative scopes

    # ---------------------------------------------------------------
    # Location helper
    # ---------------------------------------------------------------

    def _target_list(self) -> list:
        """Return the list where new children should be appended.

        - Inside an element: that element's children
        - Before root: preamble
        - After root: postamble
        """
        if self._stack:
            return self._stack[-1].children
        if self._root_closed:
            return self.postamble
        return self.preamble

    # ---------------------------------------------------------------
    # Namespace helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _parse_name(name: str) -> tuple[str | None, str]:
        """Split an expat name 'uri local' into (uri, local).

        If there is no namespace, returns (None, name).
        """
        if " " in name:
            uri, local = name.split(" ", 1)
            return uri, local
        return None, name

    def _find_prefix(self, uri: str | None) -> str | None:
        """Look up the prefix for a namespace URI in the current scope.

        Returns the prefix string, or None if the URI is the default
        namespace (empty prefix) or not found.
        """
        if not uri:
            return None
        scope = self._ns_scopes[-1]
        for prefix, u in scope.items():
            if u == uri:
                return prefix if prefix != "" else None
        return None

    # ---------------------------------------------------------------
    # Expat event handlers
    # ---------------------------------------------------------------

    def start_namespace_decl(self, prefix: str | None, uri: str) -> None:
        """Called before StartElementHandler for each xmlns declaration."""
        if prefix is None:
            prefix = ""
        self._ns_pending[prefix] = uri

    def end_namespace_decl(self, prefix: str | None) -> None:
        """Called after EndElementHandler. Scoping handled via stack."""
        pass

    def start_element(self, name: str, attrs: dict[str, str]) -> None:
        """Build an Element node and push it onto the stack."""
        ns_uri, local_name = self._parse_name(name)

        # --- Namespace scope management ---
        # Push a new scope that includes any pending declarations.
        new_scope = dict(self._ns_scopes[-1])
        ns_decls: dict[str, str] = {}
        if self._ns_pending:
            new_scope.update(self._ns_pending)
            ns_decls = dict(self._ns_pending)
            self._ns_pending = {}
        self._ns_scopes.append(new_scope)

        # Determine this element's prefix from the active scope.
        prefix = self._find_prefix(ns_uri) if ns_uri else None

        # --- Create the Element ---
        elem = Element(
            name=local_name,
            namespace_uri=ns_uri if ns_uri else None,
            prefix=prefix,
            namespace_declarations=ns_decls,
        )

        # --- Attributes ---
        for attr_name, attr_value in attrs.items():
            attr_ns, attr_local = self._parse_name(attr_name)
            attr_prefix = self._find_prefix(attr_ns) if attr_ns else None
            elem.attributes.append(Attribute(
                name=attr_local,
                value=attr_value,
                namespace_uri=attr_ns if attr_ns else None,
                prefix=attr_prefix,
            ))

        # --- Attach to parent ---
        if self._stack:
            self._stack[-1].children.append(elem)

        self._stack.append(elem)

    def end_element(self, name: str) -> None:
        """Pop the current element. If stack is empty, this was the root."""
        elem = self._stack.pop()
        self._ns_scopes.pop()

        if not self._stack:
            self.root = elem
            self._root_closed = True

    def character_data(self, data: str) -> None:
        """Append text content as a Scalar(STRING) child.

        Coalesces adjacent character data events (expat may split a
        single text node into multiple callbacks).
        """
        if not self._stack:
            return  # Ignore text outside elements (inter-element whitespace)

        children = self._stack[-1].children
        # Coalesce with previous text node if adjacent
        if (children
                and isinstance(children[-1], Scalar)
                and children[-1].scalar_type == ScalarType.STRING):
            children[-1] = Scalar(ScalarType.STRING, children[-1].value + data)
        else:
            children.append(Scalar(ScalarType.STRING, data))

    def comment(self, text: str) -> None:
        """Append a Comment node at the current position."""
        self._target_list().append(Comment(text.strip()))

    def processing_instruction(self, target: str, data: str) -> None:
        """Append a ProcessingInstruction at the current position."""
        pi = ProcessingInstruction(
            target=target,
            data=data if data else None,
        )
        self._target_list().append(pi)
