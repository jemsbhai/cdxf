# CDXF Binary Encoding Specification

**Version:** 0.1.0-draft
**Status:** Working Draft
**Author:** Muntaser Syed
**Date:** May 2026

---

## 1. Design Principles

The CDXF binary encoding is built on CBOR (RFC 8949) as its substrate.

**Principle 1: Zero overhead for JSON-model data.** A CDXF-encoded JSON
document is standard CBOR with a single framing tag. Any conforming CBOR
decoder can read the data content. CDXF-specific tags only appear when
encoding constructs that CBOR cannot natively represent.

**Principle 2: CBOR tag extension.** Each CDXF information model construct
beyond CBOR's native data model is represented by a CBOR semantic tag
wrapping a CBOR structure. This leverages CBOR's built-in extensibility
without modifying the encoding rules.

**Principle 3: Compact by default.** Common constructs use the most
compact encoding. Rare constructs (processing instructions, namespace
declarations) may use slightly more verbose encodings.

**Principle 4: Deterministic encoding.** CDXF canonical form is defined
in terms of CBOR Common Deterministic Encoding (CDE,
draft-ietf-cbor-cde) with additional rules for CDXF-specific constructs.

---

## 2. CBOR Tag Assignments

CDXF reserves a contiguous block of CBOR tag numbers for its constructs.
The following assignments are provisional and subject to IANA registration.

Tag numbers in the range 256-65535 require 3 bytes of overhead (1 tag
major type byte + 2 bytes for the tag number). Tag numbers 0-23 require
1 byte. Final IANA-assigned numbers may differ.

### 2.1 Provisional Tag Table

| Tag Number | Name             | Content Type               | Usage Frequency |
|------------|------------------|----------------------------|-----------------|
| 25700      | CDXF_STREAM      | array                      | Once per stream |
| 25701      | CDXF_DOCUMENT    | array                      | Once per doc    |
| 25702      | CDXF_ELEMENT     | array                      | XML data        |
| 25703      | CDXF_ATTRIBUTE   | array                      | XML data        |
| 25704      | CDXF_ANCHOR      | array                      | YAML graphs     |
| 25705      | CDXF_ALIAS       | text string or unsigned int | YAML graphs     |
| 25706      | CDXF_COMMENT     | text string                | All formats     |
| 25707      | CDXF_PI          | array                      | XML data        |
| 25708      | CDXF_DIRECTIVE   | array                      | YAML/XML        |
| 25709      | CDXF_TAG         | array                      | YAML tagged     |
| 25710      | CDXF_NAMESPACE   | map                        | XML data        |
| 25711      | CDXF_TIMESTAMP_L | text string                | TOML            |
| 25712      | CDXF_DATE        | text string                | TOML            |
| 25713      | CDXF_TIME        | text string                | TOML            |

Notes:
- CBOR's native tag 0 (RFC 3339 datetime string) is used for
  `timestamp_offset`. No CDXF tag needed.
- CBOR's native tag 2/3 (bignum) is used for arbitrary-precision integers.
- CBOR's native tag 4 (decimal fraction) is used for `decimal` scalars.
- CBOR's native null, bool, int, float, text string, and byte string
  major types are used directly for scalars.

---

## 3. Encoding Rules

### 3.1 Stream

A CDXF byte stream is a CBOR data item:

```
Tag(25700) [                    ; CDXF_STREAM
  <document>,                   ; one or more documents
  <document>,
  ...
]
```

**Shorthand for single JSON-model documents:** When encoding a single
document containing only JSON-model data (maps, arrays, scalars) with no
CDXF-specific constructs, the Stream and Document wrappers MAY be omitted.
The output is simply the CBOR encoding of the root value. A decoder
encountering a CBOR value without tag 25700 treats it as a single-document
stream containing the value as the root.

### 3.2 Document

```
Tag(25701) [                    ; CDXF_DOCUMENT
  <root_node>,                  ; required: the document root
  <options_map>                 ; optional: map of metadata (omit if empty)
]
```

The options map, when present, uses integer keys for compactness:

| Key | Name               | Value Type         | Default       |
|-----|--------------------|--------------------|---------------|
| 1   | source_format_hint | unsigned int (enum)| 0 (unspecified)|
| 2   | allows_cycles      | bool               | false         |
| 3   | preamble           | array of nodes     | []            |
| 4   | postamble          | array of nodes     | []            |
| 5   | directives         | array of Directives| []            |

Source format hint enum values:

| Value | Format      |
|-------|-------------|
| 0     | unspecified |
| 1     | json        |
| 2     | yaml        |
| 3     | xml         |
| 4     | toml        |

**Compact form:** When the options map would be empty (single JSON/YAML
document with no directives, no preamble/postamble, no cycles), the
Document is encoded as:

```
Tag(25701) <root_node>          ; no wrapping array needed
```

A decoder distinguishes the two forms by checking whether the tag's
content is an array whose first element could be interpreted as the full
form. In practice: if the content is a 2-element array where the second
element is a CBOR map, it is the full form. Otherwise, the content is
the root node directly.

### 3.3 Map

CBOR maps are used directly. No CDXF tag is needed.

```
{                               ; CBOR map (major type 5)
  <key>: <value>,
  <key>: <value>,
  ...
}
```

**Comment interleaving.** When a Map contains interleaved comments, the
map is encoded as a CBOR array of alternating entries and comments to
preserve order:

```
Tag(25700+TBD?) [               ; future: CDXF_ORDERED_MAP if needed
  [<key>, <value>],
  Tag(25706) "comment text",
  [<key>, <value>],
  ...
]
```

However, for the common case (no comments), a standard CBOR map is used.
A decoder uses the presence of tag 25706 children to detect the ordered
form.

**Revised approach (simpler):** Maps with interleaved comments are encoded
as a CBOR array tagged with a map-with-comments tag:

| Tag Number | Name                | Content Type |
|------------|---------------------|--------------|
| 25714      | CDXF_COMMENTED_MAP  | array        |

```
Tag(25714) [                    ; CDXF_COMMENTED_MAP
  <key>, <value>,               ; entries as alternating key, value
  Tag(25706) "comment",         ; comments interleaved at position
  <key>, <value>,
  ...
]
```

Maps without comments: standard CBOR map. Maps with comments: tag 25714
wrapping a flat array of alternating keys, values, and tagged comments.

### 3.4 Sequence

CBOR arrays are used directly. No CDXF tag is needed.

```
[                               ; CBOR array (major type 4)
  <item>,
  <item>,
  ...
]
```

**Comment interleaving.** When a Sequence contains interleaved comments,
they appear as tagged Comment nodes in the array:

```
[
  <item>,
  Tag(25706) "comment text",    ; comment between items
  <item>,
  ...
]
```

A plain CBOR array with no Comment-tagged items is an uncommented
Sequence. Comment-tagged items are filtered out during semantic
comparison.

### 3.5 Scalar

Scalars use CBOR's native encoding wherever possible:

| Scalar Type        | CBOR Encoding                                    |
|--------------------|--------------------------------------------------|
| null               | CBOR simple value null (0xf6)                    |
| boolean            | CBOR simple value true/false (0xf5/0xf4)         |
| integer            | CBOR integer (major type 0/1)                    |
| integer (bignum)   | CBOR tag 2 or 3 (positive/negative bignum)       |
| float              | CBOR float (major type 7, half/single/double)    |
| decimal            | CBOR tag 4 (decimal fraction [exponent, mantissa])|
| string             | CBOR text string (major type 3)                  |
| byte_string        | CBOR byte string (major type 2)                  |
| timestamp_offset   | CBOR tag 0 (RFC 3339 datetime string)            |
| timestamp_local    | Tag(25711) "2024-05-27T07:32:00"                 |
| date               | Tag(25712) "2024-05-27"                          |
| time               | Tag(25713) "07:32:00.000"                        |

Note: `timestamp_local`, `date`, and `time` use CDXF tags because CBOR's
native tag 0 requires a UTC offset. The values are RFC 3339 partial
strings.

### 3.6 Element (XML)

```
Tag(25702) [                    ; CDXF_ELEMENT
  <name>,                       ; text string: local name
  <namespace_uri_or_null>,      ; text string or null
  <attributes>,                 ; CBOR map {name: value, ...} or
                                ;   array of CDXF_ATTRIBUTE for namespaced attrs
  <children>                    ; CBOR array of child nodes
]
```

**Compact form for elements without namespaces:**

```
Tag(25702) [
  "div",                        ; local name
  null,                         ; no namespace
  {"class": "main"},            ; simple string-keyed attributes
  [                             ; children
    "Hello ",                   ; text content (Scalar string)
    Tag(25702) [                ; child element
      "b", null, {}, ["world"]
    ],
    "!"                         ; more text content
  ]
]
```

This naturally represents mixed content: the children array interleaves
strings (text nodes) and elements.

### 3.7 Attribute (XML, namespaced)

For simple non-namespaced attributes, a CBOR map on the Element suffices
(`{"class": "main"}`). For namespaced attributes, the Attribute tag is
used:

```
Tag(25703) [                    ; CDXF_ATTRIBUTE
  <local_name>,                 ; text string
  <namespace_uri>,              ; text string
  <value>,                      ; text string
  <prefix_or_null>              ; optional, advisory
]
```

When an Element has a mix of namespaced and non-namespaced attributes,
the attributes field is an array containing both plain key-value pairs
and CDXF_ATTRIBUTE tagged items:

```
Tag(25702) [
  "element",
  "http://example.com/ns",
  [                             ; attributes as array (mixed form)
    Tag(25703) ["href", "http://www.w3.org/1999/xlink", "doc.xml", "xlink"],
    Tag(25703) ["type", "http://www.w3.org/1999/xlink", "simple", "xlink"],
    ["class", "main"],          ; non-namespaced: [name, value] pair
  ],
  [...]                         ; children
]
```

### 3.8 Anchor

```
Tag(25704) [                    ; CDXF_ANCHOR
  <anchor_name>,                ; text string (e.g., "defaults")
  <node>                        ; the anchored node
]
```

The Anchor tag wraps the node it labels. This means anchored nodes are
encoded inline at their first occurrence, and subsequent references use
Alias.

### 3.9 Alias

```
Tag(25705) <anchor_name>        ; CDXF_ALIAS
```

The Alias tag wraps a text string containing the anchor name to resolve.
Alternatively, for compactness, if anchors are assigned sequential integer
IDs during encoding:

```
Tag(25705) <anchor_index>       ; unsigned integer (0-based)
```

A decoder MUST support both text string and unsigned integer content.
Integer aliases are preferred in canonical form for compactness.

### 3.10 Comment

```
Tag(25706) <text>               ; CDXF_COMMENT, text string
```

Comments are lightweight: a single tag wrapping the comment text.

### 3.11 Processing Instruction

```
Tag(25707) [                    ; CDXF_PI
  <target>,                     ; text string
  <data_or_null>                ; text string or null
]
```

### 3.12 Directive

```
Tag(25708) [                    ; CDXF_DIRECTIVE
  <name>,                       ; text string (e.g., "YAML", "TAG", "DOCTYPE")
  <parameters>                  ; array of text strings
]
```

### 3.13 Tag Annotation (YAML/semantic tag)

```
Tag(25709) [                    ; CDXF_TAG
  <tag_uri>,                    ; text string (the tag URI)
  <node>                        ; the tagged node
]
```

This wraps any node with a semantic tag. For example, a YAML scalar
`!!int "42"` becomes:

```
Tag(25709) [
  "tag:yaml.org,2002:int",
  42
]
```

Note: For common YAML tags that map to CBOR native types (int, float,
str, bool, null, binary), the tag annotation is unnecessary because the
CBOR encoding already carries the type. The CDXF_TAG is only needed for:
- Custom application tags
- YAML merge tag (`tag:yaml.org,2002:merge`)
- Tags whose resolved type differs from the CBOR encoding

### 3.14 Namespace Declarations

```
Tag(25710) {                    ; CDXF_NAMESPACE
  <prefix>: <uri>,              ; text string: text string
  ...                           ; e.g., "": "http://default.ns/",
                                ;       "xlink": "http://www.w3.org/1999/xlink"
}
```

Namespace declarations are attached to an Element by prepending them
in the Element's encoding:

```
Tag(25702) [
  "root",
  "http://example.com",
  {},                           ; attributes
  [...],                        ; children
  Tag(25710) {                  ; namespace declarations (optional 5th element)
    "": "http://example.com",
    "xs": "http://www.w3.org/2001/XMLSchema"
  }
]
```

When the 5th element is absent, the Element has no namespace declarations.

### 3.15 Advisory Annotations

Advisory annotations (is_cdata, is_inline, source_format_hint) are not
encoded as separate tags. They are carried in the Document options map
(Section 3.2) or, for per-node advisories, via a lightweight wrapper:

| Tag Number | Name              | Content           |
|------------|-------------------|-------------------|
| 25715      | CDXF_ADVISORY     | [flags_int, node] |

Flags are a bitmask:

| Bit | Meaning       |
|-----|---------------|
| 0   | is_cdata      |
| 1   | is_inline     |

```
Tag(25715) [1, "raw text"]      ; is_cdata = true
Tag(25715) [2, {key: "val"}]    ; is_inline = true (inline TOML table)
```

Advisory tags are stripped during canonical form computation.

---

## 4. Encoding Examples

### 4.1 JSON Document

Input:
```json
{"name": "Alice", "age": 30, "active": true}
```

CDXF binary (hex, annotated):
```
D9 6444                         ; Tag(25700) CDXF_STREAM
  81                            ; array(1) - one document
    D9 6445                     ; Tag(25701) CDXF_DOCUMENT
      A3                       ; map(3)
        64 6E616D65             ; text(4) "name"
        65 416C696365           ; text(5) "Alice"
        63 616765               ; text(3) "age"
        18 1E                   ; unsigned(30)
        66 616374697665         ; text(6) "active"
        F5                     ; true
```

**Shorthand** (omitting Stream/Document wrappers for plain JSON):
```
A3                              ; map(3)
  64 6E616D65                   ; "name"
  65 416C696365                 ; "Alice"
  63 616765                     ; "age"
  18 1E                         ; 30
  66 616374697665               ; "active"
  F5                            ; true
```

This is byte-identical to standard CBOR. Any CBOR library can read it.

### 4.2 YAML Document with Anchors

Input:
```yaml
defaults: &defaults
  timeout: 30
  retries: 3
production:
  <<: *defaults
  timeout: 60
```

CDXF binary (diagnostic notation):
```
25700([                         ; CDXF_STREAM
  25701(                        ; CDXF_DOCUMENT
    {                           ; root map
      "defaults": 25704([       ; CDXF_ANCHOR
        "defaults",             ; anchor name
        {                       ; anchored map
          "timeout": 30,
          "retries": 3
        }
      ]),
      "production": {
        25709([                 ; CDXF_TAG (merge tag)
          "tag:yaml.org,2002:merge",
          "<<"
        ]): 25705("defaults"), ; CDXF_ALIAS -> "defaults"
        "timeout": 60
      }
    }
  )
])
```

### 4.3 XML Document with Mixed Content

Input:
```xml
<?xml-stylesheet type="text/xsl" href="style.xsl"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body class="main">
    <p>Hello <b>world</b>!</p>
  </body>
</html>
```

CDXF binary (diagnostic notation):
```
25700([                         ; CDXF_STREAM
  25701([                       ; CDXF_DOCUMENT (full form)
    25702([                     ; CDXF_ELEMENT: html
      "html",                   ; local name
      "http://www.w3.org/1999/xhtml",  ; namespace
      {},                       ; no attributes
      [                         ; children
        25702([                 ; CDXF_ELEMENT: body
          "body",
          null,                 ; inherits parent namespace
          {"class": "main"},    ; attributes
          [                     ; children
            25702([             ; CDXF_ELEMENT: p
              "p", null, {},
              [                 ; children (mixed content)
                "Hello ",       ; text node
                25702([         ; CDXF_ELEMENT: b
                  "b", null, {},
                  ["world"]
                ]),
                "!"             ; text node
              ]
            ])
          ]
        ])
      ],
      25710({                   ; namespace declarations
        "": "http://www.w3.org/1999/xhtml"
      })
    ]),
    {                           ; options map
      1: 3,                     ; source_format_hint = xml
      3: [                      ; preamble
        25707(["xml-stylesheet",; CDXF_PI
          "type=\"text/xsl\" href=\"style.xsl\""
        ])
      ]
    }
  ])
])
```

### 4.4 TOML Document with Datetimes

Input:
```toml
# Server configuration
[server]
host = "localhost"
port = 8080
started = 2024-05-27T07:32:00-04:00

[server.tls]
# Enable TLS
enabled = true
cert_expiry = 2025-12-31
```

CDXF binary (diagnostic notation):
```
25700([                         ; CDXF_STREAM
  25701([                       ; CDXF_DOCUMENT (full form)
    25714([                     ; CDXF_COMMENTED_MAP
      25706("Server configuration"),  ; comment
      "server",                 ; key
      25714([                   ; nested CDXF_COMMENTED_MAP
        "host", "localhost",
        "port", 8080,
        "started", 0("2024-05-27T07:32:00-04:00"),  ; CBOR tag 0: timestamp
        "tls",
        25714([                 ; nested commented map
          25706("Enable TLS"),  ; comment
          "enabled", true,
          "cert_expiry", 25712("2025-12-31")  ; CDXF_DATE
        ])
      ])
    ]),
    {1: 4}                      ; source_format_hint = toml
  ])
])
```

---

## 5. Deterministic Encoding (Canonical Form)

CDXF canonical form extends CBOR CDE (draft-ietf-cbor-cde) with:

1. **All CDE rules apply.** Preferred integer/float encoding, sorted map
   keys (bytewise comparison of serialized keys), deterministic length
   encoding.

2. **CDXF-specific rules:**
   - Advisory tags (25715 CDXF_ADVISORY) are stripped.
   - Comment tags (25706 CDXF_COMMENT) are stripped.
   - ProcessingInstruction tags (25707 CDXF_PI) are stripped.
   - CDXF_COMMENTED_MAP (25714) is replaced with a standard CBOR map
     (comments removed, entries sorted by key).
   - Anchor names are replaced with sequential unsigned integers (0, 1,
     2, ...) assigned in depth-first encounter order.
   - Alias content is the integer anchor index.
   - Tag annotation URIs (25709) are preserved in NFC-normalized form.
   - Namespace prefixes in Element encodings are dropped (URIs preserved).
   - source_format_hint is stripped.
   - Scalar strings are NFC-normalized.

3. **Stream/Document wrappers are always present** in canonical form (no
   shorthand).

---

## 6. MIME Type and File Extension

- MIME type: `application/cdxf` (to be registered with IANA)
- File extension: `.cdxf`
- CBOR tag: 25700 (CDXF_STREAM, to be registered with IANA)

---

## 7. Summary of Tag Assignments

| Tag    | Name                | Overhead | Used For                    |
|--------|---------------------|----------|-----------------------------|
| 0      | (CBOR native)       | 1 byte   | RFC 3339 timestamp w/ offset|
| 2, 3   | (CBOR native)       | 1 byte   | Bignum (pos/neg)            |
| 4      | (CBOR native)       | 1 byte   | Decimal fraction            |
| 25700  | CDXF_STREAM         | 3 bytes  | Top-level stream container  |
| 25701  | CDXF_DOCUMENT       | 3 bytes  | Document wrapper            |
| 25702  | CDXF_ELEMENT        | 3 bytes  | XML element                 |
| 25703  | CDXF_ATTRIBUTE      | 3 bytes  | XML namespaced attribute    |
| 25704  | CDXF_ANCHOR         | 3 bytes  | YAML anchor                 |
| 25705  | CDXF_ALIAS          | 3 bytes  | YAML alias                  |
| 25706  | CDXF_COMMENT        | 3 bytes  | Comment                     |
| 25707  | CDXF_PI             | 3 bytes  | Processing instruction      |
| 25708  | CDXF_DIRECTIVE      | 3 bytes  | Directive                   |
| 25709  | CDXF_TAG            | 3 bytes  | Semantic tag annotation     |
| 25710  | CDXF_NAMESPACE      | 3 bytes  | Namespace declarations      |
| 25711  | CDXF_TIMESTAMP_L    | 3 bytes  | Local datetime              |
| 25712  | CDXF_DATE           | 3 bytes  | Local date                  |
| 25713  | CDXF_TIME           | 3 bytes  | Local time                  |
| 25714  | CDXF_COMMENTED_MAP  | 3 bytes  | Map with interleaved comments|
| 25715  | CDXF_ADVISORY       | 3 bytes  | Advisory flags wrapper      |

Total: 16 CDXF-specific tags. A future IANA registration could request
tags in the 0-255 range for the most frequent tags (ELEMENT, ANCHOR,
ALIAS, COMMENT), reducing their overhead to 1-2 bytes.
