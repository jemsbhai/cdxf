"""Diagnose XML size overhead — where do the bytes go?"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cdxf.bridges.xml_bridge import from_xml
from cdxf.codec import encode
from cdxf.model import Element

DATA_RAW = PROJECT_ROOT / "data" / "raw"

files = [
    "synthetic/xml_namespace_heavy.xml",
    "synthetic/xml_mixed_content.xml",
    "synthetic/xml_comment_heavy.xml",
    "xml/tier2_canonical/atom-namespace.xml",
]

for rel in files:
    fpath = DATA_RAW / rel
    text = fpath.read_text(encoding="utf-8")
    stream = from_xml(text)
    binary = encode(stream)
    root = stream.documents[0].root

    stats = {"elems": 0, "ns_uri_bytes": 0, "prefix_bytes": 0,
             "name_bytes": 0, "ns_decl_bytes": 0, "nulls": 0}
    ns_uri_set = set()

    def walk(node):
        if isinstance(node, Element):
            stats["elems"] += 1
            stats["name_bytes"] += len(node.name.encode())
            if node.namespace_uri:
                stats["ns_uri_bytes"] += len(node.namespace_uri.encode())
                ns_uri_set.add(node.namespace_uri)
            else:
                stats["nulls"] += 1
            if node.prefix:
                stats["prefix_bytes"] += len(node.prefix.encode())
            else:
                stats["nulls"] += 1
            for p, u in node.namespace_declarations.items():
                stats["ns_decl_bytes"] += len(p.encode()) + len(u.encode())
            for child in node.children:
                walk(child)

    walk(root)

    unique_ns_bytes = sum(len(u.encode()) for u in ns_uri_set)
    savings = stats["ns_uri_bytes"] - unique_ns_bytes
    hypothetical = len(binary) - savings

    print(f"\n{'='*60}")
    print(f"{rel}")
    print(f"  Text size:        {len(text.encode()):,} bytes")
    print(f"  CDXF size:        {len(binary):,} bytes")
    print(f"  Ratio:            {len(binary)/len(text.encode()):.3f}")
    print(f"  Elements:         {stats['elems']}")
    print(f"  Unique NS URIs:   {len(ns_uri_set)}")
    print(f"  NS URI bytes:     {stats['ns_uri_bytes']:,} (repeated)")
    print(f"  NS decl bytes:    {stats['ns_decl_bytes']:,}")
    print(f"  Prefix bytes:     {stats['prefix_bytes']:,}")
    print(f"  Name bytes:       {stats['name_bytes']:,}")
    print(f"  Null fields:      {stats['nulls']}")
    print(f"\n  If NS URIs stored once: save {savings:,} bytes")
    print(f"  Hypothetical size:      {hypothetical:,} ({hypothetical/len(text.encode()):.3f}x)")
