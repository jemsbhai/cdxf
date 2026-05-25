"""Profile CDXF encode to identify bottlenecks."""

import cProfile
import pstats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cdxf.bridges.json_bridge import from_json
from cdxf.codec import encode

# Use a medium-sized JSON file
src = PROJECT_ROOT / "data" / "raw" / "json" / "tier1_schemastore" / "jsonresume.json"
text = src.read_text(encoding="utf-8")

# Pre-parse to CDXF model
stream = from_json(text)

# Profile the codec encode (1000 iterations)
def run_encode():
    for _ in range(1000):
        encode(stream)

print("Profiling cdxf.codec.encode() x 1000...")
cProfile.run("run_encode()", "encode_profile")

stats = pstats.Stats("encode_profile")
stats.strip_dirs()
stats.sort_stats("cumulative")
print("\n=== Top 30 by cumulative time ===")
stats.print_stats(30)

stats.sort_stats("tottime")
print("\n=== Top 30 by total time ===")
stats.print_stats(30)
