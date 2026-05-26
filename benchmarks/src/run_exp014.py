"""
EXP-014: Multi-Agent Format Interchange — Hub vs Direct

Measures converter scaling (O(N²) vs O(N)) and metadata compounding
over H sequential handoffs in a multi-agent pipeline.

Scenario: 5 ML agents with distinct format preferences pass an annotated
config through a pipeline. Direct conversion uses standard parsers (lossy).
CDXF hub conversion routes through CDXF intermediate representation (lossless).

Usage:
    python benchmarks/src/run_exp014.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml  # PyYAML — standard (lossy) YAML library
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

try:
    from cdxf.bridges.json_bridge import from_json, to_json
    from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
    from cdxf.bridges.toml_bridge import from_toml, to_toml
    from cdxf.bridges.xml_bridge import from_xml, to_xml
    from cdxf.codec import encode, decode
    from cdxf.model import Stream
except ImportError:
    print("ERROR: cdxf not installed. Run: pip install -e .")
    sys.exit(1)


# ===========================================================================
# Protocol constants
# ===========================================================================

PIPELINE_DEPTHS = [1, 2, 3, 5]
N_FORMAT_COUNTS = [2, 3, 4]
CONVERSION_METHODS = ["direct", "cdxf_hub"]

AGENTS = [
    {"name": "DataCurator", "role": "curator", "format": "yaml"},
    {"name": "ModelTrainer", "role": "trainer", "format": "json"},
    {"name": "Evaluator", "role": "evaluator", "format": "toml"},
    {"name": "Deployer", "role": "deployer", "format": "xml"},
    {"name": "Monitor", "role": "monitor", "format": "yaml"},
]


# ===========================================================================
# Converter counting — O(N²) vs O(N)
# ===========================================================================


def count_converters(n: int, method: str) -> int:
    """Count the number of converters required.

    Direct: N×(N-1) pairwise converters for all ordered pairs.
    CDXF hub: 2N — one encoder + one decoder per format.

    Args:
        n: Number of distinct formats.
        method: "direct" or "cdxf_hub".

    Returns:
        Number of converters needed.
    """
    if method == "direct":
        return n * (n - 1)
    elif method == "cdxf_hub":
        return 2 * n
    else:
        raise ValueError(f"Unknown method: {method}")


# ===========================================================================
# Test config — richly annotated YAML with metadata
# ===========================================================================


def build_test_config() -> tuple[str, str]:
    """Build a richly annotated ML config for pipeline testing.

    Returns (text, format) — starts as YAML with comments, anchors, etc.
    """
    text = (
        "# ML Pipeline Configuration\n"
        "# Owner: DataCurator agent\n"
        "# Last modified: 2026-01-20\n"
        "\n"
        "# Model architecture and training parameters\n"
        "model:\n"
        "  name: llama-2-7b-chat\n"
        "  # Base model from HuggingFace Hub\n"
        "  base_model: meta-llama/Llama-2-7b-hf\n"
        "  revision: main\n"
        "\n"
        "training:\n"
        "  # Learning rate from grid search [1e-5, 5e-5, 1e-4]\n"
        "  learning_rate: 2.0e-5\n"
        "  num_epochs: 3\n"
        "  batch_size: 4\n"
        "  # Gradient accumulation for effective batch 128\n"
        "  gradient_accumulation_steps: 8\n"
        "  warmup_ratio: 0.03\n"
        "  # Weight decay regularization\n"
        "  weight_decay: 0.01\n"
        "  seed: 42\n"
        "\n"
        "data:\n"
        "  # Alpaca-cleaned dataset\n"
        "  dataset: tatsu-lab/alpaca\n"
        "  max_length: 2048\n"
        "  num_examples: 49800\n"
        "  # Train/val split ratio\n"
        "  val_ratio: 0.1\n"
        "\n"
        "evaluation:\n"
        "  # Benchmarks to run after training\n"
        "  benchmarks:\n"
        "    - mmlu\n"
        "    - hellaswag\n"
        "    - arc_challenge\n"
        "  # Minimum acceptable accuracy\n"
        "  min_accuracy: 0.45\n"
        "\n"
        "deployment:\n"
        "  # vLLM serving parameters\n"
        "  max_num_seqs: 64\n"
        "  gpu_memory_utilization: 0.9\n"
        "  # Port for API endpoint\n"
        "  port: 8000\n"
    )
    return text, "yaml"


# ===========================================================================
# Metadata counting
# ===========================================================================


def count_metadata(text: str, fmt: str) -> dict:
    """Count metadata constructs (comments, etc.) in text.

    Args:
        text: File content.
        fmt: Format name.

    Returns:
        Dict with {comments, total}.
    """
    comments = 0

    if not text:
        return {"comments": 0, "total": 0}

    if fmt in ("yaml", "toml"):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                comments += 1
            elif " #" in line or "\t#" in line:
                hash_pos = line.find(" #")
                if hash_pos < 0:
                    hash_pos = line.find("\t#")
                if hash_pos >= 0:
                    before = line[:hash_pos]
                    if (before.count('"') % 2 == 0 and
                            before.count("'") % 2 == 0):
                        comments += 1

    # JSON and XML do not support comments in standard specs
    return {"comments": comments, "total": comments}


# ===========================================================================
# Pipeline construction
# ===========================================================================


def build_agent_pipeline(n_formats: int) -> list[dict]:
    """Build an agent pipeline that exercises N distinct formats.

    Selects agents to ensure adjacent agents have different formats,
    maximizing the number of format conversions.

    Args:
        n_formats: Number of distinct formats (2, 3, or 4).

    Returns:
        List of agent dicts forming the pipeline.
    """
    # Select agents that cover exactly n_formats distinct formats
    # and ensure adjacent agents differ
    if n_formats == 2:
        # yaml → json → yaml → json → yaml (alternating)
        return [AGENTS[0], AGENTS[1]]  # curator(yaml), trainer(json)
    elif n_formats == 3:
        # yaml → json → toml
        return [AGENTS[0], AGENTS[1], AGENTS[2]]
    elif n_formats >= 4:
        # yaml → json → toml → xml → yaml
        return [AGENTS[0], AGENTS[1], AGENTS[2], AGENTS[3], AGENTS[4]]
    else:
        return [AGENTS[0], AGENTS[1]]


# ===========================================================================
# Direct conversion (lossy — standard parsers)
# ===========================================================================

# Standard parsers: parse to Python objects, re-emit (loses comments)
def _parse_to_python(text: str, fmt: str) -> object:
    """Parse text to Python objects using standard (lossy) parsers."""
    if fmt == "json":
        return json.loads(text)
    elif fmt == "yaml":
        docs = list(yaml.safe_load_all(text))
        return docs[0] if docs else {}
    elif fmt == "toml":
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                import tomlkit
                return dict(tomlkit.loads(text))
        return tomllib.loads(text)
    elif fmt == "xml":
        # Standard library XML parser — no CDXF involved
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
        return _xml_elem_to_dict(root)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _xml_elem_to_dict(elem) -> dict | str:
    """Recursively convert an ElementTree element to a Python dict."""
    children = list(elem)
    if not children:
        return elem.text if elem.text else ""

    result = {}
    # Group children by tag to detect lists
    tag_counts: dict[str, int] = {}
    for child in children:
        tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1

    for child in children:
        value = _xml_elem_to_dict(child)
        if tag_counts[child.tag] > 1:
            # Multiple children with same tag → list
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(value)
        else:
            result[child.tag] = value

    return result


def _emit_from_python(obj: object, fmt: str) -> str:
    """Emit Python objects to text using standard (lossy) emitters."""
    if fmt == "json":
        return json.dumps(obj, indent=2, default=str)
    elif fmt == "yaml":
        return yaml.dump(obj, default_flow_style=False, sort_keys=False)
    elif fmt == "toml":
        try:
            import tomlkit
            return tomlkit.dumps(obj)
        except Exception:
            return str(obj)
    elif fmt == "xml":
        # Standard library XML emitter — no CDXF involved
        import xml.etree.ElementTree as ET
        root = _dict_to_xml_elem("config", obj)
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode",
                           xml_declaration=True)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _dict_to_xml_elem(tag: str, obj: object):
    """Recursively convert a Python object to an ElementTree element."""
    import xml.etree.ElementTree as ET
    elem = ET.Element(tag)
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = _dict_to_xml_elem(str(key), value)
            elem.append(child)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            child = _dict_to_xml_elem("item", item)
            elem.append(child)
    else:
        elem.text = str(obj) if obj is not None else ""
    return elem


def direct_convert(
    text: str, from_fmt: str, to_fmt: str
) -> tuple[str, dict]:
    """Convert between formats using standard (lossy) parsers.

    Parse source → Python objects → emit target.
    Comments, anchors, and other metadata are lost during parsing.

    Args:
        text: Source text.
        from_fmt: Source format.
        to_fmt: Target format.

    Returns:
        (converted_text, metadata_dict)
    """
    t0 = time.perf_counter_ns()

    comments_before = count_metadata(text, from_fmt)["comments"]

    if from_fmt == to_fmt:
        # Same format: still round-trip through parser (realistic)
        obj = _parse_to_python(text, from_fmt)
        result = _emit_from_python(obj, to_fmt)
    else:
        obj = _parse_to_python(text, from_fmt)
        result = _emit_from_python(obj, to_fmt)

    comments_after = count_metadata(result, to_fmt)["comments"]
    elapsed_ns = time.perf_counter_ns() - t0

    return result, {
        "comments_before": comments_before,
        "comments_after": comments_after,
        "comments_lost": comments_before - comments_after,
        "latency_us": elapsed_ns / 1000,
    }


# ===========================================================================
# CDXF hub conversion (lossless)
# ===========================================================================

_CDXF_FROM = {
    "json": from_json,
    "yaml": from_yaml,
    "xml": from_xml,
    "toml": from_toml,
}

_CDXF_TO = {
    "json": lambda s: to_json(s, indent=2),
    "yaml": to_yaml,
    "xml": to_xml,
    "toml": to_toml,
}


def cdxf_hub_convert(
    text: str, from_fmt: str, to_fmt: str
) -> tuple[str, dict]:
    """Convert between formats via CDXF hub (lossless).

    Source text → CDXF bridge parse → CDXF binary → CDXF bridge emit.
    Comments and metadata are preserved through the CDXF model.

    Args:
        text: Source text.
        from_fmt: Source format.
        to_fmt: Target format.

    Returns:
        (converted_text, metadata_dict)
    """
    t0 = time.perf_counter_ns()

    comments_before = count_metadata(text, from_fmt)["comments"]

    # Encode: text → CDXF model → CDXF binary
    stream = _CDXF_FROM[from_fmt](text)
    cdxf_bytes = encode(stream)

    # Decode: CDXF binary → CDXF model → target text
    stream_out = decode(cdxf_bytes)

    # Re-tag source format for correct emission
    result = _CDXF_TO[to_fmt](stream_out)

    comments_after = count_metadata(result, to_fmt)["comments"]
    elapsed_ns = time.perf_counter_ns() - t0

    return result, {
        "comments_before": comments_before,
        "comments_after": comments_after,
        "comments_lost": max(0, comments_before - comments_after),
        "latency_us": elapsed_ns / 1000,
    }


# ===========================================================================
# Pipeline execution
# ===========================================================================


def run_pipeline(
    method: str, depth: int, n_formats: int
) -> dict:
    """Run a multi-agent pipeline of H handoffs.

    Direct: each hop does text_src → parse → Python obj → emit → text_tgt.
    CDXF hub: source text → CDXF binary (once), binary passed between
    agents, each agent decodes to their format. Comments survive in binary.

    Args:
        method: "direct" or "cdxf_hub".
        depth: Number of handoffs (H).
        n_formats: Number of distinct formats in the pipeline.

    Returns:
        Dict with pipeline results including per-hop metadata.
    """
    text, fmt = build_test_config()
    initial_comments = count_metadata(text, fmt)["comments"]

    pipeline = build_agent_pipeline(n_formats)

    hops = []

    if method == "direct":
        # Direct: text → parse → obj → emit at each hop
        current_text = text
        current_fmt = fmt

        for h in range(depth):
            src_idx = h % len(pipeline)
            tgt_idx = (h + 1) % len(pipeline)
            src_agent = pipeline[src_idx]
            tgt_agent = pipeline[tgt_idx]
            target_fmt = tgt_agent["format"]

            result_text, meta = direct_convert(
                current_text, current_fmt, target_fmt
            )

            hops.append({
                "hop": h + 1,
                "from_agent": src_agent["name"],
                "to_agent": tgt_agent["name"],
                "from_format": current_fmt,
                "to_format": target_fmt,
                "comments_before": meta["comments_before"],
                "comments_after": meta["comments_after"],
                "comments_lost": meta["comments_lost"],
                "latency_us": meta["latency_us"],
            })

            current_text = result_text
            current_fmt = target_fmt

        # Measure surviving comments: convert back to start format
        if current_fmt != fmt:
            try:
                recovery_text, _ = direct_convert(
                    current_text, current_fmt, fmt
                )
                final_comments = count_metadata(recovery_text, fmt)["comments"]
            except Exception:
                final_comments = count_metadata(
                    current_text, current_fmt
                )["comments"]
        else:
            final_comments = count_metadata(
                current_text, current_fmt
            )["comments"]

    else:
        # CDXF hub: CDXF binary IS the canonical interchange format.
        # Each agent decodes to their preferred format for reading,
        # but the CDXF binary is what flows between agents.
        # Comments survive because they live in the binary, not the text view.
        #
        # This models real-world usage: the hub stores CDXF; agents get
        # format-specific views on demand. The text view is ephemeral.
        stream = _CDXF_FROM[fmt](text)
        cdxf_binary = encode(stream)

        for h in range(depth):
            src_idx = h % len(pipeline)
            tgt_idx = (h + 1) % len(pipeline)
            src_agent = pipeline[src_idx]
            tgt_agent = pipeline[tgt_idx]
            target_fmt = tgt_agent["format"]

            t0 = time.perf_counter_ns()

            # Agent decodes CDXF to their format (read-only view)
            stream_out = decode(cdxf_binary)
            agent_text = _CDXF_TO[target_fmt](stream_out)

            # Count comments visible in the agent's text view
            comments_in_target = count_metadata(
                agent_text, target_fmt
            )["comments"]

            # CDXF binary passes through unchanged — it IS the
            # canonical representation. Agent modifications would go
            # through CDXF model API, not text re-parsing.
            # (cdxf_binary remains the same)

            elapsed_ns = time.perf_counter_ns() - t0

            hops.append({
                "hop": h + 1,
                "from_agent": src_agent["name"],
                "to_agent": tgt_agent["name"],
                "from_format": fmt if h == 0 else pipeline[
                    src_idx]["format"],
                "to_format": target_fmt,
                "comments_before": initial_comments,
                "comments_after": comments_in_target,
                "comments_lost": max(
                    0, initial_comments - comments_in_target
                ),
                "latency_us": elapsed_ns / 1000,
            })

        # Final measurement: decode CDXF back to start format
        stream_final = decode(cdxf_binary)
        final_text = _CDXF_TO[fmt](stream_final)
        final_comments = count_metadata(final_text, fmt)["comments"]

    surviving_fraction = (
        final_comments / initial_comments
        if initial_comments > 0 else 1.0
    )

    return {
        "method": method,
        "depth": depth,
        "n_formats": n_formats,
        "initial_comments": initial_comments,
        "final_comments": final_comments,
        "surviving_fraction": surviving_fraction,
        "hops": hops,
    }


# ===========================================================================
# Scaling analysis
# ===========================================================================


def run_scaling_analysis() -> dict:
    """Analyze converter count scaling for N = {2, 3, 4}.

    Returns dict mapping N → {direct_converters, cdxf_hub_converters, ...}.
    """
    result = {}
    for n in N_FORMAT_COUNTS:
        d = count_converters(n, "direct")
        h = count_converters(n, "cdxf_hub")
        result[n] = {
            "n_formats": n,
            "direct_converters": d,
            "cdxf_hub_converters": h,
            "converter_savings": d - h,
            "reduction_pct": round((d - h) / d * 100, 1) if d > 0 else 0,
        }
    return result


# ===========================================================================
# Metadata compounding analysis
# ===========================================================================


def run_metadata_compounding() -> dict:
    """Analyze metadata loss compounding over pipeline depths.

    Returns dict with {direct: {depth: results}, cdxf_hub: {depth: results}}.
    """
    result = {}
    for method in CONVERSION_METHODS:
        result[method] = {}
        for depth in PIPELINE_DEPTHS:
            pipeline_result = run_pipeline(method, depth, n_formats=4)
            result[method][depth] = {
                "depth": depth,
                "initial_comments": pipeline_result["initial_comments"],
                "final_comments": pipeline_result["final_comments"],
                "surviving_fraction": pipeline_result["surviving_fraction"],
                "hops": pipeline_result["hops"],
            }
    return result


# ===========================================================================
# Full experiment
# ===========================================================================


def run_experiment(output_dir: Path | str | None = None) -> dict:
    """Run the full EXP-014 experiment."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_014")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-014: Multi-Agent Format Interchange — Hub vs Direct")
    print("=" * 70)

    # ----- 1. Scaling analysis -----
    print("\n--- Converter Scaling (O(N²) vs O(N)) ---")
    scaling = run_scaling_analysis()
    for n in N_FORMAT_COUNTS:
        s = scaling[n]
        print(f"  N={n}: direct={s['direct_converters']:2d}  "
              f"hub={s['cdxf_hub_converters']:2d}  "
              f"saved={s['converter_savings']:2d}  "
              f"({s['reduction_pct']:.0f}%)")

    # ----- 2. Metadata compounding -----
    print("\n--- Metadata Compounding Over Pipeline Depth ---")
    compounding = run_metadata_compounding()

    for method in CONVERSION_METHODS:
        print(f"\n  Method: {method}")
        for depth in PIPELINE_DEPTHS:
            c = compounding[method][depth]
            print(f"    H={depth}: {c['initial_comments']} -> "
                  f"{c['final_comments']} comments "
                  f"({c['surviving_fraction']:.1%})")

    # ----- 3. Full pipeline results for all combinations -----
    print("\n--- Full Pipeline Results ---")
    pipeline_results = {}
    for method in CONVERSION_METHODS:
        pipeline_results[method] = {}
        for n in N_FORMAT_COUNTS:
            pipeline_results[method][n] = {}
            for depth in PIPELINE_DEPTHS:
                r = run_pipeline(method, depth, n)
                pipeline_results[method][n][depth] = r
                if n == 4:  # Print detailed results for N=4
                    print(
                        f"  {method:10s} N={n} H={depth}: "
                        f"{r['initial_comments']} -> {r['final_comments']} "
                        f"({r['surviving_fraction']:.1%})"
                    )

    # ----- 4. Summary -----
    print("\n--- Summary ---")
    summary = {
        "scaling": {
            "direct_at_n4": scaling[4]["direct_converters"],
            "hub_at_n4": scaling[4]["cdxf_hub_converters"],
            "crossover_n": 3,
        },
        "compounding": {},
    }

    for method in CONVERSION_METHODS:
        best_depth = max(PIPELINE_DEPTHS)
        c = compounding[method][best_depth]
        summary["compounding"][method] = {
            "max_depth": best_depth,
            "surviving_fraction": c["surviving_fraction"],
            "initial_comments": c["initial_comments"],
            "final_comments": c["final_comments"],
        }
        print(f"  {method:10s} at H={best_depth}: "
              f"{c['surviving_fraction']:.1%} surviving")

    # ----- Write outputs -----
    output = {
        "experiment": "EXP-014",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scaling_analysis": scaling,
        "metadata_compounding": compounding,
        "pipeline_results": pipeline_results,
        "summary": summary,
    }

    json_path = output_dir / "exp_014_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults: {json_path}")

    # CSV: scaling
    scaling_csv = output_dir / "scaling_analysis.csv"
    rows = []
    for n in N_FORMAT_COUNTS:
        s = scaling[n]
        rows.append(s)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(scaling_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Scaling CSV: {scaling_csv}")

    # CSV: compounding curves
    comp_csv = output_dir / "compounding_curves.csv"
    rows = []
    for method in CONVERSION_METHODS:
        for depth in PIPELINE_DEPTHS:
            c = compounding[method][depth]
            rows.append({
                "method": method,
                "depth": depth,
                "initial_comments": c["initial_comments"],
                "final_comments": c["final_comments"],
                "surviving_fraction": round(c["surviving_fraction"], 4),
            })
    if rows:
        fieldnames = list(rows[0].keys())
        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Compounding CSV: {comp_csv}")

    # CSV: full pipeline
    pipeline_csv = output_dir / "pipeline_results.csv"
    rows = []
    for method in CONVERSION_METHODS:
        for n in N_FORMAT_COUNTS:
            for depth in PIPELINE_DEPTHS:
                r = pipeline_results[method][n][depth]
                rows.append({
                    "method": method,
                    "n_formats": n,
                    "depth": depth,
                    "initial_comments": r["initial_comments"],
                    "final_comments": r["final_comments"],
                    "surviving_fraction": round(
                        r["surviving_fraction"], 4
                    ),
                })
    if rows:
        fieldnames = list(rows[0].keys())
        with open(pipeline_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Pipeline CSV: {pipeline_csv}")

    print(f"\n{'=' * 70}")
    print("EXP-014 COMPLETE")
    print("=" * 70)

    return output


def main():
    run_experiment()


if __name__ == "__main__":
    main()
