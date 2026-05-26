"""
EXP-015: LangGraph Stateful Agent — Config Handoff Fidelity

Build a real LangGraph StateGraph where ML agents read/modify configs
passed via state. LangGraph serializes state as JSON (default).
Compare JSON state vs CDXF-enhanced state for metadata survival.

Two modes:
  - json_default: config parsed to Python dict via yaml.safe_load,
    stored in state as dict. LangGraph JSON-serializes it → comments lost.
  - cdxf_enhanced: config encoded to CDXF binary, stored as base64 string
    in state. LangGraph JSON-serializes the string → comments preserved.

Usage:
    python benchmarks/src/run_exp015.py
"""

from __future__ import annotations

import base64
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    print("ERROR: langgraph not installed. Run: pip install langgraph")
    sys.exit(1)

try:
    from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
    from cdxf.codec import encode, decode
except ImportError:
    print("ERROR: cdxf not installed. Run: pip install -e .")
    sys.exit(1)

# Shared corpus for enhanced experiments
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from benchmarks.src.config_corpus import (
    YAML_CONFIGS,
    ROLE_MODIFICATIONS,
    EXPECTED_VALUES_AFTER_4AGENT,
    EXPECTED_VALUES_AFTER_6AGENT,
    count_config_metadata as corpus_count_metadata,
    verify_data_integrity,
    Timer,
)


# ===========================================================================
# Protocol constants
# ===========================================================================

STATE_MODES = ["json_default", "cdxf_enhanced"]

GRAPH_CONFIGS = [
    {
        "name": "linear_4node",
        "description": "Linear pipeline: curator → trainer → evaluator → deployer",
        "nodes": ["data_curator", "trainer", "evaluator", "deployer"],
    },
    {
        "name": "linear_6node",
        "description": "Extended: curator → trainer → evaluator → deployer → monitor → reviewer",
        "nodes": ["data_curator", "trainer", "evaluator", "deployer",
                  "monitor", "reviewer"],
    },
]


# ===========================================================================
# Initial config — richly annotated YAML
# ===========================================================================


def build_initial_config() -> tuple[str, str]:
    """Build an ML config with rich metadata for testing.

    Returns (text, format).
    """
    text = (
        "# ============================================\n"
        "# ML Pipeline Configuration\n"
        "# Owner: ML Platform Team\n"
        "# Created: 2026-01-15\n"
        "# ============================================\n"
        "\n"
        "# Model architecture selection\n"
        "model:\n"
        "  name: llama-2-7b-chat\n"
        "  # Base model from HuggingFace Hub\n"
        "  base_model: meta-llama/Llama-2-7b-hf\n"
        "  revision: main\n"
        "  # Task type determines head architecture\n"
        "  task_type: causal_lm\n"
        "\n"
        "# Training hyperparameters\n"
        "training:\n"
        "  # Learning rate from grid search [1e-5, 5e-5, 1e-4]\n"
        "  learning_rate: 2.0e-5\n"
        "  num_epochs: 3\n"
        "  per_device_batch_size: 4\n"
        "  # Gradient accumulation for effective batch 128\n"
        "  gradient_accumulation_steps: 8\n"
        "  warmup_ratio: 0.03\n"
        "  # Weight decay regularization\n"
        "  weight_decay: 0.01\n"
        "  # Seed for reproducibility across runs\n"
        "  seed: 42\n"
        "  max_grad_norm: 1.0\n"
        "\n"
        "# Dataset configuration\n"
        "data:\n"
        "  # Alpaca-cleaned dataset\n"
        "  dataset: tatsu-lab/alpaca\n"
        "  max_length: 2048\n"
        "  # Train/validation split\n"
        "  val_ratio: 0.1\n"
        "  num_proc: 8\n"
        "\n"
        "# Evaluation benchmarks\n"
        "evaluation:\n"
        "  # Run after each epoch\n"
        "  benchmarks:\n"
        "    - mmlu\n"
        "    - hellaswag\n"
        "    - arc_challenge\n"
        "  # Minimum acceptable accuracy\n"
        "  min_accuracy: 0.45\n"
        "\n"
        "# Deployment settings\n"
        "deployment:\n"
        "  # vLLM serving config\n"
        "  engine: vllm\n"
        "  max_num_seqs: 64\n"
        "  gpu_memory_utilization: 0.9\n"
        "  # Port for inference API\n"
        "  port: 8000\n"
    )
    return text, "yaml"


# ===========================================================================
# Metadata counting
# ===========================================================================


def count_config_metadata(text: str, fmt: str) -> dict:
    """Count metadata constructs in config text."""
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

    return {"comments": comments, "total": comments}


# ===========================================================================
# State serialization — the core comparison
# ===========================================================================


def serialize_config_for_state(
    text: str, fmt: str, mode: str
) -> Any:
    """Serialize config into a value suitable for LangGraph state.

    json_default: parse YAML to dict (standard approach — lossy).
    cdxf_enhanced: encode to CDXF binary, store as base64 string (lossless).

    Both return JSON-serializable values (LangGraph requirement).
    """
    if mode == "json_default":
        # Standard approach: parse YAML to Python dict
        # This is what developers naturally do — comments are lost
        docs = list(yaml.safe_load_all(text))
        return docs[0] if docs else {}

    elif mode == "cdxf_enhanced":
        # CDXF approach: encode to binary, store as base64
        stream = from_yaml(text)
        cdxf_bytes = encode(stream)
        return base64.b64encode(cdxf_bytes).decode("ascii")

    else:
        raise ValueError(f"Unknown mode: {mode}")


def extract_config_text(
    state_value: Any, fmt: str, mode: str
) -> str:
    """Extract config text back from a LangGraph state value.

    json_default: re-emit dict as YAML via yaml.dump (lossy).
    cdxf_enhanced: decode CDXF binary, emit as YAML (lossless).
    """
    if mode == "json_default":
        # Re-emit Python dict as YAML — comments are gone
        return yaml.dump(state_value, default_flow_style=False,
                         sort_keys=False)

    elif mode == "cdxf_enhanced":
        # Decode CDXF binary and emit as YAML
        cdxf_bytes = base64.b64decode(state_value.encode("ascii"))
        stream = decode(cdxf_bytes)
        return to_yaml(stream)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ===========================================================================
# LangGraph state definition
# ===========================================================================


class AgentState(TypedDict):
    """LangGraph state for the ML pipeline."""
    config_data: Any           # The config (dict or base64 CDXF string)
    config_format: str         # "yaml"
    state_mode: str            # "json_default" or "cdxf_enhanced"
    node_trace: list[dict]     # Trace of what each node did
    modifications: list[str]   # Log of modifications made


def _make_node_fn(node_name: str, modification: dict):
    """Create a LangGraph node function for an agent.

    Each agent:
    1. Extracts config from state
    2. Reads/validates it
    3. Makes a small modification
    4. Re-serializes back to state
    """
    def node_fn(state: AgentState) -> dict:
        mode = state["state_mode"]
        fmt = state["config_format"]

        # Extract config text from state
        config_text = extract_config_text(
            state["config_data"], fmt, mode
        )

        # Count metadata before this node's work
        meta_before = count_config_metadata(config_text, fmt)

        # Make a modification (regex-based, preserves text structure)
        modified_text = config_text
        for pattern, replacement_fn in modification.items():
            modified_text = re.sub(
                pattern, replacement_fn, modified_text, count=1
            )

        # Count metadata after modification
        meta_after = count_config_metadata(modified_text, fmt)

        # Re-serialize for state
        new_config_data = serialize_config_for_state(
            modified_text, fmt, mode
        )

        # Build trace entry
        trace_entry = {
            "node_name": node_name,
            "comments_before": meta_before["comments"],
            "comments_after": meta_after["comments"],
            "modification": f"{node_name} modified config",
        }

        return {
            "config_data": new_config_data,
            "node_trace": state.get("node_trace", []) + [trace_entry],
            "modifications": (
                state.get("modifications", []) +
                [f"{node_name}: applied changes"]
            ),
        }

    return node_fn


# Agent node modifications (realistic parameter tweaks)
_NODE_MODIFICATIONS = {
    "data_curator": {
        r"(num_proc:\s*)(\d+)":
            lambda m: m.group(1) + str(int(m.group(2)) + 4),
    },
    "trainer": {
        r"(num_epochs:\s*)(\d+)":
            lambda m: m.group(1) + str(int(m.group(2)) + 1),
    },
    "evaluator": {
        r"(min_accuracy:\s*)([\d.]+)":
            lambda m: m.group(1) + str(round(float(m.group(2)) + 0.05, 2)),
    },
    "deployer": {
        r"(max_num_seqs:\s*)(\d+)":
            lambda m: m.group(1) + str(int(m.group(2)) + 16),
    },
    "monitor": {
        r"(port:\s*)(\d+)":
            lambda m: m.group(1) + str(int(m.group(2)) + 1),
    },
    "reviewer": {
        r"(seed:\s*)(\d+)":
            lambda m: m.group(1) + str(int(m.group(2)) + 1),
    },
}


# ===========================================================================
# LangGraph pipeline construction
# ===========================================================================


def build_langgraph_pipeline(
    mode: str,
    graph_config: dict | None = None,
    checkpointer=None,
) -> Any:
    """Build and compile a LangGraph StateGraph.

    Args:
        mode: "json_default" or "cdxf_enhanced".
        graph_config: Graph topology config (default: linear_4node).
        checkpointer: Optional LangGraph checkpointer.

    Returns:
        Compiled LangGraph graph.
    """
    if graph_config is None:
        graph_config = GRAPH_CONFIGS[0]

    builder = StateGraph(AgentState)

    nodes = graph_config["nodes"]
    for node_name in nodes:
        mods = _NODE_MODIFICATIONS.get(node_name, {})
        builder.add_node(node_name, _make_node_fn(node_name, mods))

    # Linear pipeline: START → node1 → node2 → ... → END
    builder.add_edge(START, nodes[0])
    for i in range(len(nodes) - 1):
        builder.add_edge(nodes[i], nodes[i + 1])
    builder.add_edge(nodes[-1], END)

    return builder.compile(checkpointer=checkpointer)


# ===========================================================================
# Graph execution
# ===========================================================================


def run_graph(
    mode: str,
    graph_config: dict | None = None,
) -> dict:
    """Execute a LangGraph pipeline and measure metadata survival.

    Args:
        mode: "json_default" or "cdxf_enhanced".
        graph_config: Optional graph topology.

    Returns:
        Dict with execution results and metadata trace.
    """
    if graph_config is None:
        graph_config = GRAPH_CONFIGS[0]

    text, fmt = build_initial_config()
    initial_meta = count_config_metadata(text, fmt)

    # Serialize config for initial state
    config_data = serialize_config_for_state(text, fmt, mode)

    initial_state = {
        "config_data": config_data,
        "config_format": fmt,
        "state_mode": mode,
        "node_trace": [],
        "modifications": [],
    }

    # Build and run the graph
    graph = build_langgraph_pipeline(mode, graph_config)
    final_state = graph.invoke(initial_state)

    # Extract final config and measure metadata
    final_text = extract_config_text(
        final_state["config_data"], fmt, mode
    )
    final_meta = count_config_metadata(final_text, fmt)

    surviving_fraction = (
        final_meta["comments"] / initial_meta["comments"]
        if initial_meta["comments"] > 0 else 1.0
    )

    return {
        "mode": mode,
        "graph_config": graph_config["name"],
        "initial_comments": initial_meta["comments"],
        "final_comments": final_meta["comments"],
        "surviving_fraction": surviving_fraction,
        "node_trace": final_state["node_trace"],
        "n_nodes": len(graph_config["nodes"]),
        "modifications": final_state["modifications"],
    }


def run_graph_with_checkpoints(
    mode: str,
    graph_config: dict | None = None,
) -> dict:
    """Execute with LangGraph checkpointing to test serialization fidelity.

    LangGraph checkpointers serialize state to JSON. This tests whether
    config metadata survives a checkpoint → restore cycle.
    """
    if graph_config is None:
        graph_config = GRAPH_CONFIGS[0]

    text, fmt = build_initial_config()
    initial_meta = count_config_metadata(text, fmt)

    config_data = serialize_config_for_state(text, fmt, mode)

    initial_state = {
        "config_data": config_data,
        "config_format": fmt,
        "state_mode": mode,
        "node_trace": [],
        "modifications": [],
    }

    # Use InMemorySaver — it JSON-serializes state for checkpoints
    checkpointer = InMemorySaver()
    graph = build_langgraph_pipeline(mode, graph_config,
                                      checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": f"exp015-{mode}"}}
    final_state = graph.invoke(initial_state, thread_config)

    # Retrieve checkpoint data
    checkpoints = list(checkpointer.list(thread_config))

    # Extract config from final state after checkpoint
    final_text = extract_config_text(
        final_state["config_data"], fmt, mode
    )
    final_meta = count_config_metadata(final_text, fmt)

    return {
        "mode": mode,
        "initial_comments": initial_meta["comments"],
        "comments_after_restore": final_meta["comments"],
        "surviving_fraction": (
            final_meta["comments"] / initial_meta["comments"]
            if initial_meta["comments"] > 0 else 1.0
        ),
        "checkpoints": [
            {"checkpoint_id": cp.checkpoint["id"],
             "node": cp.metadata.get("source", "unknown")}
            for cp in checkpoints[:5]  # Limit output size
        ],
        "n_checkpoints": len(checkpoints),
    }


# ===========================================================================
# Enhanced experiments — scaling, timing, data integrity
# ===========================================================================


def run_scaling_experiment(
    config_sizes: list[str] | None = None,
) -> dict:
    """Test metadata fidelity across different config sizes.

    Uses the shared config corpus (small/medium/large/xlarge) to show
    that CDXF fidelity is size-invariant.
    """
    if config_sizes is None:
        config_sizes = ["small", "medium", "large", "xlarge"]

    results = []
    for size in config_sizes:
        cfg = YAML_CONFIGS[size]
        text = cfg["text"]
        fmt = cfg["format"]

        for mode in STATE_MODES:
            initial_meta = count_config_metadata(text, fmt)
            config_data = serialize_config_for_state(text, fmt, mode)

            initial_state = {
                "config_data": config_data,
                "config_format": fmt,
                "state_mode": mode,
                "node_trace": [],
                "modifications": [],
            }

            graph = build_langgraph_pipeline(mode, GRAPH_CONFIGS[0])
            final_state = graph.invoke(initial_state)

            final_text = extract_config_text(
                final_state["config_data"], fmt, mode
            )
            final_meta = count_config_metadata(final_text, fmt)

            surviving = (
                final_meta["comments"] / initial_meta["comments"]
                if initial_meta["comments"] > 0 else 1.0
            )

            results.append({
                "config_size": size,
                "mode": mode,
                "initial_comments": initial_meta["comments"],
                "final_comments": final_meta["comments"],
                "surviving_fraction": surviving,
            })

    return {"scaling_results": results}


def run_timing_experiment(
    n_iterations: int = 20,
) -> dict:
    """Measure overhead of CDXF vs JSON serialization in LangGraph.

    Times: initial serialization, per-node round-trip, full pipeline.
    Reports mean times over n_iterations.
    """
    text, fmt = build_initial_config()
    timer = Timer()

    # Time initial serialization
    timer.measure_n(
        "json_default_serialize",
        lambda: serialize_config_for_state(text, fmt, "json_default"),
        n=n_iterations,
    )
    timer.measure_n(
        "cdxf_enhanced_serialize",
        lambda: serialize_config_for_state(text, fmt, "cdxf_enhanced"),
        n=n_iterations,
    )

    # Time extract (deserialize)
    json_val = serialize_config_for_state(text, fmt, "json_default")
    cdxf_val = serialize_config_for_state(text, fmt, "cdxf_enhanced")

    timer.measure_n(
        "json_default_extract",
        lambda: extract_config_text(json_val, fmt, "json_default"),
        n=n_iterations,
    )
    timer.measure_n(
        "cdxf_enhanced_extract",
        lambda: extract_config_text(cdxf_val, fmt, "cdxf_enhanced"),
        n=n_iterations,
    )

    # Time full pipeline
    def run_pipeline(mode):
        config_data = serialize_config_for_state(text, fmt, mode)
        state = {
            "config_data": config_data,
            "config_format": fmt,
            "state_mode": mode,
            "node_trace": [],
            "modifications": [],
        }
        graph = build_langgraph_pipeline(mode, GRAPH_CONFIGS[0])
        graph.invoke(state)

    timer.measure_n(
        "json_default_pipeline",
        lambda: run_pipeline("json_default"),
        n=n_iterations,
    )
    timer.measure_n(
        "cdxf_enhanced_pipeline",
        lambda: run_pipeline("cdxf_enhanced"),
        n=n_iterations,
    )

    timings = timer.summary()
    overhead = {
        "serialize_overhead_ms": (
            (timings["cdxf_enhanced_serialize"] -
             timings["json_default_serialize"]) * 1000
        ),
        "extract_overhead_ms": (
            (timings["cdxf_enhanced_extract"] -
             timings["json_default_extract"]) * 1000
        ),
        "pipeline_overhead_ms": (
            (timings["cdxf_enhanced_pipeline"] -
             timings["json_default_pipeline"]) * 1000
        ),
    }

    return {
        "timings_seconds": timings,
        "overhead": overhead,
        "n_iterations": n_iterations,
    }


def run_integrity_experiment() -> dict:
    """Verify that agent modifications are correctly applied AND preserved.

    For each mode × topology, check that the final config contains the
    expected values after all agents have made their modifications.
    """
    results = []

    for gc in GRAPH_CONFIGS:
        n_agents = len(gc["nodes"])
        expected = (
            EXPECTED_VALUES_AFTER_6AGENT
            if n_agents >= 6 else EXPECTED_VALUES_AFTER_4AGENT
        )

        for mode in STATE_MODES:
            r = run_graph(mode, gc)
            final_text = extract_config_text(
                # Re-run to get final state
                serialize_config_for_state(
                    *build_initial_config(), mode
                ),
                "yaml", mode,
            )
            # Actually run the full pipeline to get the real final text
            text, fmt = build_initial_config()
            config_data = serialize_config_for_state(text, fmt, mode)
            state = {
                "config_data": config_data,
                "config_format": fmt,
                "state_mode": mode,
                "node_trace": [],
                "modifications": [],
            }
            graph = build_langgraph_pipeline(mode, gc)
            final_state = graph.invoke(state)
            final_text = extract_config_text(
                final_state["config_data"], fmt, mode
            )

            integrity = verify_data_integrity(final_text, expected)

            results.append({
                "graph_config": gc["name"],
                "mode": mode,
                "n_agents": n_agents,
                "integrity_passed": integrity["passed"],
                "checks": integrity["checks"],
                "failures": integrity["failures"],
            })

    return {"integrity_results": results}


# ===========================================================================
# Full experiment
# ===========================================================================


def run_experiment(output_dir: Path | str | None = None) -> dict:
    """Run the full EXP-015 experiment."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_015")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-015: LangGraph Stateful Agent — Config Handoff Fidelity")
    print("=" * 70)

    results = {}
    checkpoint_results = {}

    for gc in GRAPH_CONFIGS:
        print(f"\n--- Graph: {gc['name']} ({len(gc['nodes'])} nodes) ---")

        for mode in STATE_MODES:
            print(f"\n  Mode: {mode}")

            # Basic pipeline run
            r = run_graph(mode, gc)
            key = f"{gc['name']}_{mode}"
            results[key] = r
            print(f"    Pipeline: {r['initial_comments']} → "
                  f"{r['final_comments']} comments "
                  f"({r['surviving_fraction']:.1%})")

            # Checkpoint/restore cycle
            cr = run_graph_with_checkpoints(mode, gc)
            checkpoint_results[key] = cr
            print(f"    Checkpoint: {cr['initial_comments']} → "
                  f"{cr['comments_after_restore']} comments "
                  f"({cr['surviving_fraction']:.1%})  "
                  f"[{cr['n_checkpoints']} checkpoints]")

    # --- Enhanced experiments ---
    print("\n--- Scaling Experiment (multi-size configs) ---")
    scaling = run_scaling_experiment()
    for sr in scaling["scaling_results"]:
        if sr["mode"] == "cdxf_enhanced":
            print(f"  {sr['config_size']:8s}: {sr['initial_comments']} → "
                  f"{sr['final_comments']} ({sr['surviving_fraction']:.1%})")

    print("\n--- Timing Experiment ---")
    timing = run_timing_experiment()
    for k, v in timing["timings_seconds"].items():
        print(f"  {k:30s}: {v*1000:.3f} ms")
    for k, v in timing["overhead"].items():
        print(f"  {k:30s}: {v:+.3f} ms")

    print("\n--- Data Integrity Experiment ---")
    integrity = run_integrity_experiment()
    for ir in integrity["integrity_results"]:
        status = "PASS" if ir["integrity_passed"] else "FAIL"
        print(f"  {ir['graph_config']} / {ir['mode']}: "
              f"{status} ({ir['checks']} checks)")

    # Summary
    print("\n--- Summary ---")
    summary = {}
    for mode in STATE_MODES:
        gc = GRAPH_CONFIGS[0]
        key = f"{gc['name']}_{mode}"
        r = results[key]
        cr = checkpoint_results[key]
        summary[mode] = {
            "pipeline_surviving": r["surviving_fraction"],
            "checkpoint_surviving": cr["surviving_fraction"],
            "initial_comments": r["initial_comments"],
            "final_comments": r["final_comments"],
        }
        print(f"  {mode:18s}: pipeline={r['surviving_fraction']:.1%}  "
              f"checkpoint={cr['surviving_fraction']:.1%}")

    # Write outputs
    output = {
        "experiment": "EXP-015",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "framework": "langgraph",
        "framework_version": _get_version("langgraph"),
        "results": results,
        "checkpoint_results": checkpoint_results,
        "summary": summary,
        "scaling": scaling,
        "timing": timing,
        "integrity": integrity,
    }

    json_path = output_dir / "exp_015_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults: {json_path}")

    # CSV: comparison
    comp_csv = output_dir / "mode_comparison.csv"
    rows = []
    for key, r in results.items():
        cr = checkpoint_results[key]
        rows.append({
            "graph_config": r["graph_config"],
            "mode": r["mode"],
            "n_nodes": r["n_nodes"],
            "initial_comments": r["initial_comments"],
            "final_comments_pipeline": r["final_comments"],
            "surviving_fraction_pipeline": round(
                r["surviving_fraction"], 4
            ),
            "final_comments_checkpoint": cr["comments_after_restore"],
            "surviving_fraction_checkpoint": round(
                cr["surviving_fraction"], 4
            ),
            "n_checkpoints": cr["n_checkpoints"],
        })
    if rows:
        fieldnames = list(rows[0].keys())
        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Comparison CSV: {comp_csv}")

    print(f"\n{'=' * 70}")
    print("EXP-015 COMPLETE")
    print("=" * 70)

    return output


def _get_version(package: str) -> str:
    """Get installed package version."""
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return "unknown"


def main():
    run_experiment()


if __name__ == "__main__":
    main()
