"""
EXP-013: Agent Workflow State Persistence Across Sessions

Measures cumulative information loss when agent state is serialized across
K sequential session boundaries. Models a realistic autonomous research
agent that reads state, modifies configs, adds results, and re-serializes.

State formats:
  - CDXF: multi-document stream via CDXF bridges (lossless)
  - JSON mega-JSON: parse all files to Python objects, merge to JSON (lossy)
  - Pickle: parse all files to Python objects, pickle (lossy)
  - tar.gz: tar raw text files, but modification uses yaml.safe_load (gradual loss)

Usage:
    python benchmarks/src/run_exp013.py
"""

from __future__ import annotations

import copy
import csv
import io
import json
import pickle
import re
import sys
import tarfile
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
    from cdxf.codec import encode, decode
    from cdxf.model import (
        Comment, Document, Scalar, ScalarType,
        SourceFormat, Stream,
    )
except ImportError:
    print("ERROR: cdxf not installed. Run: pip install -e .")
    sys.exit(1)


# ===========================================================================
# Protocol constants
# ===========================================================================

SESSION_BOUNDARIES = [1, 2, 5, 10, 20]
STATE_COMPLEXITIES = {"small": 3, "medium": 8, "large": 15}
STATE_FORMATS = ["cdxf", "json_mega", "pickle", "tar_gz"]

_FORMAT_TO_SOURCE = {
    "json": SourceFormat.JSON,
    "yaml": SourceFormat.YAML,
    "toml": SourceFormat.TOML,
}

_FROM_BRIDGE = {
    "json": from_json,
    "yaml": from_yaml,
    "toml": from_toml,
}

_TO_BRIDGE = {
    "json": lambda stream: to_json(stream, indent=2),
    "yaml": to_yaml,
    "toml": to_toml,
}


# ===========================================================================
# Initial state construction
# ===========================================================================

def _make_dataset_card() -> dict:
    return {
        "name": "dataset_card",
        "filename": "dataset_card.yaml",
        "format": "yaml",
        "text": (
            "---\n"
            "# Dataset card for fine-tuning corpus\n"
            "# See https://huggingface.co/docs/datasets for schema\n"
            "dataset_info:\n"
            "  name: alpaca-cleaned\n"
            "  version: 1.0.0\n"
            "  description: Cleaned version of Stanford Alpaca dataset\n"
            "  # Original: 52K instructions, cleaned: 49.8K\n"
            "  num_examples: 49800\n"
            "  features:\n"
            "    - name: instruction\n"
            "      dtype: string\n"
            "    - name: input\n"
            "      dtype: string\n"
            "    - name: output\n"
            "      dtype: string\n"
            "  splits:\n"
            "    train: 44820\n"
            "    validation: 4980\n"
            "  # License inherited from original Alpaca\n"
            "  license: cc-by-nc-4.0\n"
            "  language: en\n"
            "---\n"
        ),
    }


def _make_preprocessing() -> dict:
    return {
        "name": "preprocessing",
        "filename": "preprocessing.json",
        "format": "json",
        "text": json.dumps({
            "tokenizer": "meta-llama/Llama-2-7b-hf",
            "max_length": 2048,
            "padding": "max_length",
            "truncation": True,
            "add_special_tokens": True,
            "template": "alpaca",
            "num_proc": 8,
            "batched": True,
        }, indent=2),
    }


def _make_training_hparams() -> dict:
    return {
        "name": "training_hparams",
        "filename": "training_hparams.yaml",
        "format": "yaml",
        "text": (
            "# Training hyperparameters for Llama-2-7B LoRA fine-tuning\n"
            "# Experiment: sft-alpaca-v3\n"
            "# Hardware: 4x A100 80GB\n"
            "\n"
            "training:\n"
            "  # LR from grid search over [1e-5, 5e-5, 1e-4]\n"
            "  learning_rate: 2.0e-5\n"
            "  # Epochs: early stopping with patience=3\n"
            "  num_train_epochs: 3\n"
            "  per_device_train_batch_size: 4\n"
            "  # Effective batch = 4 * 4 GPUs * 8 accum = 128\n"
            "  gradient_accumulation_steps: 8\n"
            "  warmup_ratio: 0.03\n"
            "  weight_decay: 0.01\n"
            "  lr_scheduler_type: cosine\n"
            "  max_grad_norm: 1.0\n"
            "\n"
            "precision:\n"
            "  bf16: true\n"
            "  tf32: true\n"
            "\n"
            "logging:\n"
            "  logging_steps: 10\n"
            "  # Checkpoint every 500 steps to resume on preemption\n"
            "  save_steps: 500\n"
            "  eval_steps: 500\n"
            "  save_total_limit: 3\n"
            "  report_to: wandb\n"
            "\n"
            "optimization:\n"
            "  # AdamW chosen over SGD for stability on LLMs\n"
            "  optim: adamw_torch\n"
            "  # Seed for reproducibility across runs\n"
            "  seed: 42\n"
        ),
    }


def _make_adapter_config() -> dict:
    return {
        "name": "adapter_config",
        "filename": "adapter_config.json",
        "format": "json",
        "text": json.dumps({
            "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
            "bias": "none",
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "task_type": "CAUSAL_LM",
        }, indent=2),
    }


def _make_quant_config() -> dict:
    return {
        "name": "quantization_config",
        "filename": "quantization_config.json",
        "format": "json",
        "text": json.dumps({
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }, indent=2),
    }


def _make_serving_config() -> dict:
    return {
        "name": "serving_config",
        "filename": "serving_config.toml",
        "format": "toml",
        "text": (
            "# vLLM serving configuration\n"
            "# Optimized for Llama-2-7B + LoRA on single A100\n"
            "\n"
            "[model]\n"
            'name = "meta-llama/Llama-2-7b-hf"\n'
            '# LoRA adapter path on shared storage\n'
            'lora_adapter = "/models/llama2-sft-alpaca-v3"\n'
            'dtype = "auto"\n'
            "max_model_len = 2048\n"
            "\n"
            "[serving]\n"
            'host = "0.0.0.0"\n'
            "port = 8000\n"
            "# Max concurrent requests based on GPU memory profiling\n"
            "max_num_seqs = 64\n"
            "max_num_batched_tokens = 4096\n"
            "gpu_memory_utilization = 0.9\n"
            "\n"
            "[logging]\n"
            '# Access log for monitoring request latency\n'
            "access_log = true\n"
        ),
    }


def _make_eval_results() -> dict:
    return {
        "name": "eval_results",
        "filename": "eval_results.json",
        "format": "json",
        "text": json.dumps({
            "model": "llama2-7b-sft-alpaca-v3",
            "eval_date": "2026-01-20",
            "benchmarks": {
                "mmlu": {"accuracy": 0.462},
                "hellaswag": {"accuracy": 0.782},
                "arc_challenge": {"accuracy": 0.534},
            },
            "training_loss_final": 0.876,
        }, indent=2),
    }


def _make_deployment_manifest() -> dict:
    return {
        "name": "deployment_manifest",
        "filename": "deployment.yaml",
        "format": "yaml",
        "text": (
            "# Kubernetes deployment for model serving\n"
            "# Cluster: ml-prod-us-east-1\n"
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: llama2-sft-alpaca\n"
            "  # Managed by ML Platform team\n"
            "  labels:\n"
            "    app: llama2-sft-alpaca\n"
            "    team: ml-platform\n"
            "spec:\n"
            "  replicas: 2\n"
            "  selector:\n"
            "    matchLabels:\n"
            "      app: llama2-sft-alpaca\n"
            "  template:\n"
            "    metadata:\n"
            "      labels:\n"
            "        app: llama2-sft-alpaca\n"
            "    spec:\n"
            "      containers:\n"
            "        - name: vllm\n"
            "          image: vllm/vllm-openai:v0.4.0\n"
            "          # Resource limits from capacity planning\n"
            "          resources:\n"
            "            requests:\n"
            "              cpu: '4'\n"
            "              memory: 32Gi\n"
        ),
    }


# --- Additional components for 'large' complexity ---

def _make_data_validation() -> dict:
    return {
        "name": "data_validation",
        "filename": "data_validation.yaml",
        "format": "yaml",
        "text": (
            "# Data quality validation config\n"
            "# Run before each training iteration\n"
            "validation:\n"
            "  # Check for empty fields\n"
            "  check_empty: true\n"
            "  min_length: 10\n"
            "  max_length: 4096\n"
            "  # Deduplication threshold\n"
            "  dedup_threshold: 0.95\n"
            "  language_filter: en\n"
        ),
    }


def _make_monitoring_config() -> dict:
    return {
        "name": "monitoring_config",
        "filename": "monitoring.toml",
        "format": "toml",
        "text": (
            "# Prometheus monitoring for model serving\n"
            "# Dashboards: grafana.ml-platform.internal\n"
            "\n"
            "[metrics]\n"
            "# Scrape interval in seconds\n"
            "scrape_interval = 15\n"
            "port = 9090\n"
            "\n"
            "[alerts]\n"
            "# Latency threshold for P99 in ms\n"
            "latency_p99_threshold = 500\n"
            "# Error rate threshold\n"
            "error_rate_threshold = 0.01\n"
        ),
    }


def _make_safety_config() -> dict:
    return {
        "name": "safety_config",
        "filename": "safety_config.yaml",
        "format": "yaml",
        "text": (
            "# Content safety filters\n"
            "# Required by policy before production deployment\n"
            "safety:\n"
            "  # Toxicity detection model\n"
            "  toxicity_model: unitary/toxic-bert\n"
            "  toxicity_threshold: 0.7\n"
            "  # PII detection\n"
            "  pii_detection: true\n"
            "  pii_types:\n"
            "    - email\n"
            "    - phone\n"
            "    - ssn\n"
            "  # Output length limits\n"
            "  max_output_tokens: 2048\n"
        ),
    }


def _make_cost_tracking() -> dict:
    return {
        "name": "cost_tracking",
        "filename": "cost_tracking.json",
        "format": "json",
        "text": json.dumps({
            "project": "llama2-sft-alpaca",
            "budget_usd": 500.0,
            "spent_usd": 168.0,
            "gpu_type": "A100-80GB",
            "gpu_hours_used": 16.8,
            "cost_per_gpu_hour": 10.0,
        }, indent=2),
    }


def _make_experiment_log() -> dict:
    return {
        "name": "experiment_log",
        "filename": "experiment_log.yaml",
        "format": "yaml",
        "text": (
            "# Experiment decision log\n"
            "# Records key decisions and their rationale\n"
            "decisions:\n"
            "  - date: 2026-01-15\n"
            "    # Chose LoRA over full fine-tuning for memory efficiency\n"
            "    decision: use LoRA (r=16)\n"
            "    rationale: 4x A100s not enough for full FT of 7B model\n"
            "  - date: 2026-01-17\n"
            "    # Switched from SGD to AdamW after loss instability\n"
            "    decision: use AdamW optimizer\n"
            "    rationale: SGD showed loss spikes at LR > 1e-5\n"
            "  - date: 2026-01-18\n"
            "    # Reduced LR after overfitting on validation set\n"
            "    decision: reduce LR from 5e-5 to 2e-5\n"
            "    rationale: validation loss diverged after epoch 2\n"
        ),
    }


def _make_tokenizer_config() -> dict:
    return {
        "name": "tokenizer_config",
        "filename": "tokenizer_config.json",
        "format": "json",
        "text": json.dumps({
            "model_type": "llama",
            "vocab_size": 32000,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "[PAD]",
            "add_bos_token": True,
            "add_eos_token": False,
            "model_max_length": 2048,
        }, indent=2),
    }


def _make_ci_pipeline() -> dict:
    return {
        "name": "ci_pipeline",
        "filename": "ci_pipeline.yaml",
        "format": "yaml",
        "text": (
            "# CI/CD pipeline for model evaluation\n"
            "# Triggered on new checkpoint push\n"
            "stages:\n"
            "  - name: eval_mmlu\n"
            "    # Run MMLU benchmark (5-shot)\n"
            "    command: lm_eval --model hf --tasks mmlu\n"
            "    timeout_minutes: 60\n"
            "  - name: eval_hellaswag\n"
            "    # Run HellaSwag (10-shot)\n"
            "    command: lm_eval --model hf --tasks hellaswag\n"
            "    timeout_minutes: 30\n"
            "  - name: safety_check\n"
            "    # Content safety evaluation\n"
            "    command: python safety_eval.py\n"
            "    timeout_minutes: 15\n"
        ),
    }


_ALL_COMPONENTS = [
    _make_dataset_card,        # 0: yaml, comments
    _make_preprocessing,       # 1: json
    _make_training_hparams,    # 2: yaml, comments
    _make_adapter_config,      # 3: json
    _make_quant_config,        # 4: json
    _make_serving_config,      # 5: toml, comments
    _make_eval_results,        # 6: json
    _make_deployment_manifest, # 7: yaml, comments
    _make_data_validation,     # 8: yaml, comments
    _make_monitoring_config,   # 9: toml, comments
    _make_safety_config,       # 10: yaml, comments
    _make_cost_tracking,       # 11: json
    _make_experiment_log,      # 12: yaml, comments
    _make_tokenizer_config,    # 13: json
    _make_ci_pipeline,         # 14: yaml, comments
]


def build_initial_state(complexity: str) -> list[dict]:
    """Build initial agent state for the given complexity.

    Args:
        complexity: "small" (3 files), "medium" (8 files), "large" (15 files).

    Returns:
        List of component dicts with {name, filename, format, text}.
    """
    n = STATE_COMPLEXITIES[complexity]
    return [fn() for fn in _ALL_COMPONENTS[:n]]


# ===========================================================================
# Metadata counting
# ===========================================================================


def count_metadata_constructs(text: str, fmt: str) -> dict:
    """Count format-specific metadata constructs in a file.

    Returns dict with {comments, anchors, typed_temporals, total}.
    """
    comments = 0
    anchors = 0
    typed_temporals = 0

    if not text:
        return {"comments": 0, "anchors": 0, "typed_temporals": 0, "total": 0}

    if fmt in ("yaml", "toml"):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                comments += 1
            elif " #" in line or "\t#" in line:
                # Inline comment
                # Check it's not inside a string
                hash_pos = line.find(" #")
                if hash_pos < 0:
                    hash_pos = line.find("\t#")
                if hash_pos >= 0:
                    # Simple heuristic: not inside quotes
                    before = line[:hash_pos]
                    if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                        comments += 1

    if fmt == "yaml":
        # Count anchors (&name)
        anchors = len(re.findall(r"&\w+", text))
        # Count typed temporals (YAML native dates)
        temporal_pattern = r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?"
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if ":" in line:
                value_part = line.split(":", 1)[1].strip() if ":" in line else ""
                if re.match(temporal_pattern, value_part):
                    typed_temporals += 1

    total = comments + anchors + typed_temporals
    return {
        "comments": comments,
        "anchors": anchors,
        "typed_temporals": typed_temporals,
        "total": total,
    }


def count_all_metadata(state: list[dict]) -> dict:
    """Count metadata across all files in a state.

    Returns dict with totals and per-file breakdown.
    """
    total_comments = 0
    total_anchors = 0
    total_temporals = 0
    per_file = []

    for comp in state:
        m = count_metadata_constructs(comp["text"], comp["format"])
        total_comments += m["comments"]
        total_anchors += m["anchors"]
        total_temporals += m["typed_temporals"]
        per_file.append({
            "filename": comp["filename"],
            "format": comp["format"],
            **m,
        })

    return {
        "total_comments": total_comments,
        "total_anchors": total_anchors,
        "total_typed_temporals": total_temporals,
        "grand_total": total_comments + total_anchors + total_temporals,
        "per_file": per_file,
    }


# ===========================================================================
# Serialization methods
# ===========================================================================


def serialize_state(state: list[dict], state_format: str) -> bytes:
    """Serialize agent state to bytes.

    Args:
        state: List of {name, filename, format, text} dicts.
        state_format: One of STATE_FORMATS.

    Returns:
        Serialized bytes.
    """
    if state_format == "cdxf":
        return _serialize_cdxf(state)
    elif state_format == "json_mega":
        return _serialize_json_mega(state)
    elif state_format == "pickle":
        return _serialize_pickle(state)
    elif state_format == "tar_gz":
        return _serialize_tar_gz(state)
    else:
        raise ValueError(f"Unknown state format: {state_format}")


def deserialize_state(data: bytes, state_format: str) -> list[dict]:
    """Deserialize agent state from bytes.

    Args:
        data: Serialized bytes.
        state_format: One of STATE_FORMATS.

    Returns:
        List of {name, filename, format, text} dicts.
    """
    if state_format == "cdxf":
        return _deserialize_cdxf(data)
    elif state_format == "json_mega":
        return _deserialize_json_mega(data)
    elif state_format == "pickle":
        return _deserialize_pickle(data)
    elif state_format == "tar_gz":
        return _deserialize_tar_gz(data)
    else:
        raise ValueError(f"Unknown state format: {state_format}")


# --- CDXF: lossless ---

def _serialize_cdxf(state: list[dict]) -> bytes:
    """Encode all files into a CDXF multi-document stream."""
    documents = []
    for comp in state:
        fmt = comp["format"]
        text = comp["text"]
        filename = comp["filename"]

        if fmt in _FROM_BRIDGE:
            stream = _FROM_BRIDGE[fmt](text)
            doc = stream.documents[0]
        else:
            root = Scalar(scalar_type=ScalarType.STRING, value=text)
            doc = Document(
                root=root,
                source_format_hint=SourceFormat.UNSPECIFIED,
            )

        doc.preamble.insert(
            0, Comment(text=f"cdxf-bundle-filename: {filename}")
        )
        doc.preamble.insert(
            1, Comment(text=f"cdxf-bundle-format: {fmt}")
        )
        doc.source_format_hint = _FORMAT_TO_SOURCE.get(
            fmt, SourceFormat.UNSPECIFIED
        )
        documents.append(doc)

    return encode(Stream(documents=documents))


def _deserialize_cdxf(data: bytes) -> list[dict]:
    """Decode a CDXF stream back to text files."""
    stream = decode(data)
    result = []

    for doc in stream.documents:
        filename = None
        fmt = None

        # Extract filename and format from preamble comments
        remaining_preamble = []
        for node in doc.preamble:
            if isinstance(node, Comment):
                if node.text.strip().startswith("cdxf-bundle-filename:"):
                    filename = node.text.strip().split(":", 1)[1].strip()
                elif node.text.strip().startswith("cdxf-bundle-format:"):
                    fmt = node.text.strip().split(":", 1)[1].strip()
                else:
                    remaining_preamble.append(node)
            else:
                remaining_preamble.append(node)

        doc.preamble = remaining_preamble

        if fmt is None:
            sf = doc.source_format_hint
            fmt_map = {
                SourceFormat.JSON: "json",
                SourceFormat.YAML: "yaml",
                SourceFormat.TOML: "toml",
            }
            fmt = fmt_map.get(sf, "json")

        if filename is None:
            filename = f"unknown.{fmt}"

        name = filename.rsplit(".", 1)[0] if "." in filename else filename

        # Re-emit as text via bridge
        rebuilt_stream = Stream(documents=[doc])
        if fmt in _TO_BRIDGE:
            text = _TO_BRIDGE[fmt](rebuilt_stream)
        else:
            text = str(doc.root.value) if doc.root else ""

        result.append({
            "name": name,
            "filename": filename,
            "format": fmt,
            "text": text,
        })

    return result


# --- JSON mega-JSON: lossy (parses everything to Python objects) ---

def _serialize_json_mega(state: list[dict]) -> bytes:
    """Parse all files to Python objects and merge into JSON.

    This models the 'just convert everything to JSON' approach.
    YAML/TOML comments, anchors, typed temporals are all lost.
    """
    merged = {}
    for comp in state:
        fmt = comp["format"]
        text = comp["text"]
        filename = comp["filename"]

        if fmt == "json":
            try:
                merged[filename] = {"format": fmt, "data": json.loads(text)}
            except json.JSONDecodeError:
                merged[filename] = {"format": fmt, "data": text}
        elif fmt == "yaml":
            try:
                # Handle multi-document YAML (e.g., frontmatter with ---)
                docs = list(yaml.safe_load_all(text))
                parsed = docs[0] if docs else {}
                merged[filename] = {"format": fmt, "data": parsed}
            except Exception:
                merged[filename] = {"format": fmt, "data": text}
        elif fmt == "toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    import tomlkit
                    parsed = dict(tomlkit.loads(text))
                    merged[filename] = {"format": fmt, "data": _strip_tomlkit(parsed)}
                    continue
            parsed = tomllib.loads(text)
            merged[filename] = {"format": fmt, "data": parsed}
        else:
            merged[filename] = {"format": fmt, "data": text}

    return json.dumps(merged, indent=2, default=str).encode("utf-8")


def _strip_tomlkit(obj):
    """Recursively convert tomlkit objects to plain Python types."""
    if isinstance(obj, dict):
        return {k: _strip_tomlkit(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_tomlkit(v) for v in obj]
    if hasattr(obj, "unwrap"):
        return obj.unwrap()
    return obj


def _deserialize_json_mega(data: bytes) -> list[dict]:
    """Deserialize JSON mega and re-emit each file as text.

    YAML/TOML files are re-emitted using standard (lossy) libraries.
    """
    merged = json.loads(data.decode("utf-8"))
    result = []

    for filename, entry in merged.items():
        fmt = entry["format"]
        obj = entry["data"]
        name = filename.rsplit(".", 1)[0] if "." in filename else filename

        if fmt == "json":
            text = json.dumps(obj, indent=2, default=str)
        elif fmt == "yaml":
            text = yaml.dump(obj, default_flow_style=False, sort_keys=False)
        elif fmt == "toml":
            try:
                import tomlkit
                text = tomlkit.dumps(obj)
            except Exception:
                text = str(obj)
        else:
            text = str(obj)

        result.append({
            "name": name,
            "filename": filename,
            "format": fmt,
            "text": text,
        })

    return result


# --- Pickle: lossy (same as JSON mega but pickled) ---

def _serialize_pickle(state: list[dict]) -> bytes:
    """Parse all files to Python objects and pickle.

    Same loss profile as JSON mega — comments destroyed during parsing.
    """
    parsed = {}
    for comp in state:
        fmt = comp["format"]
        text = comp["text"]
        filename = comp["filename"]

        if fmt == "json":
            try:
                parsed[filename] = {"format": fmt, "data": json.loads(text)}
            except json.JSONDecodeError:
                parsed[filename] = {"format": fmt, "data": text}
        elif fmt == "yaml":
            try:
                # Handle multi-document YAML (e.g., frontmatter with ---)
                docs = list(yaml.safe_load_all(text))
                parsed[filename] = {
                    "format": fmt, "data": docs[0] if docs else {}
                }
            except Exception:
                parsed[filename] = {"format": fmt, "data": text}
        elif fmt == "toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    import tomlkit
                    obj = dict(tomlkit.loads(text))
                    parsed[filename] = {
                        "format": fmt, "data": _strip_tomlkit(obj)
                    }
                    continue
            parsed[filename] = {"format": fmt, "data": tomllib.loads(text)}
        else:
            parsed[filename] = {"format": fmt, "data": text}

    return pickle.dumps(parsed)


def _deserialize_pickle(data: bytes) -> list[dict]:
    """Unpickle and re-emit files as text (lossy for YAML/TOML)."""
    parsed = pickle.loads(data)  # noqa: S301
    result = []

    for filename, entry in parsed.items():
        fmt = entry["format"]
        obj = entry["data"]
        name = filename.rsplit(".", 1)[0] if "." in filename else filename

        if fmt == "json":
            text = json.dumps(obj, indent=2, default=str)
        elif fmt == "yaml":
            text = yaml.dump(obj, default_flow_style=False, sort_keys=False)
        elif fmt == "toml":
            try:
                import tomlkit
                text = tomlkit.dumps(obj)
            except Exception:
                text = str(obj)
        else:
            text = str(obj)

        result.append({
            "name": name,
            "filename": filename,
            "format": fmt,
            "text": text,
        })

    return result


# --- tar.gz: text-preserving (but modification uses lossy parser) ---

def _serialize_tar_gz(state: list[dict]) -> bytes:
    """Tar.gz all files as raw text."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for comp in state:
            data = comp["text"].encode("utf-8")
            info = tarfile.TarInfo(name=comp["filename"])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _deserialize_tar_gz(data: bytes) -> list[dict]:
    """Extract tar.gz to text files (comments preserved)."""
    result = []
    buf = io.BytesIO(data)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is None:
                continue
            text = f.read().decode("utf-8")
            filename = member.name
            name = filename.rsplit(".", 1)[0] if "." in filename else filename
            ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            fmt_map = {"json": "json", "yaml": "yaml", "yml": "yaml",
                       "toml": "toml", "xml": "xml"}
            fmt = fmt_map.get(ext, "json")
            result.append({
                "name": name,
                "filename": filename,
                "format": fmt,
                "text": text,
            })
    return result


# ===========================================================================
# Session simulation — modify + add
# ===========================================================================


def modify_one_config(state: list[dict], session_k: int) -> list[dict]:
    """Modify one config file to simulate agent activity.

    Round-robins through modifiable files and changes a value.
    Does NOT add comment annotations — purely modifies data values,
    which is the realistic agent behavior.

    Args:
        state: Current state (list of components).
        session_k: Session number (1-indexed).

    Returns:
        New state with one file modified (deep copy).
    """
    result = copy.deepcopy(state)

    # Find modifiable files (YAML/TOML — have metadata at stake)
    modifiable = [
        i for i, c in enumerate(result)
        if c["format"] in ("yaml", "toml")
    ]
    if not modifiable:
        return result

    # Round-robin through modifiable files
    idx = modifiable[(session_k - 1) % len(modifiable)]
    comp = result[idx]

    # Value-only modification: increment a numeric value
    # This works regardless of whether text went through lossy round-trip
    text = comp["text"]
    if comp["format"] == "yaml":
        # Try common YAML numeric patterns
        for pattern, replacement_fn in [
            (r"(num_train_epochs:\s*)(\d+)",
             lambda m: m.group(1) + str(int(m.group(2)) + session_k)),
            (r"(replicas:\s*)(\d+)",
             lambda m: m.group(1) + str(int(m.group(2)) + 1)),
            (r"(seed:\s*)(\d+)",
             lambda m: m.group(1) + str(int(m.group(2)) + 1)),
        ]:
            new_text = re.sub(pattern, replacement_fn, text, count=1)
            if new_text != text:
                comp["text"] = new_text
                break
        else:
            # Fallback: append a benign YAML key (no comment)
            comp["text"] = text.rstrip("\n") + f"\nmodified_session: {session_k}\n"
    elif comp["format"] == "toml":
        for pattern, replacement_fn in [
            (r"(max_num_seqs\s*=\s*)(\d+)",
             lambda m: m.group(1) + str(int(m.group(2)) + 1)),
            (r"(port\s*=\s*)(\d+)",
             lambda m: m.group(1) + str(int(m.group(2)) + 1)),
        ]:
            new_text = re.sub(pattern, replacement_fn, text, count=1)
            if new_text != text:
                comp["text"] = new_text
                break
        else:
            comp["text"] = text.rstrip("\n") + f"\nmodified_session = {session_k}\n"

    return result


def add_result_file(state: list[dict], session_k: int) -> list[dict]:
    """Add a YAML result file with comments (simulates eval output).

    Args:
        state: Current state.
        session_k: Session number.

    Returns:
        New state with one file added.
    """
    result = copy.deepcopy(state)
    result.append({
        "name": f"eval_session_{session_k}",
        "filename": f"eval_session_{session_k}.yaml",
        "format": "yaml",
        "text": (
            f"# Evaluation results from session {session_k}\n"
            f"# Timestamp: 2026-01-{20 + session_k:02d}\n"
            f"session: {session_k}\n"
            f"metrics:\n"
            f"  # Loss measured on held-out validation set\n"
            f"  eval_loss: {0.9 - session_k * 0.01:.3f}\n"
            f"  accuracy: {0.45 + session_k * 0.005:.3f}\n"
            f"  # Perplexity tracks overall quality\n"
            f"  perplexity: {12.5 - session_k * 0.2:.1f}\n"
        ),
    })
    return result


# ===========================================================================
# Degradation loop
# ===========================================================================


def run_degradation_loop(
    complexity: str,
    state_format: str,
    max_k: int,
) -> list[dict]:
    """Run the degradation simulation for K session boundaries.

    Args:
        complexity: State complexity ("small", "medium", "large").
        state_format: One of STATE_FORMATS.
        max_k: Maximum number of session boundaries.

    Returns:
        List of dicts, one per K (including K=0 for initial state).
        Each has: k, total_comments, original_comments_surviving,
                  state_size_bytes, n_files.
    """
    state = build_initial_state(complexity)
    initial_meta = count_all_metadata(state)
    initial_comments = initial_meta["total_comments"]

    # Track which files are "original" (for tracking original comment survival)
    original_filenames = {c["filename"] for c in state}

    records = [{
        "k": 0,
        "total_comments": initial_meta["total_comments"],
        "original_comments_surviving": initial_comments,
        "state_size_bytes": len(serialize_state(state, state_format)),
        "n_files": len(state),
        "grand_total_metadata": initial_meta["grand_total"],
    }]

    for k in range(1, max_k + 1):
        # 1. Serialize current state
        serialized = serialize_state(state, state_format)

        # 2. Deserialize (this is where loss happens for JSON/Pickle)
        state = deserialize_state(serialized, state_format)

        # === MEASUREMENT POINT: right after the boundary ===
        # Count what survived the serialize->deserialize round-trip,
        # BEFORE any agent modifications that would add new metadata.
        orig_comments = 0
        for comp in state:
            if comp["filename"] in original_filenames:
                m = count_metadata_constructs(comp["text"], comp["format"])
                orig_comments += m["comments"]

        # 3. Modify one config (agent work — may add annotations)
        state = modify_one_config(state, k)

        # 4. Add a result file (agent work — adds new YAML with comments)
        state = add_result_file(state, k)

        # 5. Count total metadata (including new additions)
        all_meta = count_all_metadata(state)

        # Measure serialized size
        serialized_now = serialize_state(state, state_format)

        records.append({
            "k": k,
            "total_comments": all_meta["total_comments"],
            "original_comments_surviving": orig_comments,
            "state_size_bytes": len(serialized_now),
            "n_files": len(state),
            "grand_total_metadata": all_meta["grand_total"],
        })

    return records


# ===========================================================================
# Full experiment
# ===========================================================================


def run_experiment(output_dir: Path | str | None = None) -> dict:
    """Run the full EXP-013 experiment."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_013")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-013: Agent Workflow State Persistence Across Sessions")
    print("=" * 70)

    max_k = max(SESSION_BOUNDARIES)

    degradation = {}
    summary = {}

    for complexity in ["medium"]:  # Primary analysis on medium (8 files)
        print(f"\n--- Complexity: {complexity} "
              f"({STATE_COMPLEXITIES[complexity]} files) ---")

        for sf in STATE_FORMATS:
            key = f"{complexity}_{sf}"
            print(f"\n  Format: {sf}")

            records = run_degradation_loop(complexity, sf, max_k)
            degradation[key] = records

            initial = records[0]["original_comments_surviving"]
            print(f"    K=0: {initial} comments (initial)")
            for r in records[1:]:
                frac = (r["original_comments_surviving"] / initial
                        if initial > 0 else 0)
                print(f"    K={r['k']:2d}: "
                      f"{r['original_comments_surviving']:3d}/{initial} "
                      f"comments surviving ({frac:.1%})  "
                      f"size={r['state_size_bytes']:,}B  "
                      f"files={r['n_files']}")

            # Summary: survival at max_k
            final = records[-1]
            final_frac = (final["original_comments_surviving"] / initial
                          if initial > 0 else 0)
            summary[key] = {
                "state_format": sf,
                "complexity": complexity,
                "initial_comments": initial,
                "final_comments": final["original_comments_surviving"],
                "surviving_fraction": round(final_frac, 4),
                "max_k": max_k,
            }

    # Also run small and large for completeness (shorter loops)
    for complexity in ["small", "large"]:
        print(f"\n--- Complexity: {complexity} "
              f"({STATE_COMPLEXITIES[complexity]} files) ---")
        for sf in STATE_FORMATS:
            key = f"{complexity}_{sf}"
            records = run_degradation_loop(complexity, sf, min(max_k, 10))
            degradation[key] = records
            initial = records[0]["original_comments_surviving"]
            final = records[-1]
            final_frac = (final["original_comments_surviving"] / initial
                          if initial > 0 else 0)
            summary[key] = {
                "state_format": sf,
                "complexity": complexity,
                "initial_comments": initial,
                "final_comments": final["original_comments_surviving"],
                "surviving_fraction": round(final_frac, 4),
                "max_k": records[-1]["k"],
            }
            print(f"  {sf:12s}: {initial} -> "
                  f"{final['original_comments_surviving']} "
                  f"({final_frac:.1%}) over K={records[-1]['k']}")

    # ----- Write outputs -----
    output = {
        "experiment": "EXP-013",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "degradation": degradation,
        "summary": summary,
    }

    json_path = output_dir / "exp_013_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults: {json_path}")

    # CSV: degradation curves
    csv_path = output_dir / "degradation_curves.csv"
    rows = []
    for key, records in degradation.items():
        parts = key.split("_", 1)
        complexity = parts[0]
        sf = parts[1]
        for r in records:
            initial = records[0]["original_comments_surviving"]
            frac = (r["original_comments_surviving"] / initial
                    if initial > 0 else 0)
            rows.append({
                "complexity": complexity,
                "state_format": sf,
                "k": r["k"],
                "total_comments": r["total_comments"],
                "original_comments_surviving": r["original_comments_surviving"],
                "surviving_fraction": round(frac, 4),
                "state_size_bytes": r["state_size_bytes"],
                "n_files": r["n_files"],
            })
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Degradation CSV: {csv_path}")

    # CSV: summary
    summary_csv = output_dir / "summary.csv"
    if summary:
        fieldnames = list(next(iter(summary.values())).keys())
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in summary.values():
                writer.writerow(entry)
    print(f"Summary CSV: {summary_csv}")

    print(f"\n{'=' * 70}")
    print("EXP-013 COMPLETE")
    print("=" * 70)

    return output


def main():
    run_experiment()


if __name__ == "__main__":
    main()
