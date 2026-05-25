"""
EXP-007: Cross-Framework Configuration Migration

Measures CDXF's ability to serve as a lossless hub format for cross-framework
config migration. Compares direct format-to-format conversion (metadata lost)
vs CDXF hub conversion (metadata preserved).

Usage:
    python benchmarks/src/run_exp007.py
"""

import csv
import json
import io
import re
import statistics
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import ruamel.yaml
    import tomlkit
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Run: pip install ruamel.yaml tomlkit")
    sys.exit(1)

try:
    import cdxf
    from cdxf.bridges.json_bridge import from_json, to_json
    from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
    from cdxf.bridges.toml_bridge import from_toml, to_toml
    from cdxf.bridges.xml_bridge import from_xml, to_xml
    from cdxf.codec import encode, decode
    from cdxf.model import Comment, Anchor, Scalar, ScalarType, Stream
except ImportError:
    print("ERROR: cdxf not installed. Run: pip install -e .")
    sys.exit(1)


RESULTS_DIR = Path("benchmarks/results/exp_007")

SUPPORTED_FORMATS = {"json", "yaml", "xml", "toml"}

# Bridge dispatch tables
_FROM_BRIDGE = {
    "json": from_json,
    "yaml": from_yaml,
    "xml": from_xml,
    "toml": from_toml,
}

_TO_BRIDGE = {
    "json": lambda stream: to_json(stream, indent=2),
    "yaml": to_yaml,
    "xml": to_xml,
    "toml": to_toml,
}


# ---------------------------------------------------------------------------
# Migration scenario corpus
# ---------------------------------------------------------------------------

def build_migration_scenarios() -> list[dict]:
    """Build 8 realistic cross-framework migration scenarios.

    Each scenario has a source config with comments/annotations and a
    target format. The configs are realistic representations of actual
    ML/DevOps configuration files.
    """
    scenarios = []

    # 1. PyTorch Lightning YAML -> HuggingFace JSON
    scenarios.append({
        "name": "pytorch_lightning_to_hf",
        "source_format": "yaml",
        "target_format": "json",
        "motivation": "Framework migration",
        "source_text": (
            "# PyTorch Lightning Trainer configuration\n"
            "# Tuned for BERT fine-tuning on 4x A100\n"
            "trainer:\n"
            "  # Max epochs determined by early stopping analysis\n"
            "  max_epochs: 20\n"
            "  accelerator: gpu\n"
            "  devices: 4\n"
            "  strategy: ddp\n"
            "  precision: bf16-mixed\n"
            "  # Gradient accumulation to simulate larger batch size\n"
            "  accumulate_grad_batches: 4\n"
            "  gradient_clip_val: 1.0\n"
            "  # Logging every 50 steps balances overhead vs visibility\n"
            "  log_every_n_steps: 50\n"
            "\n"
            "model:\n"
            "  name: bert-base-uncased\n"
            "  # Learning rate from LR range test (see experiment notes)\n"
            "  learning_rate: 0.00003\n"
            "  weight_decay: 0.01\n"
            "  warmup_steps: 500\n"
            "  # Dropout increased from 0.1 after overfitting on dev set\n"
            "  hidden_dropout_prob: 0.2\n"
            "\n"
            "data:\n"
            "  dataset: glue/sst2\n"
            "  max_seq_length: 128\n"
            "  batch_size: 32\n"
            "  num_workers: 8\n"
        ),
    })

    # 2. HuggingFace JSON -> TOML (Rust trainer)
    scenarios.append({
        "name": "hf_json_to_toml",
        "source_format": "json",
        "target_format": "toml",
        "motivation": "Language migration",
        "source_text": json.dumps({
            "model_name_or_path": "meta-llama/Llama-2-7b-hf",
            "output_dir": "./results/llama2-sft",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "bf16": True,
            "tf32": True,
            "max_grad_norm": 1.0,
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": 500,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "report_to": ["wandb"],
            "seed": 42,
        }, indent=2),
    })

    # 3. Kubernetes YAML -> Terraform JSON
    scenarios.append({
        "name": "k8s_to_terraform",
        "source_format": "yaml",
        "target_format": "json",
        "motivation": "Infrastructure migration",
        "source_text": (
            "# Kubernetes deployment for model serving\n"
            "# Migrating to Terraform for multi-cloud support\n"
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: model-server\n"
            "  # Labels follow company naming convention\n"
            "  labels:\n"
            "    app: model-server\n"
            "    team: ml-platform\n"
            "    env: production\n"
            "spec:\n"
            "  replicas: 3\n"
            "  selector:\n"
            "    matchLabels:\n"
            "      app: model-server\n"
            "  template:\n"
            "    metadata:\n"
            "      labels:\n"
            "        app: model-server\n"
            "    spec:\n"
            "      containers:\n"
            "        - name: inference\n"
            "          image: registry.example.com/model-server:v2.3.1\n"
            "          # Resource limits based on load testing (see PERF-2024-Q3)\n"
            "          resources:\n"
            "            requests:\n"
            "              cpu: '2'\n"
            "              memory: 8Gi\n"
            "            limits:\n"
            "              cpu: '4'\n"
            "              memory: 16Gi\n"
            "          # Health check tuned for model loading time (~45s)\n"
            "          readinessProbe:\n"
            "            httpGet:\n"
            "              path: /health\n"
            "              port: 8080\n"
            "            initialDelaySeconds: 60\n"
            "            periodSeconds: 10\n"
            "          ports:\n"
            "            - containerPort: 8080\n"
        ),
    })

    # 4. Hydra YAML (with overrides) -> JSON (API config)
    scenarios.append({
        "name": "hydra_to_json",
        "source_format": "yaml",
        "target_format": "json",
        "motivation": "Deployment",
        "source_text": (
            "# Hydra configuration for training pipeline\n"
            "# Base config - override with +experiment=<name>\n"
            "defaults:\n"
            "  - _self_\n"
            "\n"
            "# Model architecture\n"
            "model:\n"
            "  pretrained_model_name_or_path: meta-llama/Llama-2-7b-hf\n"
            "  # Use 4-bit quantization to fit on single GPU\n"
            "  load_in_4bit: true\n"
            "  bnb_4bit_compute_dtype: bfloat16\n"
            "\n"
            "# Training hyperparameters\n"
            "# LR schedule: cosine with warmup (validated in EXP-023)\n"
            "training:\n"
            "  lr: 2.0e-5\n"
            "  weight_decay: 0.01\n"
            "  num_epochs: 3\n"
            "  warmup_ratio: 0.03\n"
            "  batch_size: 4\n"
            "  gradient_accumulation: 8\n"
            "  max_grad_norm: 1.0\n"
            "\n"
            "# Data configuration\n"
            "data:\n"
            "  dataset_name: tatsu-lab/alpaca\n"
            "  max_length: 2048\n"
            "  # Packing increases throughput ~30% (measured)\n"
            "  packing: true\n"
            "\n"
            "# Logging\n"
            "wandb:\n"
            "  project: llama2-sft\n"
            "  # Tags help filter runs in dashboard\n"
            "  tags:\n"
            "    - sft\n"
            "    - llama2\n"
            "    - qlora\n"
        ),
    })

    # 5. MLflow YAML -> W&B JSON
    scenarios.append({
        "name": "mlflow_to_wandb",
        "source_format": "yaml",
        "target_format": "json",
        "motivation": "Experiment tracker migration",
        "source_text": (
            "# MLflow experiment configuration\n"
            "# Migrating from MLflow to Weights & Biases\n"
            "experiment:\n"
            "  name: llama2-sft-v3\n"
            "  # Artifact location on S3 - needs updating for W&B\n"
            "  artifact_location: s3://ml-artifacts/llama2-sft-v3\n"
            "\n"
            "tracking:\n"
            "  # MLflow tracking server URI\n"
            "  tracking_uri: http://mlflow.internal:5000\n"
            "  # Auto-log PyTorch metrics\n"
            "  autolog:\n"
            "    framework: pytorch\n"
            "    log_models: true\n"
            "    log_input_examples: true\n"
            "\n"
            "# Run parameters - these map to W&B config\n"
            "params:\n"
            "  model: meta-llama/Llama-2-7b-hf\n"
            "  lr: 0.00002\n"
            "  epochs: 3\n"
            "  batch_size: 32\n"
            "  optimizer: adamw\n"
            "  scheduler: cosine\n"
            "  warmup_steps: 100\n"
            "  # Seed for reproducibility\n"
            "  seed: 42\n"
            "\n"
            "# Metrics to track\n"
            "metrics:\n"
            "  - train_loss\n"
            "  - eval_loss\n"
            "  - eval_accuracy\n"
            "  - learning_rate\n"
        ),
    })

    # 6. Docker Compose YAML -> JSON (CI/CD)
    scenarios.append({
        "name": "docker_compose_to_json",
        "source_format": "yaml",
        "target_format": "json",
        "motivation": "DevOps pipeline",
        "source_text": (
            "# Docker Compose for ML training infrastructure\n"
            "# Used in CI/CD pipeline - converting to JSON for GitHub Actions\n"
            "version: '3.8'\n"
            "\n"
            "services:\n"
            "  trainer:\n"
            "    # Custom training image with CUDA support\n"
            "    build:\n"
            "      context: .\n"
            "      dockerfile: Dockerfile.train\n"
            "    volumes:\n"
            "      - ./data:/app/data\n"
            "      - ./outputs:/app/outputs\n"
            "    environment:\n"
            "      - WANDB_API_KEY\n"
            "      - HF_TOKEN\n"
            "      # GPU memory fraction to leave room for monitoring\n"
            "      - CUDA_VISIBLE_DEVICES=0,1\n"
            "\n"
            "  tensorboard:\n"
            "    image: tensorflow/tensorflow:latest\n"
            "    ports:\n"
            "      # TensorBoard UI accessible on host port 6006\n"
            "      - '6006:6006'\n"
            "    volumes:\n"
            "      - ./outputs/logs:/logs\n"
            "    command: tensorboard --logdir=/logs --bind_all\n"
            "\n"
            "  # Redis for caching preprocessed data\n"
            "  cache:\n"
            "    image: redis:7-alpine\n"
            "    ports:\n"
            "      - '6379:6379'\n"
        ),
    })

    # 7. pyproject.toml -> JSON (packaging)
    scenarios.append({
        "name": "pyproject_to_json",
        "source_format": "toml",
        "target_format": "json",
        "motivation": "Python packaging",
        "source_text": (
            "# Project metadata for ML training framework\n"
            "# PEP 621 compliant\n"
            "\n"
            "[project]\n"
            'name = "ml-trainer"\n'
            'version = "0.4.2"\n'
            'description = "Distributed fine-tuning framework for LLMs"\n'
            'readme = "README.md"\n'
            'license = {text = "Apache-2.0"}\n'
            'requires-python = ">=3.10"\n'
            "authors = [\n"
            '    {name = "ML Platform Team", email = "ml-platform@example.com"},\n'
            "]\n"
            "\n"
            "# Core dependencies - pinned for reproducibility\n"
            "dependencies = [\n"
            '    "torch>=2.1.0",\n'
            '    "transformers>=4.36.0",\n'
            '    "peft>=0.7.0",\n'
            '    "datasets>=2.15.0",\n'
            '    "accelerate>=0.25.0",\n'
            '    "wandb>=0.16.0",\n'
            "]\n"
            "\n"
            "# Optional GPU dependencies\n"
            "[project.optional-dependencies]\n"
            "gpu = [\n"
            '    "bitsandbytes>=0.41.0",\n'
            '    "flash-attn>=2.3.0",\n'
            "]\n"
            "dev = [\n"
            '    "pytest>=7.0",\n'
            '    "ruff>=0.1.0",\n'
            "]\n"
            "\n"
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n'
            "\n"
            "# Ruff linter configuration\n"
            "[tool.ruff]\n"
            "line-length = 100\n"
            'target-version = "py310"\n'
        ),
    })

    # 8. ONNX XML metadata -> JSON (TensorRT config)
    scenarios.append({
        "name": "onnx_xml_to_json",
        "source_format": "xml",
        "target_format": "json",
        "motivation": "Model deployment",
        "source_text": (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<!-- ONNX model metadata for deployment to TensorRT -->\n"
            "<!-- Generated by export pipeline, do not edit manually -->\n"
            '<model xmlns="http://onnx.ai/schema"\n'
            '       xmlns:custom="http://example.com/ml-metadata">\n'
            "    <metadata>\n"
            '        <entry key="model_name">llama2-7b-sft</entry>\n'
            '        <entry key="model_version">2.3.1</entry>\n'
            '        <entry key="framework">PyTorch</entry>\n'
            '        <entry key="export_date">2026-01-15</entry>\n'
            "        <!-- Opset version must match TensorRT support matrix -->\n"
            '        <entry key="opset_version">17</entry>\n'
            "    </metadata>\n"
            "    <inputs>\n"
            "        <!-- Input shape: [batch, seq_len] -->\n"
            '        <input name="input_ids" dtype="int64" shape="dynamic,2048"/>\n'
            '        <input name="attention_mask" dtype="int64" shape="dynamic,2048"/>\n'
            "    </inputs>\n"
            "    <outputs>\n"
            '        <output name="logits" dtype="float32" shape="dynamic,2048,32000"/>\n'
            "    </outputs>\n"
            "    <!-- Quantization applied post-export -->\n"
            '    <custom:optimization>\n'
            '        <custom:quantization method="int8" calibration="entropy"/>\n'
            '        <custom:pruning sparsity="0.0"/>\n'
            "    </custom:optimization>\n"
            "</model>\n"
        ),
    })

    return scenarios


# ---------------------------------------------------------------------------
# Metadata counting
# ---------------------------------------------------------------------------

def count_metadata(text: str, fmt: str) -> dict:
    """Count format-specific metadata constructs in a text document.

    Returns dict with keys: comments, anchors, temporal_values,
    processing_instructions, merge_keys, multi_doc_markers, total.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")

    counts = {
        "comments": 0,
        "anchors": 0,
        "temporal_values": 0,
        "processing_instructions": 0,
        "merge_keys": 0,
        "multi_doc_markers": 0,
    }

    if fmt == "yaml":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                counts["comments"] += 1

        # Anchors (&name)
        counts["anchors"] = len(re.findall(r'&\w+', text))
        # Merge keys (<<:)
        counts["merge_keys"] = len(re.findall(r'<<\s*:', text))
        # Multi-doc markers (---)
        counts["multi_doc_markers"] = len(
            re.findall(r'^---\s*$', text, re.MULTILINE)
        )

    elif fmt == "toml":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                counts["comments"] += 1

        # TOML temporal: datetime values
        counts["temporal_values"] = len(re.findall(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', text
        ))
        # Date-only values after =
        counts["temporal_values"] += len(re.findall(
            r'=\s*\d{4}-\d{2}-\d{2}\s*$', text, re.MULTILINE
        ))

    elif fmt == "xml":
        counts["comments"] = len(re.findall(r'<!--.*?-->', text, re.DOTALL))
        # Processing instructions (excluding xml declaration)
        pis = re.findall(r'<\?(\w+)', text)
        counts["processing_instructions"] = sum(
            1 for pi in pis if pi.lower() != "xml"
        )

    elif fmt == "json":
        # JSON has no comment or annotation syntax
        pass

    counts["total"] = sum(counts.values())
    return counts


def _count_cdxf_metadata(stream: Stream) -> dict:
    """Count metadata constructs in a CDXF model tree."""
    counts = {
        "comments": 0,
        "anchors": 0,
        "temporal_values": 0,
        "processing_instructions": 0,
    }

    def walk(node):
        from cdxf.model import (
            Comment as CComment, Map, Sequence, Scalar as CScalar,
            ScalarType as CST, Document, Element,
            ProcessingInstruction, Anchor as CAnchor, Alias as CAlias,
        )

        if isinstance(node, CComment):
            counts["comments"] += 1
        elif isinstance(node, ProcessingInstruction):
            counts["processing_instructions"] += 1
        elif isinstance(node, CAlias):
            pass  # alias references, not metadata
        elif isinstance(node, CScalar):
            if node.anchor is not None:
                counts["anchors"] += 1
            if node.scalar_type in (
                CST.TIMESTAMP_OFFSET, CST.TIMESTAMP_LOCAL,
                CST.DATE, CST.TIME,
            ):
                counts["temporal_values"] += 1
        elif isinstance(node, Map):
            if node.anchor is not None:
                counts["anchors"] += 1
            for entry in node.entries:
                if isinstance(entry, CComment):
                    walk(entry)
                elif isinstance(entry, tuple) and len(entry) == 2:
                    walk(entry[0])  # key
                    walk(entry[1])  # value
        elif isinstance(node, Sequence):
            if node.anchor is not None:
                counts["anchors"] += 1
            for item in node.items:
                walk(item)
        elif isinstance(node, Element):
            for child in node.children:
                walk(child)
        elif isinstance(node, Document):
            if node.root is not None:
                walk(node.root)
            for child in node.preamble:
                walk(child)
            for child in node.postamble:
                walk(child)

    if isinstance(stream, Stream):
        for doc in stream.documents:
            walk(doc)

    counts["total"] = sum(counts.values())
    return counts


# ---------------------------------------------------------------------------
# Direct conversion (baseline - loses metadata)
# ---------------------------------------------------------------------------

def _parse_to_native(text: str, fmt: str):
    """Parse text to native Python data structure (dict/list).
    Format-specific metadata is lost.
    """
    if fmt == "json":
        return json.loads(text)
    elif fmt == "yaml":
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = False
        data = yaml.load(text)
        return _ruamel_to_plain(data)
    elif fmt == "toml":
        data = tomlkit.loads(text)
        return _tomlkit_to_plain(data)
    elif fmt == "xml":
        return _xml_to_dict(text)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _ruamel_to_plain(obj):
    """Recursively convert ruamel.yaml types to plain Python."""
    if isinstance(obj, dict):
        return {str(k): _ruamel_to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ruamel_to_plain(v) for v in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        return float(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif obj is None:
        return None
    else:
        return str(obj)


def _tomlkit_to_plain(obj):
    """Recursively convert tomlkit types to plain Python."""
    if isinstance(obj, dict):
        return {str(k): _tomlkit_to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_tomlkit_to_plain(v) for v in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        return float(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif obj is None:
        return None
    else:
        return str(obj)


def _xml_to_dict(text: str) -> dict:
    """Parse XML to a nested Python dict (lossy - drops comments, PIs,
    namespaces are simplified)."""
    root = ET.fromstring(text)

    def elem_to_dict(elem):
        result = {}
        tag = re.sub(r'\{[^}]+\}', '', elem.tag)

        if elem.attrib:
            result["@attributes"] = {
                re.sub(r'\{[^}]+\}', '', k): v
                for k, v in elem.attrib.items()
            }

        children = list(elem)
        if children:
            child_dict = {}
            for child in children:
                child_tag = re.sub(r'\{[^}]+\}', '', child.tag)
                child_data = elem_to_dict(child)
                if child_tag in child_dict:
                    existing = child_dict[child_tag]
                    if not isinstance(existing, list):
                        child_dict[child_tag] = [existing]
                    child_dict[child_tag].append(child_data[child_tag])
                else:
                    child_dict.update(child_data)
            result.update(child_dict)
        elif elem.text and elem.text.strip():
            return {tag: elem.text.strip()}

        return {tag: result if result else None}

    return elem_to_dict(root)


def _emit_as_format(data, fmt: str) -> str:
    """Emit a plain Python data structure as the target format."""
    if fmt == "json":
        return json.dumps(data, indent=2, default=str)
    elif fmt == "yaml":
        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = False
        buf = io.StringIO()
        yaml.dump(data, buf)
        return buf.getvalue()
    elif fmt == "toml":
        return tomlkit.dumps(data)
    elif fmt == "xml":
        return _dict_to_xml(data)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _dict_to_xml(data: dict, root_tag: str = "root") -> str:
    """Convert a plain Python dict to XML string (simple)."""
    def build_elem(tag, value):
        elem = ET.Element(tag)
        if isinstance(value, dict):
            value = dict(value)  # copy to avoid mutation
            attrs = value.pop("@attributes", {})
            for ak, av in attrs.items():
                elem.set(ak, str(av))
            for k, v in value.items():
                if isinstance(v, list):
                    for item in v:
                        child = build_elem(k, item)
                        elem.append(child)
                else:
                    child = build_elem(k, v)
                    elem.append(child)
        elif isinstance(value, list):
            for item in value:
                child = build_elem("item", item)
                elem.append(child)
        elif value is not None:
            elem.text = str(value)
        return elem

    if isinstance(data, dict) and len(data) == 1:
        tag = list(data.keys())[0]
        root = build_elem(tag, data[tag])
    else:
        root = build_elem(root_tag, data)

    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    )


def direct_convert(source_text: str, source_format: str,
                   target_format: str) -> dict:
    """Convert source text directly to target format via native Python.

    This is the baseline: parse to Python dict, emit as target.
    All format-specific metadata (comments, anchors, etc.) is lost.
    """
    result = {
        "success": False,
        "output_text": "",
        "metadata_survived": {"comments": 0, "anchors": 0,
                              "temporal_values": 0, "total": 0},
        "conversion_time_ns": 0,
        "error": None,
    }

    try:
        t_start = time.perf_counter_ns()
        native = _parse_to_native(source_text, source_format)
        output = _emit_as_format(native, target_format)
        t_end = time.perf_counter_ns()

        result["success"] = True
        result["output_text"] = output
        result["conversion_time_ns"] = t_end - t_start
        result["metadata_survived"] = count_metadata(output, target_format)

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# CDXF hub conversion (preserves metadata)
# ---------------------------------------------------------------------------

def cdxf_hub_convert(source_text: str, source_format: str,
                     target_format: str) -> dict:
    """Convert source text to target format via CDXF hub.

    Path: source -> CDXF bridge -> encode -> decode -> target bridge.
    Metadata is preserved in the CDXF intermediate representation.
    """
    result = {
        "success": False,
        "output_text": "",
        "cdxf_metadata": {"comments": 0, "anchors": 0,
                          "temporal_values": 0, "total": 0},
        "cdxf_size_bytes": 0,
        "conversion_time_ns": 0,
        "error": None,
    }

    try:
        t_start = time.perf_counter_ns()

        # Step 1: Parse source via CDXF bridge
        from_bridge = _FROM_BRIDGE[source_format]
        stream = from_bridge(source_text)

        # Step 2: Encode to CDXF binary
        cdxf_bytes = encode(stream)
        result["cdxf_size_bytes"] = len(cdxf_bytes)

        # Step 3: Decode back to CDXF model
        decoded_stream = decode(cdxf_bytes)

        # Step 4: Count metadata in CDXF intermediate
        result["cdxf_metadata"] = _count_cdxf_metadata(decoded_stream)

        # Step 5: Emit via target bridge
        to_bridge = _TO_BRIDGE[target_format]
        output = to_bridge(decoded_stream)

        t_end = time.perf_counter_ns()

        result["success"] = True
        result["output_text"] = output
        result["conversion_time_ns"] = t_end - t_start

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Migration measurement
# ---------------------------------------------------------------------------

def measure_migration(scenario: dict) -> dict:
    """Run both migration methods on a scenario and compare."""
    name = scenario["name"]
    src_fmt = scenario["source_format"]
    tgt_fmt = scenario["target_format"]
    src_text = scenario["source_text"]

    source_meta = count_metadata(src_text, src_fmt)
    direct_result = direct_convert(src_text, src_fmt, tgt_fmt)
    hub_result = cdxf_hub_convert(src_text, src_fmt, tgt_fmt)

    return {
        "name": name,
        "source_format": src_fmt,
        "target_format": tgt_fmt,
        "motivation": scenario["motivation"],
        "source_size_bytes": len(src_text.encode("utf-8")),
        "source_metadata": source_meta,
        "direct": direct_result,
        "cdxf_hub": hub_result,
    }


# ---------------------------------------------------------------------------
# Converter count analysis
# ---------------------------------------------------------------------------

def compute_converter_counts(n_formats: int) -> dict:
    """Compute converter implementations needed: direct vs CDXF hub.

    Direct: N * (N-1) pairwise converters.
    CDXF hub: 2N converters (one encode + one decode per format).
    """
    return {
        "n_formats": n_formats,
        "direct": n_formats * (n_formats - 1),
        "cdxf_hub": 2 * n_formats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run EXP-007: Cross-Framework Configuration Migration."""
    print("=" * 70)
    print("EXP-007: Cross-Framework Configuration Migration")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = build_migration_scenarios()
    print(f"\nScenarios: {len(scenarios)}")

    # Run all migrations
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}/8: {scenario['name']} ---")
        print(f"    {scenario['source_format']} -> {scenario['target_format']}"
              f" ({scenario['motivation']})")

        result = measure_migration(scenario)
        results.append(result)

        src_meta = result["source_metadata"]
        print(f"    Source metadata: {src_meta['total']} constructs "
              f"({src_meta['comments']} comments, "
              f"{src_meta['anchors']} anchors)")

        if result["direct"]["success"]:
            d_meta = result["direct"]["metadata_survived"]
            t_ms = result["direct"]["conversion_time_ns"] / 1e6
            print(f"    Direct:   OK survived={d_meta['total']} "
                  f"({t_ms:.2f} ms)")
        else:
            print(f"    Direct:   FAIL {result['direct']['error']}")

        if result["cdxf_hub"]["success"]:
            h_meta = result["cdxf_hub"]["cdxf_metadata"]
            h_size = result["cdxf_hub"]["cdxf_size_bytes"]
            t_ms = result["cdxf_hub"]["conversion_time_ns"] / 1e6
            print(f"    CDXF hub: OK preserved={h_meta['total']} "
                  f"(CDXF={h_size}B, {t_ms:.2f} ms)")
        else:
            print(f"    CDXF hub: FAIL {result['cdxf_hub']['error']}")

    # Write per-scenario CSV
    csv_path = RESULTS_DIR / "migration_results.csv"
    fieldnames = [
        "name", "source_format", "target_format", "motivation",
        "source_size_bytes",
        "source_comments", "source_anchors", "source_total_metadata",
        "direct_success", "direct_metadata_survived", "direct_time_ns",
        "cdxf_success", "cdxf_metadata_preserved", "cdxf_size_bytes",
        "cdxf_time_ns",
        "metadata_advantage",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            src_m = r["source_metadata"]
            d = r["direct"]
            h = r["cdxf_hub"]
            d_survived = d["metadata_survived"]["total"] if d["success"] else 0
            h_preserved = h["cdxf_metadata"]["total"] if h["success"] else 0
            writer.writerow({
                "name": r["name"],
                "source_format": r["source_format"],
                "target_format": r["target_format"],
                "motivation": r["motivation"],
                "source_size_bytes": r["source_size_bytes"],
                "source_comments": src_m["comments"],
                "source_anchors": src_m["anchors"],
                "source_total_metadata": src_m["total"],
                "direct_success": d["success"],
                "direct_metadata_survived": d_survived,
                "direct_time_ns": d["conversion_time_ns"],
                "cdxf_success": h["success"],
                "cdxf_metadata_preserved": h_preserved,
                "cdxf_size_bytes": h.get("cdxf_size_bytes", 0),
                "cdxf_time_ns": h["conversion_time_ns"],
                "metadata_advantage": h_preserved - d_survived,
            })
    print(f"\nPer-scenario results: {csv_path}")

    # Converter count scaling analysis
    print("\n" + "=" * 70)
    print("CONVERTER COUNT SCALING: Direct O(N^2) vs CDXF Hub O(N)")
    print("=" * 70)

    scaling = {}
    for n in range(2, 8):
        counts = compute_converter_counts(n)
        scaling[str(n)] = counts
        savings = counts["direct"] - counts["cdxf_hub"]
        pct = (savings / counts["direct"] * 100) if counts["direct"] > 0 else 0
        print(f"  N={n}: direct={counts['direct']:3d}  "
              f"hub={counts['cdxf_hub']:3d}  "
              f"savings={savings:3d} ({pct:.0f}%)")

    # Write scaling analysis
    scaling_path = RESULTS_DIR / "converter_scaling.json"
    with open(scaling_path, "w") as f:
        json.dump(scaling, f, indent=2)
    print(f"\nScaling analysis: {scaling_path}")

    # Summary
    print("\n" + "=" * 70)
    print("HEADLINE FINDINGS")
    print("=" * 70)

    n_direct_success = sum(1 for r in results if r["direct"]["success"])
    n_hub_success = sum(1 for r in results if r["cdxf_hub"]["success"])
    print(f"Direct conversion success: {n_direct_success}/{len(results)}")
    print(f"CDXF hub conversion success: {n_hub_success}/{len(results)}")

    total_source_meta = sum(
        r["source_metadata"]["total"] for r in results
    )
    total_direct_survived = sum(
        r["direct"]["metadata_survived"]["total"]
        for r in results if r["direct"]["success"]
    )
    total_hub_preserved = sum(
        r["cdxf_hub"]["cdxf_metadata"]["total"]
        for r in results if r["cdxf_hub"]["success"]
    )

    print(f"\nTotal source metadata constructs: {total_source_meta}")
    print(f"Direct conversion survived: {total_direct_survived} "
          f"({total_direct_survived / max(total_source_meta, 1):.1%})")
    print(f"CDXF hub preserved: {total_hub_preserved} "
          f"({total_hub_preserved / max(total_source_meta, 1):.1%})")

    c4 = compute_converter_counts(4)
    print(f"\nAt N=4 formats: {c4['direct']} direct converters vs "
          f"{c4['cdxf_hub']} CDXF hub converters "
          f"({c4['direct'] - c4['cdxf_hub']} saved)")

    # Write full summary JSON
    summary_path = RESULTS_DIR / "summary.json"
    summary = {
        "n_scenarios": len(results),
        "direct_success": n_direct_success,
        "hub_success": n_hub_success,
        "total_source_metadata": total_source_meta,
        "total_direct_survived": total_direct_survived,
        "total_hub_preserved": total_hub_preserved,
        "converter_scaling": scaling,
        "scenarios": [],
    }
    for r in results:
        summary["scenarios"].append({
            "name": r["name"],
            "source_format": r["source_format"],
            "target_format": r["target_format"],
            "motivation": r["motivation"],
            "source_metadata_total": r["source_metadata"]["total"],
            "direct_success": r["direct"]["success"],
            "direct_survived": (
                r["direct"]["metadata_survived"]["total"]
                if r["direct"]["success"] else None
            ),
            "hub_success": r["cdxf_hub"]["success"],
            "hub_preserved": (
                r["cdxf_hub"]["cdxf_metadata"]["total"]
                if r["cdxf_hub"]["success"] else None
            ),
            "cdxf_size_bytes": (
                r["cdxf_hub"]["cdxf_size_bytes"]
                if r["cdxf_hub"]["success"] else None
            ),
        })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
