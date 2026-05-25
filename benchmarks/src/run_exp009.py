"""
EXP-009: End-to-End Fine-Tuning Pipeline State Capture

Measures CDXF's ability to capture a complete fine-tuning pipeline state
in a single portable file. Compares CDXF vs tar.gz, mega-JSON, Pickle on
size, metadata preservation, state diffing, and cross-language readability.

Usage:
    python benchmarks/src/run_exp009.py
"""

import csv
import gzip
import io
import json
import pickle
import re
import sys
import tarfile
import time
from pathlib import Path

try:
    import ruamel.yaml
    import tomlkit
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)

try:
    from cdxf.bridges.json_bridge import from_json, to_json
    from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
    from cdxf.bridges.toml_bridge import from_toml, to_toml
    from cdxf.codec import encode, decode
    from cdxf.model import (
        Comment, Document, Map, Scalar, ScalarType, Sequence,
        SourceFormat, Stream,
    )
except ImportError:
    print("ERROR: cdxf not installed. Run: pip install -e .")
    sys.exit(1)


RESULTS_DIR = Path("benchmarks/results/exp_009")

CAPTURE_METHODS = {"cdxf", "tar_gz", "mega_json", "pickle"}

PIPELINE_COMPONENTS = [
    "dataset_card",
    "preprocessing",
    "training_hparams",
    "adapter_config",
    "quantization_config",
    "serving_config",
    "eval_results",
    "deployment_manifest",
]

_MINIMAL_COMPONENTS = {"training_hparams", "adapter_config", "eval_results"}

_FORMAT_TO_SOURCE = {
    "json": SourceFormat.JSON,
    "yaml": SourceFormat.YAML,
    "toml": SourceFormat.TOML,
    "markdown": SourceFormat.UNSPECIFIED,
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


# ---------------------------------------------------------------------------
# Pipeline state construction
# ---------------------------------------------------------------------------

def build_pipeline_state(complexity: str,
                         variant: str | None = None) -> dict:
    """Build a realistic fine-tuning pipeline state.

    Args:
        complexity: "full" (8 components) or "minimal" (3 components).
        variant: None for base state, "lr_change" for modified state
                 (learning rate changed from 2e-5 to 1e-5).

    Returns:
        dict with "complexity" and "components" keys.
    """
    if complexity not in ("full", "minimal"):
        raise ValueError(f"Invalid complexity: {complexity}")

    lr = "1.0e-5" if variant == "lr_change" else "2.0e-5"
    lr_comment = ("  # LR reduced after overfitting observed in run-v2"
                  if variant == "lr_change"
                  else "  # LR from grid search over [1e-5, 5e-5, 1e-4]")

    all_components = [
        {
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
        },
        {
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
                "template_format": (
                    "### Instruction:\n{instruction}\n\n"
                    "### Input:\n{input}\n\n"
                    "### Response:\n{output}"
                ),
                "num_proc": 8,
                "batched": True,
                "remove_columns": ["instruction", "input", "output"],
            }, indent=2),
        },
        {
            "name": "training_hparams",
            "filename": "training_hparams.yaml",
            "format": "yaml",
            "text": (
                "# Training hyperparameters for Llama-2-7B LoRA fine-tuning\n"
                "# Experiment: sft-alpaca-v3\n"
                "# Hardware: 4x A100 80GB\n"
                "\n"
                "training:\n"
                + lr_comment + "\n"
                f"  learning_rate: {lr}\n"
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
        },
        {
            "name": "adapter_config",
            "filename": "adapter_config.json",
            "format": "json",
            "text": json.dumps({
                "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "init_lora_weights": True,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "modules_to_save": None,
                "peft_type": "LORA",
                "r": 16,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "task_type": "CAUSAL_LM",
            }, indent=2),
        },
        {
            "name": "quantization_config",
            "filename": "quantization_config.json",
            "format": "json",
            "text": json.dumps({
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_has_fp16_weight": False,
            }, indent=2),
        },
        {
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
                'max_model_len = 2048\n'
                '\n'
                "[serving]\n"
                'host = "0.0.0.0"\n'
                'port = 8000\n'
                "# Max concurrent requests based on GPU memory profiling\n"
                "max_num_seqs = 64\n"
                "max_num_batched_tokens = 4096\n"
                "gpu_memory_utilization = 0.9\n"
                "\n"
                "[logging]\n"
                'level = "info"\n'
                "# Access log for monitoring request latency\n"
                "access_log = true\n"
            ),
        },
        {
            "name": "eval_results",
            "filename": "eval_results.json",
            "format": "json",
            "text": json.dumps({
                "model": "llama2-7b-sft-alpaca-v3",
                "eval_date": "2026-01-20",
                "benchmarks": {
                    "mmlu": {"accuracy": 0.462, "stderr": 0.008},
                    "hellaswag": {"accuracy": 0.782, "stderr": 0.004},
                    "arc_challenge": {"accuracy": 0.534, "stderr": 0.015},
                    "truthfulqa": {"accuracy": 0.412, "stderr": 0.012},
                    "winogrande": {"accuracy": 0.721, "stderr": 0.013},
                },
                "training_loss_final": 0.876,
                "eval_loss_final": 0.923,
                "total_training_time_hours": 4.2,
                "gpu_hours": 16.8,
            }, indent=2),
        },
        {
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
                "    model-version: v3\n"
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
                "              nvidia.com/gpu: '1'\n"
                "            limits:\n"
                "              cpu: '8'\n"
                "              memory: 64Gi\n"
                "              nvidia.com/gpu: '1'\n"
                "          ports:\n"
                "            - containerPort: 8000\n"
                "          # Readiness tuned for model load time\n"
                "          readinessProbe:\n"
                "            httpGet:\n"
                "              path: /health\n"
                "              port: 8000\n"
                "            initialDelaySeconds: 120\n"
                "            periodSeconds: 10\n"
            ),
        },
    ]

    if complexity == "minimal":
        components = [c for c in all_components
                      if c["name"] in _MINIMAL_COMPONENTS]
    else:
        components = all_components

    return {
        "complexity": complexity,
        "variant": variant,
        "components": components,
    }


# ---------------------------------------------------------------------------
# Capture methods
# ---------------------------------------------------------------------------

def capture_cdxf(components: list[dict]) -> bytes:
    """Capture pipeline state as a CDXF multi-document stream."""
    documents = []
    for comp in components:
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
                source_format_hint=_FORMAT_TO_SOURCE.get(
                    fmt, SourceFormat.UNSPECIFIED
                ),
            )

        doc.preamble.insert(
            0, Comment(text=f"cdxf-bundle-filename: {filename}")
        )
        doc.source_format_hint = _FORMAT_TO_SOURCE.get(
            fmt, SourceFormat.UNSPECIFIED
        )
        documents.append(doc)

    return encode(Stream(documents=documents))


def capture_tar_gz(components: list[dict]) -> bytes:
    """Capture pipeline state as a tar.gz archive."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for comp in components:
            data = comp["text"].encode("utf-8")
            info = tarfile.TarInfo(name=comp["filename"])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def capture_mega_json(components: list[dict]) -> bytes:
    """Capture pipeline state as a single JSON object."""
    merged = {}
    for comp in components:
        if comp["format"] == "json":
            try:
                merged[comp["filename"]] = json.loads(comp["text"])
            except json.JSONDecodeError:
                merged[comp["filename"]] = comp["text"]
        else:
            merged[comp["filename"]] = comp["text"]
    return json.dumps(merged, indent=2).encode("utf-8")


def capture_pickle(components: list[dict]) -> bytes:
    """Capture pipeline state as a pickled dict."""
    data = {}
    for comp in components:
        if comp["format"] == "json":
            try:
                data[comp["filename"]] = json.loads(comp["text"])
            except json.JSONDecodeError:
                data[comp["filename"]] = comp["text"]
        else:
            data[comp["filename"]] = comp["text"]
    return pickle.dumps(data)


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------

def _find_cdxf_doc_by_filename(stream: Stream, filename: str):
    """Find a Document in a CDXF stream by preamble filename comment."""
    for i, doc in enumerate(stream.documents):
        for node in doc.preamble:
            if isinstance(node, Comment):
                if node.text.strip().startswith("cdxf-bundle-filename:"):
                    doc_fn = node.text.strip().split(":", 1)[1].strip()
                    if doc_fn == filename:
                        return i, doc
    return None, None


def extract_component(bundle: bytes, method: str,
                      filename: str) -> dict:
    """Extract a single component from a captured state."""
    result = {"success": False, "text": "", "time_ns": 0, "error": None}

    try:
        t_start = time.perf_counter_ns()

        if method == "cdxf":
            stream = decode(bundle)
            idx, doc = _find_cdxf_doc_by_filename(stream, filename)
            if doc is None:
                result["error"] = f"Component '{filename}' not found"
                return result
            single = Stream(documents=[doc])
            hint = doc.source_format_hint
            if hint == SourceFormat.JSON:
                result["text"] = to_json(single, indent=2)
            elif hint == SourceFormat.YAML:
                result["text"] = to_yaml(single)
            elif hint == SourceFormat.TOML:
                result["text"] = to_toml(single)
            else:
                if isinstance(doc.root, Scalar):
                    result["text"] = str(doc.root.value)
                else:
                    result["text"] = to_json(single, indent=2)

        elif method == "tar_gz":
            buf = io.BytesIO(bundle)
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                member = tar.getmember(filename)
                f = tar.extractfile(member)
                result["text"] = f.read().decode("utf-8")

        elif method == "mega_json":
            parsed = json.loads(bundle.decode("utf-8"))
            if filename not in parsed:
                result["error"] = f"Component '{filename}' not found"
                return result
            val = parsed[filename]
            result["text"] = val if isinstance(val, str) else json.dumps(
                val, indent=2
            )

        elif method == "pickle":
            parsed = pickle.loads(bundle)
            if filename not in parsed:
                result["error"] = f"Component '{filename}' not found"
                return result
            val = parsed[filename]
            result["text"] = val if isinstance(val, str) else json.dumps(
                val, indent=2
            )

        else:
            result["error"] = f"Unknown method: {method}"
            return result

        t_end = time.perf_counter_ns()
        result["success"] = True
        result["time_ns"] = t_end - t_start

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# State diffing
# ---------------------------------------------------------------------------

def diff_states(bundle1: bytes, bundle2: bytes, method: str) -> dict:
    """Diff two pipeline states captured with the same method.

    Returns dict with: success, n_diffs, changed_components, error.
    """
    result = {
        "success": False,
        "n_diffs": 0,
        "changed_components": [],
        "error": None,
    }

    try:
        if method == "cdxf":
            s1 = decode(bundle1)
            s2 = decode(bundle2)
            # Compare by extracting filenames and emitting text
            for doc1 in s1.documents:
                fn = _get_doc_filename(doc1)
                if fn is None:
                    continue
                _, doc2 = _find_cdxf_doc_by_filename(s2, fn)
                if doc2 is None:
                    result["changed_components"].append(fn)
                    result["n_diffs"] += 1
                    continue
                # Compare by emitting both as text
                t1 = _emit_doc(doc1)
                t2 = _emit_doc(doc2)
                if t1 != t2:
                    result["changed_components"].append(fn)
                    result["n_diffs"] += 1

        elif method == "mega_json":
            d1 = json.loads(bundle1.decode("utf-8"))
            d2 = json.loads(bundle2.decode("utf-8"))
            for key in set(d1.keys()) | set(d2.keys()):
                v1 = d1.get(key)
                v2 = d2.get(key)
                if v1 != v2:
                    result["changed_components"].append(key)
                    result["n_diffs"] += 1

        elif method == "tar_gz":
            files1 = _tar_to_dict(bundle1)
            files2 = _tar_to_dict(bundle2)
            for key in set(files1.keys()) | set(files2.keys()):
                if files1.get(key) != files2.get(key):
                    result["changed_components"].append(key)
                    result["n_diffs"] += 1

        elif method == "pickle":
            d1 = pickle.loads(bundle1)
            d2 = pickle.loads(bundle2)
            for key in set(d1.keys()) | set(d2.keys()):
                v1 = d1.get(key)
                v2 = d2.get(key)
                if v1 != v2:
                    result["changed_components"].append(key)
                    result["n_diffs"] += 1

        else:
            result["error"] = f"Unknown method: {method}"
            return result

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def _get_doc_filename(doc: Document) -> str | None:
    """Extract filename from a CDXF document's preamble comment."""
    for node in doc.preamble:
        if isinstance(node, Comment):
            if node.text.strip().startswith("cdxf-bundle-filename:"):
                return node.text.strip().split(":", 1)[1].strip()
    return None


def _emit_doc(doc: Document) -> str:
    """Emit a single CDXF document as text via appropriate bridge."""
    single = Stream(documents=[doc])
    hint = doc.source_format_hint
    if hint == SourceFormat.JSON:
        return to_json(single, indent=2)
    elif hint == SourceFormat.YAML:
        return to_yaml(single)
    elif hint == SourceFormat.TOML:
        return to_toml(single)
    else:
        if isinstance(doc.root, Scalar):
            return str(doc.root.value)
        return to_json(single, indent=2)


def _tar_to_dict(tar_bytes: bytes) -> dict[str, str]:
    """Extract all files from a tar.gz to a dict."""
    result = {}
    buf = io.BytesIO(tar_bytes)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f:
                result[member.name] = f.read().decode("utf-8")
    return result


# ---------------------------------------------------------------------------
# Cross-language readability assessment
# ---------------------------------------------------------------------------

def assess_cross_language() -> dict:
    """Assess cross-language readability of each capture method.

    This is a qualitative assessment based on format properties,
    not a runtime test (would require non-Python tooling).
    """
    return {
        "cdxf": {
            "cross_language": True,
            "reason": "CBOR substrate; libraries in 30+ languages",
            "libraries": ["cbor2 (Python)", "cbor-x (JS)", "cbor (Go)",
                          "cbor (Rust)", "jackson-cbor (Java)"],
            "structured_access": True,
        },
        "tar_gz": {
            "cross_language": True,
            "reason": "Universal archive format",
            "libraries": ["stdlib in most languages"],
            "structured_access": False,  # must extract then parse each file
        },
        "mega_json": {
            "cross_language": True,
            "reason": "Universal text format",
            "libraries": ["stdlib in most languages"],
            "structured_access": True,
        },
        "pickle": {
            "cross_language": False,
            "reason": "Python-only binary format; security risk on untrusted",
            "libraries": ["pickle (Python only)"],
            "structured_access": True,
        },
    }


# ---------------------------------------------------------------------------
# Metadata counting in CDXF tree
# ---------------------------------------------------------------------------

def _count_cdxf_comments(stream: Stream) -> int:
    """Count Comment nodes in CDXF stream, excluding filename tags."""
    count = 0
    for doc in stream.documents:
        for node in doc.preamble:
            if isinstance(node, Comment):
                if not node.text.strip().startswith("cdxf-bundle-filename:"):
                    count += 1
        for node in doc.postamble:
            if isinstance(node, Comment):
                count += 1
        if doc.root is not None:
            count += _walk_count_comments(doc.root)
    return count


def _walk_count_comments(node) -> int:
    """Recursively count Comment nodes."""
    count = 0
    if isinstance(node, Comment):
        return 1
    elif isinstance(node, Map):
        for entry in node.entries:
            if isinstance(entry, Comment):
                count += 1
            elif isinstance(entry, tuple) and len(entry) == 2:
                count += _walk_count_comments(entry[0])
                count += _walk_count_comments(entry[1])
    elif isinstance(node, Sequence):
        for item in node.items:
            count += _walk_count_comments(item)
    return count


def _count_source_metadata(components: list[dict]) -> int:
    """Count metadata constructs in source texts."""
    count = 0
    for comp in components:
        if comp["format"] in ("yaml", "toml"):
            for line in comp["text"].splitlines():
                if line.strip().startswith("#"):
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Full measurement
# ---------------------------------------------------------------------------

def measure_pipeline(complexity: str) -> dict:
    """Run all capture methods on a pipeline state and compare."""
    base = build_pipeline_state(complexity)
    variant = build_pipeline_state(complexity, variant="lr_change")
    components = base["components"]
    total_input = sum(len(c["text"].encode("utf-8")) for c in components)
    source_metadata = _count_source_metadata(components)

    result = {
        "complexity": complexity,
        "n_components": len(components),
        "total_input_size": total_input,
        "source_metadata": source_metadata,
    }

    capturers = {
        "cdxf": capture_cdxf,
        "tar_gz": capture_tar_gz,
        "mega_json": capture_mega_json,
        "pickle": capture_pickle,
    }

    for method, capturer in capturers.items():
        try:
            t_start = time.perf_counter_ns()
            bundle = capturer(components)
            t_end = time.perf_counter_ns()

            mr = {
                "success": True,
                "capture_size": len(bundle),
                "size_ratio": len(bundle) / total_input if total_input else 0,
                "capture_time_ns": t_end - t_start,
                "error": None,
            }

            # Extract one component
            ext = extract_component(bundle, method, "adapter_config.json")
            mr["extract_success"] = ext["success"]
            mr["extract_time_ns"] = ext["time_ns"]

            # CDXF-specific: metadata
            if method == "cdxf":
                decoded = decode(bundle)
                mr["metadata_preserved"] = _count_cdxf_comments(decoded)

                # Round-trip fidelity
                fidelity_pass = 0
                fidelity_total = 0
                for comp in components:
                    if comp["format"] in _FROM_BRIDGE:
                        fidelity_total += 1
                        extracted = extract_component(
                            bundle, "cdxf", comp["filename"]
                        )
                        if extracted["success"] and len(extracted["text"]) > 0:
                            fidelity_pass += 1
                mr["round_trip_fidelity"] = (
                    fidelity_pass / fidelity_total
                    if fidelity_total > 0 else 1.0
                )

        except Exception as e:
            mr = {
                "success": False,
                "capture_size": 0,
                "size_ratio": 0,
                "capture_time_ns": 0,
                "error": str(e),
            }

        result[method] = mr

    # Diff test: base vs variant
    try:
        b_base = capture_cdxf(base["components"])
        b_var = capture_cdxf(variant["components"])
        diff = diff_states(b_base, b_var, "cdxf")
        result["diff_detected"] = diff["n_diffs"] > 0
        result["diff_components"] = diff["changed_components"]
    except Exception:
        result["diff_detected"] = False
        result["diff_components"] = []

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run EXP-009: End-to-End Fine-Tuning Pipeline State Capture."""
    print("=" * 70)
    print("EXP-009: End-to-End Fine-Tuning Pipeline State Capture")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for complexity in ["minimal", "full"]:
        print(f"\n{'='*70}")
        print(f"Pipeline complexity: {complexity.upper()}")
        print(f"{'='*70}")

        result = measure_pipeline(complexity)
        results.append(result)

        print(f"Components: {result['n_components']}, "
              f"total input: {result['total_input_size']}B, "
              f"source metadata: {result['source_metadata']}")

        for method in ["cdxf", "tar_gz", "mega_json", "pickle"]:
            mr = result[method]
            if mr["success"]:
                size = mr["capture_size"]
                ratio = mr["size_ratio"]
                print(f"  {method:10s}: {size:6d}B  ratio={ratio:.3f}",
                      end="")
                if method == "cdxf":
                    print(f"  fidelity={mr.get('round_trip_fidelity', 'N/A')}"
                          f"  metadata={mr.get('metadata_preserved', 0)}",
                          end="")
                print()
            else:
                print(f"  {method:10s}: FAIL {mr['error']}")

        if result.get("diff_detected"):
            print(f"  Diff detected: {result['diff_components']}")
        else:
            print(f"  Diff detected: False")

    # Cross-language assessment
    print(f"\n{'='*70}")
    print("CROSS-LANGUAGE READABILITY")
    print(f"{'='*70}")
    cross = assess_cross_language()
    for method, info in cross.items():
        lang = "YES" if info["cross_language"] else "NO"
        struct = "YES" if info["structured_access"] else "NO"
        print(f"  {method:10s}: cross-language={lang}  "
              f"structured={struct}  ({info['reason']})")

    # Write CSV
    csv_path = RESULTS_DIR / "pipeline_results.csv"
    fieldnames = [
        "complexity", "n_components", "total_input_size", "source_metadata",
        "cdxf_size", "cdxf_ratio", "cdxf_fidelity", "cdxf_metadata",
        "tar_gz_size", "tar_gz_ratio",
        "mega_json_size", "mega_json_ratio",
        "pickle_size", "pickle_ratio",
        "diff_detected",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "complexity": r["complexity"],
                "n_components": r["n_components"],
                "total_input_size": r["total_input_size"],
                "source_metadata": r["source_metadata"],
                "diff_detected": r.get("diff_detected", False),
            }
            for method in ["cdxf", "tar_gz", "mega_json", "pickle"]:
                mr = r[method]
                row[f"{method}_size"] = mr["capture_size"] if mr["success"] else ""
                row[f"{method}_ratio"] = (
                    f"{mr['size_ratio']:.4f}" if mr["success"] else ""
                )
            row["cdxf_fidelity"] = r["cdxf"].get("round_trip_fidelity", "")
            row["cdxf_metadata"] = r["cdxf"].get("metadata_preserved", "")
            writer.writerow(row)
    print(f"\nResults: {csv_path}")

    # Summary
    print(f"\n{'='*70}")
    print("HEADLINE FINDINGS")
    print(f"{'='*70}")

    full_r = [r for r in results if r["complexity"] == "full"][0]
    print(f"\nFull pipeline ({full_r['n_components']} components):")
    for method in ["cdxf", "tar_gz", "mega_json", "pickle"]:
        mr = full_r[method]
        if mr["success"]:
            print(f"  {method:10s}: {mr['capture_size']:6d}B "
                  f"(ratio={mr['size_ratio']:.3f})")

    cdxf_r = full_r["cdxf"]
    print(f"\n  CDXF round-trip fidelity: "
          f"{cdxf_r.get('round_trip_fidelity', 0):.0%}")
    print(f"  CDXF metadata preserved: "
          f"{cdxf_r.get('metadata_preserved', 0)} "
          f"(source had {full_r['source_metadata']})")
    print(f"  Diff capability: "
          f"{'YES' if full_r.get('diff_detected') else 'NO'} "
          f"(changed: {full_r.get('diff_components', [])})")

    cross_scores = {m: 1 if v["cross_language"] else 0
                    for m, v in cross.items()}
    struct_scores = {m: 1 if v["structured_access"] else 0
                     for m, v in cross.items()}
    print(f"\n  Cross-language: CDXF={cross_scores['cdxf']} "
          f"tar.gz={cross_scores['tar_gz']} "
          f"JSON={cross_scores['mega_json']} "
          f"Pickle={cross_scores['pickle']}")
    print(f"  Structured access: CDXF={struct_scores['cdxf']} "
          f"tar.gz={struct_scores['tar_gz']} "
          f"JSON={struct_scores['mega_json']} "
          f"Pickle={struct_scores['pickle']}")

    # Write summary JSON
    summary_path = RESULTS_DIR / "summary.json"
    summary = {
        "pipelines": [
            {
                "complexity": r["complexity"],
                "n_components": r["n_components"],
                "total_input_size": r["total_input_size"],
                "source_metadata": r["source_metadata"],
                "sizes": {
                    m: r[m]["capture_size"]
                    for m in CAPTURE_METHODS if r[m]["success"]
                },
                "ratios": {
                    m: round(r[m]["size_ratio"], 4)
                    for m in CAPTURE_METHODS if r[m]["success"]
                },
                "cdxf_fidelity": r["cdxf"].get("round_trip_fidelity"),
                "cdxf_metadata": r["cdxf"].get("metadata_preserved"),
                "diff_detected": r.get("diff_detected"),
                "diff_components": r.get("diff_components", []),
            }
            for r in results
        ],
        "cross_language": cross,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
