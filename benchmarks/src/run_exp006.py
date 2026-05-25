"""
EXP-006: ML Configuration Fidelity Under Serialization

Measures how much reproducibility-critical metadata (comments, anchors,
typed values) is destroyed when real ML configuration files are serialized
with existing binary interchange formats vs CDXF.

Usage:
    python benchmarks/src/run_exp006.py
"""

import csv
import json
import pickle
import re
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import cbor2
    import msgpack
    import bson
    import amazon.ion.simpleion as ion
    import ruamel.yaml
    import tomlkit
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Run: pip install cbor2 msgpack pymongo amazon.ion ruamel.yaml tomlkit")
    sys.exit(1)

# CDXF imports
try:
    import cdxf
    from cdxf.bridges.json_bridge import from_json, to_json
    from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
    from cdxf.bridges.toml_bridge import from_toml, to_toml
    from cdxf.bridges.xml_bridge import from_xml, to_xml
    from cdxf.codec import encode, decode
except ImportError:
    print("ERROR: cdxf not installed. Run: pip install -e .")
    sys.exit(1)


RESULTS_DIR = Path("benchmarks/results/exp_006")

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

BASELINES = {
    "cdxf": "CDXF (ours)",
    "cbor": "CBOR (cbor2)",
    "msgpack": "MsgPack",
    "bson": "BSON (pymongo)",
    "ion": "Amazon Ion",
    "json_stdlib": "JSON (stdlib)",
    "pickle": "Pickle (stdlib)",
}


# ---------------------------------------------------------------------------
# ML Config Corpus — realistic files with known constructs
# ---------------------------------------------------------------------------

def build_ml_corpus() -> list[dict]:
    """Build a corpus of realistic ML configuration files.

    Each entry: {name, format, text, category}
    Files are designed to contain format-specific constructs that are
    critical for ML reproducibility.
    """
    corpus = []

    # -----------------------------------------------------------------------
    # YAML configs with comments (the core reproducibility case)
    # -----------------------------------------------------------------------

    corpus.append({
        "name": "hydra_training_config",
        "format": "yaml",
        "category": "training",
        "text": """\
# Hydra training config for BERT fine-tuning on GLUE
# Author: researcher@lab.edu
# Last grid search: 2024-11-15

model:
  name: bert-base-uncased
  # Tried large but OOM on single GPU
  hidden_dropout_prob: 0.1  # default, don't change

training:
  # lr=1e-4 selected after grid search over [5e-5, 1e-4, 3e-4, 5e-4]
  # See experiment log: exp_042_lr_sweep.md
  learning_rate: 0.0001
  weight_decay: 0.01  # AdamW default
  warmup_steps: 500  # 10% of total steps
  max_epochs: 3
  # batch_size=32 is max for 24GB VRAM with fp16
  batch_size: 32
  gradient_accumulation_steps: 2  # effective batch = 64
  fp16: true
  # Early stopping patience increased from 3 to 5 after
  # observing late convergence on MNLI
  early_stopping_patience: 5
  seed: 42  # for reproducibility

data:
  dataset: glue
  task: mnli
  max_seq_length: 128
  # Truncation strategy: longest_first works best for NLI
  truncation: longest_first

optimizer:
  type: adamw
  # Betas from "Decoupled Weight Decay Regularization" (Loshchilov 2019)
  beta1: 0.9
  beta2: 0.999
  epsilon: 1.0e-8

scheduler:
  type: linear_with_warmup
  # Tried cosine but linear converged faster on this task
  num_training_steps: 5000
""",
    })

    corpus.append({
        "name": "lightning_config_with_anchors",
        "format": "yaml",
        "category": "training",
        "text": """\
# PyTorch Lightning training configuration
# Uses anchors for shared config blocks across experiments

defaults: &defaults
  precision: 16
  accelerator: gpu
  devices: 1
  max_epochs: 10
  # Gradient clipping prevents exploding gradients on long sequences
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4
  log_every_n_steps: 50

# Experiment variants — all inherit from defaults
experiment_small: &exp_small
  <<: *defaults
  max_epochs: 5
  # Smaller model needs fewer epochs
  accumulate_grad_batches: 2

experiment_large:
  <<: *defaults
  max_epochs: 20
  devices: 4
  # Multi-GPU needs larger effective batch
  accumulate_grad_batches: 1

# Callbacks
callbacks:
  early_stopping:
    monitor: val_loss
    patience: 3
    mode: min
  model_checkpoint:
    monitor: val_loss
    save_top_k: 3
    # Save weights only, not optimizer state (saves disk)
    save_weights_only: true
""",
    })

    corpus.append({
        "name": "multi_doc_experiment_variants",
        "format": "yaml",
        "category": "training",
        "text": """\
# Experiment variant 1: baseline
---
name: baseline_bert
model: bert-base-uncased
learning_rate: 0.00003
batch_size: 32
# Baseline: no regularization changes
dropout: 0.1
---
# Experiment variant 2: higher LR
name: high_lr_bert
model: bert-base-uncased
learning_rate: 0.0001
batch_size: 32
dropout: 0.1
---
# Experiment variant 3: regularization sweep
name: reg_sweep_bert
model: bert-base-uncased
learning_rate: 0.00003
batch_size: 32
# Increased dropout based on validation overfitting
dropout: 0.3
...
""",
    })

    corpus.append({
        "name": "dataset_card_frontmatter",
        "format": "yaml",
        "category": "dataset",
        "text": """\
# Dataset card metadata for fine-tuning corpus
annotations_creators:
  - machine-generated
  - expert-generated
language:
  - en
  - zh
  - es
license: apache-2.0
multilinguality: multilingual
size_categories:
  - 100K<n<1M
task_categories:
  - text-classification
  - token-classification
task_ids:
  - sentiment-analysis
  - named-entity-recognition
# Data collected between 2023-01 and 2024-06
# Filtered for quality using perplexity threshold < 50
pretty_name: MultiTask NLP Corpus v2
""",
    })

    corpus.append({
        "name": "k8s_deployment_ml_serving",
        "format": "yaml",
        "category": "deployment",
        "text": """\
# Kubernetes deployment for vLLM model serving
# Requires GPU node pool with >=24GB VRAM
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
  namespace: ml-inference
  labels:
    app: llm-serving
    # Version tracks the model, not the deployment
    version: "llama-3.1-8b-v2"
spec:
  replicas: 2  # scale based on QPS, currently ~100 req/s
  selector:
    matchLabels:
      app: llm-serving
  template:
    metadata:
      labels:
        app: llm-serving
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          # GPU memory fraction: 0.9 leaves room for KV cache
          args:
            - --model
            - meta-llama/Llama-3.1-8B-Instruct
            - --gpu-memory-utilization
            - "0.9"
            - --max-model-len
            - "4096"  # reduced from 8192 to fit on single GPU
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              memory: "32Gi"
              cpu: "4"
          ports:
            - containerPort: 8000
          # Health check: model must be loaded
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120  # model loading takes ~2min
            periodSeconds: 10
""",
    })

    # -----------------------------------------------------------------------
    # JSON configs (plain — no constructs, baseline comparison)
    # -----------------------------------------------------------------------

    corpus.append({
        "name": "hf_config_json",
        "format": "json",
        "category": "model",
        "text": json.dumps({
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 131072,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 128256,
        }, indent=2),
    })

    corpus.append({
        "name": "adapter_config_json",
        "format": "json",
        "category": "adapter",
        "text": json.dumps({
            "auto_mapping": None,
            "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 64,
            "revision": None,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "task_type": "CAUSAL_LM",
        }, indent=2),
    })

    corpus.append({
        "name": "eval_results_json",
        "format": "json",
        "category": "evaluation",
        "text": json.dumps({
            "results": {
                "mmlu": {"acc": 0.6834, "acc_stderr": 0.0042},
                "hellaswag": {"acc_norm": 0.8123, "acc_norm_stderr": 0.0039},
                "arc_challenge": {"acc_norm": 0.5734, "acc_norm_stderr": 0.0145},
                "truthfulqa_mc2": {"acc": 0.4521, "acc_stderr": 0.0156},
                "winogrande": {"acc": 0.7693, "acc_stderr": 0.0118},
                "gsm8k": {"exact_match": 0.5621, "exact_match_stderr": 0.0137},
            },
            "config": {
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "model_args": "pretrained=meta-llama/Llama-3.1-8B-Instruct",
                "batch_size": "auto",
                "device": "cuda",
                "num_fewshot": 5,
                "datetime": "2024-11-15T14:30:00Z",
            },
            "versions": {
                "lm_eval": "0.4.2",
                "transformers": "4.44.0",
                "torch": "2.4.0",
            },
        }, indent=2),
    })

    corpus.append({
        "name": "training_args_json",
        "format": "json",
        "category": "training",
        "text": json.dumps({
            "output_dir": "./results/llama-3.1-8b-sft",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": 500,
            "eval_strategy": "steps",
            "eval_steps": 500,
            "bf16": True,
            "gradient_checkpointing": True,
            "max_grad_norm": 1.0,
            "seed": 42,
            "dataloader_num_workers": 4,
            "report_to": ["wandb"],
        }, indent=2),
    })

    # -----------------------------------------------------------------------
    # TOML configs with comments and typed values
    # -----------------------------------------------------------------------

    corpus.append({
        "name": "pyproject_ml_package",
        "format": "toml",
        "category": "packaging",
        "text": """\
# ML training package configuration
[project]
name = "llm-finetune"
version = "0.3.1"
description = "Fine-tuning pipeline for LLMs"
# License changed from GPL to Apache after legal review 2024-06
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [
    {name = "ML Research Team", email = "ml@example.com"},
]

[project.dependencies]
# Pin transformers to avoid breaking changes in tokenizer API
transformers = ">=4.40.0,<4.50.0"
torch = ">=2.2.0"
peft = ">=0.10.0"
datasets = ">=2.18.0"
accelerate = ">=0.28.0"
# bitsandbytes for 4-bit quantization
bitsandbytes = ">=0.43.0"

[project.optional-dependencies]
# eval deps are heavy, keep separate
eval = ["lm-eval>=0.4.0", "vllm>=0.4.0"]
dev = ["pytest", "ruff", "mypy"]

[tool.ruff]
line-length = 100
# Ignore E501 in notebooks
extend-exclude = ["notebooks/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
# Slow tests (GPU) marked separately
markers = ["slow: GPU-required tests", "integration: end-to-end tests"]

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"
""",
    })

    corpus.append({
        "name": "serving_config_toml",
        "format": "toml",
        "category": "deployment",
        "text": """\
# vLLM serving configuration
# Last updated: 2024-11-20

[server]
host = "0.0.0.0"
port = 8000
# Timeout increased from 30s to 120s for long generation
timeout = 120

[model]
name = "meta-llama/Llama-3.1-8B-Instruct"
# Quantization: AWQ gives best quality/speed tradeoff
# Tested GPTQ, AWQ, and GGUF — see bench_quant_comparison.md
quantization = "awq"
max_model_len = 4096
gpu_memory_utilization = 0.90
# dtype auto-detects from model config
dtype = "auto"

[generation]
max_tokens = 2048
temperature = 0.7
top_p = 0.9
# Repetition penalty helps with long outputs
repetition_penalty = 1.1

[logging]
level = "INFO"
# Structured JSON logs for ELK stack
format = "json"
# Created this config on 2024-11-20
created = 2024-11-20T14:30:00
""",
    })

    corpus.append({
        "name": "experiment_tracker_toml",
        "format": "toml",
        "category": "experiment",
        "text": """\
# Experiment tracking configuration
# Tracks all runs for the Q4 LLM evaluation project

[project]
name = "q4-llm-eval"
# Started 2024-10-01, target completion 2024-12-15
start_date = 2024-10-01
target_date = 2024-12-15
status = "in_progress"

[wandb]
project = "q4-llm-eval"
entity = "ml-research-team"
# Tags for filtering runs
tags = ["llm", "evaluation", "q4-2024"]

[mlflow]
tracking_uri = "http://mlflow.internal:5000"
experiment_name = "q4-llm-eval"
# Artifact store on S3
artifact_location = "s3://ml-artifacts/q4-eval"

# Metrics to track across all experiments
[metrics]
primary = "mmlu_accuracy"
secondary = ["hellaswag", "arc_challenge", "truthfulqa"]
# Report threshold: only log runs above this accuracy
min_report_threshold = 0.5
""",
    })

    # -----------------------------------------------------------------------
    # XML configs (PMML, ONNX metadata, config files)
    # -----------------------------------------------------------------------

    corpus.append({
        "name": "pmml_model_metadata",
        "format": "xml",
        "category": "model",
        "text": """\
<?xml version="1.0" encoding="UTF-8"?>
<!-- PMML export of trained logistic regression model -->
<!-- Exported by sklearn2pmml v0.90.0 -->
<?model-info version="4.4" exported="2024-11-15"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      version="4.4">
  <Header copyright="ML Research Lab 2024">
    <Application name="sklearn2pmml" version="0.90.0"/>
    <!-- Training completed after 150 epochs, val_acc=0.923 -->
    <Timestamp>2024-11-15T10:30:00Z</Timestamp>
  </Header>
  <DataDictionary numberOfFields="5">
    <DataField name="sepal_length" optype="continuous" dataType="double"/>
    <DataField name="sepal_width" optype="continuous" dataType="double"/>
    <DataField name="petal_length" optype="continuous" dataType="double"/>
    <DataField name="petal_width" optype="continuous" dataType="double"/>
    <DataField name="species" optype="categorical" dataType="string">
      <Value value="setosa"/>
      <Value value="versicolor"/>
      <Value value="virginica"/>
    </DataField>
  </DataDictionary>
  <!-- Model section: LogisticRegression with L2 regularization -->
  <RegressionModel modelName="iris_classifier"
                   functionName="classification"
                   algorithmName="logisticRegression"
                   normalizationMethod="logit">
    <MiningSchema>
      <MiningField name="species" usageType="target"/>
      <MiningField name="sepal_length"/>
      <MiningField name="sepal_width"/>
      <MiningField name="petal_length"/>
      <MiningField name="petal_width"/>
    </MiningSchema>
  </RegressionModel>
</PMML>
""",
    })

    corpus.append({
        "name": "onnx_metadata_xml",
        "format": "xml",
        "category": "model",
        "text": """\
<?xml version="1.0" encoding="UTF-8"?>
<!-- ONNX model metadata for deployment validation -->
<onnx:ModelProto xmlns:onnx="http://onnx.ai/onnx"
                 xmlns:custom="http://example.com/custom-ops">
  <onnx:ir_version>9</onnx:ir_version>
  <onnx:producer_name>pytorch</onnx:producer_name>
  <onnx:producer_version>2.4.0</onnx:producer_version>
  <!-- Opset 18 required for GroupNorm -->
  <onnx:opset_import>
    <onnx:OperatorSetIdProto domain="" version="18"/>
    <onnx:OperatorSetIdProto domain="com.microsoft" version="1"/>
  </onnx:opset_import>
  <onnx:metadata_props>
    <onnx:StringStringEntryProto key="model_name" value="bert-base-uncased"/>
    <onnx:StringStringEntryProto key="task" value="sequence-classification"/>
    <onnx:StringStringEntryProto key="framework" value="pytorch"/>
    <!-- Dynamic axes for variable batch size and sequence length -->
    <onnx:StringStringEntryProto key="dynamic_axes"
      value="input_ids:{0:batch,1:seq},attention_mask:{0:batch,1:seq}"/>
  </onnx:metadata_props>
</onnx:ModelProto>
""",
    })

    # -----------------------------------------------------------------------
    # Additional YAML edge cases
    # -----------------------------------------------------------------------

    corpus.append({
        "name": "docker_compose_ml",
        "format": "yaml",
        "category": "deployment",
        "text": """\
# Docker Compose for ML training environment
# Requires NVIDIA Container Toolkit
version: "3.8"

services:
  trainer:
    image: nvcr.io/nvidia/pytorch:24.01-py3
    # Mount training data and output
    volumes:
      - ./data:/data:ro
      - ./outputs:/outputs
      - ./configs:/configs:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - WANDB_API_KEY=${WANDB_API_KEY}
      # Disable tokenizer parallelism warning
      - TOKENIZERS_PARALLELISM=false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    # Health check: training script creates a .alive file
    healthcheck:
      test: ["CMD", "test", "-f", "/outputs/.alive"]
      interval: 30s
      timeout: 10s
      retries: 3

  # TensorBoard for monitoring
  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./outputs/logs:/logs:ro
    command: tensorboard --logdir /logs --bind_all
""",
    })

    corpus.append({
        "name": "wandb_sweep_config",
        "format": "yaml",
        "category": "experiment",
        "text": """\
# W&B Hyperparameter Sweep Configuration
# Bayesian optimization for LLM fine-tuning
program: train.py
method: bayes
metric:
  name: eval/loss
  goal: minimize

parameters:
  learning_rate:
    # Log-uniform between 1e-5 and 1e-3
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001
  weight_decay:
    values: [0.0, 0.01, 0.05, 0.1]
  warmup_ratio:
    # Linear warmup fraction
    distribution: uniform
    min: 0.0
    max: 0.2
  lora_r:
    # LoRA rank: powers of 2
    values: [8, 16, 32, 64, 128]
  lora_alpha:
    # Usually 2x rank
    values: [16, 32, 64, 128, 256]
  # Dropout: keep low for LoRA
  lora_dropout:
    values: [0.0, 0.05, 0.1]

# Early termination: stop bad runs quickly
early_terminate:
  type: hyperband
  min_iter: 100
  # Aggressiveness: 3 means keep top 1/3 at each bracket
  eta: 3
""",
    })

    return corpus


# ---------------------------------------------------------------------------
# Construct counting (reused from EXP-005 with extensions)
# ---------------------------------------------------------------------------

def count_constructs(text: str, fmt: str) -> dict:
    """Count format-specific constructs in a file.

    Returns a dict of construct_name → count.
    """
    if fmt == "yaml":
        return _count_yaml(text)
    elif fmt == "json":
        return _count_json(text)
    elif fmt == "toml":
        return _count_toml(text)
    elif fmt == "xml":
        return _count_xml(text)
    else:
        return {}


def _count_yaml(text: str) -> dict:
    counts = {
        "comments": 0, "anchors": 0, "aliases": 0,
        "merge_keys": 0, "multi_doc_markers": 0,
    }
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            counts["comments"] += 1
        elif "#" in stripped:
            in_sq, in_dq = False, False
            for i, ch in enumerate(stripped):
                if ch == "'" and not in_dq:
                    in_sq = not in_sq
                elif ch == '"' and not in_sq:
                    in_dq = not in_dq
                elif ch == "#" and not in_sq and not in_dq and i > 0:
                    counts["comments"] += 1
                    break
        counts["anchors"] += len(re.findall(r"&\w+", stripped))
        counts["aliases"] += len(re.findall(r"\*\w+", stripped))
        if stripped.startswith("<<"):
            counts["merge_keys"] += 1
        if stripped == "---" or stripped == "...":
            counts["multi_doc_markers"] += 1
    return counts


def _count_json(text: str) -> dict:
    counts = {"key_count": 0, "max_depth": 0, "array_count": 0}
    try:
        data = json.loads(text)
        _json_depth(data, 0, counts)
    except (json.JSONDecodeError, RecursionError):
        pass
    return counts


def _json_depth(obj, depth, counts):
    if depth > counts["max_depth"]:
        counts["max_depth"] = depth
    if isinstance(obj, dict):
        counts["key_count"] += len(obj)
        for v in obj.values():
            _json_depth(v, depth + 1, counts)
    elif isinstance(obj, list):
        counts["array_count"] += 1
        for item in obj:
            _json_depth(item, depth + 1, counts)


def _count_toml(text: str) -> dict:
    counts = {"comments": 0, "sections": 0, "inline_tables": 0, "temporal_values": 0}
    dt_pat = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            counts["comments"] += 1
        elif "#" in stripped:
            in_sq, in_dq = False, False
            for i, ch in enumerate(stripped):
                if ch == "'" and not in_dq:
                    in_sq = not in_sq
                elif ch == '"' and not in_sq:
                    in_dq = not in_dq
                elif ch == "#" and not in_sq and not in_dq and i > 0:
                    counts["comments"] += 1
                    break
        if stripped.startswith("["):
            counts["sections"] += 1
        if "{" in stripped and "=" in stripped:
            counts["inline_tables"] += 1
        if "=" in stripped:
            rhs = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            if dt_pat.match(rhs):
                counts["temporal_values"] += 1
    return counts


def _count_xml(text: str) -> dict:
    return {
        "comments": len(re.findall(r"<!--", text)),
        "processing_instructions": len(re.findall(r"<\?(?!xml\s)", text)),
        "namespaces": len(re.findall(r"xmlns[:\w]*=", text)),
        "elements": len(re.findall(r"<(?!/|\?|!)[a-zA-Z]", text)),
        "attributes": len(re.findall(r'\s\w+="', text)),
    }


# Constructs that represent metadata at risk of loss
METADATA_CONSTRUCT_KEYS = {
    "yaml": ["comments", "anchors", "aliases", "merge_keys", "multi_doc_markers"],
    "json": [],  # JSON has no metadata constructs
    "toml": ["comments", "temporal_values"],
    "xml": ["comments", "processing_instructions", "namespaces"],
}


def total_metadata_constructs(text: str, fmt: str) -> int:
    """Count total metadata constructs at risk of loss."""
    c = count_constructs(text, fmt)
    keys = METADATA_CONSTRUCT_KEYS.get(fmt, [])
    return sum(c.get(k, 0) for k in keys)


# ---------------------------------------------------------------------------
# Serialization round-trip implementations
# ---------------------------------------------------------------------------

def _parse_to_python(text: str, fmt: str):
    """Parse text to a plain Python data structure (dict/list/scalar)."""
    if fmt == "yaml":
        yaml = ruamel.yaml.YAML()
        import io
        docs = list(yaml.load_all(io.StringIO(text)))
        if len(docs) == 1:
            return _to_plain(docs[0])
        # Multi-doc: return list of plain dicts
        return [_to_plain(d) for d in docs]
    elif fmt == "json":
        return json.loads(text)
    elif fmt == "toml":
        parsed = tomlkit.loads(text)
        return _unwrap_tomlkit(parsed)
    elif fmt == "xml":
        # For baselines, convert XML to a simplified dict
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
        return _xml_to_dict(root)
    return None


def _to_plain(obj):
    """Convert ruamel YAML objects to plain Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_plain(item) for item in obj]
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    else:
        return str(obj)


def _unwrap_tomlkit(obj):
    """Recursively unwrap tomlkit types to plain Python."""
    if hasattr(obj, 'unwrap'):
        obj = obj.unwrap()
    if isinstance(obj, dict):
        return {str(k): _unwrap_tomlkit(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_unwrap_tomlkit(item) for item in obj]
    else:
        return obj


def _xml_to_dict(element):
    """Simple XML element to dict conversion."""
    result = {"_tag": element.tag, "_attrib": dict(element.attrib)}
    children = []
    if element.text and element.text.strip():
        result["_text"] = element.text.strip()
    for child in element:
        children.append(_xml_to_dict(child))
    if children:
        result["_children"] = children
    return result


def _emit_from_python(data, fmt: str) -> str:
    """Emit plain Python data back to text format."""
    if fmt == "yaml":
        yaml = ruamel.yaml.YAML()
        import io
        stream = io.StringIO()
        yaml.dump(data, stream)
        return stream.getvalue()
    elif fmt == "json":
        return json.dumps(data, indent=2, default=str)
    elif fmt == "toml":
        return tomlkit.dumps(data)
    elif fmt == "xml":
        return _dict_to_xml(data)
    return str(data)


def _dict_to_xml(d: dict) -> str:
    """Simple dict to XML conversion."""
    import xml.etree.ElementTree as ET
    tag = d.get("_tag", "root")
    # Strip namespace for simplicity
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    elem = ET.Element(tag, d.get("_attrib", {}))
    if "_text" in d:
        elem.text = d["_text"]
    for child in d.get("_children", []):
        elem.append(_dict_to_xml_element(child))
    return ET.tostring(elem, encoding="unicode")


def _dict_to_xml_element(d: dict):
    import xml.etree.ElementTree as ET
    tag = d.get("_tag", "element")
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    elem = ET.Element(tag, d.get("_attrib", {}))
    if "_text" in d:
        elem.text = d["_text"]
    for child in d.get("_children", []):
        elem.append(_dict_to_xml_element(child))
    return elem


def serialize_roundtrip(text: str, fmt: str, baseline: str) -> dict:
    """Serialize text through a baseline format and back.

    Returns: {success, serialized_size, re_emitted_text, error}
    """
    result = {
        "success": False,
        "serialized_size": 0,
        "re_emitted_text": "",
        "error": "",
    }

    try:
        if baseline == "cdxf":
            return _roundtrip_cdxf(text, fmt)

        # All other baselines: parse → Python → serialize → deserialize → emit
        data = _parse_to_python(text, fmt)
        if data is None:
            result["error"] = "parse_failed"
            return result

        # Serialize
        if baseline == "cbor":
            serialized = cbor2.dumps(data)
            deserialized = cbor2.loads(serialized)
        elif baseline == "msgpack":
            serialized = msgpack.packb(data, use_bin_type=True)
            deserialized = msgpack.unpackb(serialized, raw=False)
        elif baseline == "bson":
            # BSON requires top-level dict
            if not isinstance(data, dict):
                data = {"_root": data}
            serialized = bson.BSON.encode(data)
            deserialized = bson.BSON.decode(serialized)
        elif baseline == "ion":
            serialized = ion.dumps(data, binary=True)
            deserialized = ion.loads(serialized)
        elif baseline == "json_stdlib":
            serialized = json.dumps(data, default=str).encode("utf-8")
            deserialized = json.loads(serialized)
        elif baseline == "pickle":
            serialized = pickle.dumps(data)
            deserialized = pickle.loads(serialized)
        else:
            result["error"] = f"unknown_baseline: {baseline}"
            return result

        # Re-emit as original format
        re_emitted = _emit_from_python(deserialized, fmt)

        result["success"] = True
        result["serialized_size"] = len(serialized)
        result["re_emitted_text"] = re_emitted

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return result


def _roundtrip_cdxf(text: str, fmt: str) -> dict:
    """Round-trip through CDXF using the proper bridges."""
    result = {
        "success": False,
        "serialized_size": 0,
        "re_emitted_text": "",
        "error": "",
    }

    try:
        # Parse with CDXF bridge (preserves comments, anchors, etc.)
        if fmt == "yaml":
            tree = from_yaml(text)
        elif fmt == "json":
            tree = from_json(text)
        elif fmt == "toml":
            tree = from_toml(text)
        elif fmt == "xml":
            tree = from_xml(text)
        else:
            result["error"] = f"unsupported_format: {fmt}"
            return result

        # Encode to CDXF binary
        cdxf_bytes = encode(tree)

        # Decode back
        decoded_tree = decode(cdxf_bytes)

        # Re-emit in original format
        if fmt == "yaml":
            re_emitted = to_yaml(decoded_tree)
        elif fmt == "json":
            re_emitted = to_json(decoded_tree)
        elif fmt == "toml":
            re_emitted = to_toml(decoded_tree)
        elif fmt == "xml":
            re_emitted = to_xml(decoded_tree)

        result["success"] = True
        result["serialized_size"] = len(cdxf_bytes)
        result["re_emitted_text"] = re_emitted

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return result


# ---------------------------------------------------------------------------
# Fidelity measurement
# ---------------------------------------------------------------------------

def measure_fidelity(text: str, fmt: str, baseline: str) -> dict:
    """Measure construct survival after round-trip through a baseline.

    Returns counts of original vs preserved constructs plus data fidelity.
    """
    original_constructs = count_constructs(text, fmt)
    rt = serialize_roundtrip(text, fmt, baseline)

    result = {
        "success": rt["success"],
        "serialized_size": rt["serialized_size"],
        "original_size": len(text.encode("utf-8")),
        "error": rt["error"],
    }

    # Map construct counts
    metadata_keys = METADATA_CONSTRUCT_KEYS.get(fmt, [])
    for key in ["comments", "anchors", "aliases", "merge_keys",
                "multi_doc_markers", "temporal_values",
                "processing_instructions", "namespaces"]:
        result[f"{key}_original"] = original_constructs.get(key, 0)

        if rt["success"] and rt["re_emitted_text"]:
            re_emitted_constructs = count_constructs(rt["re_emitted_text"], fmt)
            result[f"{key}_preserved"] = re_emitted_constructs.get(key, 0)
        else:
            result[f"{key}_preserved"] = 0

    # Total metadata at risk
    total_original = sum(
        original_constructs.get(k, 0) for k in metadata_keys
    )
    total_preserved = sum(
        result.get(f"{k}_preserved", 0) for k in metadata_keys
    )
    result["total_metadata_original"] = total_original
    result["total_metadata_preserved"] = total_preserved
    result["metadata_loss_rate"] = (
        1.0 - (total_preserved / total_original) if total_original > 0 else 0.0
    )

    # Data fidelity: does the semantic content survive?
    if rt["success"]:
        try:
            orig_data = _parse_to_python(text, fmt)
            rt_data = _parse_to_python(rt["re_emitted_text"], fmt)
            result["data_fidelity"] = _deep_equal(orig_data, rt_data)
        except Exception:
            result["data_fidelity"] = False
    else:
        result["data_fidelity"] = False

    return result


def _deep_equal(a, b) -> bool:
    """Deep equality check tolerant of type differences (int vs float)."""
    if type(a) != type(b):
        # Tolerate int/float mismatch
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-10
        # Tolerate str(x) matches
        return str(a) == str(b)
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(corpus: list[dict]) -> dict:
    """Build the AI/ML feature preservation matrix.

    Tests each baseline's ability to preserve specific construct types.
    Returns: {baseline: {construct: True/False}}
    """
    # Test constructs with specific files
    construct_tests = {
        "HP comments": ("yaml", None),  # any YAML with comments
        "Config anchors/aliases": ("yaml", None),  # YAML with anchors
        "Training timestamps": ("toml", None),  # TOML with temporal
        "Multi-doc streams": ("yaml", None),  # multi-doc YAML
        "XML namespaces": ("xml", None),  # XML with namespaces
        "XML comments & PIs": ("xml", None),  # XML with comments/PIs
        "Non-string keys": ("yaml", None),
        "Round-trip fidelity": ("json", None),  # data survives
        "Cross-language safe": (None, None),  # boolean property
    }

    # Find test files from corpus
    yaml_with_comments = next(
        (e for e in corpus if e["format"] == "yaml"
         and count_constructs(e["text"], "yaml")["comments"] > 0), None
    )
    yaml_with_anchors = next(
        (e for e in corpus if e["format"] == "yaml"
         and count_constructs(e["text"], "yaml")["anchors"] > 0), None
    )
    yaml_multi_doc = next(
        (e for e in corpus if e["format"] == "yaml"
         and count_constructs(e["text"], "yaml")["multi_doc_markers"] > 1), None
    )
    toml_with_temporal = next(
        (e for e in corpus if e["format"] == "toml"
         and count_constructs(e["text"], "toml")["temporal_values"] > 0), None
    )
    xml_with_ns = next(
        (e for e in corpus if e["format"] == "xml"
         and count_constructs(e["text"], "xml")["namespaces"] > 0), None
    )
    json_file = next(
        (e for e in corpus if e["format"] == "json"), None
    )

    matrix = {}
    for baseline_name in BASELINES:
        row = {}

        # HP comments
        if yaml_with_comments:
            f = measure_fidelity(yaml_with_comments["text"], "yaml", baseline_name)
            row["HP comments"] = f["comments_preserved"] > 0
        else:
            row["HP comments"] = False

        # Config anchors
        if yaml_with_anchors:
            f = measure_fidelity(yaml_with_anchors["text"], "yaml", baseline_name)
            row["Config anchors/aliases"] = f["anchors_preserved"] > 0
        else:
            row["Config anchors/aliases"] = False

        # Training timestamps
        if toml_with_temporal:
            f = measure_fidelity(toml_with_temporal["text"], "toml", baseline_name)
            row["Training timestamps"] = f["temporal_values_preserved"] > 0
        else:
            row["Training timestamps"] = False

        # Multi-doc streams
        if yaml_multi_doc:
            f = measure_fidelity(yaml_multi_doc["text"], "yaml", baseline_name)
            row["Multi-doc streams"] = f["multi_doc_markers_preserved"] > 1
        else:
            row["Multi-doc streams"] = False

        # XML namespaces
        if xml_with_ns:
            f = measure_fidelity(xml_with_ns["text"], "xml", baseline_name)
            row["XML namespaces"] = f["namespaces_preserved"] > 0
        else:
            row["XML namespaces"] = False

        # XML comments & PIs
        if xml_with_ns:
            f = measure_fidelity(xml_with_ns["text"], "xml", baseline_name)
            row["XML comments & PIs"] = (
                f["comments_preserved"] > 0 or
                f["processing_instructions_preserved"] > 0
            )
        else:
            row["XML comments & PIs"] = False

        # Round-trip data fidelity (on JSON — no metadata to lose)
        if json_file:
            f = measure_fidelity(json_file["text"], "json", baseline_name)
            row["Round-trip fidelity"] = f["data_fidelity"]
        else:
            row["Round-trip fidelity"] = False

        # Cross-language safe (inherent property, not measured)
        row["Cross-language safe"] = baseline_name != "pickle"

        matrix[baseline_name] = row

    return matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXP-006: ML Configuration Fidelity Under Serialization")
    print("=" * 70)
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python: {sys.version}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build corpus
    print("Building ML configuration corpus...")
    corpus = build_ml_corpus()
    print(f"  {len(corpus)} files across formats: "
          f"{dict(Counter(e['format'] for e in corpus))}")

    # Count total constructs
    for entry in corpus:
        entry["_constructs"] = count_constructs(entry["text"], entry["format"])
        entry["_total_metadata"] = total_metadata_constructs(
            entry["text"], entry["format"]
        )
    total_constructs = sum(e["_total_metadata"] for e in corpus)
    print(f"  Total metadata constructs in corpus: {total_constructs}")

    # Run fidelity measurements
    print("\nRunning serialization round-trips...")
    results = []

    for entry in corpus:
        for baseline_name in BASELINES:
            print(f"  {entry['name']} × {baseline_name}...", end=" ")

            fidelity = measure_fidelity(entry["text"], entry["format"], baseline_name)

            row = {
                "file": entry["name"],
                "format": entry["format"],
                "category": entry["category"],
                "baseline": baseline_name,
                "baseline_label": BASELINES[baseline_name],
                **fidelity,
            }
            results.append(row)

            status = "✓" if fidelity["success"] else "✗"
            loss = fidelity["metadata_loss_rate"]
            print(f"{status} loss={loss:.1%}")

    # Write per-file results CSV
    csv_path = RESULTS_DIR / "fidelity_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPer-file results: {csv_path}")

    # Aggregate statistics per baseline
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS: Metadata Loss Rate by Baseline")
    print("=" * 70)

    aggregate = {}
    for baseline_name, label in BASELINES.items():
        baseline_results = [r for r in results if r["baseline"] == baseline_name]
        loss_rates = [r["metadata_loss_rate"] for r in baseline_results
                      if r["total_metadata_original"] > 0]
        fidelity_rates = [r["data_fidelity"] for r in baseline_results]
        sizes = [r["serialized_size"] for r in baseline_results if r["success"]]
        orig_sizes = [r["original_size"] for r in baseline_results if r["success"]]

        agg = {
            "baseline": baseline_name,
            "label": label,
            "n_files": len(baseline_results),
            "n_success": sum(1 for r in baseline_results if r["success"]),
            "median_loss_rate": statistics.median(loss_rates) if loss_rates else None,
            "mean_loss_rate": statistics.mean(loss_rates) if loss_rates else None,
            "data_fidelity_rate": (
                sum(fidelity_rates) / len(fidelity_rates)
                if fidelity_rates else 0
            ),
            "median_size_ratio": (
                statistics.median(s / o for s, o in zip(sizes, orig_sizes))
                if sizes and orig_sizes else None
            ),
        }
        aggregate[baseline_name] = agg

        lr_str = f"{agg['median_loss_rate']:.1%}" if agg['median_loss_rate'] is not None else "N/A"
        df_str = f"{agg['data_fidelity_rate']:.1%}"
        sr_str = f"{agg['median_size_ratio']:.3f}" if agg['median_size_ratio'] is not None else "N/A"
        print(f"  {label:20s}  loss={lr_str:>6s}  fidelity={df_str:>6s}  size_ratio={sr_str}")

    # Write aggregate
    agg_path = RESULTS_DIR / "aggregate_results.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"\nAggregate results: {agg_path}")

    # Build and print feature matrix
    print("\n" + "=" * 70)
    print("AI/ML FEATURE PRESERVATION MATRIX")
    print("=" * 70)

    matrix = build_feature_matrix(corpus)
    constructs = list(next(iter(matrix.values())).keys())

    # Header
    header = f"{'Construct':30s}"
    for bl in BASELINES:
        header += f" {bl:>10s}"
    print(header)
    print("-" * len(header))

    for construct in constructs:
        line = f"{construct:30s}"
        for bl in BASELINES:
            val = matrix[bl][construct]
            line += f" {'✓':>10s}" if val else f" {'✗':>10s}"
        print(line)

    # Save matrix
    matrix_path = RESULTS_DIR / "feature_matrix.json"
    with open(matrix_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"\nFeature matrix: {matrix_path}")

    # Summary
    print("\n" + "=" * 70)
    print("HEADLINE FINDINGS")
    print("=" * 70)

    cdxf_agg = aggregate.get("cdxf", {})
    print(f"CDXF metadata loss rate: {cdxf_agg.get('median_loss_rate', 'N/A')}")
    print(f"CDXF data fidelity: {cdxf_agg.get('data_fidelity_rate', 'N/A')}")
    print(f"CDXF features preserved: {sum(matrix.get('cdxf', {}).values())}/{len(constructs)}")

    for bl, label in BASELINES.items():
        if bl == "cdxf":
            continue
        agg = aggregate.get(bl, {})
        feats = sum(matrix.get(bl, {}).values())
        print(f"{label}: loss={agg.get('median_loss_rate', 'N/A')}, "
              f"features={feats}/{len(constructs)}")

    print("\nDone.")


# For convenience
from collections import Counter

if __name__ == "__main__":
    main()
