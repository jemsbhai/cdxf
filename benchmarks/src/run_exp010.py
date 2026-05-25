"""
EXP-010: AI/ML Configuration Throughput Benchmark

Extends EXP-002 to AI/ML-specific workloads. Adds Pickle as a baseline
(the de facto ML serialization) and measures the "format tax" — config
serialization overhead relative to actual training time.

Statistical protocol: 10 warm-up, 1000 measurement iterations,
gc.disable() during measurement, time.perf_counter_ns().

Usage:
    python benchmarks/src/run_exp010.py
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import pickle
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import cbor2
    import msgpack
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)

try:
    import bson

    HAS_BSON = True
except ImportError:
    HAS_BSON = False

try:
    import amazon.ion.simpleion as ion

    HAS_ION = True
except ImportError:
    HAS_ION = False

try:
    from cdxf.bridges.json_bridge import from_json, to_json
    from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
    from cdxf.bridges.xml_bridge import from_xml, to_xml
    from cdxf.bridges.toml_bridge import from_toml, to_toml
    from cdxf.codec import encode as cdxf_encode, decode as cdxf_decode
except ImportError:
    print("ERROR: cdxf not installed. Run: pip install -e .")
    sys.exit(1)


# ===========================================================================
# Protocol constants
# ===========================================================================

WARMUP = 10
ITERATIONS = 1000

SIZE_THRESHOLDS = {
    "small_upper": 1024,      # <1KB
    "medium_upper": 10240,    # 1-10KB
}

FORMAT_TAX_T_STEPS = [0.2, 0.5]           # seconds per training step
FORMAT_TAX_STEPS = [100, 1000, 10000]     # steps per epoch

BASELINES = [
    "cdxf_full",
    "cdxf_codec",
    "json_stdlib",
    "cbor",
    "msgpack",
    "bson",
    "ion",
    "pickle",
]

BASELINE_LABELS = {
    "cdxf_full": "CDXF (full pipeline)",
    "cdxf_codec": "CDXF (codec only)",
    "json_stdlib": "JSON (stdlib)",
    "cbor": "CBOR (cbor2)",
    "msgpack": "MsgPack",
    "bson": "BSON (pymongo)",
    "ion": "Amazon Ion",
    "pickle": "Pickle (stdlib)",
}

_BRIDGES_FROM = {
    "json": from_json,
    "yaml": from_yaml,
    "xml": from_xml,
    "toml": from_toml,
}

_BRIDGES_TO = {
    "json": lambda s: to_json(s, indent=2),
    "yaml": to_yaml,
    "xml": to_xml,
    "toml": to_toml,
}


# ===========================================================================
# Size classification
# ===========================================================================


def classify_size(size_bytes: int) -> str:
    """Classify file size into small/medium/large bins.

    Protocol: small (<1KB), medium (1-10KB), large (>=10KB).
    """
    if size_bytes < SIZE_THRESHOLDS["small_upper"]:
        return "small"
    elif size_bytes < SIZE_THRESHOLDS["medium_upper"]:
        return "medium"
    else:
        return "large"


# ===========================================================================
# Statistical summary
# ===========================================================================


def summarize_stats(times: list[float]) -> dict:
    """Compute summary statistics for a list of timing measurements.

    Returns: n, median, mean, std, 95% CI, min, max.
    Protocol: NO outlier removal. All data reported.
    """
    n = len(times)
    med = statistics.median(times)
    avg = statistics.mean(times)

    if n > 1:
        std = statistics.stdev(times)
        ci_half = 1.96 * std / math.sqrt(n)
    else:
        std = 0.0
        ci_half = 0.0

    return {
        "n": n,
        "median_s": round(med, 12),
        "mean_s": round(avg, 12),
        "std_s": round(std, 12),
        "ci95_lower_s": round(avg - ci_half, 12),
        "ci95_upper_s": round(avg + ci_half, 12),
        "min_s": round(min(times), 12),
        "max_s": round(max(times), 12),
    }


# ===========================================================================
# Bench function (timing with GC control)
# ===========================================================================


def bench(func, iterations: int = ITERATIONS, warmup: int = WARMUP) -> list[float]:
    """Time a function over multiple iterations with GC control.

    Protocol:
    - `warmup` calls are executed and discarded.
    - `iterations` measurement calls are timed with time.perf_counter_ns().
    - gc.disable() during measurement, gc.enable() after.
    - Returns list of seconds (float), length == iterations.
    """
    # Warm-up phase
    for _ in range(warmup):
        func()

    # Measurement phase
    gc.disable()
    try:
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            func()
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e9)
    finally:
        gc.enable()

    return times


# ===========================================================================
# Native parsing (for baseline formats)
# ===========================================================================


def _to_native(text: str, fmt: str):
    """Parse text to plain Python types for baseline encoding."""
    try:
        if fmt == "json":
            return json.loads(text)
        if fmt == "yaml":
            from ruamel.yaml import YAML

            y = YAML(typ="safe")
            docs = list(y.load_all(text))
            return docs[0] if len(docs) == 1 else docs
        if fmt == "toml":
            import tomlkit

            return tomlkit.parse(text).unwrap()
        if fmt == "xml":
            # XML doesn't have a natural Python-dict mapping for baselines.
            # Use a simple dict representation.
            import xml.etree.ElementTree as ET

            root = ET.fromstring(text)
            return _xml_elem_to_dict(root)
    except Exception:
        return None
    return None


def _xml_elem_to_dict(elem):
    """Convert XML element to a plain dict."""
    d = {"_tag": elem.tag}
    if elem.attrib:
        d["_attrib"] = dict(elem.attrib)
    if elem.text and elem.text.strip():
        d["_text"] = elem.text.strip()
    children = [_xml_elem_to_dict(ch) for ch in elem]
    if children:
        d["_children"] = children
    return d


# ===========================================================================
# Encode/decode function registry
# ===========================================================================


def encode_decode_fns(text: str, fmt: str) -> dict:
    """Build a dict of {baseline: {encode: callable, decode: callable}}.

    Each encode callable returns the serialized form (bytes or str).
    Each decode callable returns the deserialized form.
    Baselines that cannot handle the format are omitted (not errored).
    """
    fns = {}

    # --- CDXF full pipeline: text → bridge → stream → binary ---
    bridge_from = _BRIDGES_FROM.get(fmt)
    bridge_to = _BRIDGES_TO.get(fmt)

    if bridge_from and bridge_to:
        # Pre-parse for decode benchmark
        try:
            stream = bridge_from(text)
            binary = cdxf_encode(stream)

            def _cdxf_full_enc(t=text, bf=bridge_from):
                return cdxf_encode(bf(t))

            def _cdxf_full_dec(b=binary, bt=bridge_to):
                return bt(cdxf_decode(b))

            fns["cdxf_full"] = {"encode": _cdxf_full_enc, "decode": _cdxf_full_dec}
        except Exception:
            pass

    # --- CDXF codec only: stream → binary ---
    if bridge_from:
        try:
            stream = bridge_from(text)
            binary = cdxf_encode(stream)

            def _cdxf_codec_enc(s=stream):
                return cdxf_encode(s)

            def _cdxf_codec_dec(b=binary):
                return cdxf_decode(b)

            fns["cdxf_codec"] = {"encode": _cdxf_codec_enc, "decode": _cdxf_codec_dec}
        except Exception:
            pass

    # --- Native Python data for all non-CDXF baselines ---
    native = _to_native(text, fmt)

    # --- JSON stdlib ---
    if native is not None:
        try:
            json_encoded = json.dumps(native, default=str).encode("utf-8")

            def _json_enc(n=native):
                return json.dumps(n, default=str)

            def _json_dec(e=json_encoded):
                return json.loads(e)

            fns["json_stdlib"] = {"encode": _json_enc, "decode": _json_dec}
        except Exception:
            pass

    # --- CBOR ---
    if native is not None:
        try:
            cbor_encoded = cbor2.dumps(native)

            def _cbor_enc(n=native):
                return cbor2.dumps(n)

            def _cbor_dec(e=cbor_encoded):
                return cbor2.loads(e)

            fns["cbor"] = {"encode": _cbor_enc, "decode": _cbor_dec}
        except Exception:
            pass

    # --- MsgPack ---
    if native is not None:
        try:
            mp_encoded = msgpack.packb(native, use_bin_type=True, default=str)

            def _mp_enc(n=native):
                return msgpack.packb(n, use_bin_type=True, default=str)

            def _mp_dec(e=mp_encoded):
                return msgpack.unpackb(e, raw=False)

            fns["msgpack"] = {"encode": _mp_enc, "decode": _mp_dec}
        except Exception:
            pass

    # --- BSON ---
    if HAS_BSON and native is not None and isinstance(native, dict):
        try:
            bson_encoded = bson.encode(native)

            def _bson_enc(n=native):
                return bson.encode(n)

            def _bson_dec(e=bson_encoded):
                return bson.decode(e)

            fns["bson"] = {"encode": _bson_enc, "decode": _bson_dec}
        except Exception:
            pass

    # --- Amazon Ion ---
    if HAS_ION and native is not None:
        try:
            ion_encoded = ion.dumps(native, binary=True)

            def _ion_enc(n=native):
                return ion.dumps(n, binary=True)

            def _ion_dec(e=ion_encoded):
                return ion.loads(e)

            fns["ion"] = {"encode": _ion_enc, "decode": _ion_dec}
        except Exception:
            pass

    # --- Pickle ---
    if native is not None:
        try:
            pkl_encoded = pickle.dumps(native)

            def _pkl_enc(n=native):
                return pickle.dumps(n)

            def _pkl_dec(e=pkl_encoded):
                return pickle.loads(e)

            fns["pickle"] = {"encode": _pkl_enc, "decode": _pkl_dec}
        except Exception:
            pass

    return fns


# ===========================================================================
# Format tax computation
# ===========================================================================


def compute_format_tax(t_config_s: float, steps_per_epoch: int,
                       t_step_s: float) -> float:
    """Compute the format tax fraction.

    format_tax = T_config / (steps_per_epoch * T_step)

    This is the fraction of total epoch training time spent on
    config serialization (assuming config is serialized once per epoch).
    """
    denominator = steps_per_epoch * t_step_s
    if denominator == 0:
        return 0.0
    return t_config_s / denominator


# ===========================================================================
# Environment capture
# ===========================================================================


def capture_environment() -> dict:
    """Capture the runtime environment for reproducibility."""
    import importlib.metadata

    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "warmup": WARMUP,
        "iterations": ITERATIONS,
    }
    for pkg in [
        "cbor2", "msgpack", "ruamel.yaml", "tomlkit",
        "pymongo", "amazon.ion", "cdxf",
    ]:
        try:
            env[f"pkg_{pkg}"] = importlib.metadata.version(pkg)
        except Exception:
            env[f"pkg_{pkg}"] = "not installed"
    return env


# ===========================================================================
# ML corpus
# ===========================================================================


def build_ml_corpus() -> list[dict]:
    """Build a corpus of realistic ML configuration files.

    Each entry: {name, format, text, category}.
    Covers JSON, YAML, TOML, XML with small/medium/large size bins.
    """
    corpus = []

    # -------------------------------------------------------------------
    # YAML — training configs with comments
    # -------------------------------------------------------------------
    corpus.append({
        "name": "hydra_train_bert",
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
  seed: 42

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
        "name": "lightning_anchored_config",
        "format": "yaml",
        "category": "training",
        "text": """\
# PyTorch Lightning configuration with anchors
defaults: &defaults
  precision: 16
  accelerator: gpu
  devices: 1
  max_epochs: 10
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4
  log_every_n_steps: 50

experiment_small:
  <<: *defaults
  max_epochs: 5
  accumulate_grad_batches: 2

experiment_large:
  <<: *defaults
  max_epochs: 20
  devices: 4
  accumulate_grad_batches: 1

callbacks:
  early_stopping:
    monitor: val_loss
    patience: 3
    mode: min
  model_checkpoint:
    monitor: val_loss
    save_top_k: 3
    save_weights_only: true
""",
    })

    corpus.append({
        "name": "wandb_sweep_config",
        "format": "yaml",
        "category": "experiment",
        "text": """\
# W&B Hyperparameter Sweep — Bayesian optimization
program: train.py
method: bayes
metric:
  name: eval/loss
  goal: minimize

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001
  weight_decay:
    values: [0.0, 0.01, 0.05, 0.1]
  warmup_ratio:
    distribution: uniform
    min: 0.0
    max: 0.2
  lora_r:
    values: [8, 16, 32, 64, 128]
  lora_alpha:
    values: [16, 32, 64, 128, 256]
  lora_dropout:
    values: [0.0, 0.05, 0.1]

early_terminate:
  type: hyperband
  min_iter: 100
  eta: 3
""",
    })

    corpus.append({
        "name": "k8s_ml_serving",
        "format": "yaml",
        "category": "deployment",
        "text": """\
# Kubernetes deployment for vLLM model serving
# Requires GPU node pool >= 24GB VRAM
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
  namespace: ml-inference
  labels:
    app: llm-serving
    version: "llama-3.1-8b-v2"
spec:
  replicas: 2
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
          args:
            - --model
            - meta-llama/Llama-3.1-8B-Instruct
            - --gpu-memory-utilization
            - "0.9"
            - --max-model-len
            - "4096"
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              memory: "32Gi"
              cpu: "4"
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 10
""",
    })

    corpus.append({
        "name": "dataset_card_yaml",
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
pretty_name: MultiTask NLP Corpus v2
""",
    })

    # -------------------------------------------------------------------
    # JSON — model configs, adapter configs, eval results, training args
    # -------------------------------------------------------------------
    corpus.append({
        "name": "hf_model_config",
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
                "rope_type": "llama3",
            },
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 128256,
        }, indent=2),
    })

    corpus.append({
        "name": "adapter_config",
        "format": "json",
        "category": "adapter",
        "text": json.dumps({
            "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "peft_type": "LORA",
            "r": 64,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "task_type": "CAUSAL_LM",
        }, indent=2),
    })

    corpus.append({
        "name": "eval_results",
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
                "batch_size": "auto",
                "device": "cuda",
                "num_fewshot": 5,
            },
            "versions": {
                "lm_eval": "0.4.2",
                "transformers": "4.44.0",
                "torch": "2.4.0",
            },
        }, indent=2),
    })

    corpus.append({
        "name": "training_args",
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

    # Small JSON (< 1KB guaranteed)
    corpus.append({
        "name": "tokenizer_config_small",
        "format": "json",
        "category": "model",
        "text": json.dumps({
            "add_bos_token": True,
            "add_eos_token": False,
            "model_max_length": 131072,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": True,
        }, indent=2),
    })

    # -------------------------------------------------------------------
    # TOML — packaging and serving configs
    # -------------------------------------------------------------------
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
# License changed from GPL to Apache after legal review
license = "Apache-2.0"
requires-python = ">=3.10"

[project.dependencies]
# Pin transformers to avoid breaking tokenizer API changes
transformers = ">=4.40.0,<4.50.0"
torch = ">=2.2.0"
peft = ">=0.10.0"
datasets = ">=2.18.0"
accelerate = ">=0.28.0"
bitsandbytes = ">=0.43.0"

[project.optional-dependencies]
eval = ["lm-eval>=0.4.0", "vllm>=0.4.0"]
dev = ["pytest", "ruff", "mypy"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
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
[server]
host = "0.0.0.0"
port = 8000
# Timeout increased from 30s to 120s for long generation
timeout = 120

[model]
name = "meta-llama/Llama-3.1-8B-Instruct"
# Quantization: AWQ gives best quality/speed tradeoff
quantization = "awq"
max_model_len = 4096
gpu_memory_utilization = 0.90
dtype = "auto"

[generation]
max_tokens = 2048
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.1

[logging]
level = "INFO"
format = "json"
""",
    })

    # -------------------------------------------------------------------
    # XML — model metadata
    # -------------------------------------------------------------------
    corpus.append({
        "name": "pmml_model_metadata",
        "format": "xml",
        "category": "model",
        "text": """\
<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">
  <Header copyright="ML Research Lab 2024">
    <Application name="sklearn2pmml" version="0.90.0"/>
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

    # -------------------------------------------------------------------
    # Large JSON (>10KB) — comprehensive eval results
    # -------------------------------------------------------------------
    large_eval = {
        "results": {},
        "config": {
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "model_args": "pretrained=meta-llama/Llama-3.1-70B-Instruct,dtype=bfloat16",
            "batch_size": "auto:4",
            "device": "cuda",
            "num_fewshot": 5,
        },
        "versions": {
            "lm_eval": "0.4.2",
            "transformers": "4.44.0",
            "torch": "2.4.0",
            "vllm": "0.5.0",
        },
    }
    # Fill in enough benchmarks to push past 10KB
    benchmarks = [
        "mmlu", "mmlu_abstract_algebra", "mmlu_anatomy",
        "mmlu_astronomy", "mmlu_business_ethics", "mmlu_clinical_knowledge",
        "mmlu_college_biology", "mmlu_college_chemistry",
        "mmlu_college_computer_science", "mmlu_college_mathematics",
        "mmlu_college_medicine", "mmlu_college_physics",
        "mmlu_computer_security", "mmlu_conceptual_physics",
        "mmlu_econometrics", "mmlu_electrical_engineering",
        "mmlu_elementary_mathematics", "mmlu_formal_logic",
        "mmlu_global_facts", "mmlu_high_school_biology",
        "mmlu_high_school_chemistry", "mmlu_high_school_computer_science",
        "mmlu_high_school_european_history", "mmlu_high_school_geography",
        "mmlu_high_school_government_and_politics",
        "mmlu_high_school_macroeconomics", "mmlu_high_school_mathematics",
        "mmlu_high_school_microeconomics", "mmlu_high_school_physics",
        "mmlu_high_school_psychology", "mmlu_high_school_statistics",
        "mmlu_high_school_us_history", "mmlu_high_school_world_history",
        "mmlu_human_aging", "mmlu_human_sexuality",
        "mmlu_international_law", "mmlu_jurisprudence",
        "mmlu_logical_fallacies", "mmlu_machine_learning",
        "mmlu_management", "mmlu_marketing", "mmlu_medical_genetics",
        "mmlu_miscellaneous", "mmlu_moral_disputes", "mmlu_moral_scenarios",
        "mmlu_nutrition", "mmlu_philosophy", "mmlu_prehistory",
        "mmlu_professional_accounting", "mmlu_professional_law",
        "mmlu_professional_medicine", "mmlu_professional_psychology",
        "mmlu_public_relations", "mmlu_security_studies",
        "mmlu_sociology", "mmlu_us_foreign_policy", "mmlu_virology",
        "mmlu_world_religions",
        "hellaswag", "arc_easy", "arc_challenge",
        "truthfulqa_mc1", "truthfulqa_mc2",
        "winogrande", "gsm8k", "drop",
    ]
    for i, bm in enumerate(benchmarks):
        large_eval["results"][bm] = {
            "acc": round(0.3 + 0.5 * (i % 7) / 7, 4),
            "acc_stderr": round(0.001 + 0.01 * (i % 5) / 5, 4),
            "acc_norm": round(0.35 + 0.5 * ((i + 3) % 7) / 7, 4),
            "acc_norm_stderr": round(0.002 + 0.01 * ((i + 2) % 5) / 5, 4),
            "samples": 100 + (i * 37) % 900,
            "alias": f"{bm}_5shot",
        }
    corpus.append({
        "name": "large_eval_results",
        "format": "json",
        "category": "evaluation",
        "text": json.dumps(large_eval, indent=2),
    })

    # -------------------------------------------------------------------
    # Large YAML — comprehensive pipeline config (>10KB)
    # -------------------------------------------------------------------
    large_yaml_parts = [
        "# Comprehensive ML pipeline configuration\n",
        "# Spans data preprocessing through deployment\n\n",
    ]
    for stage_i in range(8):
        stage_name = [
            "data_loading", "preprocessing", "tokenization",
            "model_init", "training", "evaluation", "export", "deployment",
        ][stage_i]
        large_yaml_parts.append(f"# --- Stage {stage_i + 1}: {stage_name} ---\n")
        large_yaml_parts.append(f"{stage_name}:\n")
        for p in range(30):
            large_yaml_parts.append(f"  param_{p}: {p * 0.1 + stage_i}\n")
            if p % 3 == 0:
                large_yaml_parts.append(f"  # Tuned on 2024-11-{10 + (p % 20)}\n")
        large_yaml_parts.append("\n")
    corpus.append({
        "name": "large_pipeline_yaml",
        "format": "yaml",
        "category": "training",
        "text": "".join(large_yaml_parts),
    })

    return corpus


# ===========================================================================
# Single-file benchmark runner
# ===========================================================================


def run_single_file(entry: dict, iterations: int = ITERATIONS,
                    warmup: int = WARMUP) -> dict:
    """Benchmark one corpus entry across all baselines.

    Returns a dict with file metadata and per-baseline timing stats.
    """
    text = entry["text"]
    fmt = entry["format"]
    size_bytes = len(text.encode("utf-8"))

    result = {
        "name": entry["name"],
        "format": fmt,
        "category": entry["category"],
        "size_bytes": size_bytes,
        "size_category": classify_size(size_bytes),
        "baselines": {},
    }

    fns = encode_decode_fns(text, fmt)

    for bl_name in BASELINES:
        gc.collect()

        if bl_name not in fns:
            result["baselines"][bl_name] = {"error": "unsupported_format"}
            continue

        ops = fns[bl_name]

        try:
            bl_result = {}

            for op_name in ("encode", "decode"):
                fn = ops[op_name]
                times = bench(fn, iterations=iterations, warmup=warmup)
                stats = summarize_stats(times)

                # Derived metrics
                med = stats["median_s"]
                if med > 0:
                    stats["ops_per_sec"] = round(1.0 / med, 2)
                    stats["throughput_bytes_per_sec"] = round(size_bytes / med)
                else:
                    stats["ops_per_sec"] = float("inf")
                    stats["throughput_bytes_per_sec"] = float("inf")

                # Format tax (encode only — that's the serialization cost)
                if op_name == "encode":
                    ft = {}
                    for t_step in FORMAT_TAX_T_STEPS:
                        for steps in FORMAT_TAX_STEPS:
                            key = f"tstep_{t_step}_steps_{steps}"
                            ft[key] = compute_format_tax(med, steps, t_step)
                    stats["format_tax"] = ft

                bl_result[op_name] = stats

            result["baselines"][bl_name] = bl_result

        except Exception as e:
            result["baselines"][bl_name] = {"error": str(e)}

    return result


# ===========================================================================
# Full experiment
# ===========================================================================


def run_experiment(output_dir: Path | str | None = None,
                   iterations: int = ITERATIONS,
                   warmup: int = WARMUP) -> dict:
    """Run the full EXP-010 benchmark.

    Args:
        output_dir: Directory for results. Defaults to benchmarks/results/exp_010.
        iterations: Number of measurement iterations per benchmark.
        warmup: Number of warm-up iterations.

    Returns:
        Complete results dict (also written to output_dir).
    """
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_010")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-010: AI/ML Configuration Throughput Benchmark")
    print(f"Warm-up: {warmup}, Iterations: {iterations}")
    print("=" * 70)

    # Environment
    env = capture_environment()
    env_path = output_dir / "environment.json"
    env_path.write_text(json.dumps(env, indent=2), encoding="utf-8")

    # Build corpus
    print("\nBuilding ML configuration corpus...")
    corpus = build_ml_corpus()
    print(f"  {len(corpus)} files")
    for entry in corpus:
        sz = len(entry["text"].encode("utf-8"))
        print(f"    {entry['name']:35s} {entry['format']:5s} "
              f"{sz:>6,}B ({classify_size(sz)})")

    # Run benchmarks
    print(f"\nBenchmarking ({iterations} iterations, {warmup} warm-up)...")
    results = []

    for entry in corpus:
        print(f"\n  {entry['name']}...", flush=True)
        file_result = run_single_file(entry, iterations=iterations,
                                      warmup=warmup)
        results.append(file_result)

        # Print summary line
        for bl in BASELINES:
            bl_data = file_result["baselines"].get(bl, {})
            if "error" in bl_data:
                continue
            enc = bl_data.get("encode", {})
            ops = enc.get("ops_per_sec", 0)
            med_us = enc.get("median_s", 0) * 1e6
            print(f"    {bl:15s} enc={med_us:>8.1f}µs ({ops:>12,.0f} ops/s)")

    # --- Aggregate summary ---
    print(f"\n{'=' * 70}")
    print("AGGREGATE SUMMARY")
    print("=" * 70)

    aggregate = {}
    for bl_name in BASELINES:
        enc_ops = []
        dec_ops = []
        enc_throughput = []
        dec_throughput = []

        for r in results:
            bl_data = r["baselines"].get(bl_name, {})
            if "error" in bl_data:
                continue
            enc = bl_data.get("encode", {})
            dec = bl_data.get("decode", {})
            if "ops_per_sec" in enc and enc["ops_per_sec"] != float("inf"):
                enc_ops.append(enc["ops_per_sec"])
            if "ops_per_sec" in dec and dec["ops_per_sec"] != float("inf"):
                dec_ops.append(dec["ops_per_sec"])
            if "throughput_bytes_per_sec" in enc and enc["throughput_bytes_per_sec"] != float("inf"):
                enc_throughput.append(enc["throughput_bytes_per_sec"])
            if "throughput_bytes_per_sec" in dec and dec["throughput_bytes_per_sec"] != float("inf"):
                dec_throughput.append(dec["throughput_bytes_per_sec"])

        agg_entry = {
            "n_files": len(enc_ops),
        }
        if enc_ops:
            agg_entry["encode_median_ops_per_sec"] = round(statistics.median(enc_ops), 1)
            agg_entry["encode_mean_ops_per_sec"] = round(statistics.mean(enc_ops), 1)
        if dec_ops:
            agg_entry["decode_median_ops_per_sec"] = round(statistics.median(dec_ops), 1)
            agg_entry["decode_mean_ops_per_sec"] = round(statistics.mean(dec_ops), 1)
        if enc_throughput:
            agg_entry["encode_median_bytes_per_sec"] = round(
                statistics.median(enc_throughput))
        if dec_throughput:
            agg_entry["decode_median_bytes_per_sec"] = round(
                statistics.median(dec_throughput))

        aggregate[bl_name] = agg_entry

        label = BASELINE_LABELS.get(bl_name, bl_name)
        enc_med = agg_entry.get("encode_median_ops_per_sec", "N/A")
        dec_med = agg_entry.get("decode_median_ops_per_sec", "N/A")
        n = agg_entry["n_files"]
        if isinstance(enc_med, (int, float)):
            print(f"  {label:25s} enc={enc_med:>12,.0f} ops/s  "
                  f"dec={dec_med:>12,.0f} ops/s  (n={n})")
        else:
            print(f"  {label:25s} no data (n={n})")

    # --- Format tax summary ---
    print(f"\n{'=' * 70}")
    print("FORMAT TAX ANALYSIS")
    print("-" * 70)
    print(f"{'Baseline':20s} {'T_step':>8s} {'Steps':>8s} {'Median Tax':>14s}")
    print("-" * 70)

    format_tax_rows = []

    for bl_name in BASELINES:
        for t_step in FORMAT_TAX_T_STEPS:
            for steps in FORMAT_TAX_STEPS:
                key = f"tstep_{t_step}_steps_{steps}"
                taxes = []
                for r in results:
                    bl_data = r["baselines"].get(bl_name, {})
                    if "error" in bl_data:
                        continue
                    enc = bl_data.get("encode", {})
                    ft = enc.get("format_tax", {})
                    if key in ft:
                        taxes.append(ft[key])

                if taxes:
                    median_tax = statistics.median(taxes)
                    mean_tax = statistics.mean(taxes)
                else:
                    median_tax = None
                    mean_tax = None

                row = {
                    "baseline": bl_name,
                    "t_step_s": t_step,
                    "steps_per_epoch": steps,
                    "format_tax": median_tax,
                    "format_tax_mean": mean_tax,
                    "n_files": len(taxes),
                }
                format_tax_rows.append(row)

                if median_tax is not None:
                    print(f"  {bl_name:20s} {t_step:>8.1f}s {steps:>8,d} "
                          f"{median_tax:>14.2e}")

    # --- Write outputs ---

    # 1. Full JSON results
    output = {
        "experiment": "EXP-010",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "corpus_size": len(results),
        "results": results,
        "aggregate": aggregate,
    }
    json_path = output_dir / "exp_010_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nFull results: {json_path}")

    # 2. Throughput CSV (flat: one row per file × baseline × operation)
    csv_path = output_dir / "throughput_results.csv"
    csv_rows = []
    for r in results:
        for bl_name, bl_data in r["baselines"].items():
            if "error" in bl_data:
                continue
            for op in ("encode", "decode"):
                if op not in bl_data:
                    continue
                stats = bl_data[op]
                csv_rows.append({
                    "name": r["name"],
                    "format": r["format"],
                    "category": r["category"],
                    "size_bytes": r["size_bytes"],
                    "size_category": r["size_category"],
                    "baseline": bl_name,
                    "operation": op,
                    "n": stats["n"],
                    "median_s": stats["median_s"],
                    "mean_s": stats["mean_s"],
                    "std_s": stats["std_s"],
                    "ci95_lower_s": stats["ci95_lower_s"],
                    "ci95_upper_s": stats["ci95_upper_s"],
                    "min_s": stats["min_s"],
                    "max_s": stats["max_s"],
                    "ops_per_sec": stats.get("ops_per_sec", ""),
                    "throughput_bytes_per_sec": stats.get(
                        "throughput_bytes_per_sec", ""),
                })
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Throughput CSV: {csv_path}")

    # 3. Format tax CSV
    tax_path = output_dir / "format_tax.csv"
    if format_tax_rows:
        fieldnames = list(format_tax_rows[0].keys())
        with open(tax_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(format_tax_rows)
    print(f"Format tax CSV: {tax_path}")

    print(f"\n{'=' * 70}")
    print("EXP-010 COMPLETE")
    print("=" * 70)

    return output


# ===========================================================================
# CLI entry point
# ===========================================================================


def main():
    run_experiment()


if __name__ == "__main__":
    main()
