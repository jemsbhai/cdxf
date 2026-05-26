"""
Shared config corpus for EXP-015, EXP-016, EXP-017 enhancements.

Provides:
- Multi-size YAML configs (small/medium/large/xlarge) with known comment counts
- Cross-format configs (JSON, XML, TOML) for EXP-016
- Data integrity verification (expected values after agent modifications)
- Timing utilities for overhead measurement
- Comment counting shared across experiments
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import yaml


# ===========================================================================
# YAML configs — graduated sizes for scaling analysis
# ===========================================================================

YAML_CONFIGS = {
    "small": {
        "name": "small",
        "description": "Minimal config — 4 comments, 4 keys",
        "text": (
            "# Learning rate setting\n"
            "learning_rate: 0.001\n"
            "# Seed for reproducibility\n"
            "seed: 42\n"
            "num_epochs: 3\n"
            "# Data path\n"
            "dataset: alpaca\n"
            "# Batch size\n"
            "batch_size: 4\n"
        ),
        "expected_comments": 4,
        "format": "yaml",
    },
    "medium": {
        "name": "medium",
        "description": "Typical config — 10 comments, ~15 keys",
        "text": (
            "# Training Configuration\n"
            "# Owner: ML Platform Team\n"
            "\n"
            "# Model settings\n"
            "model:\n"
            "  name: llama-2-7b-chat\n"
            "  # Base model from HuggingFace Hub\n"
            "  base_model: meta-llama/Llama-2-7b-hf\n"
            "  task_type: causal_lm\n"
            "\n"
            "# Training hyperparameters\n"
            "training:\n"
            "  # Grid search winner: [1e-5, 5e-5, 1e-4]\n"
            "  learning_rate: 2.0e-5\n"
            "  num_epochs: 3\n"
            "  per_device_batch_size: 4\n"
            "  # Effective batch = 4 * 8 = 32\n"
            "  gradient_accumulation_steps: 8\n"
            "  warmup_ratio: 0.03\n"
            "\n"
            "# Dataset\n"
            "data:\n"
            "  dataset: tatsu-lab/alpaca\n"
            "  max_length: 2048\n"
            "  num_proc: 8\n"
        ),
        "expected_comments": 8,
        "format": "yaml",
    },
    "large": {
        "name": "large",
        "description": "Full pipeline config — 22 comments (identical to EXP-015)",
        "text": (
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
        ),
        "expected_comments": 22,
        "format": "yaml",
    },
    "xlarge": {
        "name": "xlarge",
        "description": "Enterprise config — 42 comments, ~40 keys",
        "text": (
            "# ============================================================\n"
            "# Enterprise ML Pipeline Configuration\n"
            "# Organization: Acme ML Platform\n"
            "# Author: ML Infrastructure Team\n"
            "# Created: 2026-01-15\n"
            "# Last Modified: 2026-03-20\n"
            "# Review: Approved by ML Architecture Board\n"
            "# ============================================================\n"
            "\n"
            "# Model architecture and weights\n"
            "model:\n"
            "  name: llama-2-70b-chat\n"
            "  # Base model — 70B parameter variant\n"
            "  base_model: meta-llama/Llama-2-70b-hf\n"
            "  revision: main\n"
            "  # Task type determines output head\n"
            "  task_type: causal_lm\n"
            "  # Quantization for memory efficiency\n"
            "  quantization: bitsandbytes-4bit\n"
            "  # Device placement strategy\n"
            "  device_map: auto\n"
            "\n"
            "# LoRA adapter configuration\n"
            "lora:\n"
            "  # Rank — tradeoff between capacity and memory\n"
            "  r: 16\n"
            "  lora_alpha: 32\n"
            "  # Dropout for regularization\n"
            "  lora_dropout: 0.05\n"
            "  # Target modules for adaptation\n"
            "  target_modules:\n"
            "    - q_proj\n"
            "    - v_proj\n"
            "    - k_proj\n"
            "    - o_proj\n"
            "\n"
            "# Training hyperparameters\n"
            "training:\n"
            "  # Learning rate from Bayesian sweep\n"
            "  learning_rate: 1.5e-5\n"
            "  num_epochs: 5\n"
            "  per_device_batch_size: 2\n"
            "  # Gradient accumulation — effective batch 64\n"
            "  gradient_accumulation_steps: 16\n"
            "  warmup_ratio: 0.05\n"
            "  # Weight decay regularization\n"
            "  weight_decay: 0.01\n"
            "  # Seed for reproducibility\n"
            "  seed: 42\n"
            "  max_grad_norm: 1.0\n"
            "  # Mixed precision training\n"
            "  fp16: true\n"
            "  # Gradient checkpointing for memory\n"
            "  gradient_checkpointing: true\n"
            "\n"
            "# Dataset pipeline\n"
            "data:\n"
            "  # Primary instruction dataset\n"
            "  dataset: tatsu-lab/alpaca\n"
            "  # Supplementary datasets for diversity\n"
            "  supplementary:\n"
            "    - openassistant/oasst1\n"
            "    - sahil2801/CodeAlpaca-20k\n"
            "  max_length: 4096\n"
            "  # Train/validation split ratio\n"
            "  val_ratio: 0.05\n"
            "  num_proc: 16\n"
            "  # Data filtering thresholds\n"
            "  min_quality_score: 0.7\n"
            "\n"
            "# Evaluation configuration\n"
            "evaluation:\n"
            "  # Comprehensive benchmark suite\n"
            "  benchmarks:\n"
            "    - mmlu\n"
            "    - hellaswag\n"
            "    - arc_challenge\n"
            "    - truthfulqa\n"
            "    - winogrande\n"
            "  # Quality gates\n"
            "  min_accuracy: 0.55\n"
            "  # Evaluation frequency\n"
            "  eval_steps: 500\n"
            "\n"
            "# Deployment and serving\n"
            "deployment:\n"
            "  # vLLM serving configuration\n"
            "  engine: vllm\n"
            "  max_num_seqs: 128\n"
            "  gpu_memory_utilization: 0.92\n"
            "  # API endpoint settings\n"
            "  port: 8000\n"
            "  # Auto-scaling parameters\n"
            "  min_replicas: 2\n"
            "  max_replicas: 8\n"
            "  # Health check interval (seconds)\n"
            "  health_check_interval: 30\n"
        ),
        "expected_comments": 38,
        "format": "yaml",
    },
}


# ===========================================================================
# Cross-format configs for EXP-016
# ===========================================================================

XML_CONFIG = {
    "name": "xml_medium",
    "description": "XML config with comments — 6 comments",
    "text": (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!-- ML Pipeline Configuration -->\n"
        "<!-- Owner: ML Platform Team -->\n"
        "<config>\n"
        "  <!-- Model settings -->\n"
        "  <model>\n"
        "    <name>llama-2-7b-chat</name>\n"
        "    <!-- Base model from HuggingFace -->\n"
        "    <base_model>meta-llama/Llama-2-7b-hf</base_model>\n"
        "    <task_type>causal_lm</task_type>\n"
        "  </model>\n"
        "  <!-- Training hyperparameters -->\n"
        "  <training>\n"
        "    <!-- Grid search winner -->\n"
        "    <learning_rate>2.0e-5</learning_rate>\n"
        "    <num_epochs>3</num_epochs>\n"
        "    <batch_size>4</batch_size>\n"
        "  </training>\n"
        "</config>\n"
    ),
    "expected_comments": 6,
    "format": "xml",
}

TOML_CONFIG = {
    "name": "toml_medium",
    "description": "TOML config with comments — 8 comments",
    "text": (
        "# ML Pipeline Configuration\n"
        "# Owner: ML Platform Team\n"
        "\n"
        "# Model settings\n"
        "[model]\n"
        "name = \"llama-2-7b-chat\"\n"
        "# Base model from HuggingFace Hub\n"
        "base_model = \"meta-llama/Llama-2-7b-hf\"\n"
        "task_type = \"causal_lm\"\n"
        "\n"
        "# Training hyperparameters\n"
        "[training]\n"
        "# Grid search winner\n"
        "learning_rate = 2.0e-5\n"
        "num_epochs = 3\n"
        "batch_size = 4\n"
        "# Effective batch via accumulation\n"
        "gradient_accumulation_steps = 8\n"
    ),
    "expected_comments": 7,
    "format": "toml",
}


# ===========================================================================
# Agent modifications — deterministic parameter tweaks
# ===========================================================================

# Maps role → (regex_pattern, replacement_fn)
# Used by EXP-015, EXP-017 for consistent agent behavior
ROLE_MODIFICATIONS = {
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


# Expected values after applying all 4-agent modifications
# to the "large" config (the default pipeline)
EXPECTED_VALUES_AFTER_4AGENT = {
    "num_proc": 12,         # 8 + 4 (data_curator)
    "num_epochs": 4,        # 3 + 1 (trainer)
    "min_accuracy": 0.5,    # 0.45 + 0.05 (evaluator)
    "max_num_seqs": 80,     # 64 + 16 (deployer)
}

# Expected values after applying all 6-agent modifications
EXPECTED_VALUES_AFTER_6AGENT = {
    "num_proc": 12,         # 8 + 4 (data_curator)
    "num_epochs": 4,        # 3 + 1 (trainer)
    "min_accuracy": 0.5,    # 0.45 + 0.05 (evaluator)
    "max_num_seqs": 80,     # 64 + 16 (deployer)
    "port": 8001,           # 8000 + 1 (monitor)
    "seed": 43,             # 42 + 1 (reviewer)
}


def verify_data_integrity(
    config_text: str, expected_values: dict[str, Any]
) -> dict:
    """Verify that agent modifications were correctly applied.

    Parses the final config and checks that specific values match
    what we expect after the deterministic agent pipeline.

    Returns:
        {"passed": bool, "checks": int, "failures": list[str]}
    """
    try:
        parsed = yaml.safe_load(config_text)
    except yaml.YAMLError:
        return {"passed": False, "checks": 0,
                "failures": ["YAML parse error"]}

    if not isinstance(parsed, dict):
        return {"passed": False, "checks": 0,
                "failures": ["Not a dict"]}

    failures = []
    checks = 0

    # Flatten nested config for lookup
    flat = _flatten_dict(parsed)

    for key, expected in expected_values.items():
        checks += 1
        actual = flat.get(key)
        if actual is None:
            failures.append(f"{key}: missing (expected {expected})")
        elif isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-6:
                failures.append(
                    f"{key}: {actual} != {expected}"
                )
        elif int(actual) != int(expected):
            failures.append(f"{key}: {actual} != {expected}")

    return {
        "passed": len(failures) == 0,
        "checks": checks,
        "failures": failures,
    }


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict to {leaf_key: value}."""
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            items.update(_flatten_dict(v, f"{prefix}{k}."))
        else:
            items[k] = v
            # Also store without prefix for simple lookups
            if prefix:
                items[k] = v
    return items


# ===========================================================================
# Comment counting (shared)
# ===========================================================================


def count_config_metadata(text: str, fmt: str) -> dict:
    """Count metadata constructs in config text.

    Supports yaml, toml (# comments) and xml (<!-- --> comments).
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
    elif fmt == "xml":
        import re
        comments = len(re.findall(r"<!--.*?-->", text, re.DOTALL))

    return {"comments": comments, "total": comments}


# ===========================================================================
# Timing utilities
# ===========================================================================


class Timer:
    """Context manager for precise timing of operations.

    Usage:
        timer = Timer()
        with timer.measure("encode"):
            encode(data)
        with timer.measure("decode"):
            decode(data)
        print(timer.results)  # {"encode": 0.00123, "decode": 0.00098}
    """

    def __init__(self):
        self.results: dict[str, float] = {}
        self._start: float = 0.0
        self._label: str = ""

    @contextmanager
    def measure(self, label: str):
        """Measure wall-clock time for a block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.results[label] = elapsed

    def measure_n(self, label: str, fn, n: int = 10) -> float:
        """Run fn() n times and record the mean time.

        Returns mean time in seconds.
        """
        times = []
        for _ in range(n):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        mean_t = sum(times) / len(times)
        self.results[label] = mean_t
        return mean_t

    def summary(self) -> dict[str, float]:
        """Return all recorded timings."""
        return dict(self.results)
