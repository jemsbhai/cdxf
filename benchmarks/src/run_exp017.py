"""
EXP-017: CrewAI Pipeline — ML Config Handoff Fidelity

Build a real CrewAI crew with FakeLLM where ML agents pass annotated
YAML configs as task outputs. CrewAI serializes task results as strings
via TaskOutput.raw. Compare:
  - json_default: agent parses YAML → dict, modifies, re-emits → comments lost
  - cdxf_enhanced: agent encodes to CDXF, passes base64, decodes → comments preserved

Uses a deterministic FakeLLM (subclass of BaseLLM) so results are
100% reproducible without API calls.

Usage:
    python benchmarks/src/run_exp017.py
"""

from __future__ import annotations

import base64
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.llms.base_llm import BaseLLM
except ImportError:
    print("ERROR: crewai not installed. Run: pip install crewai")
    sys.exit(1)

try:
    from cdxf.bridges.yaml_bridge import from_yaml, to_yaml
    from cdxf.codec import encode, decode
except ImportError:
    print("ERROR: cdxf not installed. Run: pip install -e .")
    sys.exit(1)


# ===========================================================================
# Protocol constants
# ===========================================================================

STATE_MODES = ["json_default", "cdxf_enhanced"]

CREW_CONFIGS = [
    {
        "name": "linear_4agent",
        "description": "Linear pipeline: curator → trainer → evaluator → deployer",
        "roles": ["data_curator", "trainer", "evaluator", "deployer"],
    },
    {
        "name": "linear_6agent",
        "description": "Extended: curator → trainer → evaluator → deployer → monitor → reviewer",
        "roles": ["data_curator", "trainer", "evaluator", "deployer",
                  "monitor", "reviewer"],
    },
]


# ===========================================================================
# Agent role descriptions (for CrewAI Agent construction)
# ===========================================================================

_ROLE_DESCRIPTIONS = {
    "data_curator": {
        "role": "Data Curator",
        "goal": "Prepare and validate the training data configuration",
        "backstory": "Expert in ML data pipelines who ensures data configs are correct.",
    },
    "trainer": {
        "role": "ML Trainer",
        "goal": "Configure and optimize the training hyperparameters",
        "backstory": "Senior ML engineer specializing in training optimization.",
    },
    "evaluator": {
        "role": "Model Evaluator",
        "goal": "Set up evaluation benchmarks and acceptance criteria",
        "backstory": "Evaluation specialist who ensures model quality gates.",
    },
    "deployer": {
        "role": "Deployment Engineer",
        "goal": "Configure the deployment and serving parameters",
        "backstory": "MLOps engineer managing model serving infrastructure.",
    },
    "monitor": {
        "role": "Production Monitor",
        "goal": "Set up monitoring and alerting for deployed models",
        "backstory": "SRE focused on ML system observability.",
    },
    "reviewer": {
        "role": "Config Reviewer",
        "goal": "Final review and validation of the complete config",
        "backstory": "Senior architect who reviews all ML configs before production.",
    },
}


# ===========================================================================
# Initial config — richly annotated YAML (identical to EXP-015)
# ===========================================================================


def build_initial_config() -> tuple[str, str]:
    """Build an ML config with rich metadata for testing.

    Returns (text, format). Identical to EXP-015 for cross-experiment
    comparability.
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
# Metadata counting (same as EXP-015)
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
# Config serialization for task output strings
# ===========================================================================


def serialize_config_for_output(
    text: str, fmt: str, mode: str
) -> str:
    """Serialize config into a string suitable for CrewAI TaskOutput.

    json_default: parse YAML to dict, re-emit via yaml.dump (lossy).
    cdxf_enhanced: encode to CDXF binary, return as base64 string (lossless).

    Both return plain strings (CrewAI TaskOutput.raw is always a string).
    """
    if mode == "json_default":
        # Standard approach: parse YAML to Python dict, re-emit
        # This is what developers naturally do — comments are lost
        docs = list(yaml.safe_load_all(text))
        data = docs[0] if docs else {}
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    elif mode == "cdxf_enhanced":
        # CDXF approach: encode to binary, return as base64
        stream = from_yaml(text)
        cdxf_bytes = encode(stream)
        return base64.b64encode(cdxf_bytes).decode("ascii")

    else:
        raise ValueError(f"Unknown mode: {mode}")


def extract_config_from_output(
    output_text: str, fmt: str, mode: str
) -> str:
    """Extract config text from a CrewAI TaskOutput.raw string.

    json_default: the output IS yaml text (already lossy).
    cdxf_enhanced: decode base64 → CDXF binary → YAML text (lossless).
    """
    if mode == "json_default":
        # The output is already YAML text (from yaml.dump)
        return output_text

    elif mode == "cdxf_enhanced":
        # Decode base64 → CDXF binary → YAML
        cdxf_bytes = base64.b64decode(output_text.encode("ascii"))
        stream = decode(cdxf_bytes)
        return to_yaml(stream)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ===========================================================================
# Agent modifications (same as EXP-015)
# ===========================================================================

_ROLE_MODIFICATIONS = {
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
# FakeLLM — deterministic BaseLLM subclass
# ===========================================================================


class FakeLLM(BaseLLM):
    """Deterministic LLM for reproducible CrewAI experiments.

    Parses config from task context, applies a predetermined modification
    based on agent role, and returns the modified config as a string.
    No API calls, no randomness, 100% reproducible.
    """

    mode: str = "json_default"
    _current_role: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, mode: str = "json_default", **kwargs):
        super().__init__(
            model="fake-model",
            llm_type="fake",
            provider="fake",
            **kwargs,
        )
        self.mode = mode

    def call(
        self,
        messages: str | list[dict],
        tools: list | None = None,
        callbacks: list | None = None,
        available_functions: dict | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        """Process config deterministically based on agent role."""
        # Determine the agent role
        role_key = self._current_role
        if from_agent is not None:
            # Map CrewAI role name back to our internal key
            role_name = getattr(from_agent, "role", "")
            role_key = self._role_name_to_key(role_name)

        # Extract the message text
        msg_text = self._extract_message_text(messages)

        # Find config in the message
        config_text = self._extract_config(msg_text)

        if not config_text:
            # First task — use initial config from task description
            config_text, _ = build_initial_config()

        # Apply modification
        modified = self._apply_modification(config_text, role_key)

        # Re-serialize according to mode
        return serialize_config_for_output(modified, "yaml", self.mode)

    def _extract_message_text(self, messages: str | list[dict]) -> str:
        """Extract text content from messages."""
        if isinstance(messages, str):
            return messages
        text_parts = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
        return "\n".join(text_parts)

    def _extract_config(self, text: str) -> str | None:
        """Extract config from message text.

        In json_default mode: the prior output is YAML text (from yaml.dump).
        In cdxf_enhanced mode: the prior output is a base64 string.
        """
        if not text:
            return None

        if self.mode == "cdxf_enhanced":
            # Look for base64-encoded CDXF data in the text
            # Base64 strings are long alphanumeric sequences
            for line in text.splitlines():
                line = line.strip()
                if len(line) > 50 and self._is_base64(line):
                    try:
                        config = extract_config_from_output(
                            line, "yaml", "cdxf_enhanced"
                        )
                        if config and (":" in config or "model" in config):
                            return config
                    except Exception:
                        continue

        # Try to find YAML config in text
        # Look for lines that look like YAML config
        lines = text.splitlines()
        yaml_lines = []
        in_config = False
        for line in lines:
            stripped = line.strip()
            # Detect start of YAML config
            if (stripped.startswith("#") or
                    (": " in stripped and not stripped.startswith("-")) or
                    stripped.endswith(":") or
                    (stripped.startswith("- ") and in_config)):
                in_config = True
                yaml_lines.append(line)
            elif in_config and stripped == "":
                yaml_lines.append(line)
            elif in_config and not stripped:
                continue

        if yaml_lines:
            candidate = "\n".join(yaml_lines).strip() + "\n"
            # Verify it's valid YAML
            try:
                parsed = yaml.safe_load(candidate)
                if isinstance(parsed, dict) and len(parsed) >= 2:
                    return candidate
            except yaml.YAMLError:
                pass

        return None

    def _is_base64(self, s: str) -> bool:
        """Check if a string looks like base64."""
        import string
        b64_chars = set(string.ascii_letters + string.digits + "+/=")
        return all(c in b64_chars for c in s) and len(s) > 20

    def _apply_modification(self, config_text: str, role_key: str) -> str:
        """Apply a deterministic modification based on role."""
        modifications = _ROLE_MODIFICATIONS.get(role_key, {})
        modified = config_text
        for pattern, replacement_fn in modifications.items():
            modified = re.sub(pattern, replacement_fn, modified, count=1)
        return modified

    @staticmethod
    def _role_name_to_key(role_name: str) -> str:
        """Map CrewAI role name to internal key."""
        mapping = {
            "Data Curator": "data_curator",
            "ML Trainer": "trainer",
            "Model Evaluator": "evaluator",
            "Deployment Engineer": "deployer",
            "Production Monitor": "monitor",
            "Config Reviewer": "reviewer",
        }
        return mapping.get(role_name, "")

    def get_context_window_size(self) -> int:
        return 128000

    def supports_stop_words(self) -> bool:
        return False

    def supports_multimodal(self) -> bool:
        return False


# ===========================================================================
# CrewAI pipeline construction
# ===========================================================================


def build_crew_pipeline(
    mode: str,
    crew_config: dict | None = None,
) -> tuple[Crew, list[Task]]:
    """Build a CrewAI crew for the ML config pipeline.

    Args:
        mode: "json_default" or "cdxf_enhanced".
        crew_config: Crew topology config (default: linear_4agent).

    Returns:
        (Crew, list[Task]) — the crew and its tasks.
    """
    if crew_config is None:
        crew_config = CREW_CONFIGS[0]

    llm = FakeLLM(mode=mode)
    text, fmt = build_initial_config()

    # Build agents
    agents = []
    for role_key in crew_config["roles"]:
        desc = _ROLE_DESCRIPTIONS[role_key]
        agent = Agent(
            role=desc["role"],
            goal=desc["goal"],
            backstory=desc["backstory"],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        agents.append(agent)

    # Build tasks — sequential with context chaining
    initial_config_output = serialize_config_for_output(text, fmt, mode)

    tasks = []
    for i, (role_key, agent) in enumerate(
        zip(crew_config["roles"], agents)
    ):
        if i == 0:
            # First task: include initial config in description
            task = Task(
                description=(
                    f"Process the following ML configuration. "
                    f"Apply your expertise as {agent.role}.\n\n"
                    f"{initial_config_output}"
                ),
                expected_output="Modified ML configuration",
                agent=agent,
            )
        else:
            # Subsequent tasks: reference prior task output as context
            task = Task(
                description=(
                    f"Review and update the ML configuration. "
                    f"Apply your expertise as {agent.role}."
                ),
                expected_output="Modified ML configuration",
                agent=agent,
                context=[tasks[-1]],
            )
        tasks.append(task)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    return crew, tasks


# ===========================================================================
# Crew execution
# ===========================================================================


def run_crew(
    mode: str,
    crew_config: dict | None = None,
) -> dict:
    """Execute a CrewAI pipeline and measure metadata survival.

    Args:
        mode: "json_default" or "cdxf_enhanced".
        crew_config: Optional crew topology.

    Returns:
        Dict with execution results and metadata metrics.
    """
    if crew_config is None:
        crew_config = CREW_CONFIGS[0]

    text, fmt = build_initial_config()
    initial_meta = count_config_metadata(text, fmt)

    # Build and run the crew
    crew, tasks = build_crew_pipeline(mode, crew_config)
    crew_output = crew.kickoff()

    # Extract final config from crew output
    final_output_raw = crew_output.raw
    final_config_text = extract_config_from_output(
        final_output_raw, fmt, mode
    )
    final_meta = count_config_metadata(final_config_text, fmt)

    surviving_fraction = (
        final_meta["comments"] / initial_meta["comments"]
        if initial_meta["comments"] > 0 else 1.0
    )

    # Collect per-task trace
    task_trace = []
    for task in tasks:
        if task.output is not None:
            task_output_text = extract_config_from_output(
                task.output.raw, fmt, mode
            )
            task_meta = count_config_metadata(task_output_text, fmt)
            task_trace.append({
                "agent": task.output.agent,
                "comments": task_meta["comments"],
            })

    return {
        "mode": mode,
        "crew_config": crew_config["name"],
        "initial_comments": initial_meta["comments"],
        "final_comments": final_meta["comments"],
        "surviving_fraction": surviving_fraction,
        "task_trace": task_trace,
        "n_agents": len(crew_config["roles"]),
    }


# ===========================================================================
# Full experiment
# ===========================================================================


def run_experiment(output_dir: Path | str | None = None) -> dict:
    """Run the full EXP-017 experiment."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_017")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-017: CrewAI Pipeline — ML Config Handoff Fidelity")
    print("=" * 70)

    results = {}

    for cc in CREW_CONFIGS:
        print(f"\n--- Crew: {cc['name']} ({len(cc['roles'])} agents) ---")

        for mode in STATE_MODES:
            print(f"\n  Mode: {mode}")

            r = run_crew(mode, cc)
            key = f"{cc['name']}_{mode}"
            results[key] = r
            print(f"    Pipeline: {r['initial_comments']} → "
                  f"{r['final_comments']} comments "
                  f"({r['surviving_fraction']:.1%})")

    # Summary
    print("\n--- Summary ---")
    summary = {}
    for mode in STATE_MODES:
        gc = CREW_CONFIGS[0]
        key = f"{gc['name']}_{mode}"
        r = results[key]
        summary[mode] = {
            "pipeline_surviving": r["surviving_fraction"],
            "initial_comments": r["initial_comments"],
            "final_comments": r["final_comments"],
        }
        print(f"  {mode:18s}: pipeline={r['surviving_fraction']:.1%}")

    # Write outputs
    output = {
        "experiment": "EXP-017",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "framework": "crewai",
        "framework_version": _get_version("crewai"),
        "results": results,
        "summary": summary,
    }

    json_path = output_dir / "exp_017_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults: {json_path}")

    # CSV: comparison
    comp_csv = output_dir / "mode_comparison.csv"
    rows = []
    for key, r in results.items():
        rows.append({
            "crew_config": r["crew_config"],
            "mode": r["mode"],
            "n_agents": r["n_agents"],
            "initial_comments": r["initial_comments"],
            "final_comments": r["final_comments"],
            "surviving_fraction": round(r["surviving_fraction"], 4),
        })
    if rows:
        fieldnames = list(rows[0].keys())
        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Comparison CSV: {comp_csv}")

    print(f"\n{'=' * 70}")
    print("EXP-017 COMPLETE")
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
