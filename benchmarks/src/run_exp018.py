"""
EXP-018: AutoGen Group Chat — ML Config Handoff Fidelity

Build a real AutoGen RoundRobinGroupChat with FakeChatCompletionClient
where ML agents pass annotated YAML configs as TextMessages. Compare:
  - json_default: agent parses YAML → dict, modifies, re-emits → comments lost
  - cdxf_enhanced: agent encodes to CDXF, passes base64, decodes → comments preserved

Uses a deterministic FakeChatCompletionClient so results are 100%
reproducible without API calls.

Usage:
    python benchmarks/src/run_exp018.py
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Mapping, Optional, Sequence

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_core.models import (
        ChatCompletionClient,
        CreateResult,
        RequestUsage,
        LLMMessage,
        UserMessage,
    )
    from autogen_core.tools import Tool, ToolSchema
    from autogen_core import ComponentModel, CancellationToken
except ImportError:
    print("ERROR: autogen-agentchat not installed. Run: pip install autogen-agentchat")
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

GROUP_CONFIGS = [
    {
        "name": "group_4agent",
        "description": "4-agent round robin: curator → trainer → evaluator → deployer",
        "roles": ["data_curator", "trainer", "evaluator", "deployer"],
    },
    {
        "name": "group_6agent",
        "description": "6-agent round robin: + monitor + reviewer",
        "roles": ["data_curator", "trainer", "evaluator", "deployer",
                  "monitor", "reviewer"],
    },
]

_ROLE_DESCRIPTIONS = {
    "data_curator": {
        "name": "data_curator",
        "description": "Expert in ML data pipelines who ensures data configs are correct.",
    },
    "trainer": {
        "name": "trainer",
        "description": "Senior ML engineer specializing in training optimization.",
    },
    "evaluator": {
        "name": "evaluator",
        "description": "Evaluation specialist who ensures model quality gates.",
    },
    "deployer": {
        "name": "deployer",
        "description": "MLOps engineer managing model serving infrastructure.",
    },
    "monitor": {
        "name": "monitor",
        "description": "SRE focused on ML system observability.",
    },
    "reviewer": {
        "name": "reviewer",
        "description": "Senior architect who reviews all ML configs before production.",
    },
}


# ===========================================================================
# Initial config (identical to EXP-015/017)
# ===========================================================================


def build_initial_config() -> tuple[str, str]:
    """Build an ML config with rich metadata. Identical to EXP-015."""
    return YAML_CONFIGS["large"]["text"], "yaml"


# ===========================================================================
# Metadata counting
# ===========================================================================


def count_config_metadata(text: str, fmt: str) -> dict:
    """Count metadata constructs in config text."""
    return corpus_count_metadata(text, fmt)


# ===========================================================================
# Config serialization
# ===========================================================================


def serialize_config_for_output(
    text: str, fmt: str, mode: str
) -> str:
    """Serialize config into a string for AutoGen messages."""
    if mode == "json_default":
        docs = list(yaml.safe_load_all(text))
        data = docs[0] if docs else {}
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    elif mode == "cdxf_enhanced":
        stream = from_yaml(text)
        cdxf_bytes = encode(stream)
        return base64.b64encode(cdxf_bytes).decode("ascii")
    else:
        raise ValueError(f"Unknown mode: {mode}")


def extract_config_from_output(
    output_text: str, fmt: str, mode: str
) -> str:
    """Extract config text from an AutoGen message."""
    if mode == "json_default":
        return output_text
    elif mode == "cdxf_enhanced":
        cdxf_bytes = base64.b64decode(output_text.encode("ascii"))
        stream = decode(cdxf_bytes)
        return to_yaml(stream)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ===========================================================================
# FakeChatCompletionClient
# ===========================================================================

_ZERO_USAGE = RequestUsage(prompt_tokens=0, completion_tokens=0)


class FakeChatCompletionClient(ChatCompletionClient):
    """Deterministic ChatCompletionClient for reproducible experiments.

    Parses config from incoming messages, applies a predetermined
    modification based on agent name, returns the modified config.
    """

    def __init__(self, mode: str = "json_default"):
        self._mode = mode
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def capabilities(self) -> Any:
        """Return model capabilities."""
        return {"vision": False, "function_calling": False, "json_output": False}

    @property
    def model_info(self) -> Any:
        """Return model info (required abstract property)."""
        from autogen_core.models import ModelInfo
        return ModelInfo(
            vision=False, function_calling=False,
            json_output=False, family="fake",
        )

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool | type] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        """Process config deterministically based on agent context."""
        role_key = self._detect_role(messages)

        # Extract config from the LATEST non-system message (reverse order)
        config_text = None
        for msg in reversed(messages):
            msg_type = type(msg).__name__
            if msg_type == "SystemMessage":
                continue
            if hasattr(msg, "content"):
                content = msg.content if isinstance(msg.content, str) else ""
                config_text = self._extract_config(content)
                if config_text:
                    break

        if not config_text:
            config_text, _ = build_initial_config()

        modified = self._apply_modification(config_text, role_key)
        output = serialize_config_for_output(modified, "yaml", self._mode)

        return CreateResult(
            finish_reason="stop",
            content=output,
            usage=_ZERO_USAGE,
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool | type] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        result = await self.create(
            messages, tools=tools, tool_choice=tool_choice,
            json_output=json_output, extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result

    def actual_usage(self) -> RequestUsage:
        return _ZERO_USAGE

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
    ) -> int:
        return 0

    def remaining_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
    ) -> int:
        return 128000

    def close(self) -> None:
        pass

    @classmethod
    def _from_config(cls, config: dict) -> "FakeChatCompletionClient":
        return cls(mode=config.get("mode", "json_default"))

    def dump_component(self) -> ComponentModel:
        return ComponentModel(
            provider="benchmarks.src.run_exp018.FakeChatCompletionClient",
            component_type="model",
            version=1,
            description="Fake client for EXP-018",
            config={"mode": self._mode},
        )

    @classmethod
    def load_component(
        cls, model: ComponentModel, **kwargs: Any
    ) -> "FakeChatCompletionClient":
        return cls._from_config(model.config)

    def _extract_message_text(self, messages: Sequence[LLMMessage]) -> str:
        parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                content = msg.content
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif hasattr(item, "text"):
                            parts.append(item.text)
        return "\n".join(parts)

    def _detect_role(self, messages: Sequence[LLMMessage]) -> str:
        """Detect agent role from the SystemMessage (first in list)."""
        # AutoGen puts the current agent's system message first
        for msg in messages:
            msg_type = type(msg).__name__
            if msg_type == "SystemMessage":
                content = msg.content if isinstance(msg.content, str) else ""
                for role_key, desc in _ROLE_DESCRIPTIONS.items():
                    if desc["name"] in content.lower():
                        return role_key
                break  # Only check the first SystemMessage
        return ""

    def _extract_config(self, text: str) -> str | None:
        if not text:
            return None

        if self._mode == "cdxf_enhanced":
            import string
            b64_chars = set(string.ascii_letters + string.digits + "+/=")
            for line in text.splitlines():
                line = line.strip()
                if len(line) > 50 and all(c in b64_chars for c in line):
                    try:
                        config = extract_config_from_output(
                            line, "yaml", "cdxf_enhanced"
                        )
                        if config and ":" in config:
                            return config
                    except Exception:
                        continue

        # Try to find YAML config
        lines = text.splitlines()
        yaml_lines = []
        in_config = False
        for line in lines:
            stripped = line.strip()
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
            try:
                parsed = yaml.safe_load(candidate)
                if isinstance(parsed, dict) and len(parsed) >= 2:
                    return candidate
            except yaml.YAMLError:
                pass

        return None

    def _apply_modification(self, config_text: str, role_key: str) -> str:
        modifications = ROLE_MODIFICATIONS.get(role_key, {})
        modified = config_text
        for pattern, replacement_fn in modifications.items():
            modified = re.sub(pattern, replacement_fn, modified, count=1)
        return modified


# ===========================================================================
# Group chat construction
# ===========================================================================


def build_group_chat(
    mode: str,
    group_config: dict | None = None,
) -> tuple[RoundRobinGroupChat, list[AssistantAgent]]:
    """Build an AutoGen RoundRobinGroupChat."""
    if group_config is None:
        group_config = GROUP_CONFIGS[0]

    client = FakeChatCompletionClient(mode=mode)

    agents = []
    for role_key in group_config["roles"]:
        desc = _ROLE_DESCRIPTIONS[role_key]
        agent = AssistantAgent(
            name=desc["name"],
            model_client=client,
            description=desc["description"],
            system_message=(
                f"You are {desc['name']}: {desc['description']} "
                f"Process the ML configuration and apply your expertise."
            ),
        )
        agents.append(agent)

    n_agents = len(agents)
    # +1 because the task message counts toward the message limit
    termination = MaxMessageTermination(max_messages=n_agents + 1)

    team = RoundRobinGroupChat(
        participants=agents,
        termination_condition=termination,
    )

    return team, agents


# ===========================================================================
# Group chat execution
# ===========================================================================


async def run_group_chat(
    mode: str,
    group_config: dict | None = None,
) -> dict:
    """Execute an AutoGen group chat and measure metadata survival."""
    if group_config is None:
        group_config = GROUP_CONFIGS[0]

    text, fmt = build_initial_config()
    initial_meta = count_config_metadata(text, fmt)

    initial_output = serialize_config_for_output(text, fmt, mode)
    task_text = f"Process this ML configuration:\n\n{initial_output}"

    team, agents = build_group_chat(mode, group_config)
    result = await team.run(task=task_text)

    # Extract final config from the last message
    final_message_content = ""
    if result.messages:
        for msg in reversed(result.messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                final_message_content = msg.content
                break

    final_config_text = extract_config_from_output(
        final_message_content, fmt, mode
    )
    final_meta = count_config_metadata(final_config_text, fmt)

    surviving_fraction = (
        final_meta["comments"] / initial_meta["comments"]
        if initial_meta["comments"] > 0 else 1.0
    )

    return {
        "mode": mode,
        "group_config": group_config["name"],
        "initial_comments": initial_meta["comments"],
        "final_comments": final_meta["comments"],
        "surviving_fraction": surviving_fraction,
        "n_agents": len(group_config["roles"]),
        "n_messages": len(result.messages) if result.messages else 0,
    }


# ===========================================================================
# Enhanced experiments — scaling, timing, data integrity
# ===========================================================================


async def run_scaling_experiment(
    config_sizes: list[str] | None = None,
) -> dict:
    """Test fidelity across config sizes."""
    if config_sizes is None:
        config_sizes = ["small", "medium", "large", "xlarge"]

    results = []
    for size in config_sizes:
        cfg = YAML_CONFIGS[size]
        text = cfg["text"]
        fmt = cfg["format"]

        for mode in STATE_MODES:
            initial_meta = count_config_metadata(text, fmt)
            initial_output = serialize_config_for_output(text, fmt, mode)

            client = FakeChatCompletionClient(mode=mode)
            agent1 = AssistantAgent(
                name="data_curator", model_client=client,
                description="Config processor",
                system_message="You are data_curator: Expert in ML data pipelines.",
            )
            agent2 = AssistantAgent(
                name="trainer", model_client=client,
                description="Config updater",
                system_message="You are trainer: Senior ML engineer.",
            )

            team = RoundRobinGroupChat(
                participants=[agent1, agent2],
                termination_condition=MaxMessageTermination(max_messages=3),  # task + 2 agents
            )

            result = await team.run(
                task=f"Process config:\n\n{initial_output}"
            )

            final_content = ""
            if result.messages:
                for msg in reversed(result.messages):
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        final_content = msg.content
                        break

            final_text = extract_config_from_output(final_content, fmt, mode)
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


async def run_timing_experiment(
    n_iterations: int = 10,
) -> dict:
    """Measure overhead of CDXF vs JSON in AutoGen."""
    text, fmt = build_initial_config()
    timer = Timer()

    timer.measure_n(
        "json_default_serialize",
        lambda: serialize_config_for_output(text, fmt, "json_default"),
        n=n_iterations,
    )
    timer.measure_n(
        "cdxf_enhanced_serialize",
        lambda: serialize_config_for_output(text, fmt, "cdxf_enhanced"),
        n=n_iterations,
    )

    json_val = serialize_config_for_output(text, fmt, "json_default")
    cdxf_val = serialize_config_for_output(text, fmt, "cdxf_enhanced")

    timer.measure_n(
        "json_default_extract",
        lambda: extract_config_from_output(json_val, fmt, "json_default"),
        n=n_iterations,
    )
    timer.measure_n(
        "cdxf_enhanced_extract",
        lambda: extract_config_from_output(cdxf_val, fmt, "cdxf_enhanced"),
        n=n_iterations,
    )

    # Time full pipeline
    async def run_pipeline(mode):
        team, _ = build_group_chat(mode, GROUP_CONFIGS[0])
        initial_output = serialize_config_for_output(text, fmt, mode)
        await team.run(task=f"Process:\n\n{initial_output}")

    # Async timing for pipelines
    import time

    async def measure_async(label, coro_fn, n):
        times = []
        for _ in range(n):
            start = time.perf_counter()
            await coro_fn()
            times.append(time.perf_counter() - start)
        mean_t = sum(times) / len(times)
        timer.results[label] = mean_t

    await measure_async(
        "json_default_pipeline",
        lambda: run_pipeline("json_default"),
        n_iterations,
    )
    await measure_async(
        "cdxf_enhanced_pipeline",
        lambda: run_pipeline("cdxf_enhanced"),
        n_iterations,
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


async def run_integrity_experiment() -> dict:
    """Verify that agent modifications are correctly applied."""
    results = []

    for gc in GROUP_CONFIGS:
        n_agents = len(gc["roles"])
        expected = (
            EXPECTED_VALUES_AFTER_6AGENT
            if n_agents >= 6 else EXPECTED_VALUES_AFTER_4AGENT
        )

        for mode in STATE_MODES:
            r = await run_group_chat(mode, gc)

            # Re-run to get the actual final text
            text, fmt = build_initial_config()
            initial_output = serialize_config_for_output(text, fmt, mode)
            team, _ = build_group_chat(mode, gc)
            result = await team.run(
                task=f"Process config:\n\n{initial_output}"
            )

            final_content = ""
            if result.messages:
                for msg in reversed(result.messages):
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        final_content = msg.content
                        break

            final_text = extract_config_from_output(final_content, fmt, mode)
            integrity = verify_data_integrity(final_text, expected)

            results.append({
                "group_config": gc["name"],
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


async def run_experiment(output_dir: Path | str | None = None) -> dict:
    """Run the full EXP-018 experiment."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_018")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-018: AutoGen Group Chat — ML Config Handoff Fidelity")
    print("=" * 70)

    results = {}

    for gc in GROUP_CONFIGS:
        print(f"\n--- Group: {gc['name']} ({len(gc['roles'])} agents) ---")

        for mode in STATE_MODES:
            print(f"\n  Mode: {mode}")
            r = await run_group_chat(mode, gc)
            key = f"{gc['name']}_{mode}"
            results[key] = r
            print(f"    Pipeline: {r['initial_comments']} → "
                  f"{r['final_comments']} comments "
                  f"({r['surviving_fraction']:.1%})")

    # Enhanced experiments
    print("\n--- Scaling Experiment (multi-size configs) ---")
    scaling = await run_scaling_experiment()
    for sr in scaling["scaling_results"]:
        if sr["mode"] == "cdxf_enhanced":
            print(f"  {sr['config_size']:8s}: {sr['initial_comments']} → "
                  f"{sr['final_comments']} ({sr['surviving_fraction']:.1%})")

    print("\n--- Timing Experiment ---")
    timing = await run_timing_experiment()
    for k, v in timing["timings_seconds"].items():
        print(f"  {k:30s}: {v*1000:.3f} ms")
    for k, v in timing["overhead"].items():
        print(f"  {k:30s}: {v:+.3f} ms")

    print("\n--- Data Integrity Experiment ---")
    integrity = await run_integrity_experiment()
    for ir in integrity["integrity_results"]:
        status = "PASS" if ir["integrity_passed"] else "FAIL"
        print(f"  {ir['group_config']} / {ir['mode']}: "
              f"{status} ({ir['checks']} checks)")

    # Summary
    print("\n--- Summary ---")
    summary = {}
    for mode in STATE_MODES:
        gc = GROUP_CONFIGS[0]
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
        "experiment": "EXP-018",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "framework": "autogen-agentchat",
        "framework_version": _get_version("autogen-agentchat"),
        "results": results,
        "summary": summary,
        "scaling": scaling,
        "timing": timing,
        "integrity": integrity,
    }

    json_path = output_dir / "exp_018_results.json"
    json_path.write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults: {json_path}")

    comp_csv = output_dir / "mode_comparison.csv"
    rows = []
    for key, r in results.items():
        rows.append({
            "group_config": r["group_config"],
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
    print("EXP-018 COMPLETE")
    print("=" * 70)

    return output


def _get_version(package: str) -> str:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return "unknown"


def main():
    asyncio.run(run_experiment())


if __name__ == "__main__":
    main()
