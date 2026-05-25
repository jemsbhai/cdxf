"""
EXP-005: Format Heterogeneity Census of the HuggingFace AI Ecosystem

Collects and analyzes configuration/metadata files from the top models,
LoRA adapters, and datasets on HuggingFace Hub to quantify format
heterogeneity in the AI/ML ecosystem.

Usage:
    python benchmarks/src/run_exp005.py [--resume] [--max-models N]

Requires: HF_TOKEN environment variable set.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests as _requests_lib
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        GatedRepoError,
    )
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("benchmarks/results/exp_005")
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
REPO_SUMMARY_CSV = RESULTS_DIR / "repo_summary.csv"
FILE_DETAIL_CSV = RESULTS_DIR / "file_detail.csv"
STATS_JSON = RESULTS_DIR / "summary_stats.json"

# File extensions we care about (config/metadata files)
CONFIG_EXTENSIONS = {
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".xml",
    ".cfg", ".ini", ".conf", ".properties",
}

# Files to always grab regardless of extension (for YAML frontmatter, etc.)
SPECIAL_FILES = {"README.md", "MODEL_CARD.md", "DATASET_CARD.md"}

# Files/patterns to skip (weights, binaries, large data)
SKIP_PATTERNS = {
    ".bin", ".safetensors", ".pt", ".pth", ".gguf", ".ggml",
    ".model", ".spiece", ".vocab", ".msgpack",  # tokenizer binaries
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",  # images
    ".mp4", ".avi", ".wav", ".mp3",  # media
    ".parquet", ".arrow", ".h5", ".hdf5",  # large data
    ".onnx", ".tflite",  # model formats (large)
    ".lock", ".whl", ".tar.gz", ".zip",
}

# Max file size to download (skip very large configs)
MAX_FILE_SIZE_BYTES = 1_000_000  # 1MB

# Repos known to hang (massive file trees)
SKIP_REPOS = {
    "banned-historical-archives/banned-historical-archives",
}

# Max files to iterate per repo before giving up
MAX_FILES_PER_REPO = 500

# Rate limiting
REQUEST_DELAY_SEC = 0.1  # 100ms between API calls
CHECKPOINT_INTERVAL = 50  # Save progress every N repos

# Canonical ML/AI datasets to always include regardless of download ranking
CANONICAL_DATASETS = [
    # NLP benchmarks
    "rajpurkar/squad", "rajpurkar/squad_v2",
    "nyu-mll/glue", "nyu-mll/multi_nli",
    "stanfordnlp/imdb", "stanfordnlp/sst2",
    "cais/mmlu", "openai/gsm8k",
    "tatsu-lab/alpaca", "databricks/databricks-dolly-15k",
    "Open-Orca/OpenOrca", "HuggingFaceH4/ultrachat_200k",
    "yahma/alpaca-cleaned", "teknium/OpenHermes-2.5",
    # Vision
    "imagenet-1k", "cifar10", "cifar100",
    "mnist", "fashion_mnist",
    "huggingface/cats-image",
    # Code
    "bigcode/the-stack", "codeparrot/github-code",
    "bigcode/starcoderdata",
    # Multilingual
    "mc4", "cc100", "allenai/c4",
    "facebook/flores", "Helsinki-NLP/opus-100",
    # Audio
    "mozilla-foundation/common_voice_17_0",
    "facebook/voxpopuli", "librispeech_asr",
    # Instruction tuning / RLHF
    "Anthropic/hh-rlhf", "OpenAssistant/oasst1",
    "HuggingFaceH4/no_robots",
    "argilla/ultrafeedback-binarized-preferences-cleaned",
    # Reasoning / Math
    "lighteval/MATH", "hendrycks/competition_math",
    "allenai/ai2_arc", "EleutherAI/truthful_qa_mc",
    "lukaemon/bbh", "TIGER-Lab/MMLU-Pro",
    # Retrieval / QA
    "ms_marco", "natural_questions", "trivia_qa",
    "BeIR/hotpotqa", "neural-bridge/rag-dataset-12000",
    # Safety
    "allenai/wildjailbreak", "PKU-Alignment/BeaverTails",
    # Medical / Science
    "bigbio/pubmed_qa", "epfl-llm/guidelines",
    # Table / Structured
    "wikitablequestions", "msr_sqa",
    # Popular fine-tuning / chat
    "mlabonne/guanaco-llama2-1k",
    "Intel/orca_dpo_pairs",
    "garage-bAInd/Open-Platypus",
    # Eval
    "lmsys/chatbot_arena_conversations",
    "HuggingFaceH4/mt-bench-prompts",
    # Image-text
    "laion/laion2B-en", "ChristophSchuhmann/improved_aesthetics_6.5plus",
    # Recent popular
    "cognitivecomputations/dolphin",
    "BAAI/Infinity-Instruct",
    "nvidia/HelpSteer2",
]


# ---------------------------------------------------------------------------
# Format classification
# ---------------------------------------------------------------------------

def classify_format(filename: str) -> str:
    """Classify a file's format based on its name and extension.

    Returns one of: json, jsonl, yaml, toml, xml, ini, markdown, other
    """
    name = filename.lower()
    ext = Path(name).suffix

    # Special cases first
    if name in {"readme.md", "model_card.md", "dataset_card.md"}:
        return "markdown"  # will check for YAML frontmatter separately

    ext_map = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".cfg": "ini",
        ".ini": "ini",
        ".conf": "ini",
        ".properties": "ini",
    }
    return ext_map.get(ext, "other")


# ---------------------------------------------------------------------------
# Construct counting (format-specific metadata analysis)
# ---------------------------------------------------------------------------

def count_yaml_constructs(text: str) -> dict:
    """Count format-specific constructs in a YAML file.

    Returns dict with: comments, anchors, aliases, merge_keys,
    multi_doc_markers, typed_temporals
    """
    counts = {
        "comments": 0,
        "anchors": 0,
        "aliases": 0,
        "merge_keys": 0,
        "multi_doc_markers": 0,
    }

    for line in text.splitlines():
        stripped = line.strip()

        # Comments: lines starting with # or inline # (but not inside strings)
        # Simple heuristic: count # that appear outside of quotes
        if stripped.startswith("#"):
            counts["comments"] += 1
        elif "#" in stripped:
            # Rough heuristic: # after a value (not inside a quoted string)
            # This undercounts but avoids false positives from # in URLs etc.
            in_single = False
            in_double = False
            for i, ch in enumerate(stripped):
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif ch == "#" and not in_single and not in_double and i > 0:
                    counts["comments"] += 1
                    break

        # Anchors: &anchor_name
        counts["anchors"] += len(re.findall(r"&\w+", stripped))

        # Aliases: *alias_name
        counts["aliases"] += len(re.findall(r"\*\w+", stripped))

        # Merge keys: <<:
        if stripped.startswith("<<"):
            counts["merge_keys"] += 1

        # Multi-document markers: ---
        if stripped == "---" or stripped == "...":
            counts["multi_doc_markers"] += 1

    return counts


def count_json_constructs(text: str) -> dict:
    """Count structural properties of a JSON file."""
    counts = {
        "max_depth": 0,
        "key_count": 0,
        "array_count": 0,
    }
    try:
        data = json.loads(text)
        _json_depth(data, 0, counts)
    except (json.JSONDecodeError, RecursionError):
        pass
    return counts


def _json_depth(obj, depth: int, counts: dict):
    """Recursively compute JSON depth and counts."""
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


def count_toml_constructs(text: str) -> dict:
    """Count format-specific constructs in a TOML file."""
    counts = {
        "comments": 0,
        "sections": 0,
        "inline_tables": 0,
        "temporal_values": 0,
    }

    # ISO 8601 date/datetime pattern (rough)
    datetime_pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?"
    )

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            counts["comments"] += 1
        elif "#" in stripped:
            # Inline comment (rough heuristic, same as YAML)
            in_single = False
            in_double = False
            for i, ch in enumerate(stripped):
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif ch == "#" and not in_single and not in_double and i > 0:
                    counts["comments"] += 1
                    break

        if stripped.startswith("["):
            counts["sections"] += 1

        if "{" in stripped and "=" in stripped:
            counts["inline_tables"] += 1

        # Check for datetime values on RHS of assignment
        if "=" in stripped:
            rhs = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            if datetime_pattern.match(rhs):
                counts["temporal_values"] += 1

    return counts


def count_xml_constructs(text: str) -> dict:
    """Count format-specific constructs in an XML file."""
    counts = {
        "comments": 0,
        "processing_instructions": 0,
        "namespaces": 0,
        "elements": 0,
        "attributes": 0,
    }

    counts["comments"] = len(re.findall(r"<!--", text))
    counts["processing_instructions"] = len(re.findall(r"<\?(?!xml\s)", text))
    counts["namespaces"] = len(re.findall(r"xmlns[:\w]*=", text))
    counts["elements"] = len(re.findall(r"<(?!/|\?|!)[a-zA-Z]", text))
    counts["attributes"] = len(re.findall(r'\s\w+="', text))

    return counts


def detect_yaml_frontmatter(text: str) -> dict | None:
    """Check if a markdown file has YAML frontmatter. Returns construct
    counts if found, None otherwise."""
    if not text.startswith("---"):
        return None
    # Find closing ---
    end = text.find("\n---", 3)
    if end == -1:
        return None
    frontmatter = text[3:end]
    counts = count_yaml_constructs(frontmatter)
    counts["frontmatter_size_bytes"] = len(frontmatter.encode("utf-8"))
    return counts


def shannon_entropy(format_counts: dict[str, int]) -> float:
    """Compute Shannon entropy of format distribution."""
    total = sum(format_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in format_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


# ---------------------------------------------------------------------------
# HuggingFace Hub data collection
# ---------------------------------------------------------------------------

def fetch_top_models(api: HfApi, n: int = 500) -> list[dict]:
    """Fetch top N models by downloads from HuggingFace Hub."""
    print(f"Fetching top {n} models by downloads...")
    models = []
    for model in api.list_models(
        sort="downloads",
        limit=n,
    ):
        models.append({
            "repo_id": model.id,
            "repo_type": "model",
            "task": getattr(model, "pipeline_tag", None) or "unknown",
            "downloads": getattr(model, "downloads", 0),
            "library": getattr(model, "library_name", None) or "unknown",
            "author": model.id.split("/")[0] if "/" in model.id else "unknown",
            "last_modified": str(getattr(model, "last_modified", "")),
        })
    print(f"  Got {len(models)} models")
    return models


def fetch_lora_adapters(api: HfApi, n: int = 50) -> list[dict]:
    """Fetch top N LoRA/PEFT adapters by downloads."""
    print(f"Fetching top {n} LoRA adapters...")
    adapters = []
    for model in api.list_models(
        sort="downloads",
        limit=n,
        filter="peft",
    ):
        adapters.append({
            "repo_id": model.id,
            "repo_type": "adapter",
            "task": getattr(model, "pipeline_tag", None) or "unknown",
            "downloads": getattr(model, "downloads", 0),
            "library": "peft",
            "author": model.id.split("/")[0] if "/" in model.id else "unknown",
            "last_modified": str(getattr(model, "last_modified", "")),
        })
    print(f"  Got {len(adapters)} adapters")
    return adapters


def fetch_top_datasets(api: HfApi, n: int = 50) -> list[dict]:
    """Fetch top N datasets by downloads."""
    print(f"Fetching top {n} datasets...")
    datasets = []
    for ds in api.list_datasets(
        sort="downloads",
        limit=n,
    ):
        datasets.append({
            "repo_id": ds.id,
            "repo_type": "dataset",
            "task": getattr(ds, "pipeline_tag", None) or "unknown",
            "downloads": getattr(ds, "downloads", 0),
            "library": "datasets",
            "author": ds.id.split("/")[0] if "/" in ds.id else "unknown",
            "last_modified": str(getattr(ds, "last_modified", "")),
        })
    print(f"  Got {len(datasets)} datasets")
    return datasets


def fetch_canonical_datasets(api: HfApi) -> list[dict]:
    """Fetch canonical/well-known ML datasets from curated list."""
    print(f"Fetching {len(CANONICAL_DATASETS)} canonical datasets...")
    canonical = []
    for ds_id in CANONICAL_DATASETS:
        try:
            info = api.dataset_info(ds_id)
            canonical.append({
                "repo_id": info.id,
                "repo_type": "dataset",
                "task": getattr(info, "pipeline_tag", None) or "unknown",
                "downloads": getattr(info, "downloads", 0),
                "library": "datasets",
                "author": info.id.split("/")[0] if "/" in info.id else "unknown",
                "last_modified": str(getattr(info, "last_modified", "")),
            })
        except (RepositoryNotFoundError, GatedRepoError) as e:
            print(f"  [skip] {ds_id}: {type(e).__name__}")
        except Exception as e:
            print(f"  [skip] {ds_id}: {e}")
        time.sleep(REQUEST_DELAY_SEC)
    print(f"  Got {len(canonical)} canonical datasets")
    return canonical


def should_download(filename: str, size: int | None) -> bool:
    """Determine if a file should be downloaded for analysis."""
    name_lower = filename.lower()
    base = Path(filename).name

    # Always grab special files
    if base in SPECIAL_FILES:
        return True

    # Skip weight files and binaries
    for skip_ext in SKIP_PATTERNS:
        if name_lower.endswith(skip_ext):
            return False

    # Check if it's a config file by extension
    ext = Path(name_lower).suffix
    if ext in CONFIG_EXTENSIONS:
        # Skip if too large
        if size is not None and size > MAX_FILE_SIZE_BYTES:
            return False
        return True

    return False


def fetch_file_content(
    repo_id: str,
    filename: str,
    repo_type: str,
    token: str,
) -> str | None:
    """Fetch file content from HF Hub in memory (no disk write).

    Uses the raw file API endpoint to read content directly.
    Returns file text or None on failure.
    """
    if repo_type == "dataset":
        url = f"https://huggingface.co/datasets/{repo_id}/raw/main/{filename}"
    else:
        url = f"https://huggingface.co/{repo_id}/raw/main/{filename}"

    try:
        resp = _requests_lib.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.text
        return None
    except _requests_lib.RequestException:
        return None


def analyze_repo(
    api: HfApi,
    repo_info: dict,
    token: str,
) -> tuple[dict, list[dict]]:
    """Analyze a single repo's format heterogeneity.

    Fetches config/metadata files IN MEMORY (no disk writes).

    Returns:
        (repo_summary, file_details) where repo_summary is a dict of
        aggregate metrics and file_details is a list of per-file dicts.
    """
    repo_id = repo_info["repo_id"]
    repo_type_api = "dataset" if repo_info["repo_type"] == "dataset" else "model"

    # List all files
    try:
        files = list(api.list_repo_tree(
            repo_id,
            repo_type=repo_type_api,
            recursive=True,
        ))
    except (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError) as e:
        return _error_summary(repo_info, str(e)), []
    except Exception as e:
        return _error_summary(repo_info, str(e)), []

    format_counts: dict[str, int] = Counter()
    file_details: list[dict] = []
    total_config_bytes = 0
    total_constructs = 0
    has_frontmatter = False

    for file_entry in files:
        # list_repo_tree returns RepoFile/RepoFolder objects
        if not hasattr(file_entry, "rfilename"):
            continue

        # Guard against repos with enormous file trees
        if len(file_details) >= MAX_FILES_PER_REPO:
            break

        filename = file_entry.rfilename
        file_size = getattr(file_entry, "size", None)

        if not should_download(filename, file_size):
            continue

        fmt = classify_format(filename)
        format_counts[fmt] += 1

        # Try to download and analyze
        file_detail = {
            "repo_id": repo_id,
            "filename": filename,
            "format": fmt,
            "size_bytes": file_size or 0,
            "sha256": "",
            "comments": 0,
            "anchors": 0,
            "aliases": 0,
            "merge_keys": 0,
            "multi_doc_markers": 0,
            "temporal_values": 0,
            "sections": 0,
            "max_depth": 0,
            "key_count": 0,
            "has_frontmatter": False,
            "frontmatter_size": 0,
            "xml_namespaces": 0,
            "xml_elements": 0,
            "xml_pis": 0,
            "error": "",
        }

        try:
            # Fetch file content in memory (no disk write)
            text = fetch_file_content(
                repo_id, filename, repo_type_api, token
            )
            if text is None:
                file_detail["error"] = "fetch_failed"
                file_details.append(file_detail)
                continue

            # SHA-256
            file_detail["sha256"] = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()[:16]
            file_detail["size_bytes"] = len(text.encode("utf-8"))
            total_config_bytes += file_detail["size_bytes"]

            # Analyze constructs based on format
            if fmt == "yaml":
                c = count_yaml_constructs(text)
                file_detail.update({
                    "comments": c["comments"],
                    "anchors": c["anchors"],
                    "aliases": c["aliases"],
                    "merge_keys": c["merge_keys"],
                    "multi_doc_markers": c["multi_doc_markers"],
                })
                total_constructs += sum(c.values())

            elif fmt == "json":
                c = count_json_constructs(text)
                file_detail.update({
                    "max_depth": c["max_depth"],
                    "key_count": c["key_count"],
                })

            elif fmt == "toml":
                c = count_toml_constructs(text)
                file_detail.update({
                    "comments": c["comments"],
                    "sections": c["sections"],
                    "temporal_values": c["temporal_values"],
                })
                total_constructs += c["comments"] + c["temporal_values"]

            elif fmt == "xml":
                c = count_xml_constructs(text)
                file_detail.update({
                    "comments": c["comments"],
                    "xml_namespaces": c["namespaces"],
                    "xml_elements": c["elements"],
                    "xml_pis": c["processing_instructions"],
                })
                total_constructs += c["comments"] + c["processing_instructions"]

            elif fmt == "markdown":
                fm = detect_yaml_frontmatter(text)
                if fm is not None:
                    has_frontmatter = True
                    file_detail["has_frontmatter"] = True
                    file_detail["frontmatter_size"] = fm.get(
                        "frontmatter_size_bytes", 0
                    )
                    file_detail["comments"] = fm.get("comments", 0)
                    total_constructs += fm.get("comments", 0)
                    # Count frontmatter as YAML too
                    format_counts["yaml_frontmatter"] = (
                        format_counts.get("yaml_frontmatter", 0) + 1
                    )

            time.sleep(REQUEST_DELAY_SEC)

        except Exception as e:
            file_detail["error"] = f"{type(e).__name__}: {str(e)[:80]}"

        file_details.append(file_detail)

    # Compute repo-level summary
    n_formats = len([k for k, v in format_counts.items() if v > 0])
    entropy = shannon_entropy(format_counts)

    # Count files with comments
    files_with_comments = sum(
        1 for fd in file_details if fd["comments"] > 0
    )
    yaml_files = [fd for fd in file_details if fd["format"] == "yaml"]
    files_with_anchors = sum(
        1 for fd in file_details if fd["anchors"] > 0
    )
    files_with_temporals = sum(
        1 for fd in file_details if fd["temporal_values"] > 0
    )

    repo_summary = {
        "repo_id": repo_id,
        "repo_type": repo_info["repo_type"],
        "task": repo_info["task"],
        "downloads": repo_info["downloads"],
        "library": repo_info["library"],
        "n_config_files": len(file_details),
        "n_formats": n_formats,
        "format_distribution": json.dumps(dict(format_counts)),
        "format_diversity_index": round(entropy, 4),
        "total_config_bytes": total_config_bytes,
        "total_constructs": total_constructs,
        "files_with_comments": files_with_comments,
        "comment_prevalence": (
            round(files_with_comments / len(yaml_files), 4)
            if yaml_files else 0.0
        ),
        "files_with_anchors": files_with_anchors,
        "anchor_prevalence": (
            round(files_with_anchors / len(yaml_files), 4)
            if yaml_files else 0.0
        ),
        "files_with_temporals": files_with_temporals,
        "has_frontmatter": has_frontmatter,
        "error": "",
    }

    return repo_summary, file_details


def _error_summary(repo_info: dict, error: str) -> dict:
    """Create an error repo summary for repos that couldn't be accessed."""
    return {
        "repo_id": repo_info["repo_id"],
        "repo_type": repo_info["repo_type"],
        "task": repo_info["task"],
        "downloads": repo_info["downloads"],
        "library": repo_info["library"],
        "n_config_files": 0,
        "n_formats": 0,
        "format_distribution": "{}",
        "format_diversity_index": 0.0,
        "total_config_bytes": 0,
        "total_constructs": 0,
        "files_with_comments": 0,
        "comment_prevalence": 0.0,
        "files_with_anchors": 0,
        "anchor_prevalence": 0.0,
        "files_with_temporals": 0,
        "has_frontmatter": False,
        "error": error[:200],
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(processed_ids: list[str], phase: str):
    """Save checkpoint with list of processed repo IDs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "processed_ids": processed_ids,
            "phase": phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def load_checkpoint() -> set[str]:
    """Load checkpoint and return set of already-processed repo IDs."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        ids = set(data.get("processed_ids", []))
        print(f"Resuming from checkpoint: {len(ids)} repos already processed")
        return ids
    return set()


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

REPO_FIELDS = [
    "repo_id", "repo_type", "task", "downloads", "library",
    "n_config_files", "n_formats", "format_distribution",
    "format_diversity_index", "total_config_bytes", "total_constructs",
    "files_with_comments", "comment_prevalence",
    "files_with_anchors", "anchor_prevalence",
    "files_with_temporals", "has_frontmatter", "error",
]

FILE_FIELDS = [
    "repo_id", "filename", "format", "size_bytes", "sha256",
    "comments", "anchors", "aliases", "merge_keys",
    "multi_doc_markers", "temporal_values", "sections",
    "max_depth", "key_count",
    "has_frontmatter", "frontmatter_size",
    "xml_namespaces", "xml_elements", "xml_pis",
    "error",
]


def init_csvs(resume: bool):
    """Initialize CSV files with headers (unless resuming)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not resume or not REPO_SUMMARY_CSV.exists():
        with open(REPO_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=REPO_FIELDS).writeheader()
    if not resume or not FILE_DETAIL_CSV.exists():
        with open(FILE_DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FILE_FIELDS).writeheader()


def append_repo_csv(summary: dict):
    """Append one repo summary row to CSV."""
    with open(REPO_SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPO_FIELDS)
        writer.writerow(summary)


def append_file_csv(details: list[dict]):
    """Append file detail rows to CSV."""
    with open(FILE_DETAIL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FILE_FIELDS)
        for d in details:
            writer.writerow(d)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(repo_csv_path: Path) -> dict:
    """Compute aggregate statistics from repo summary CSV."""
    import statistics

    rows = []
    with open(repo_csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue  # skip errored repos
            rows.append(row)

    if not rows:
        return {"error": "no valid rows"}

    n_formats = [int(r["n_formats"]) for r in rows]
    diversities = [float(r["format_diversity_index"]) for r in rows]
    config_bytes = [int(r["total_config_bytes"]) for r in rows]
    constructs = [int(r["total_constructs"]) for r in rows]
    comment_prev = [float(r["comment_prevalence"]) for r in rows if r["comment_prevalence"]]
    has_fm = sum(1 for r in rows if r["has_frontmatter"] == "True")

    def _stats(values):
        if not values:
            return {}
        return {
            "n": len(values),
            "median": round(statistics.median(values), 4),
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "q1": round(statistics.quantiles(values, n=4)[0], 4) if len(values) >= 4 else None,
            "q3": round(statistics.quantiles(values, n=4)[2], 4) if len(values) >= 4 else None,
        }

    # Format distribution across all repos
    all_formats = Counter()
    for r in rows:
        try:
            fd = json.loads(r["format_distribution"])
            all_formats.update(fd)
        except json.JSONDecodeError:
            pass

    # By repo type
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["repo_type"]].append(int(r["n_formats"]))

    stats = {
        "total_repos_analyzed": len(rows),
        "total_repos_with_errors": 0,  # filled below
        "formats_per_repo": _stats(n_formats),
        "format_diversity_index": _stats(diversities),
        "total_config_bytes": _stats(config_bytes),
        "total_constructs": _stats(constructs),
        "comment_prevalence": _stats(comment_prev),
        "frontmatter_prevalence": round(has_fm / len(rows), 4),
        "global_format_distribution": dict(all_formats.most_common()),
        "formats_per_repo_by_type": {
            rtype: _stats(vals) for rtype, vals in by_type.items()
        },
    }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EXP-005: HuggingFace Format Census")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-models", type=int, default=500,
                        help="Max models to fetch (default: 500)")
    parser.add_argument("--max-adapters", type=int, default=50,
                        help="Max LoRA adapters to fetch (default: 50)")
    parser.add_argument("--max-datasets", type=int, default=50,
                        help="Max datasets to fetch (default: 50)")
    args = parser.parse_args()

    # Verify HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set")
        sys.exit(1)

    api = HfApi(token=token)

    # Record environment
    print("=" * 70)
    print("EXP-005: Format Heterogeneity Census of HuggingFace AI Ecosystem")
    print("=" * 70)
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python: {sys.version}")
    print(f"Target: {args.max_models} models + {args.max_adapters} adapters + {args.max_datasets} datasets")
    print()

    # Fetch repo lists
    all_repos = []
    all_repos.extend(fetch_top_models(api, args.max_models))
    all_repos.extend(fetch_lora_adapters(api, args.max_adapters))
    all_repos.extend(fetch_top_datasets(api, args.max_datasets))
    all_repos.extend(fetch_canonical_datasets(api))

    # Deduplicate (a PEFT model might also be in the top models list)
    seen = set()
    deduped = []
    for r in all_repos:
        if r["repo_id"] not in seen:
            seen.add(r["repo_id"])
            deduped.append(r)
    all_repos = deduped
    print(f"\nTotal unique repos to analyze: {len(all_repos)}")

    # Load checkpoint if resuming
    processed_ids = load_checkpoint() if args.resume else set()

    # Init CSVs
    init_csvs(resume=args.resume)

    # Process each repo
    n_total = len(all_repos)
    n_done = len(processed_ids)
    t_start = time.time()

    for i, repo_info in enumerate(all_repos):
        repo_id = repo_info["repo_id"]

        if repo_id in processed_ids:
            continue

        if repo_id in SKIP_REPOS:
            print(f"  [skipping known-problematic repo: {repo_id}]")
            processed_ids.add(repo_id)
            append_repo_csv(_error_summary(repo_info, "skipped: known problematic"))
            n_done += 1
            continue

        elapsed = time.time() - t_start
        rate = (n_done + 1) / max(elapsed, 1)
        remaining = (n_total - n_done - 1) / max(rate, 0.001)

        print(
            f"[{n_done + 1}/{n_total}] {repo_id} "
            f"({repo_info['repo_type']}, {repo_info['task']}) "
            f"ETA: {remaining/60:.0f}m"
        )

        try:
            summary, details = analyze_repo(api, repo_info, token)
            append_repo_csv(summary)
            if details:
                append_file_csv(details)

            if not summary.get("error"):
                print(
                    f"  -> {summary['n_config_files']} files, "
                    f"{summary['n_formats']} formats, "
                    f"diversity={summary['format_diversity_index']}, "
                    f"constructs={summary['total_constructs']}"
                )
            else:
                print(f"  -> ERROR: {summary['error'][:80]}")

        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            save_checkpoint(list(processed_ids), "interrupted")
            sys.exit(1)
        except Exception as e:
            print(f"  -> UNEXPECTED ERROR: {e}")
            append_repo_csv(_error_summary(repo_info, str(e)))

        processed_ids.add(repo_id)
        n_done += 1

        # Checkpoint periodically
        if n_done % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(list(processed_ids), "in_progress")
            print(f"  [checkpoint saved: {n_done} repos]")

    # Final checkpoint
    save_checkpoint(list(processed_ids), "completed")

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("Computing summary statistics...")
    stats = compute_summary_stats(REPO_SUMMARY_CSV)
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  Repo summaries: {REPO_SUMMARY_CSV}")
    print(f"  File details:   {FILE_DETAIL_CSV}")
    print(f"  Statistics:     {STATS_JSON}")

    # Print headline numbers
    print("\n" + "=" * 70)
    print("HEADLINE RESULTS")
    print("=" * 70)
    if "formats_per_repo" in stats:
        fpr = stats["formats_per_repo"]
        print(f"Formats per repo:  median={fpr['median']}, mean={fpr['mean']:.2f}")
    if "format_diversity_index" in stats:
        fdi = stats["format_diversity_index"]
        print(f"Format diversity:  median={fdi['median']}, mean={fdi['mean']:.2f}")
    if "global_format_distribution" in stats:
        print(f"Global format distribution: {stats['global_format_distribution']}")
    if "frontmatter_prevalence" in stats:
        print(f"YAML frontmatter prevalence: {stats['frontmatter_prevalence']:.1%}")
    if "total_constructs" in stats:
        tc = stats["total_constructs"]
        print(f"Constructs at risk: median={tc['median']}, total sum={tc.get('mean', 0) * tc.get('n', 0):.0f}")

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total/60:.1f} minutes")
    print("Done.")


if __name__ == "__main__":
    main()
