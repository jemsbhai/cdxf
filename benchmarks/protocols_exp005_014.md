
---

## EXP-005: Format Heterogeneity Census of the HuggingFace AI Ecosystem

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational (empirical survey)
**Status:** planned
**Motivation:** To establish CDXF's relevance for ICTAI, we need quantitative
evidence that the AI/ML ecosystem has a format heterogeneity problem. No prior
work has systematically measured this. This census produces both a novel
empirical dataset and the corpus for EXP-006 through EXP-010.

### Hypothesis

The HuggingFace Hub model ecosystem exhibits significant format heterogeneity:
the median model repository contains configuration and metadata files in 3+
distinct text serialization formats (JSON, YAML, TOML, XML/HTML). A substantial
fraction of these files contain format-specific constructs (comments, typed
temporal values, anchors) that are destroyed by existing binary interchange
formats, creating a reproducibility and provenance gap.

### Independent Variables

- **Model popularity tier:** Top 50 by downloads, Top 51-200, Top 201-500
- **Model task type:** text-generation, text-classification, image-classification,
  translation, question-answering, feature-extraction, other
- **Repository type:** base model, fine-tuned model, LoRA adapter, dataset card

### Dependent Variables / Metrics

1. `formats_per_repo` — count of distinct file formats per model repo
   (JSON, YAML, TOML, XML, Markdown-with-YAML-frontmatter, INI, other)
2. `files_per_format` — count of files per format per repo
3. `total_config_bytes` — total size of all config/metadata files per repo
4. `comment_prevalence` — fraction of YAML/TOML files containing comments
5. `anchor_prevalence` — fraction of YAML files containing anchors/aliases
6. `temporal_prevalence` — fraction of files containing typed datetime values
7. `frontmatter_prevalence` — fraction of repos with YAML frontmatter in README
8. `construct_density` — per-file count of format-specific constructs that
   existing binary formats would destroy (comments, anchors, merge keys,
   typed temporals, multi-doc streams)
9. `format_diversity_index` — Shannon entropy of format distribution per repo
   (higher = more heterogeneous)

### Control Conditions

This is a survey experiment — no intervention. The control is the null
hypothesis: "AI model repositories are format-homogeneous (predominantly
JSON-only)."

### Corpus Construction

**Source:** HuggingFace Hub API (https://huggingface.co/api/models)
**Access:** Authenticated via HF Pro account token (HF_TOKEN env var)
**Sample:**
- Top 500 models by total downloads (ensures real-world relevance)
- Stratified by task type to avoid bias toward text-generation
- Include at least 50 LoRA adapter repos (filter: library=peft)
- Include at least 50 dataset cards (filter: type=dataset)
- Total target: 600 repos (500 models + 50 adapters + 50 datasets)

**File collection per repo:**
- List all files via HF Hub API (model.siblings or list_repo_files)
- Download all files matching: *.json, *.yaml, *.yml, *.toml, *.xml,
  *.cfg, *.ini, *.conf, README.md (for YAML frontmatter), *.jsonl
- Exclude: model weights (*.bin, *.safetensors, *.pt, *.gguf),
  tokenizer binaries (*.model), images, videos
- Record: filename, format, size_bytes, SHA-256

### Protocol

1. **Environment setup:**
   ```powershell
   pip install huggingface_hub tiktoken --break-system-packages
   ```
   Record exact library versions.

2. **Fetch model list:**
   - Query HF Hub API for top 500 models by downloads
   - Query top 50 LoRA adapters (filter: library name contains "peft")
   - Query top 50 datasets by downloads
   - Record: model_id, task, downloads, library, author, last_modified

3. **For each repo (rate-limited, with checkpointing):**
   a. List all files via API
   b. Classify each file by format (extension + content sniffing for ambiguous)
   c. Download config/metadata files only (skip weights)
   d. For each downloaded file:
      - Record format, size, SHA-256
      - Parse and count format-specific constructs:
        - JSON: nested depth, array lengths, key count
        - YAML: comment count, anchor count, merge key count, multi-doc count
        - TOML: comment count, temporal value count, inline table count
        - Markdown: YAML frontmatter presence and size
   e. Checkpoint progress every 50 repos (resume-safe)

4. **Analysis:**
   a. Compute all dependent variables per repo
   b. Aggregate: median, mean, std, IQR across population
   c. Stratify by task type and popularity tier
   d. Compute Shannon entropy of format distribution per repo
   e. Count total constructs that would be destroyed by each baseline
      (CBOR, MsgPack, BSON, Ion) — using the feature matrix from EXP-001

5. **Outputs:**
   - CSV: per-repo summary (model_id, task, n_formats, n_files, constructs...)
   - CSV: per-file detail (model_id, filename, format, size, construct_counts)
   - Summary statistics table
   - Distribution plots (histograms, box plots)

### Environment

- **Hardware:** Windows laptop, 64GB RAM, NVIDIA RTX 4090
- **Software:** Python 3.12, Windows 11
- **Key packages:** huggingface_hub, tiktoken (exact versions at runtime)
- **API:** HuggingFace Hub (authenticated, Pro account)
- **Network:** Rate-limited to respect HF API limits (max 10 req/s)
- **Git commit:** TBD (recorded at execution time)
- **Config file:** benchmarks/configs/exp_005_hf_census.yaml
- **Seeds:** N/A (deterministic survey)

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp005.py
- Config: benchmarks/configs/exp_005_hf_census.yaml
- Data: benchmarks/results/exp_005/
- Corpus: benchmarks/results/exp_005/corpus/ (downloaded config files)

---

## EXP-006: ML Configuration Fidelity Under Serialization

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned
**Motivation:** EXP-005 quantifies format heterogeneity. This experiment
measures what happens when you try to serialize those configs with existing
binary formats — how much reproducibility-critical metadata is destroyed?

### Hypothesis

Existing binary interchange formats destroy a significant fraction of the
format-specific metadata in real ML configuration files. Comments documenting
hyperparameter decisions, YAML anchors encoding config cross-references, and
typed temporal values for experiment tracking are lost by all baselines except
CDXF. This metadata loss constitutes a measurable reproducibility gap.

### Independent Variables

- **Source format:** JSON, YAML, TOML (from EXP-005 corpus)
- **Serialization format:** CDXF, CBOR, MsgPack, BSON, Ion, JSON ("just use
  JSON" approach), Pickle
- **Document type:** model config, training config, adapter config, dataset card

### Dependent Variables / Metrics

1. `round_trip_fidelity` — binary: does serialize→deserialize→re-emit produce
   semantically equivalent output?
2. `comments_preserved` — count of comments surviving round-trip / total comments
3. `anchors_preserved` — count of anchors surviving / total anchors
4. `temporal_types_preserved` — typed datetime values surviving / total
5. `merge_keys_preserved` — YAML merge keys surviving / total
6. `metadata_loss_count` — total format-specific constructs lost per format
7. `metadata_loss_rate` — metadata_loss_count / total_constructs (fraction)
8. `size_ratio` — serialized size / original text size

### Control Conditions

**Baselines:**

| Format | Library | Cross-lang? | Preserves comments? |
|--------|---------|-------------|---------------------|
| CDXF | ours | Yes (CBOR) | Yes |
| CBOR | cbor2 | Yes | No |
| MsgPack | msgpack | Yes | No |
| BSON | pymongo | Yes | No |
| Ion | amazon.ion | Yes | No |
| JSON | stdlib | Yes | No |
| Pickle | stdlib | Python-only | No |

### AI/ML-Specific Feature Matrix

Extends EXP-001 matrix with AI/ML-relevant constructs:

| Construct | CDXF | CBOR | MsgPack | BSON | Ion | JSON | Pickle |
|-----------|------|------|---------|------|-----|------|--------|
| HP comments | ? | ? | ? | ? | ? | ? | ? |
| Config anchors | ? | ? | ? | ? | ? | ? | ? |
| Training timestamps | ? | ? | ? | ? | ? | ? | ? |
| Multi-doc streams | ? | ? | ? | ? | ? | ? | ? |
| Cross-language safe | ? | ? | ? | ? | ? | ? | ? |
| Non-string keys | ? | ? | ? | ? | ? | ? | ? |
| Typed local datetime | ? | ? | ? | ? | ? | ? | ? |
| Round-trip fidelity | ? | ? | ? | ? | ? | ? | ? |

### Corpus

Subset of EXP-005 corpus: select the 100 most construct-rich files (highest
comment counts, anchor counts, temporal value counts) plus 50 randomly
sampled files for unbiased comparison.

### Protocol

1. Select corpus subset from EXP-005 results (100 richest + 50 random).
2. For each file:
   a. Count all format-specific constructs (comments, anchors, merge keys,
      typed temporals, multi-doc markers).
   b. Serialize with each baseline format.
   c. Deserialize back.
   d. Re-emit as original format.
   e. Compare: count surviving constructs.
   f. Record per-construct survival, size ratio, fidelity pass/fail.
3. Aggregate: compute metadata_loss_rate per baseline across corpus.
4. Build AI/ML Feature Matrix (fill the table above).
5. Compute "reproducibility impact" metric: for each lost comment, classify
   whether it documents a hyperparameter decision (manual review of 50 files).

### Environment

Same as EXP-005.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp006.py
- Results: benchmarks/results/exp_006/

---

## EXP-007: Cross-Framework Configuration Migration

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned
**Motivation:** AI practitioners frequently migrate between frameworks
(PyTorch Lightning → HuggingFace Trainer, Kubernetes YAML → Terraform JSON).
This experiment measures CDXF's ability to serve as a lossless hub format
for cross-framework config migration.

### Hypothesis

CDXF enables N×N cross-framework configuration migration through a single
hub format with zero data loss for the shared data model, whereas direct
format-to-format conversion loses metadata. For N formats, CDXF requires
2N converter implementations vs N×(N-1) for direct conversion.

### Independent Variables

- **Migration scenario:** (see Protocol for specific pairs)
- **Migration method:** Direct conversion vs CDXF hub
- **Metadata density:** plain config vs annotated config (with comments)

### Dependent Variables / Metrics

1. `data_fidelity` — binary: are all key-value pairs preserved?
2. `metadata_survival` — fraction of comments/annotations surviving
3. `type_preservation` — are numeric/boolean/temporal types preserved?
4. `converter_count` — number of distinct converter implementations needed
5. `total_conversion_time` — wall-clock time for end-to-end migration

### Migration Scenarios

| # | Source | Target | Real-world motivation |
|---|--------|--------|----------------------|
| 1 | PyTorch Lightning YAML | HuggingFace JSON | Framework migration |
| 2 | HuggingFace JSON | TOML (Rust trainer) | Language migration |
| 3 | Kubernetes YAML | Terraform JSON | Infrastructure migration |
| 4 | Hydra YAML (with overrides) | JSON (API config) | Deployment |
| 5 | MLflow YAML | W&B JSON | Experiment tracker migration |
| 6 | Docker Compose YAML | JSON (CI/CD) | DevOps pipeline |
| 7 | pyproject.toml | setup.cfg (INI-like) → JSON | Python packaging |
| 8 | ONNX XML metadata | JSON (TensorRT config) | Model deployment |

### Protocol

1. For each scenario, create or collect a realistic config file in the source
   format. Annotate with comments explaining key decisions.
2. Direct conversion path: parse source → emit target directly (using
   appropriate Python libraries). Record fidelity and metadata survival.
3. CDXF hub path: parse source → CDXF encode → CDXF decode → emit target.
   Record fidelity and metadata survival.
4. Compare: metadata_survival(direct) vs metadata_survival(CDXF_hub).
5. Count total converter implementations for N formats: N×(N-1) vs 2N.

### Environment

Same as EXP-005.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp007.py
- Configs: benchmarks/results/exp_007/scenarios/
- Results: benchmarks/results/exp_007/

---

## EXP-008: LoRA Adapter Registry Metadata Bundle

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned
**Motivation:** LoRA adapters are the dominant parameter-efficient fine-tuning
method. Their metadata is scattered across multiple formats. CDXF can unify
this into a single portable bundle.

### Hypothesis

A LoRA adapter's full metadata provenance chain (adapter config, training
arguments, model card, package metadata) can be captured in a single CDXF
multi-document stream that is (a) smaller than the sum of individual files,
(b) losslessly round-trips every component to its original format, and
(c) enables cross-format querying (e.g., emit training args as TOML).

### Independent Variables

- **Adapter source:** Top 50 LoRA adapters from HuggingFace by downloads
- **Bundle method:** CDXF multi-doc stream vs tar.gz vs mega-JSON vs Pickle
- **Query operation:** extract single component, emit in alternate format

### Dependent Variables / Metrics

1. `bundle_size` — total bytes of the unified bundle
2. `size_vs_sum` — bundle_size / sum(individual file sizes)
3. `round_trip_fidelity` — per-component: does extract→re-emit match original?
4. `cross_format_emit` — can each component be emitted in any target format?
5. `component_access_time` — time to extract a single component from bundle

### Protocol

1. From EXP-005 LoRA adapter subset, select top 50 by downloads.
2. For each adapter, collect: adapter_config.json, training_args (JSON/YAML),
   README.md (YAML frontmatter), any pyproject.toml or requirements.txt.
3. Bundle all components into:
   a. CDXF multi-document stream (one Document per file, preserving format)
   b. tar.gz of originals
   c. Single JSON (merge all, converting non-JSON to JSON)
   d. Pickle (serialize Python objects)
4. Measure bundle size, round-trip fidelity, cross-format emission.
5. Benchmark: extract adapter_config from each bundle format. Measure time.

### Environment

Same as EXP-005.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp008.py
- Results: benchmarks/results/exp_008/

---

## EXP-009: End-to-End Fine-Tuning Pipeline State Capture

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned
**Motivation:** ML reproducibility requires capturing the complete pipeline
state: dataset card, preprocessing config, training hyperparameters, adapter
config, quantization config, serving config. These span multiple formats.
No existing single-file format captures all of them losslessly.

### Hypothesis

CDXF can capture a complete fine-tuning pipeline state in a single portable
binary file that (a) is smaller than a tar.gz of the originals, (b) preserves
all format-specific metadata (comments documenting decisions, typed values),
(c) enables diffing between experiment runs at the config level, and
(d) is cross-language readable (unlike Pickle).

### Independent Variables

- **Pipeline complexity:** minimal (3 configs) vs full (8 configs)
- **State format:** CDXF stream vs tar.gz vs mega-JSON vs Pickle
- **Operation:** full capture, single-component extract, state diff

### Dependent Variables / Metrics

1. `capture_size` — total bytes for full pipeline state
2. `capture_time` — time to serialize all components
3. `extract_time` — time to extract a single component
4. `diff_capability` — can two states be meaningfully compared? (binary)
5. `cross_language` — is the format readable without Python? (binary)
6. `metadata_preserved` — fraction of comments/annotations surviving

### Pipeline Components (Realistic Fine-Tuning)

| Component | Format | Source |
|-----------|--------|--------|
| Dataset card | YAML frontmatter | HuggingFace dataset card |
| Data preprocessing | JSON | Custom preprocessing config |
| Training hyperparameters | YAML (with comments) | Hydra/Lightning config |
| LoRA adapter config | JSON | PEFT adapter_config |
| Quantization config | JSON | BitsAndBytes config |
| Serving config | TOML | vLLM/TGI serving config |
| Evaluation results | JSON | lm-eval-harness output |
| Deployment manifest | YAML | Kubernetes deployment |

### Protocol

1. Construct a realistic pipeline state using real or realistic configs for
   each component. Annotate YAML files with comments documenting decisions.
2. Capture full state with each method (CDXF, tar.gz, mega-JSON, Pickle).
3. Measure size, capture time, extract time.
4. Test metadata preservation: round-trip each component.
5. Test diff capability: create a second state with one HP change (lr: 1e-4
   → 1e-5). Can each format surface the diff?
6. Test cross-language: can a non-Python tool read the state? (CDXF → any
   CBOR library; JSON → universal; Pickle → Python only)

### Environment

Same as EXP-005.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp009.py
- Pipeline configs: benchmarks/results/exp_009/pipeline/
- Results: benchmarks/results/exp_009/

---

## EXP-010: AI/ML Configuration Throughput Benchmark

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational (performance)
**Status:** planned
**Motivation:** Extends EXP-002 to AI/ML-specific workloads. Adds Pickle as
a baseline (the de facto ML serialization) and measures the "format tax" —
config serialization overhead relative to actual training time.

### Hypothesis

CDXF encode/decode throughput for ML configuration files is fast enough that
the "format tax" (time spent on config serialization vs actual training) is
negligible (<0.01% of a typical fine-tuning run). CDXF is competitive with
Pickle on speed while being cross-language safe and metadata-preserving.

### Independent Variables

- **Operation:** encode vs decode
- **Format:** CDXF, CBOR, MsgPack, BSON, Ion, JSON, Pickle
- **Document source:** ML configs from EXP-005 corpus
- **Document size:** small (<1KB), medium (1-10KB), large (>10KB)

### Dependent Variables / Metrics

1. `ops_per_sec` — operations per second (median, mean, std, 95% CI)
2. `throughput_bytes_per_sec` — bytes processed per second
3. `format_tax_fraction` — serialization time / typical training step time
4. `format_tax_per_epoch` — total serialization overhead per training epoch
   (assuming configs read once per epoch)

### Statistical Protocol

- **Warm-up:** 10 iterations discarded
- **Measurement:** 1000 iterations per (file, operation, format) triple
- **Reporting:** median, mean, std, 95% CI, min, max
- **GC control:** gc.disable() during measurement, gc.collect() between files
- **Timer:** time.perf_counter_ns()
- **Outlier handling:** report all data, do NOT remove outliers

### "Format Tax" Calculation

To contextualize: a single fine-tuning step on a 7B model takes ~200-500ms
on an A100. A typical epoch has ~1000 steps. If config serialization takes
T_config ms and is done once per epoch, the format tax is:
  format_tax = T_config / (steps_per_epoch × T_step)

We will compute this for: T_step = {200ms, 500ms}, steps_per_epoch = {100,
1000, 10000}, using measured T_config values.

### Environment

Same as EXP-005.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp010.py
- Results: benchmarks/results/exp_010/

---

## EXP-011: Token Cost of Format Syntax — The "Syntax Tax"

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned
**Motivation:** LLM-based AI agents consume tokens proportional to input
length. Text serialization formats waste tokens on syntax characters (braces,
brackets, closing tags, indentation, quotes) that carry structural information
already encoded in the data model. This experiment measures the "syntax tax"
— the fraction of tokens wasted on format syntax rather than data content —
and projects the context window savings if agents used a non-text
intermediate representation.

### Hypothesis

Text serialization formats impose a measurable "syntax tax" on LLM agents:
a significant fraction (20-50%) of tokens consumed when processing
configuration files encode format syntax rather than semantic content. XML
has the highest syntax tax (redundant closing tags), followed by JSON
(braces, brackets, quotes), TOML, and YAML. For an agent managing N config
files in a 128K context window, eliminating the syntax tax frees a
quantifiable number of tokens for reasoning.

### Independent Variables

- **Format:** JSON, YAML, XML, TOML
- **Tokenizer:** cl100k_base (GPT-4/Claude), o200k_base (GPT-4o)
- **Document source:** ML configs from EXP-005 corpus + all 43 files from
  EXP-001 corpus
- **Document size:** small (<1KB), medium (1-10KB), large (>10KB)

### Dependent Variables / Metrics

1. `total_tokens` — token count for the full document
2. `syntax_tokens` — tokens consumed by format syntax (see classification below)
3. `semantic_tokens` — tokens consumed by data content (keys + values)
4. `syntax_tax_rate` — syntax_tokens / total_tokens (fraction)
5. `tokens_per_byte` — total_tokens / size_bytes (tokenization efficiency)
6. `context_waste_projection` — for K files of median size, total syntax
   tokens wasted as fraction of 128K context window
7. `effective_context_gain` — additional files that could fit in context if
   syntax were eliminated

### Token Classification Protocol

For each format, define syntax vs semantic tokens:

**JSON syntax:** `{`, `}`, `[`, `]`, `:`, `,`, `"` around keys
**JSON semantic:** key names, string values, numbers, booleans, null

**YAML syntax:** `:`, `-`, indentation whitespace, `---`, `...`, `>`, `|`
**YAML semantic:** key names, values, comments (they ARE semantic content)

**XML syntax:** `<`, `>`, `</`, `/>`, attribute `=`, `"` around attr values,
  closing tag names (redundant — repeat the opening tag name)
**XML semantic:** element names (first occurrence), attribute names, text
  content, attribute values, comments, PI content

**TOML syntax:** `[`, `]`, `=`, `"`, `,`, `{`, `}`
**TOML semantic:** key names, values, section names, comments

Classification method: tokenize the full document, then parse and identify
which tokens correspond to syntax characters vs data content. For tokens that
span syntax and content (e.g., a token containing both `"` and the start of
a key), attribute proportionally by character count.

### Protocol

1. Install tiktoken. Record version.
2. For each file in corpus:
   a. Tokenize with cl100k_base and o200k_base.
   b. Parse the file to identify syntax vs semantic spans.
   c. Map each token to its source characters.
   d. Classify each token as syntax, semantic, or mixed.
   e. For mixed tokens, attribute proportionally.
   f. Record all metrics.
3. Aggregate per format: median, mean, std, 95% CI of syntax_tax_rate.
4. Project context savings:
   a. For a typical agentic session touching N={10, 25, 50, 100} config files
   b. Compute total syntax tokens wasted
   c. Express as fraction of 128K context window
   d. Express as equivalent number of additional files that could fit

### Environment

Same as EXP-005. Additional: tiktoken library.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp011.py
- Results: benchmarks/results/exp_011/

---

## EXP-012: Agentic Tool Schema Consolidation Overhead

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned
**Motivation:** LLM agents that handle multiple config formats need multiple
tool definitions in their function-calling schema. Each tool description
consumes context tokens. CDXF consolidates N format-specific tools into a
small set of universal tools. This experiment measures the token savings.

### Hypothesis

An agentic AI system handling 4 text formats (JSON, YAML, XML, TOML)
requires at least 8 format-specific tool definitions (parse + emit per
format). CDXF consolidates this to 3 universal tools (encode, decode,
convert). The token savings from tool schema consolidation represent a
measurable fraction of the agent's context budget, with the savings
growing linearly with the number of supported formats.

### Independent Variables

- **Approach:** Format-specific tools (N per format) vs CDXF universal tools
- **Number of formats:** 2, 3, 4, 5 (to show scaling)
- **Tool schema format:** OpenAI function-calling format, Anthropic tool format
- **Schema detail level:** minimal (name + description) vs full (with
  parameter schemas and examples)

### Dependent Variables / Metrics

1. `tool_schema_tokens` — total tokens for all tool definitions
2. `tokens_saved` — format_specific_tokens - cdxf_tokens
3. `savings_fraction` — tokens_saved / 128K context window
4. `scaling_factor` — how does savings grow with N formats?
5. `call_result_tokens` — tokens consumed by tool call results (parsed data
   representation) per format vs CDXF
6. `total_session_overhead` — tool_schema_tokens + expected_call_result_tokens
   for a typical session with M tool calls

### Protocol

1. Write realistic tool definitions for each format-specific tool:
   - parse_json, emit_json (with JSON Schema for parameters)
   - parse_yaml, emit_yaml
   - parse_xml, emit_xml
   - parse_toml, emit_toml
   Total: 8 tools for 4 formats

2. Write equivalent CDXF tool definitions:
   - cdxf_encode (any format → CDXF binary)
   - cdxf_decode (CDXF binary → any format)
   - cdxf_convert (source format → target format via CDXF)
   Total: 3 tools

3. Tokenize all tool definitions with tiktoken (cl100k_base, o200k_base).

4. Measure tool call result sizes: for 20 sample configs from EXP-005 corpus,
   measure the token count of the tool's return value (parsed data
   representation) for each approach.

5. Project total session overhead for typical agentic scenarios:
   - Light session: 5 tool calls
   - Medium session: 20 tool calls
   - Heavy session: 100 tool calls

6. Scale analysis: repeat steps 1-3 for N = {2, 3, 4, 5, 6} formats to
   show O(N) vs O(1) scaling.

### Environment

Same as EXP-005.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp012.py
- Tool definitions: benchmarks/results/exp_012/tool_schemas/
- Results: benchmarks/results/exp_012/

---

## EXP-013: Agent Workflow State Persistence Across Sessions

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational (simulation)
**Status:** planned
**Motivation:** Agentic AI researchers operate across multiple sessions with
limited context windows. Between sessions, the agent must serialize its
working state (configs read, results obtained, decisions made). If the
serialization format loses metadata, information degrades with each session
boundary. This experiment measures cumulative information loss over simulated
multi-session workflows.

### Hypothesis

Over K sequential session boundaries, storing agent state in a lossy format
(JSON, Pickle) causes cumulative information loss that grows linearly with K.
CDXF maintains zero information loss regardless of the number of session
boundaries. After 10 session boundaries, the lossy formats have lost a
quantifiable fraction of the original metadata (comments, typed values,
structural annotations).

### Independent Variables

- **Number of session boundaries (K):** 1, 2, 5, 10, 20
- **State format:** CDXF, JSON ("mega-JSON"), Pickle, tar.gz of originals
- **State complexity:** small (3 files), medium (8 files), large (15 files)
- **Operations per session:** read state, modify one config, add one result
  file, re-serialize

### Dependent Variables / Metrics

1. `metadata_surviving_fraction` — (comments + anchors + typed values
   surviving after K boundaries) / original count
2. `cumulative_loss_count` — total constructs lost after K boundaries
3. `information_entropy` — Shannon entropy of the state content after K
   boundaries (lower = more information lost / collapsed)
4. `state_size_drift` — how does serialized size change over K boundaries?
   (lossy formats may shrink as metadata is stripped)
5. `round_trip_fidelity_K` — does original→K boundaries→original match?

### Simulation Protocol

The simulation models a realistic autonomous research agent workflow:

**Session 0 (initialization):**
- Create initial state with 8 config files (from EXP-009 pipeline):
  - dataset_card.yaml (with comments, 15 comments documenting data decisions)
  - preprocess_config.json
  - training_config.yaml (with comments on HP choices, 20 comments)
  - adapter_config.json
  - quant_config.json
  - serving_config.toml (with comments, 8 comments)
  - eval_results.json
  - deployment.yaml (with comments, 12 comments)
- Count total metadata constructs: comments, anchors, typed temporals.

**Session k (for k = 1 to K):**
1. Deserialize state from previous session's format.
2. Read one config file (randomly selected).
3. Modify one value (e.g., change learning_rate).
4. Add a new result file (eval_results_k.json).
5. Re-serialize entire state.
6. Count surviving metadata constructs.

**Measurement:**
- After each session boundary, count surviving constructs.
- Plot: metadata_surviving_fraction vs K for each format.
- Expected: CDXF = flat line at 1.0; JSON/Pickle = monotonically decreasing.

### Protocol

1. Create the initial state files with known construct counts.
2. Implement the session simulation loop.
3. For each format, run the loop for K = 1 to 20.
4. Record construct counts after each boundary.
5. Plot degradation curves.
6. Compute the "half-life" of metadata for lossy formats: at what K does
   50% of metadata survive?

### Environment

Same as EXP-005.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp013.py
- Initial state: benchmarks/results/exp_013/initial_state/
- Results: benchmarks/results/exp_013/

---

## EXP-014: Multi-Agent Format Interchange — Hub vs Direct Conversion

**Date:** 2026-05-24
**Researcher:** Muntaser Syed
**Type:** computational (simulation)
**Status:** planned
**Motivation:** Multi-agent AI systems involve agents with different format
preferences (one emits YAML, another needs JSON, a third needs TOML). Direct
format conversion requires O(N²) converters. CDXF as a hub format requires
O(N) converters. This experiment measures the engineering complexity, latency,
and fidelity trade-offs.

### Hypothesis

For a multi-agent system with A agents using N distinct formats:
(a) CDXF hub architecture requires 2N converter implementations vs N×(N-1)
    for direct conversion — a quadratic-to-linear reduction.
(b) CDXF hub introduces at most 2× the latency of a single direct conversion
    (encode + decode vs single conversion), which is negligible for config-
    sized documents.
(c) CDXF hub preserves all metadata at every handoff, while direct converters
    lose metadata at each step (compounding over multi-hop transfers).
(d) For a pipeline of H sequential handoffs, CDXF hub has zero cumulative
    metadata loss while direct conversion loses metadata proportional to H.

### Independent Variables

- **Number of agents (A):** 2, 3, 4, 5
- **Number of formats (N):** 2, 3, 4 (subset of JSON, YAML, XML, TOML)
- **Pipeline depth (H):** 1, 2, 3, 5 sequential handoffs
- **Conversion method:** Direct vs CDXF hub

### Dependent Variables / Metrics

1. `converter_count` — number of distinct converter implementations needed
2. `handoff_latency` — time for a single inter-agent format conversion
3. `pipeline_latency` — total time for H sequential handoffs
4. `metadata_after_H_hops` — constructs surviving after H handoffs (fraction)
5. `total_bytes_transferred` — sum of all inter-agent data transfers
6. `implementation_complexity` — lines of code for all converters (proxy)

### Multi-Agent Pipeline Scenario

| Agent | Role | Preferred Format | Consumes | Produces |
|-------|------|------------------|----------|----------|
| A (Curator) | Data preparation | JSON | raw data | dataset manifest (JSON) |
| B (Trainer) | Model training | YAML | dataset manifest, HP config | training logs, checkpoints |
| C (Evaluator) | Evaluation | JSON | model, eval config | eval results (JSON) |
| D (Deployer) | Deployment | TOML | model, serving config | deployment manifest (TOML) |
| E (Monitor) | Monitoring/reporting | XML | all outputs | monitoring report (XML) |

### Protocol

1. Implement the 5-agent pipeline scenario above.
2. For direct conversion: implement all N×(N-1) pairwise converters.
3. For CDXF hub: implement N encode + N decode converters.
4. Run the pipeline for H = {1, 2, 3, 5} sequential handoffs.
5. At each handoff, measure: latency, bytes transferred, metadata survival.
6. Vary N from 2 to 4 and plot converter_count scaling.
7. Compare cumulative metadata loss: direct (compounding) vs CDXF (zero).

### Environment

Same as EXP-005.

### Results

*To be filled after experiment completes.*

### Artifacts

- Script: benchmarks/src/run_exp014.py
- Agent implementations: benchmarks/results/exp_014/agents/
- Results: benchmarks/results/exp_014/
