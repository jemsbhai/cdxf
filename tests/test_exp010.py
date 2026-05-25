"""Tests for EXP-010: AI/ML Configuration Throughput Benchmark.

Extends EXP-002 to AI/ML-specific workloads with Pickle baseline
and "format tax" analysis. Tests corpus construction, size classification,
statistical computation, benchmarking mechanics, format tax math,
all 8 baselines, environment capture, and scientific rigor constraints.

TDD: These tests are written BEFORE the implementation.
"""

import gc
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.src.run_exp010 import (
    BASELINES,
    FORMAT_TAX_STEPS,
    FORMAT_TAX_T_STEPS,
    ITERATIONS,
    SIZE_THRESHOLDS,
    WARMUP,
    bench,
    build_ml_corpus,
    classify_size,
    compute_format_tax,
    encode_decode_fns,
    summarize_stats,
    capture_environment,
    run_single_file,
    run_experiment,
)


# ===========================================================================
# Constants and protocol compliance
# ===========================================================================


class TestProtocolConstants:
    """Verify protocol-mandated constants are correct."""

    def test_warmup_is_10(self):
        assert WARMUP == 10

    def test_iterations_is_1000(self):
        assert ITERATIONS == 1000

    def test_size_thresholds(self):
        """Protocol: small (<1KB), medium (1-10KB), large (>10KB)."""
        assert SIZE_THRESHOLDS["small_upper"] == 1024
        assert SIZE_THRESHOLDS["medium_upper"] == 10240

    def test_format_tax_t_steps(self):
        """Protocol: T_step = {200ms, 500ms}."""
        assert set(FORMAT_TAX_T_STEPS) == {0.2, 0.5}

    def test_format_tax_steps_per_epoch(self):
        """Protocol: steps_per_epoch = {100, 1000, 10000}."""
        assert set(FORMAT_TAX_STEPS) == {100, 1000, 10000}

    def test_baselines_include_all_required(self):
        """Protocol mandates 7 formats: CDXF, CBOR, MsgPack, BSON, Ion, JSON, Pickle."""
        required = {"cdxf_full", "cdxf_codec", "cbor", "msgpack", "bson", "ion", "json_stdlib", "pickle"}
        assert required.issubset(set(BASELINES))

    def test_pickle_is_a_baseline(self):
        """EXP-010 specifically adds Pickle as a new baseline vs EXP-002."""
        assert "pickle" in BASELINES


# ===========================================================================
# Corpus construction
# ===========================================================================


class TestBuildMlCorpus:
    """Corpus must be realistic ML configs with known properties."""

    def test_returns_list(self):
        corpus = build_ml_corpus()
        assert isinstance(corpus, list)

    def test_non_empty(self):
        corpus = build_ml_corpus()
        assert len(corpus) > 0

    def test_minimum_corpus_size(self):
        """Need enough files for statistical validity across size bins."""
        corpus = build_ml_corpus()
        assert len(corpus) >= 10

    def test_entry_required_fields(self):
        """Each corpus entry must have name, format, text, category."""
        corpus = build_ml_corpus()
        for entry in corpus:
            assert "name" in entry, f"Missing 'name' in {entry.keys()}"
            assert "format" in entry, f"Missing 'format' in {entry.keys()}"
            assert "text" in entry, f"Missing 'text' in {entry.keys()}"
            assert "category" in entry, f"Missing 'category' in {entry.keys()}"

    def test_text_is_non_empty_string(self):
        corpus = build_ml_corpus()
        for entry in corpus:
            assert isinstance(entry["text"], str)
            assert len(entry["text"]) > 0, f"Empty text for {entry['name']}"

    def test_format_is_valid(self):
        """Only JSON, YAML, TOML, XML are valid source formats."""
        valid = {"json", "yaml", "toml", "xml"}
        corpus = build_ml_corpus()
        for entry in corpus:
            assert entry["format"] in valid, f"Bad format '{entry['format']}' for {entry['name']}"

    def test_has_multiple_formats(self):
        """Corpus must span at least JSON and YAML (dominant ML formats)."""
        corpus = build_ml_corpus()
        formats = {e["format"] for e in corpus}
        assert "json" in formats
        assert "yaml" in formats

    def test_has_multiple_size_bins(self):
        """Corpus should cover small, medium, and large files."""
        corpus = build_ml_corpus()
        sizes = [len(e["text"].encode("utf-8")) for e in corpus]
        has_small = any(s < 1024 for s in sizes)
        has_medium = any(1024 <= s < 10240 for s in sizes)
        has_large = any(s >= 10240 for s in sizes)
        assert has_small, "No small (<1KB) files in corpus"
        assert has_medium, "No medium (1-10KB) files in corpus"
        assert has_large, "No large (>10KB) files in corpus"

    def test_names_are_unique(self):
        corpus = build_ml_corpus()
        names = [e["name"] for e in corpus]
        assert len(names) == len(set(names)), "Duplicate corpus names"

    def test_parseable(self):
        """Every file must actually parse in its declared format."""
        corpus = build_ml_corpus()
        for entry in corpus:
            fmt = entry["format"]
            text = entry["text"]
            try:
                if fmt == "json":
                    json.loads(text)
                elif fmt == "yaml":
                    from ruamel.yaml import YAML
                    y = YAML(typ="safe")
                    list(y.load_all(text))
                elif fmt == "toml":
                    import tomlkit
                    tomlkit.parse(text)
                elif fmt == "xml":
                    import xml.etree.ElementTree as ET
                    ET.fromstring(text)
            except Exception as e:
                pytest.fail(f"Corpus file '{entry['name']}' ({fmt}) does not parse: {e}")

    def test_has_yaml_with_comments(self):
        """ML configs in YAML typically have comments — must be present."""
        corpus = build_ml_corpus()
        yaml_with_comments = [
            e for e in corpus
            if e["format"] == "yaml" and "#" in e["text"]
        ]
        assert len(yaml_with_comments) >= 1, "No YAML files with comments"

    def test_has_ml_relevant_categories(self):
        """Categories should reflect ML config types."""
        corpus = build_ml_corpus()
        categories = {e["category"] for e in corpus}
        # Must have at least training and model categories
        assert len(categories) >= 2, f"Only {categories} — need variety"


# ===========================================================================
# Size classification
# ===========================================================================


class TestClassifySize:
    def test_small_below_1kb(self):
        assert classify_size(0) == "small"
        assert classify_size(512) == "small"
        assert classify_size(1023) == "small"

    def test_medium_1kb_to_10kb(self):
        assert classify_size(1024) == "medium"
        assert classify_size(5000) == "medium"
        assert classify_size(10239) == "medium"

    def test_large_above_10kb(self):
        assert classify_size(10240) == "large"
        assert classify_size(100000) == "large"
        assert classify_size(1_000_000) == "large"

    def test_boundary_small_medium(self):
        """Exact boundary: 1024 bytes should be medium (>=1KB)."""
        assert classify_size(1023) == "small"
        assert classify_size(1024) == "medium"

    def test_boundary_medium_large(self):
        """Exact boundary: 10240 bytes should be large (>=10KB)."""
        assert classify_size(10239) == "medium"
        assert classify_size(10240) == "large"


# ===========================================================================
# Statistical summary
# ===========================================================================


class TestSummarizeStats:
    """Verify correctness of statistical computations."""

    def test_returns_dict(self):
        result = summarize_stats([1.0, 2.0, 3.0])
        assert isinstance(result, dict)

    def test_required_fields(self):
        """Protocol requires: n, median, mean, std, 95% CI, min, max."""
        result = summarize_stats([1.0, 2.0, 3.0])
        required = {"n", "median_s", "mean_s", "std_s",
                     "ci95_lower_s", "ci95_upper_s", "min_s", "max_s"}
        assert required.issubset(set(result.keys()))

    def test_n_correct(self):
        result = summarize_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["n"] == 5

    def test_median_correct(self):
        result = summarize_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["median_s"] == pytest.approx(3.0)

    def test_mean_correct(self):
        result = summarize_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["mean_s"] == pytest.approx(3.0)

    def test_std_correct(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = summarize_stats(data)
        expected_std = statistics.stdev(data)
        assert result["std_s"] == pytest.approx(expected_std)

    def test_ci95_symmetric_around_mean(self):
        """95% CI must be symmetric around the mean."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = summarize_stats(data)
        mean = result["mean_s"]
        lower_delta = mean - result["ci95_lower_s"]
        upper_delta = result["ci95_upper_s"] - mean
        assert lower_delta == pytest.approx(upper_delta, abs=1e-12)

    def test_ci95_width_correct(self):
        """CI half-width = 1.96 * std / sqrt(n)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = summarize_stats(data)
        expected_half = 1.96 * statistics.stdev(data) / math.sqrt(len(data))
        actual_half = result["ci95_upper_s"] - result["mean_s"]
        assert actual_half == pytest.approx(expected_half, abs=1e-12)

    def test_min_max_correct(self):
        data = [5.0, 1.0, 3.0, 2.0, 4.0]
        result = summarize_stats(data)
        assert result["min_s"] == pytest.approx(1.0)
        assert result["max_s"] == pytest.approx(5.0)

    def test_single_element(self):
        """Single-element list should work with std=0, CI=0."""
        result = summarize_stats([42.0])
        assert result["n"] == 1
        assert result["median_s"] == pytest.approx(42.0)
        assert result["mean_s"] == pytest.approx(42.0)
        assert result["std_s"] == pytest.approx(0.0)
        # CI should collapse to a point
        assert result["ci95_lower_s"] == pytest.approx(42.0)
        assert result["ci95_upper_s"] == pytest.approx(42.0)

    def test_identical_values(self):
        """All identical values => std=0, CI width=0."""
        data = [7.0] * 100
        result = summarize_stats(data)
        assert result["std_s"] == pytest.approx(0.0)
        assert result["ci95_lower_s"] == pytest.approx(result["mean_s"])
        assert result["ci95_upper_s"] == pytest.approx(result["mean_s"])

    def test_no_outlier_removal(self):
        """Protocol: report ALL data, do NOT remove outliers.
        Verify min/max reflect actual extremes even with outlier present."""
        data = [1.0] * 99 + [1000.0]  # extreme outlier
        result = summarize_stats(data)
        assert result["max_s"] == pytest.approx(1000.0)
        assert result["n"] == 100


# ===========================================================================
# Bench function (timing mechanics)
# ===========================================================================


class TestBenchFunction:
    """Verify the bench() function follows the statistical protocol."""

    def test_returns_list_of_floats(self):
        times = bench(lambda: None, iterations=10, warmup=2)
        assert isinstance(times, list)
        assert all(isinstance(t, float) for t in times)

    def test_returns_correct_count(self):
        """Must return exactly `iterations` measurements (warm-up excluded)."""
        times = bench(lambda: None, iterations=50, warmup=5)
        assert len(times) == 50

    def test_warmup_excluded_from_results(self):
        """Warm-up calls must not appear in the returned timings."""
        call_count = [0]

        def counter():
            call_count[0] += 1

        times = bench(counter, iterations=20, warmup=5)
        assert len(times) == 20
        assert call_count[0] == 25  # 5 warm-up + 20 measurement

    def test_times_are_non_negative(self):
        times = bench(lambda: time.sleep(0), iterations=10, warmup=1)
        assert all(t >= 0 for t in times)

    def test_uses_perf_counter_ns(self):
        """Ensure timing resolution is sub-microsecond (perf_counter_ns)."""
        times = bench(lambda: None, iterations=100, warmup=5)
        # At least some measurements should be distinct (not all zero)
        # A trivial lambda may occasionally round to 0, but not all 100
        distinct = len(set(times))
        assert distinct >= 2, "All 100 timings identical — timer resolution suspect"

    def test_gc_disabled_during_measurement(self):
        """GC must be disabled during the measurement loop.

        We verify by checking that gc.isenabled() is True after bench()
        returns (it was re-enabled), and that the function ran successfully.
        """
        times = bench(lambda: None, iterations=10, warmup=2)
        assert gc.isenabled(), "GC not re-enabled after bench()"
        assert len(times) == 10

    def test_gc_restored_on_exception(self):
        """If the benchmarked function raises, GC must still be re-enabled."""
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] > 8:  # fail during measurement
                raise ValueError("boom")

        with pytest.raises(ValueError):
            bench(flaky, iterations=20, warmup=2)
        assert gc.isenabled(), "GC not re-enabled after exception in bench()"

    def test_measurement_precision(self):
        """A ~1ms sleep should measure in the ~0.001s range, not 0."""
        times = bench(lambda: time.sleep(0.001), iterations=5, warmup=1)
        median_t = statistics.median(times)
        assert median_t > 0.0005, f"Median {median_t} too low for 1ms sleep"
        assert median_t < 0.01, f"Median {median_t} too high for 1ms sleep"


# ===========================================================================
# Encode/decode function registry
# ===========================================================================


class TestEncodeDecodeFns:
    """Each baseline must provide encode and decode callables."""

    SIMPLE_JSON = '{"lr": 0.001, "epochs": 10}'
    SIMPLE_YAML = "lr: 0.001\nepochs: 10\n"
    SIMPLE_TOML = 'lr = 0.001\nepochs = 10\n'

    def test_returns_dict(self):
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        assert isinstance(fns, dict)

    def test_all_baselines_present_for_json(self):
        """JSON files should work with all baselines."""
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        for bl in BASELINES:
            assert bl in fns, f"Baseline '{bl}' missing for JSON"

    def test_each_baseline_has_encode_and_decode(self):
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        for bl, ops in fns.items():
            assert "encode" in ops, f"{bl} missing 'encode'"
            assert "decode" in ops, f"{bl} missing 'decode'"

    def test_encode_is_callable(self):
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        for bl, ops in fns.items():
            assert callable(ops["encode"]), f"{bl} encode not callable"
            assert callable(ops["decode"]), f"{bl} decode not callable"

    def test_cdxf_full_roundtrip_json(self):
        """CDXF full pipeline must round-trip JSON data."""
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        encoded = fns["cdxf_full"]["encode"]()
        assert isinstance(encoded, (bytes, str))

    def test_cdxf_codec_roundtrip_json(self):
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        encoded = fns["cdxf_codec"]["encode"]()
        assert isinstance(encoded, bytes)

    def test_pickle_roundtrip(self):
        """Pickle must serialize/deserialize native Python dicts."""
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        encoded = fns["pickle"]["encode"]()
        assert isinstance(encoded, bytes)
        decoded = fns["pickle"]["decode"]()
        assert isinstance(decoded, dict)

    def test_json_stdlib_encode_returns_bytes_or_str(self):
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        result = fns["json_stdlib"]["encode"]()
        assert isinstance(result, (bytes, str))

    def test_cbor_encode_returns_bytes(self):
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        result = fns["cbor"]["encode"]()
        assert isinstance(result, bytes)

    def test_msgpack_encode_returns_bytes(self):
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        result = fns["msgpack"]["encode"]()
        assert isinstance(result, bytes)

    def test_yaml_source_has_cdxf_and_pickle(self):
        """YAML-source files must also work with CDXF and Pickle."""
        fns = encode_decode_fns(self.SIMPLE_YAML, "yaml")
        assert "cdxf_full" in fns
        assert "cdxf_codec" in fns
        assert "pickle" in fns

    def test_toml_source_has_cdxf_and_pickle(self):
        fns = encode_decode_fns(self.SIMPLE_TOML, "toml")
        assert "cdxf_full" in fns
        assert "cdxf_codec" in fns
        assert "pickle" in fns

    def test_unsupported_format_skips_gracefully(self):
        """Baselines that can't handle a format should be absent, not error."""
        fns = encode_decode_fns("<root/>", "xml")
        # XML has no native dict representation, some baselines may be absent
        # But CDXF should always be present
        assert "cdxf_full" in fns

    def test_encode_output_size_gt_zero(self):
        """Encoded output must be non-empty."""
        fns = encode_decode_fns(self.SIMPLE_JSON, "json")
        for bl, ops in fns.items():
            encoded = ops["encode"]()
            if isinstance(encoded, bytes):
                assert len(encoded) > 0, f"{bl} encode returned empty bytes"
            elif isinstance(encoded, str):
                assert len(encoded) > 0, f"{bl} encode returned empty string"


# ===========================================================================
# Format tax computation
# ===========================================================================


class TestComputeFormatTax:
    """Verify format tax = T_config / (steps_per_epoch * T_step)."""

    def test_basic_calculation(self):
        # T_config = 0.001s, steps = 1000, T_step = 0.2s
        # tax = 0.001 / (1000 * 0.2) = 0.001 / 200 = 5e-6
        result = compute_format_tax(0.001, steps_per_epoch=1000, t_step_s=0.2)
        assert result == pytest.approx(5e-6)

    def test_zero_config_time(self):
        result = compute_format_tax(0.0, steps_per_epoch=1000, t_step_s=0.2)
        assert result == pytest.approx(0.0)

    def test_small_config_time(self):
        """A 10μs config time should be negligible."""
        result = compute_format_tax(1e-5, steps_per_epoch=1000, t_step_s=0.5)
        # 1e-5 / (1000 * 0.5) = 1e-5 / 500 = 2e-8
        assert result == pytest.approx(2e-8)

    def test_large_config_time(self):
        """Even a 1s config time should be small relative to training."""
        result = compute_format_tax(1.0, steps_per_epoch=10000, t_step_s=0.5)
        # 1.0 / (10000 * 0.5) = 1.0 / 5000 = 0.0002
        assert result == pytest.approx(0.0002)

    def test_all_protocol_combinations(self):
        """Verify formula for all (T_step, steps) combinations in protocol."""
        t_config = 0.01  # 10ms
        for t_step in FORMAT_TAX_T_STEPS:
            for steps in FORMAT_TAX_STEPS:
                result = compute_format_tax(t_config, steps, t_step)
                expected = t_config / (steps * t_step)
                assert result == pytest.approx(expected), (
                    f"Failed for t_step={t_step}, steps={steps}"
                )

    def test_returns_float(self):
        result = compute_format_tax(0.001, 1000, 0.2)
        assert isinstance(result, float)

    def test_format_tax_always_non_negative(self):
        """Tax cannot be negative (both time and steps are positive)."""
        result = compute_format_tax(0.001, 100, 0.2)
        assert result >= 0

    def test_inversely_proportional_to_steps(self):
        """Doubling steps should halve the tax."""
        tax_100 = compute_format_tax(0.001, 100, 0.2)
        tax_200 = compute_format_tax(0.001, 200, 0.2)
        assert tax_100 == pytest.approx(2 * tax_200)

    def test_inversely_proportional_to_t_step(self):
        """Doubling T_step should halve the tax."""
        tax_02 = compute_format_tax(0.001, 1000, 0.2)
        tax_04 = compute_format_tax(0.001, 1000, 0.4)
        assert tax_02 == pytest.approx(2 * tax_04)


# ===========================================================================
# Environment capture
# ===========================================================================


class TestCaptureEnvironment:
    def test_returns_dict(self):
        env = capture_environment()
        assert isinstance(env, dict)

    def test_has_timestamp(self):
        env = capture_environment()
        assert "timestamp" in env

    def test_has_python_version(self):
        env = capture_environment()
        assert "python_version" in env

    def test_has_os(self):
        env = capture_environment()
        assert "os" in env

    def test_has_machine(self):
        env = capture_environment()
        assert "machine" in env

    def test_has_iterations_and_warmup(self):
        env = capture_environment()
        assert env["warmup"] == WARMUP
        assert env["iterations"] == ITERATIONS

    def test_has_package_versions(self):
        """Must record versions of all key packages."""
        env = capture_environment()
        required_pkgs = ["cbor2", "msgpack"]
        for pkg in required_pkgs:
            key = f"pkg_{pkg}"
            assert key in env, f"Missing package version for {pkg}"


# ===========================================================================
# Single-file benchmark runner
# ===========================================================================


class TestRunSingleFile:
    """run_single_file should benchmark one corpus entry across all baselines."""

    @pytest.fixture
    def simple_entry(self):
        return {
            "name": "test_config",
            "format": "json",
            "text": '{"learning_rate": 0.001, "epochs": 10, "batch_size": 32}',
            "category": "training",
        }

    def test_returns_dict(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        assert isinstance(result, dict)

    def test_has_file_metadata(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        assert result["name"] == "test_config"
        assert result["format"] == "json"
        assert result["category"] == "training"
        assert "size_bytes" in result
        assert "size_category" in result

    def test_size_bytes_correct(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        expected = len(simple_entry["text"].encode("utf-8"))
        assert result["size_bytes"] == expected

    def test_size_category_assigned(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        assert result["size_category"] in {"small", "medium", "large"}

    def test_has_baselines_dict(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        assert "baselines" in result
        assert isinstance(result["baselines"], dict)

    def test_each_baseline_has_encode_decode_stats(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        for bl_name, bl_data in result["baselines"].items():
            if "error" in bl_data:
                continue
            assert "encode" in bl_data, f"{bl_name} missing encode stats"
            assert "decode" in bl_data, f"{bl_name} missing decode stats"

    def test_stats_have_ops_per_sec(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        for bl_name, bl_data in result["baselines"].items():
            if "error" in bl_data:
                continue
            for op in ("encode", "decode"):
                if op in bl_data:
                    assert "ops_per_sec" in bl_data[op], (
                        f"{bl_name}/{op} missing ops_per_sec"
                    )

    def test_stats_have_throughput(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        for bl_name, bl_data in result["baselines"].items():
            if "error" in bl_data:
                continue
            for op in ("encode", "decode"):
                if op in bl_data:
                    assert "throughput_bytes_per_sec" in bl_data[op], (
                        f"{bl_name}/{op} missing throughput_bytes_per_sec"
                    )

    def test_has_format_tax(self, simple_entry):
        """Each baseline's encode stats must include format_tax entries."""
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        for bl_name, bl_data in result["baselines"].items():
            if "error" in bl_data:
                continue
            if "encode" in bl_data:
                assert "format_tax" in bl_data["encode"], (
                    f"{bl_name}/encode missing format_tax"
                )
                ft = bl_data["encode"]["format_tax"]
                assert isinstance(ft, dict)
                # Should have entries for each (t_step, steps) combination
                assert len(ft) > 0

    def test_cdxf_full_present(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        assert "cdxf_full" in result["baselines"]
        assert "error" not in result["baselines"]["cdxf_full"]

    def test_pickle_present(self, simple_entry):
        result = run_single_file(simple_entry, iterations=5, warmup=1)
        assert "pickle" in result["baselines"]
        assert "error" not in result["baselines"]["pickle"]


# ===========================================================================
# Full experiment runner
# ===========================================================================


class TestRunExperiment:
    """Integration test for the full experiment runner."""

    @pytest.fixture
    def mini_results(self, tmp_path):
        """Run the experiment with minimal iterations for speed."""
        return run_experiment(
            output_dir=tmp_path,
            iterations=3,
            warmup=1,
        )

    def test_returns_dict(self, mini_results):
        assert isinstance(mini_results, dict)

    def test_has_experiment_id(self, mini_results):
        assert mini_results["experiment"] == "EXP-010"

    def test_has_timestamp(self, mini_results):
        assert "timestamp" in mini_results

    def test_has_environment(self, mini_results):
        assert "environment" in mini_results
        assert isinstance(mini_results["environment"], dict)

    def test_has_corpus_size(self, mini_results):
        assert "corpus_size" in mini_results
        assert mini_results["corpus_size"] > 0

    def test_has_results_list(self, mini_results):
        assert "results" in mini_results
        assert isinstance(mini_results["results"], list)
        assert len(mini_results["results"]) > 0

    def test_has_aggregate_summary(self, mini_results):
        """Must include aggregate statistics across the corpus."""
        assert "aggregate" in mini_results
        assert isinstance(mini_results["aggregate"], dict)

    def test_aggregate_has_per_baseline_stats(self, mini_results):
        agg = mini_results["aggregate"]
        # Should have at least cdxf_full and pickle
        assert "cdxf_full" in agg or "cdxf_codec" in agg

    def test_writes_json_output(self, mini_results, tmp_path):
        output_file = tmp_path / "exp_010_results.json"
        assert output_file.exists()
        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert data["experiment"] == "EXP-010"

    def test_writes_csv_output(self, mini_results, tmp_path):
        """CSV with per-file per-baseline results for easy analysis."""
        csv_file = tmp_path / "throughput_results.csv"
        assert csv_file.exists()
        import csv
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        # Check required columns
        required_cols = {"name", "format", "size_bytes", "size_category",
                         "baseline", "operation", "median_s", "mean_s",
                         "std_s", "ops_per_sec"}
        actual_cols = set(rows[0].keys())
        assert required_cols.issubset(actual_cols), (
            f"Missing CSV columns: {required_cols - actual_cols}"
        )

    def test_writes_format_tax_csv(self, mini_results, tmp_path):
        """Separate CSV for format tax analysis."""
        tax_file = tmp_path / "format_tax.csv"
        assert tax_file.exists()
        import csv
        with open(tax_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        required_cols = {"baseline", "t_step_s", "steps_per_epoch", "format_tax"}
        actual_cols = set(rows[0].keys())
        assert required_cols.issubset(actual_cols), (
            f"Missing format_tax CSV columns: {required_cols - actual_cols}"
        )


# ===========================================================================
# Scientific rigor constraints
# ===========================================================================


class TestScientificRigor:
    """Verify the experiment follows the scientific protocol strictly."""

    def test_warmup_discarded_not_measured(self):
        """Warm-up iterations must NOT appear in measurements."""
        call_log = []

        def tracked():
            call_log.append(time.perf_counter_ns())

        times = bench(tracked, iterations=20, warmup=5)
        assert len(times) == 20  # Only measurement iterations returned
        assert len(call_log) == 25  # But all 25 calls happened

    def test_no_outlier_removal_in_stats(self):
        """Protocol explicitly says: report ALL data, do NOT remove outliers."""
        data = [0.001] * 999 + [10.0]  # 1 extreme outlier in 1000
        result = summarize_stats(data)
        assert result["n"] == 1000
        assert result["max_s"] == pytest.approx(10.0)
        # Mean should be affected by the outlier
        assert result["mean_s"] > 0.01  # would be ~0.001 without outlier

    def test_ci95_uses_z_196(self):
        """95% CI uses z=1.96 (normal approximation for n=1000)."""
        data = [float(i) for i in range(1, 101)]
        result = summarize_stats(data)
        std = statistics.stdev(data)
        n = len(data)
        expected_half = 1.96 * std / math.sqrt(n)
        actual_half = result["ci95_upper_s"] - result["mean_s"]
        assert actual_half == pytest.approx(expected_half, rel=1e-6)

    def test_gc_collect_between_baselines(self):
        """run_single_file should gc.collect() between baselines.

        We verify indirectly: the function should not crash or produce
        wildly different results when GC pressure is high.
        """
        entry = {
            "name": "gc_test",
            "format": "json",
            "text": '{"key": "value"}',
            "category": "test",
        }
        # Create GC pressure
        _garbage = [{"a": list(range(100))} for _ in range(1000)]
        result = run_single_file(entry, iterations=5, warmup=1)
        del _garbage
        assert "baselines" in result
        # Should have results for multiple baselines
        successful = [bl for bl, data in result["baselines"].items()
                       if "error" not in data]
        assert len(successful) >= 3

    def test_perf_counter_ns_resolution(self):
        """Verify we're using perf_counter_ns (nanosecond resolution)."""
        # Take two close readings — they should differ by < 1ms
        # but be representable with sub-microsecond precision
        t0 = time.perf_counter_ns()
        t1 = time.perf_counter_ns()
        delta_ns = t1 - t0
        assert delta_ns >= 0
        # Convert to seconds with fractional precision
        delta_s = delta_ns / 1e9
        # Should be representable as a float with >6 decimal digits of precision
        assert isinstance(delta_s, float)

    def test_format_tax_uses_encode_median(self):
        """Format tax must be based on median encode time, not mean.

        The median is more robust to outliers — protocol says T_config
        from measured values. We use median as the representative time.
        """
        entry = {
            "name": "tax_check",
            "format": "json",
            "text": '{"lr": 0.001}',
            "category": "training",
        }
        result = run_single_file(entry, iterations=10, warmup=2)
        cdxf_data = result["baselines"].get("cdxf_full", {})
        if "error" in cdxf_data:
            pytest.skip("CDXF failed on this input")
        enc = cdxf_data["encode"]
        ft = enc["format_tax"]
        median_s = enc["median_s"]
        # Verify tax is computed from median
        for key, tax_val in ft.items():
            # Parse key to get t_step and steps
            parts = key.split("_")
            # key format: "tstep_{t_step}_steps_{steps}"
            t_step = float(parts[1])
            steps = int(parts[3])
            expected = median_s / (steps * t_step)
            assert tax_val == pytest.approx(expected, rel=1e-3), (
                f"Format tax mismatch for {key}: {tax_val} vs {expected}"
            )
