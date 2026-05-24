"""Tests for the CDXF CLI — written BEFORE implementation (TDD)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def run_cdxf(*args: str, input_data: bytes | None = None) -> subprocess.CompletedProcess:
    """Run the cdxf CLI as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "cdxf.cli", *args],
        capture_output=True,
        input=input_data,
    )


# ===================================================================
# cdxf encode
# ===================================================================

class TestCliEncode:
    def test_encode_json(self, tmp_path):
        src = tmp_path / "test.json"
        src.write_text('{"key": "value"}')
        out = tmp_path / "test.cdxf"
        r = run_cdxf("encode", str(src), "-o", str(out))
        assert r.returncode == 0, r.stderr.decode()
        assert out.exists()
        assert out.stat().st_size > 0

    def test_encode_yaml(self, tmp_path):
        src = tmp_path / "test.yaml"
        src.write_text("key: value\n")
        out = tmp_path / "test.cdxf"
        r = run_cdxf("encode", str(src), "-o", str(out))
        assert r.returncode == 0, r.stderr.decode()
        assert out.exists()

    def test_encode_xml(self, tmp_path):
        src = tmp_path / "test.xml"
        src.write_text("<root><child>text</child></root>")
        out = tmp_path / "test.cdxf"
        r = run_cdxf("encode", str(src), "-o", str(out))
        assert r.returncode == 0, r.stderr.decode()
        assert out.exists()

    def test_encode_toml(self, tmp_path):
        src = tmp_path / "test.toml"
        src.write_text('key = "value"\n')
        out = tmp_path / "test.cdxf"
        r = run_cdxf("encode", str(src), "-o", str(out))
        assert r.returncode == 0, r.stderr.decode()
        assert out.exists()

    def test_encode_default_output(self, tmp_path):
        """Without -o, output filename is input stem + .cdxf."""
        src = tmp_path / "data.json"
        src.write_text('{"a": 1}')
        r = run_cdxf("encode", str(src))
        assert r.returncode == 0, r.stderr.decode()
        expected = tmp_path / "data.cdxf"
        assert expected.exists()

    def test_encode_stdout(self, tmp_path):
        """With -o -, write binary to stdout."""
        src = tmp_path / "data.json"
        src.write_text('{"a": 1}')
        r = run_cdxf("encode", str(src), "-o", "-")
        assert r.returncode == 0, r.stderr.decode()
        assert len(r.stdout) > 0

    def test_encode_bad_file(self):
        r = run_cdxf("encode", "nonexistent.json")
        assert r.returncode != 0


# ===================================================================
# cdxf decode
# ===================================================================

class TestCliDecode:
    def _encode_file(self, tmp_path, name, content):
        src = tmp_path / name
        src.write_text(content)
        out = tmp_path / (Path(name).stem + ".cdxf")
        r = run_cdxf("encode", str(src), "-o", str(out))
        assert r.returncode == 0, r.stderr.decode()
        return out

    def test_decode_json(self, tmp_path):
        cdxf = self._encode_file(tmp_path, "test.json", '{"key": "value"}')
        out = tmp_path / "restored.json"
        r = run_cdxf("decode", str(cdxf), "-o", str(out), "-f", "json")
        assert r.returncode == 0, r.stderr.decode()
        restored = json.loads(out.read_text())
        assert restored == {"key": "value"}

    def test_decode_yaml(self, tmp_path):
        cdxf = self._encode_file(tmp_path, "test.yaml", "key: value\n")
        out = tmp_path / "restored.yaml"
        r = run_cdxf("decode", str(cdxf), "-o", str(out), "-f", "yaml")
        assert r.returncode == 0, r.stderr.decode()
        assert "key" in out.read_text()

    def test_decode_xml(self, tmp_path):
        cdxf = self._encode_file(tmp_path, "test.xml", "<root>text</root>")
        out = tmp_path / "restored.xml"
        r = run_cdxf("decode", str(cdxf), "-o", str(out), "-f", "xml")
        assert r.returncode == 0, r.stderr.decode()
        assert "root" in out.read_text()

    def test_decode_toml(self, tmp_path):
        cdxf = self._encode_file(tmp_path, "test.toml", 'key = "value"\n')
        out = tmp_path / "restored.toml"
        r = run_cdxf("decode", str(cdxf), "-o", str(out), "-f", "toml")
        assert r.returncode == 0, r.stderr.decode()
        assert "key" in out.read_text()

    def test_decode_auto_format(self, tmp_path):
        """If source_format_hint is set, -f is optional."""
        cdxf = self._encode_file(tmp_path, "test.json", '{"a": 1}')
        out = tmp_path / "restored.json"
        r = run_cdxf("decode", str(cdxf), "-o", str(out))
        assert r.returncode == 0, r.stderr.decode()
        assert out.exists()

    def test_decode_stdout(self, tmp_path):
        cdxf = self._encode_file(tmp_path, "test.json", '{"a": 1}')
        r = run_cdxf("decode", str(cdxf), "-o", "-")
        assert r.returncode == 0, r.stderr.decode()
        assert len(r.stdout) > 0


# ===================================================================
# cdxf convert
# ===================================================================

class TestCliConvert:
    def test_json_to_yaml(self, tmp_path):
        src = tmp_path / "data.json"
        src.write_text('{"name": "Alice", "age": 30}')
        out = tmp_path / "data.yaml"
        r = run_cdxf("convert", str(src), "--to", "yaml", "-o", str(out))
        assert r.returncode == 0, r.stderr.decode()
        text = out.read_text()
        assert "name" in text
        assert "Alice" in text

    def test_yaml_to_json(self, tmp_path):
        src = tmp_path / "data.yaml"
        src.write_text("name: Alice\nage: 30\n")
        out = tmp_path / "data.json"
        r = run_cdxf("convert", str(src), "--to", "json", "-o", str(out))
        assert r.returncode == 0, r.stderr.decode()
        restored = json.loads(out.read_text())
        assert restored["name"] == "Alice"

    def test_json_to_toml(self, tmp_path):
        src = tmp_path / "data.json"
        src.write_text('{"name": "Alice", "count": 42}')
        out = tmp_path / "data.toml"
        r = run_cdxf("convert", str(src), "--to", "toml", "-o", str(out))
        assert r.returncode == 0, r.stderr.decode()
        assert "name" in out.read_text()

    def test_convert_missing_to(self, tmp_path):
        src = tmp_path / "data.json"
        src.write_text('{"a": 1}')
        r = run_cdxf("convert", str(src))
        assert r.returncode != 0


# ===================================================================
# cdxf info
# ===================================================================

class TestCliInfo:
    def test_info_json(self, tmp_path):
        src = tmp_path / "test.json"
        src.write_text('{"key": "value"}')
        r = run_cdxf("info", str(src))
        assert r.returncode == 0, r.stderr.decode()
        output = r.stdout.decode()
        assert "json" in output.lower()

    def test_info_cdxf(self, tmp_path):
        src = tmp_path / "test.json"
        src.write_text('{"key": "value"}')
        cdxf = tmp_path / "test.cdxf"
        run_cdxf("encode", str(src), "-o", str(cdxf))
        r = run_cdxf("info", str(cdxf))
        assert r.returncode == 0, r.stderr.decode()
        output = r.stdout.decode()
        assert "cdxf" in output.lower() or "size" in output.lower()

    def test_info_xml(self, tmp_path):
        src = tmp_path / "test.xml"
        src.write_text("<root><child/></root>")
        r = run_cdxf("info", str(src))
        assert r.returncode == 0, r.stderr.decode()
        output = r.stdout.decode()
        assert "xml" in output.lower()


# ===================================================================
# Edge cases
# ===================================================================

class TestCliEdgeCases:
    def test_no_args(self):
        r = run_cdxf()
        # Should print help, not crash
        assert r.returncode in (0, 2)  # argparse returns 2 for missing args

    def test_help(self):
        r = run_cdxf("--help")
        assert r.returncode == 0
        assert b"cdxf" in r.stdout.lower() or b"CDXF" in r.stdout

    def test_encode_roundtrip(self, tmp_path):
        """Full CLI round-trip: text -> encode -> decode -> text."""
        src = tmp_path / "original.json"
        src.write_text('{"x": 1, "y": [2, 3]}')
        cdxf = tmp_path / "data.cdxf"
        restored = tmp_path / "restored.json"

        r1 = run_cdxf("encode", str(src), "-o", str(cdxf))
        assert r1.returncode == 0, r1.stderr.decode()

        r2 = run_cdxf("decode", str(cdxf), "-o", str(restored))
        assert r2.returncode == 0, r2.stderr.decode()

        original = json.loads(src.read_text())
        result = json.loads(restored.read_text())
        assert original == result
