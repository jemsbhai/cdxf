"""
EXP-011: Token Cost of Format Syntax — The "Syntax Tax"

Measures the fraction of LLM tokens consumed by format syntax (braces,
brackets, closing tags, quotes) vs semantic content (keys, values, comments).
Projects context window savings if agents used a non-text representation.

Usage:
    python benchmarks/src/run_exp011.py
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken not installed. Run: pip install tiktoken")
    sys.exit(1)

# ===========================================================================
# Protocol constants
# ===========================================================================

TOKENIZERS = ["cl100k_base", "o200k_base"]
CONTEXT_WINDOW = 128_000
N_FILES_PROJECTIONS = [10, 25, 50, 100]
FORMATS = ["json", "yaml", "xml", "toml"]


# ===========================================================================
# Character-level role identification
# ===========================================================================


def identify_char_roles(text: str, fmt: str) -> list[str]:
    """Classify every character in text as 'syntax' or 'semantic'.

    Returns a list of len(text), each element is 'syntax' or 'semantic'.
    """
    if not text:
        return []
    if fmt == "json":
        return _json_char_roles(text)
    elif fmt == "yaml":
        return _yaml_char_roles(text)
    elif fmt == "xml":
        return _xml_char_roles(text)
    elif fmt == "toml":
        return _toml_char_roles(text)
    else:
        # Unknown format — treat everything as semantic
        return ["semantic"] * len(text)


# ---------------------------------------------------------------------------
# JSON character roles
# ---------------------------------------------------------------------------

def _json_char_roles(text: str) -> list[str]:
    """JSON syntax: { } [ ] : , " whitespace
    JSON semantic: key names, string values, numbers, booleans, null.
    """
    roles = ["syntax"] * len(text)
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch == '"':
            # String literal — quotes are syntax, content is semantic
            roles[i] = "syntax"  # opening quote
            i += 1
            while i < n and text[i] != '"':
                if text[i] == '\\':
                    # Escape sequence: backslash is syntax, next char is semantic
                    roles[i] = "syntax"
                    i += 1
                    if i < n:
                        roles[i] = "semantic"
                        i += 1
                else:
                    roles[i] = "semantic"
                    i += 1
            if i < n:
                roles[i] = "syntax"  # closing quote
                i += 1

        elif ch in "{}[]:,":
            roles[i] = "syntax"
            i += 1

        elif ch in " \t\n\r":
            roles[i] = "syntax"
            i += 1

        elif ch == '-' or ch.isdigit():
            # Number literal
            while i < n and text[i] in "-0123456789.eE+":
                roles[i] = "semantic"
                i += 1

        elif text[i:i + 4] == "true":
            for j in range(4):
                roles[i + j] = "semantic"
            i += 4

        elif text[i:i + 5] == "false":
            for j in range(5):
                roles[i + j] = "semantic"
            i += 5

        elif text[i:i + 4] == "null":
            for j in range(4):
                roles[i + j] = "semantic"
            i += 4

        else:
            # Unknown — default syntax
            i += 1

    return roles


# ---------------------------------------------------------------------------
# YAML character roles
# ---------------------------------------------------------------------------

def _yaml_char_roles(text: str) -> list[str]:
    """YAML syntax: : (separator), - (list), indentation, ---, ...
    YAML semantic: keys, values, comments (comments ARE semantic content).
    """
    roles = ["syntax"] * len(text)
    lines = text.split("\n")
    pos = 0

    for line_idx, line in enumerate(lines):
        if not line:
            # Empty line — the \n is syntax
            if pos < len(text) and pos < len(roles):
                pass  # already syntax
            pos += 1  # for the \n
            continue

        # Document markers
        stripped = line.strip()
        if stripped in ("---", "..."):
            for j in range(len(line)):
                roles[pos + j] = "syntax"
            pos += len(line) + 1  # +1 for \n
            continue

        i = 0
        line_len = len(line)

        # Leading indentation — syntax
        while i < line_len and line[i] in " \t":
            roles[pos + i] = "syntax"
            i += 1

        # Check what follows indentation
        rest = line[i:]

        if rest.startswith("#"):
            # Full-line comment: # is syntax, content is semantic
            roles[pos + i] = "syntax"  # the #
            i += 1
            # Space after # is syntax
            if i < line_len and line[i] == " ":
                roles[pos + i] = "syntax"
                i += 1
            # Rest is semantic (comment content)
            while i < line_len:
                roles[pos + i] = "semantic"
                i += 1

        elif rest.startswith("- ") or rest == "-":
            # List indicator
            roles[pos + i] = "syntax"  # -
            i += 1
            if i < line_len and line[i] == " ":
                roles[pos + i] = "syntax"
                i += 1
            # Remaining content on this line
            _yaml_classify_value_region(line, i, pos, roles)

        else:
            # Key-value or plain value
            _yaml_classify_value_region(line, i, pos, roles)

        pos += line_len + 1  # +1 for \n

    # Fix: pos might overshoot by 1 (no trailing \n)
    return roles[:len(text)]


def _yaml_classify_value_region(line: str, start: int, line_pos: int,
                                 roles: list[str]):
    """Classify a region of a YAML line that may contain key: value # comment."""
    i = start
    line_len = len(line)
    max_idx = len(roles)

    # Look for key: value pattern
    colon_idx = -1
    in_quote = False
    quote_char = None
    for j in range(i, line_len):
        ch = line[j]
        if not in_quote and ch in ("'", '"'):
            in_quote = True
            quote_char = ch
        elif in_quote and ch == quote_char:
            in_quote = False
        elif not in_quote and ch == ':':
            # Must be followed by space, newline, or end of line to be a key separator
            if j + 1 >= line_len or line[j + 1] in (" ", "\t"):
                colon_idx = j
                break

    if colon_idx >= 0:
        # Key part: i to colon_idx — semantic
        for j in range(i, colon_idx):
            if line_pos + j < max_idx:
                roles[line_pos + j] = "semantic"

        # Colon — syntax
        if line_pos + colon_idx < max_idx:
            roles[line_pos + colon_idx] = "syntax"

        # Space after colon — syntax
        val_start = colon_idx + 1
        while val_start < line_len and line[val_start] in " \t":
            if line_pos + val_start < max_idx:
                roles[line_pos + val_start] = "syntax"
            val_start += 1

        # Value and possible inline comment
        _yaml_classify_value_and_comment(line, val_start, line_pos, roles)
    else:
        # No colon — plain value or continuation
        _yaml_classify_value_and_comment(line, i, line_pos, roles)


def _yaml_classify_value_and_comment(line: str, start: int, line_pos: int,
                                      roles: list[str]):
    """Classify value text with possible trailing inline comment."""
    i = start
    line_len = len(line)
    max_idx = len(roles)

    # Find inline comment (# preceded by space, not inside quotes)
    comment_start = -1
    in_quote = False
    quote_char = None
    for j in range(i, line_len):
        ch = line[j]
        if not in_quote and ch in ("'", '"'):
            in_quote = True
            quote_char = ch
        elif in_quote and ch == quote_char:
            in_quote = False
        elif not in_quote and ch == "#" and j > 0 and line[j - 1] in (" ", "\t"):
            comment_start = j
            break

    if comment_start >= 0:
        # Value before comment — semantic
        for j in range(i, comment_start):
            if line_pos + j < max_idx:
                ch = line[j]
                # Trailing spaces before comment are syntax
                roles[line_pos + j] = "semantic"
        # Trim trailing spaces from value
        val_end = comment_start
        while val_end > i and line[val_end - 1] in " \t":
            if line_pos + val_end - 1 < max_idx:
                roles[line_pos + val_end - 1] = "syntax"
            val_end -= 1

        # Comment: # is syntax, space after # is syntax, rest is semantic
        if line_pos + comment_start < max_idx:
            roles[line_pos + comment_start] = "syntax"  # #
        j = comment_start + 1
        if j < line_len and line[j] == " ":
            if line_pos + j < max_idx:
                roles[line_pos + j] = "syntax"
            j += 1
        while j < line_len:
            if line_pos + j < max_idx:
                roles[line_pos + j] = "semantic"
            j += 1
    else:
        # No comment — everything is value (semantic)
        for j in range(i, line_len):
            if line_pos + j < max_idx:
                roles[line_pos + j] = "semantic"


# ---------------------------------------------------------------------------
# XML character roles
# ---------------------------------------------------------------------------

def _xml_char_roles(text: str) -> list[str]:
    """XML syntax: < > </ /> = attr quotes, closing tag names (redundant).
    XML semantic: element names (opening), attr names, attr values, text, comment content.
    """
    roles = ["syntax"] * len(text)
    i = 0
    n = len(text)

    # Track seen element names to mark closing tag names as syntax
    while i < n:
        if text[i:i + 4] == "<!--":
            # Comment: <!-- is syntax, content is semantic, --> is syntax
            for j in range(4):
                roles[i + j] = "syntax"
            i += 4
            # Content until -->
            while i < n and text[i:i + 3] != "-->":
                ch = text[i]
                if ch in " \t\n\r":
                    roles[i] = "syntax"
                else:
                    roles[i] = "semantic"
                i += 1
            # -->
            if i < n:
                for j in range(min(3, n - i)):
                    roles[i + j] = "syntax"
                i += 3

        elif text[i:i + 2] == "<?":
            # Processing instruction
            roles[i] = "syntax"      # <
            roles[i + 1] = "syntax"  # ?
            i += 2
            # PI target is semantic
            while i < n and text[i] not in " \t\n\r?":
                roles[i] = "semantic"
                i += 1
            # PI content
            while i < n and text[i:i + 2] != "?>":
                ch = text[i]
                if ch in " \t\n\r":
                    roles[i] = "syntax"
                else:
                    roles[i] = "semantic"
                i += 1
            # ?>
            if i < n:
                roles[i] = "syntax"
                i += 1
            if i < n:
                roles[i] = "syntax"
                i += 1

        elif text[i:i + 2] == "</":
            # Closing tag — ENTIRE tag is syntax (name is redundant)
            roles[i] = "syntax"      # <
            roles[i + 1] = "syntax"  # /
            i += 2
            while i < n and text[i] != ">":
                roles[i] = "syntax"  # closing tag name = redundant = syntax
                i += 1
            if i < n:
                roles[i] = "syntax"  # >
                i += 1

        elif text[i] == "<":
            # Opening tag
            roles[i] = "syntax"  # <
            i += 1

            # Self-closing shorthand check
            if i < n and text[i] == "/":
                roles[i] = "syntax"
                i += 1
                continue

            # Element name — semantic
            while i < n and text[i] not in " \t\n\r/>":
                roles[i] = "semantic"
                i += 1

            # Attributes
            while i < n and text[i] != ">" and text[i:i + 2] != "/>":
                if text[i] in " \t\n\r":
                    roles[i] = "syntax"
                    i += 1
                elif text[i] == "=":
                    roles[i] = "syntax"
                    i += 1
                elif text[i] in ('"', "'"):
                    # Attribute value
                    quote = text[i]
                    roles[i] = "syntax"  # opening quote
                    i += 1
                    while i < n and text[i] != quote:
                        roles[i] = "semantic"  # attr value content
                        i += 1
                    if i < n:
                        roles[i] = "syntax"  # closing quote
                        i += 1
                else:
                    # Attribute name — semantic
                    roles[i] = "semantic"
                    i += 1

            # Handle /> or >
            if i < n and text[i:i + 2] == "/>":
                roles[i] = "syntax"
                roles[i + 1] = "syntax"
                i += 2
            elif i < n and text[i] == ">":
                roles[i] = "syntax"
                i += 1

        elif text[i] in " \t\n\r":
            # Whitespace between elements — syntax
            roles[i] = "syntax"
            i += 1

        else:
            # Text content — semantic
            roles[i] = "semantic"
            i += 1

    return roles


# ---------------------------------------------------------------------------
# TOML character roles
# ---------------------------------------------------------------------------

def _toml_char_roles(text: str) -> list[str]:
    """TOML syntax: [ ] = " , { } whitespace
    TOML semantic: key names, values, section names, comments.
    """
    roles = ["syntax"] * len(text)
    lines = text.split("\n")
    pos = 0

    for line in lines:
        stripped = line.strip()
        line_len = len(line)

        if not stripped:
            # Empty line
            pos += line_len + 1
            continue

        i = 0

        # Leading whitespace
        while i < line_len and line[i] in " \t":
            roles[pos + i] = "syntax"
            i += 1

        rest = line[i:]

        if rest.startswith("#"):
            # Comment line: # is syntax, content is semantic
            roles[pos + i] = "syntax"  # #
            i += 1
            if i < line_len and line[i] == " ":
                roles[pos + i] = "syntax"
                i += 1
            while i < line_len:
                roles[pos + i] = "semantic"
                i += 1

        elif rest.startswith("[["):
            # Array of tables header
            roles[pos + i] = "syntax"      # [
            roles[pos + i + 1] = "syntax"  # [
            i += 2
            # Section name — semantic
            while i < line_len and line[i:i + 2] != "]]":
                if line[i] in " \t":
                    roles[pos + i] = "syntax"
                else:
                    roles[pos + i] = "semantic"
                i += 1
            if i < line_len:
                roles[pos + i] = "syntax"      # ]
                roles[pos + i + 1] = "syntax"  # ]
                i += 2

        elif rest.startswith("["):
            # Section header
            roles[pos + i] = "syntax"  # [
            i += 1
            while i < line_len and line[i] != "]":
                if line[i] in " \t":
                    roles[pos + i] = "syntax"
                else:
                    roles[pos + i] = "semantic"
                i += 1
            if i < line_len:
                roles[pos + i] = "syntax"  # ]
                i += 1

        else:
            # Key = value line
            _toml_classify_kv_line(line, i, pos, roles)

        pos += line_len + 1

    return roles[:len(text)]


def _toml_classify_kv_line(line: str, start: int, line_pos: int,
                            roles: list[str]):
    """Classify a TOML key = value line."""
    i = start
    line_len = len(line)
    max_idx = len(roles)

    # Find the = separator (not inside quotes)
    eq_idx = -1
    in_quote = False
    quote_char = None
    for j in range(i, line_len):
        ch = line[j]
        if not in_quote and ch in ('"', "'"):
            in_quote = True
            quote_char = ch
        elif in_quote and ch == quote_char:
            in_quote = False
        elif not in_quote and ch == "=":
            eq_idx = j
            break

    if eq_idx < 0:
        # No = found — treat as value continuation or bare value
        for j in range(i, line_len):
            if line_pos + j < max_idx:
                roles[line_pos + j] = "semantic"
        return

    # Key: everything before = is key (semantic), whitespace is syntax
    for j in range(i, eq_idx):
        if line_pos + j < max_idx:
            if line[j] in " \t":
                roles[line_pos + j] = "syntax"
            else:
                roles[line_pos + j] = "semantic"

    # = is syntax
    if line_pos + eq_idx < max_idx:
        roles[line_pos + eq_idx] = "syntax"

    # Value after =
    val_start = eq_idx + 1
    # Skip whitespace after =
    while val_start < line_len and line[val_start] in " \t":
        if line_pos + val_start < max_idx:
            roles[line_pos + val_start] = "syntax"
        val_start += 1

    # Classify value with possible inline comment
    _toml_classify_value(line, val_start, line_pos, roles)


def _toml_classify_value(line: str, start: int, line_pos: int,
                          roles: list[str]):
    """Classify a TOML value region (may have trailing comment)."""
    i = start
    line_len = len(line)
    max_idx = len(roles)

    # Find inline comment
    comment_start = -1
    in_quote = False
    quote_char = None
    for j in range(i, line_len):
        ch = line[j]
        if not in_quote and ch in ('"', "'"):
            in_quote = True
            quote_char = ch
        elif in_quote and ch == quote_char:
            in_quote = False
        elif not in_quote and ch == "#":
            comment_start = j
            break

    end = comment_start if comment_start >= 0 else line_len

    # Value region
    for j in range(i, end):
        if line_pos + j >= max_idx:
            break
        ch = line[j]
        if ch in ('"', "'"):
            roles[line_pos + j] = "syntax"  # quote delimiter
        elif ch in (",", "{", "}", "[", "]"):
            roles[line_pos + j] = "syntax"
        elif ch in (" ", "\t"):
            roles[line_pos + j] = "syntax"
        else:
            roles[line_pos + j] = "semantic"  # value content

    # Inline comment
    if comment_start >= 0:
        if line_pos + comment_start < max_idx:
            roles[line_pos + comment_start] = "syntax"  # #
        j = comment_start + 1
        if j < line_len and line[j] == " ":
            if line_pos + j < max_idx:
                roles[line_pos + j] = "syntax"
            j += 1
        while j < line_len:
            if line_pos + j < max_idx:
                roles[line_pos + j] = "semantic"
            j += 1


# ===========================================================================
# Token classification
# ===========================================================================


def classify_tokens(text: str, fmt: str, tokenizer_name: str) -> dict:
    """Classify tokens as syntax or semantic using character-level roles.

    For tokens spanning both syntax and semantic characters, attribution
    is proportional by character count.

    Returns: total_tokens, syntax_tokens, semantic_tokens, syntax_tax_rate,
             tokens_per_byte.
    """
    if not text:
        return {
            "total_tokens": 0,
            "syntax_tokens": 0.0,
            "semantic_tokens": 0.0,
            "syntax_tax_rate": 0.0,
            "tokens_per_byte": 0.0,
        }

    enc = tiktoken.get_encoding(tokenizer_name)
    tokens = enc.encode(text)
    total = len(tokens)

    if total == 0:
        return {
            "total_tokens": 0,
            "syntax_tokens": 0.0,
            "semantic_tokens": 0.0,
            "syntax_tax_rate": 0.0,
            "tokens_per_byte": 0.0,
        }

    roles = identify_char_roles(text, fmt)
    text_bytes = text.encode("utf-8")

    # Map each token back to its source characters
    syntax_total = 0.0
    semantic_total = 0.0

    byte_offset = 0
    for token_id in tokens:
        token_bytes = enc.decode_single_token_bytes(token_id)
        token_len = len(token_bytes)

        # Find the character range for this token
        token_text = text_bytes[byte_offset:byte_offset + token_len]
        # Count chars in this token's range
        char_start = len(text_bytes[:byte_offset].decode("utf-8", errors="replace"))
        char_end_bytes = text_bytes[:byte_offset + token_len]
        char_end = len(char_end_bytes.decode("utf-8", errors="replace"))

        # Count syntax vs semantic chars in this token
        syn_chars = 0
        sem_chars = 0
        for ci in range(char_start, min(char_end, len(roles))):
            if roles[ci] == "syntax":
                syn_chars += 1
            else:
                sem_chars += 1

        total_chars = syn_chars + sem_chars
        if total_chars > 0:
            syntax_total += syn_chars / total_chars
            semantic_total += sem_chars / total_chars
        else:
            syntax_total += 1.0  # default to syntax

        byte_offset += token_len

    size_bytes = len(text_bytes)

    return {
        "total_tokens": total,
        "syntax_tokens": round(syntax_total, 4),
        "semantic_tokens": round(semantic_total, 4),
        "syntax_tax_rate": round(syntax_total / total, 6),
        "tokens_per_byte": round(total / size_bytes, 6),
    }


# ===========================================================================
# Context projection
# ===========================================================================


def compute_context_projection(median_total_tokens: float,
                                syntax_tax_rate: float,
                                n_files: int,
                                context_window: int = CONTEXT_WINDOW) -> dict:
    """Project context window waste for n_files of median size.

    Returns: n_files, total_tokens_used, syntax_tokens_wasted,
             context_waste_fraction, effective_context_gain_files.
    """
    total_used = median_total_tokens * n_files
    wasted = total_used * syntax_tax_rate
    waste_fraction = wasted / context_window if context_window > 0 else 0

    semantic_per_file = median_total_tokens * (1 - syntax_tax_rate)
    if semantic_per_file > 0:
        gain = wasted / semantic_per_file
    else:
        gain = 0

    return {
        "n_files": n_files,
        "total_tokens_used": total_used,
        "syntax_tokens_wasted": wasted,
        "context_waste_fraction": waste_fraction,
        "effective_context_gain_files": gain,
    }


# ===========================================================================
# Comment identification
# ===========================================================================


def identify_comment_chars(text: str, fmt: str) -> list[bool]:
    """Return a boolean mask: True where the character is comment content.

    Only the semantic part of comments (the text), not the # or <!-- delimiters.
    JSON has no comments, so the mask is all False.
    """
    if not text:
        return []

    mask = [False] * len(text)

    if fmt == "json":
        return mask

    if fmt == "yaml":
        lines = text.split("\n")
        pos = 0
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if stripped.startswith("#"):
                j = indent + 1
                if j < len(line) and line[j] == " ":
                    j += 1
                while j < len(line):
                    if pos + j < len(mask):
                        mask[pos + j] = True
                    j += 1
            else:
                in_quote = False
                quote_char = None
                for j in range(indent, len(line)):
                    ch = line[j]
                    if not in_quote and ch in ("'", '"'):
                        in_quote = True
                        quote_char = ch
                    elif in_quote and ch == quote_char:
                        in_quote = False
                    elif not in_quote and ch == "#" and j > 0 and line[j - 1] in (" ", "\t"):
                        k = j + 1
                        if k < len(line) and line[k] == " ":
                            k += 1
                        while k < len(line):
                            if pos + k < len(mask):
                                mask[pos + k] = True
                            k += 1
                        break

            pos += len(line) + 1
        return mask[:len(text)]

    if fmt == "toml":
        lines = text.split("\n")
        pos = 0
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if stripped.startswith("#"):
                j = indent + 1
                if j < len(line) and line[j] == " ":
                    j += 1
                while j < len(line):
                    if pos + j < len(mask):
                        mask[pos + j] = True
                    j += 1
            else:
                in_quote = False
                quote_char = None
                for j in range(indent, len(line)):
                    ch = line[j]
                    if not in_quote and ch in ("'", '"'):
                        in_quote = True
                        quote_char = ch
                    elif in_quote and ch == quote_char:
                        in_quote = False
                    elif not in_quote and ch == "#":
                        k = j + 1
                        if k < len(line) and line[k] == " ":
                            k += 1
                        while k < len(line):
                            if pos + k < len(mask):
                                mask[pos + k] = True
                            k += 1
                        break

            pos += len(line) + 1
        return mask[:len(text)]

    if fmt == "xml":
        i = 0
        n = len(text)
        while i < n:
            if text[i:i + 4] == "<!--":
                i += 4
                while i < n and text[i:i + 3] != "-->":
                    if text[i] not in " \t\n\r":
                        mask[i] = True
                    i += 1
                if i < n:
                    i += 3
            else:
                i += 1
        return mask

    return mask


# ===========================================================================
# Comment contribution analysis
# ===========================================================================


def analyze_comment_contribution(text: str, fmt: str,
                                  tokenizer_name: str) -> dict:
    """Quantify what fraction of semantic tokens come from comments."""
    if not text:
        return {"comment_tokens": 0, "semantic_tokens": 0,
                "comment_fraction_of_semantic": 0}

    enc = tiktoken.get_encoding(tokenizer_name)
    tokens = enc.encode(text)
    roles = identify_char_roles(text, fmt)
    comment_mask = identify_comment_chars(text, fmt)
    text_bytes = text.encode("utf-8")

    semantic_total = 0.0
    comment_total = 0.0

    byte_offset = 0
    for token_id in tokens:
        token_bytes = enc.decode_single_token_bytes(token_id)
        token_len = len(token_bytes)

        char_start = len(text_bytes[:byte_offset].decode("utf-8", errors="replace"))
        char_end = len(text_bytes[:byte_offset + token_len].decode("utf-8", errors="replace"))

        sem_chars = 0
        comment_chars = 0
        total_chars = 0
        for ci in range(char_start, min(char_end, len(roles))):
            total_chars += 1
            if roles[ci] == "semantic":
                sem_chars += 1
                if ci < len(comment_mask) and comment_mask[ci]:
                    comment_chars += 1

        if total_chars > 0:
            semantic_total += sem_chars / total_chars
            comment_total += comment_chars / total_chars

        byte_offset += token_len

    return {
        "comment_tokens": round(comment_total, 4),
        "semantic_tokens": round(semantic_total, 4),
        "comment_fraction_of_semantic": (
            round(comment_total / semantic_total, 4)
            if semantic_total > 0 else 0
        ),
    }


# ===========================================================================
# Comment ablation
# ===========================================================================


def ablate_comments(text: str, fmt: str) -> str:
    """Strip all comments from text, preserving data content."""
    if fmt == "json":
        return text

    if fmt in ("yaml", "toml"):
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue

            in_quote = False
            quote_char = None
            cut = len(line)
            for j, ch in enumerate(line):
                if not in_quote and ch in ("'", '"'):
                    in_quote = True
                    quote_char = ch
                elif in_quote and ch == quote_char:
                    in_quote = False
                elif not in_quote and ch == "#" and j > 0 and line[j - 1] in (" ", "\t"):
                    cut = j
                    break
            cleaned = line[:cut].rstrip()
            result.append(cleaned)

        return "\n".join(result)

    if fmt == "xml":
        import re
        return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    return text


def syntax_tax_ablation(text: str, fmt: str, tokenizer_name: str) -> dict:
    """Compare syntax tax with vs without comments."""
    with_result = classify_tokens(text, fmt, tokenizer_name)
    stripped = ablate_comments(text, fmt)
    without_result = classify_tokens(stripped, fmt, tokenizer_name)

    with_tax = with_result["syntax_tax_rate"]
    without_tax = without_result["syntax_tax_rate"]
    sem_lost = with_result["semantic_tokens"] - without_result["semantic_tokens"]

    return {
        "with_comments_tax": round(with_tax, 6),
        "without_comments_tax": round(without_tax, 6),
        "tax_increase_without_comments": round(without_tax - with_tax, 6),
        "semantic_tokens_lost": round(sem_lost, 4),
        "total_tokens_with": with_result["total_tokens"],
        "total_tokens_without": without_result["total_tokens"],
    }


# ===========================================================================
# Syntax breakdown by construct type
# ===========================================================================


def classify_syntax_breakdown(text: str, fmt: str) -> dict:
    """Break down syntax characters into sub-categories.

    Returns a dict of {category: char_count} that sums to total syntax chars.
    """
    roles = identify_char_roles(text, fmt)

    if fmt == "json":
        counts = {"quotes": 0, "braces_brackets": 0,
                  "colons_commas": 0, "whitespace": 0}
        for i, ch in enumerate(text):
            if roles[i] != "syntax":
                continue
            if ch == '"':
                counts["quotes"] += 1
            elif ch in "{}[]":
                counts["braces_brackets"] += 1
            elif ch in ":,":
                counts["colons_commas"] += 1
            else:
                counts["whitespace"] += 1
        return counts

    if fmt == "yaml":
        counts = {"colons": 0, "dashes": 0, "indentation": 0,
                  "doc_markers": 0, "whitespace_other": 0,
                  "comment_delimiters": 0}
        lines = text.split("\n")
        pos = 0
        for line in lines:
            for j, ch in enumerate(line):
                idx = pos + j
                if idx >= len(roles) or roles[idx] != "syntax":
                    continue
                stripped = line.lstrip()
                indent_len = len(line) - len(stripped)
                if j < indent_len:
                    counts["indentation"] += 1
                elif ch == ":":
                    counts["colons"] += 1
                elif ch == "-" and stripped.startswith("-"):
                    if stripped in ("---", "..."):
                        counts["doc_markers"] += 1
                    else:
                        counts["dashes"] += 1
                elif ch == "#":
                    counts["comment_delimiters"] += 1
                else:
                    counts["whitespace_other"] += 1
            pos += len(line) + 1
        return counts

    if fmt == "xml":
        counts = {"angle_brackets": 0, "closing_tags": 0,
                  "attr_delimiters": 0, "comment_delimiters": 0,
                  "whitespace": 0}
        i = 0
        n = len(text)
        while i < n:
            if roles[i] != "syntax":
                i += 1
                continue

            if text[i:i + 4] == "<!--":
                for j in range(4):
                    if i + j < n:
                        counts["comment_delimiters"] += 1
                i += 4
                while i < n and text[i:i + 3] != "-->":
                    if roles[i] == "syntax":
                        counts["whitespace"] += 1
                    i += 1
                for j in range(3):
                    if i + j < n:
                        counts["comment_delimiters"] += 1
                i += 3
            elif text[i:i + 2] == "</":
                while i < n and text[i] != ">":
                    if roles[i] == "syntax":
                        counts["closing_tags"] += 1
                    i += 1
                if i < n:
                    counts["closing_tags"] += 1
                    i += 1
            elif text[i] in "<>":
                counts["angle_brackets"] += 1
                i += 1
            elif text[i] == "/" and i + 1 < n and text[i + 1] == ">":
                counts["angle_brackets"] += 2
                i += 2
            elif text[i] in ('"', "'", "="):
                counts["attr_delimiters"] += 1
                i += 1
            elif text[i] in " \t\n\r":
                counts["whitespace"] += 1
                i += 1
            else:
                counts["whitespace"] += 1
                i += 1
        return counts

    if fmt == "toml":
        counts = {"quotes": 0, "brackets": 0, "delimiters": 0,
                  "whitespace": 0, "comment_delimiters": 0}
        for i, ch in enumerate(text):
            if roles[i] != "syntax":
                continue
            if ch in ('"', "'"):
                counts["quotes"] += 1
            elif ch in "[]{}":
                counts["brackets"] += 1
            elif ch in "=,":
                counts["delimiters"] += 1
            elif ch == "#":
                counts["comment_delimiters"] += 1
            else:
                counts["whitespace"] += 1
        return counts

    return {}


# ===========================================================================
# Cross-format equivalent comparison
# ===========================================================================


def cross_format_comparison(data: dict, tokenizer_name: str) -> dict:
    """Express the same Python dict in JSON, YAML, and TOML. Compare taxes.

    XML is excluded because arbitrary dicts don't have a canonical XML form.
    """
    import io
    from ruamel.yaml import YAML
    import tomlkit

    results = {}

    # JSON
    json_text = json.dumps(data, indent=2, default=str)
    r = classify_tokens(json_text, "json", tokenizer_name)
    results["json"] = {"text": json_text, **r}

    # YAML
    yaml = YAML()
    yaml.default_flow_style = False
    stream = io.StringIO()
    yaml.dump(data, stream)
    yaml_text = stream.getvalue()
    r = classify_tokens(yaml_text, "yaml", tokenizer_name)
    results["yaml"] = {"text": yaml_text, **r}

    # TOML
    toml_text = tomlkit.dumps(data)
    r = classify_tokens(toml_text, "toml", tokenizer_name)
    results["toml"] = {"text": toml_text, **r}

    return results


# ===========================================================================
# Corpus
# ===========================================================================


def build_corpus() -> list[dict]:
    """Build corpus from data/raw (EXP-001) + ML configs (EXP-010).

    Returns list of {name, format, text}.
    """
    corpus = []
    seen_names = set()

    # 1. Load data/raw files (EXP-001 corpus)
    data_raw = Path("data/raw")
    if data_raw.exists():
        format_map = {
            ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".xml": "xml", ".toml": "toml",
        }
        for root, _dirs, files in os.walk(data_raw):
            for fname in sorted(files):
                fpath = Path(root) / fname
                ext = fpath.suffix.lower()
                fmt = format_map.get(ext)
                if fmt:
                    try:
                        text = fpath.read_text(encoding="utf-8", errors="replace")
                        name = str(fpath.relative_to(data_raw)).replace("\\", "/")
                        if name not in seen_names:
                            corpus.append({
                                "name": name,
                                "format": fmt,
                                "text": text,
                            })
                            seen_names.add(name)
                    except Exception:
                        pass

    # 2. Add ML configs from EXP-010 corpus
    try:
        from benchmarks.src.run_exp010 import build_ml_corpus
        for entry in build_ml_corpus():
            name = f"ml/{entry['name']}"
            if name not in seen_names:
                corpus.append({
                    "name": name,
                    "format": entry["format"],
                    "text": entry["text"],
                })
                seen_names.add(name)
    except ImportError:
        # Try absolute path fallback
        try:
            import importlib.util
            exp010_path = Path(__file__).parent / "run_exp010.py"
            if exp010_path.exists():
                spec = importlib.util.spec_from_file_location("run_exp010", exp010_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for entry in mod.build_ml_corpus():
                    name = f"ml/{entry['name']}"
                    if name not in seen_names:
                        corpus.append({
                            "name": name,
                            "format": entry["format"],
                            "text": entry["text"],
                        })
                        seen_names.add(name)
        except Exception:
            pass

    return corpus


# ===========================================================================
# Single-file analysis
# ===========================================================================


def run_single_file(entry: dict, tokenizer_name: str) -> dict:
    """Analyze one file with one tokenizer."""
    text = entry["text"]
    fmt = entry["format"]
    size_bytes = len(text.encode("utf-8"))

    result = classify_tokens(text, fmt, tokenizer_name)
    result["name"] = entry["name"]
    result["format"] = fmt
    result["size_bytes"] = size_bytes
    result["tokenizer"] = tokenizer_name

    return result


# ===========================================================================
# Full experiment
# ===========================================================================


def run_experiment(output_dir: Path | str | None = None) -> dict:
    """Run the full EXP-011 experiment."""
    if output_dir is None:
        output_dir = Path("benchmarks/results/exp_011")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-011: Token Cost of Format Syntax — The 'Syntax Tax'")
    print("=" * 70)

    corpus = build_corpus()
    print(f"\nCorpus: {len(corpus)} files")
    for fmt in FORMATS:
        count = sum(1 for e in corpus if e["format"] == fmt)
        print(f"  {fmt}: {count} files")

    # Run analysis
    results = []
    for entry in corpus:
        for tok_name in TOKENIZERS:
            r = run_single_file(entry, tok_name)
            results.append(r)
            if tok_name == TOKENIZERS[0]:  # print once per file
                print(f"  {entry['name']:50s} {entry['format']:5s} "
                      f"tax={r['syntax_tax_rate']:.1%} "
                      f"({r['total_tokens']} tokens)")

    # Aggregate per format per tokenizer
    print(f"\n{'=' * 70}")
    print("AGGREGATE: Median Syntax Tax Rate by Format × Tokenizer")
    print("-" * 70)

    aggregate = {}
    for fmt in FORMATS:
        aggregate[fmt] = {}
        for tok_name in TOKENIZERS:
            fmt_results = [
                r for r in results
                if r["format"] == fmt and r["tokenizer"] == tok_name
            ]
            if fmt_results:
                rates = [r["syntax_tax_rate"] for r in fmt_results]
                tpb = [r["tokens_per_byte"] for r in fmt_results]
                agg = {
                    "n_files": len(rates),
                    "median_syntax_tax": round(statistics.median(rates), 4),
                    "mean_syntax_tax": round(statistics.mean(rates), 4),
                    "std_syntax_tax": round(statistics.stdev(rates), 4) if len(rates) > 1 else 0,
                    "min_syntax_tax": round(min(rates), 4),
                    "max_syntax_tax": round(max(rates), 4),
                    "median_tokens_per_byte": round(statistics.median(tpb), 4),
                }
                aggregate[fmt][tok_name] = agg

                print(f"  {fmt:5s} × {tok_name:15s}  "
                      f"median={agg['median_syntax_tax']:.1%}  "
                      f"range=[{agg['min_syntax_tax']:.1%}–{agg['max_syntax_tax']:.1%}]  "
                      f"n={agg['n_files']}")

    # Context projections
    print(f"\n{'=' * 70}")
    print("CONTEXT WINDOW PROJECTIONS (cl100k_base)")
    print("-" * 70)

    context_projections = {}
    for fmt in FORMATS:
        tok_agg = aggregate.get(fmt, {}).get("cl100k_base", {})
        if not tok_agg:
            continue

        fmt_results_cl = [
            r for r in results
            if r["format"] == fmt and r["tokenizer"] == "cl100k_base"
        ]
        median_tokens = statistics.median([r["total_tokens"] for r in fmt_results_cl])
        tax_rate = tok_agg["median_syntax_tax"]

        fmt_projections = []
        for n in N_FILES_PROJECTIONS:
            proj = compute_context_projection(median_tokens, tax_rate, n)
            fmt_projections.append(proj)
            print(f"  {fmt:5s}  N={n:>3d}  "
                  f"wasted={proj['syntax_tokens_wasted']:>8,.0f} tokens  "
                  f"({proj['context_waste_fraction']:.2%} of 128K)  "
                  f"+{proj['effective_context_gain_files']:.1f} files freed")

        context_projections[fmt] = fmt_projections

    # --- Comment contribution analysis ---
    print(f"\n{'=' * 70}")
    print("COMMENT CONTRIBUTION ANALYSIS (cl100k_base)")
    print("-" * 70)

    comment_analysis = []
    for entry in corpus:
        if entry["format"] == "json":
            continue  # JSON has no comments
        ca = analyze_comment_contribution(entry["text"], entry["format"],
                                          "cl100k_base")
        ca["name"] = entry["name"]
        ca["format"] = entry["format"]
        comment_analysis.append(ca)
        if ca["comment_tokens"] > 0:
            print(f"  {entry['name']:50s} {entry['format']:5s} "
                  f"comment_frac={ca['comment_fraction_of_semantic']:.1%} "
                  f"({ca['comment_tokens']:.1f} comment tokens / "
                  f"{ca['semantic_tokens']:.1f} semantic)")

    # Aggregate comment contribution by format
    comment_agg = {}
    for fmt in ["yaml", "toml", "xml"]:
        fmt_ca = [c for c in comment_analysis if c["format"] == fmt
                  and c["comment_tokens"] > 0]
        if fmt_ca:
            fracs = [c["comment_fraction_of_semantic"] for c in fmt_ca]
            comment_agg[fmt] = {
                "n_files_with_comments": len(fmt_ca),
                "median_comment_fraction": round(statistics.median(fracs), 4),
                "mean_comment_fraction": round(statistics.mean(fracs), 4),
            }
            print(f"  {fmt:5s} median comment fraction: "
                  f"{comment_agg[fmt]['median_comment_fraction']:.1%} "
                  f"(n={len(fmt_ca)})")

    # --- Comment ablation ---
    print(f"\n{'=' * 70}")
    print("COMMENT ABLATION: With vs Without Comments (cl100k_base)")
    print("-" * 70)

    ablation_results = []
    for entry in corpus:
        abl = syntax_tax_ablation(entry["text"], entry["format"], "cl100k_base")
        abl["name"] = entry["name"]
        abl["format"] = entry["format"]
        ablation_results.append(abl)
        if abl["semantic_tokens_lost"] > 0.5:
            print(f"  {entry['name']:50s} {entry['format']:5s} "
                  f"tax: {abl['with_comments_tax']:.1%} -> "
                  f"{abl['without_comments_tax']:.1%} "
                  f"(+{abl['tax_increase_without_comments']:.1%})  "
                  f"lost: {abl['semantic_tokens_lost']:.0f} semantic tokens")

    # Aggregate ablation
    ablation_agg = {}
    for fmt in FORMATS:
        fmt_abl = [a for a in ablation_results
                   if a["format"] == fmt and a["semantic_tokens_lost"] > 0.5]
        if fmt_abl:
            increases = [a["tax_increase_without_comments"] for a in fmt_abl]
            losses = [a["semantic_tokens_lost"] for a in fmt_abl]
            ablation_agg[fmt] = {
                "n_files": len(fmt_abl),
                "median_tax_increase": round(statistics.median(increases), 4),
                "median_tokens_lost": round(statistics.median(losses), 1),
                "total_tokens_lost": round(sum(losses), 1),
            }
            print(f"  {fmt:5s} median tax increase: "
                  f"+{ablation_agg[fmt]['median_tax_increase']:.1%}, "
                  f"median tokens lost: {ablation_agg[fmt]['median_tokens_lost']:.0f} "
                  f"(n={len(fmt_abl)})")

    # --- Syntax breakdown ---
    print(f"\n{'=' * 70}")
    print("SYNTAX BREAKDOWN BY CONSTRUCT TYPE (character counts)")
    print("-" * 70)

    breakdown_results = []
    for entry in corpus:
        bd = classify_syntax_breakdown(entry["text"], entry["format"])
        bd["name"] = entry["name"]
        bd["format"] = entry["format"]
        breakdown_results.append(bd)

    # Aggregate breakdown by format
    breakdown_agg = {}
    for fmt in FORMATS:
        fmt_bds = [b for b in breakdown_results if b["format"] == fmt]
        if not fmt_bds:
            continue
        # Sum all char counts across files
        all_keys = [k for k in fmt_bds[0].keys() if k not in ("name", "format")]
        totals = {k: sum(b.get(k, 0) for b in fmt_bds) for k in all_keys}
        grand_total = sum(totals.values())
        fractions = {k: round(v / grand_total, 4) if grand_total > 0 else 0
                     for k, v in totals.items()}
        breakdown_agg[fmt] = {"totals": totals, "fractions": fractions}
        print(f"  {fmt.upper()}:")
        for k in sorted(fractions.keys(), key=lambda x: fractions[x], reverse=True):
            print(f"    {k:25s} {totals[k]:>8,} chars ({fractions[k]:.1%})")

    # --- Cross-format comparison ---
    print(f"\n{'=' * 70}")
    print("CROSS-FORMAT COMPARISON: Same Data, Different Formats (cl100k_base)")
    print("-" * 70)

    cross_configs = [
        ("flat_config", {
            "learning_rate": 0.0001, "weight_decay": 0.01,
            "batch_size": 32, "max_epochs": 10, "seed": 42,
            "gradient_accumulation_steps": 4, "warmup_steps": 500,
        }),
        ("nested_config", {
            "model": {"name": "llama-3.1-8b", "hidden_size": 4096,
                      "num_layers": 32, "num_heads": 32, "vocab_size": 128256},
            "training": {"lr": 0.0001, "batch_size": 4, "epochs": 3,
                         "warmup_ratio": 0.1, "weight_decay": 0.01,
                         "scheduler": "cosine", "gradient_checkpointing": True},
            "data": {"dataset": "alpaca", "max_length": 2048,
                     "split_ratio": 0.9, "num_workers": 4},
        }),
        ("adapter_config", {
            "peft_type": "LORA", "r": 64, "lora_alpha": 16,
            "lora_dropout": 0.05, "bias": "none",
            "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "task_type": "CAUSAL_LM", "inference_mode": True,
        }),
    ]

    cross_format_results = {}
    for config_name, data in cross_configs:
        cmp = cross_format_comparison(data, "cl100k_base")
        cross_format_results[config_name] = cmp
        print(f"  {config_name}:")
        for fmt in ["json", "yaml", "toml"]:
            r = cmp[fmt]
            print(f"    {fmt:5s} tax={r['syntax_tax_rate']:.1%}  "
                  f"tokens={r['total_tokens']}  "
                  f"bytes={r.get('size_bytes', len(r['text'].encode()))}")

    # Write outputs
    output = {
        "experiment": "EXP-011",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(corpus),
        "results": results,
        "aggregate": aggregate,
        "context_projections": context_projections,
        "comment_analysis": comment_analysis,
        "comment_aggregate": comment_agg,
        "ablation_results": ablation_results,
        "ablation_aggregate": ablation_agg,
        "breakdown_aggregate": breakdown_agg,
        "cross_format_comparison": cross_format_results,
    }

    json_path = output_dir / "exp_011_results.json"
    json_path.write_text(json.dumps(output, indent=2, default=str),
                         encoding="utf-8")
    print(f"\nResults: {json_path}")

    # CSV: per-file results
    csv_path = output_dir / "syntax_tax_results.csv"
    if results:
        fieldnames = ["name", "format", "size_bytes", "tokenizer",
                       "total_tokens", "syntax_tokens", "semantic_tokens",
                       "syntax_tax_rate", "tokens_per_byte"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})
    print(f"CSV: {csv_path}")

    # CSV: context projections
    proj_csv_path = output_dir / "context_projections.csv"
    proj_rows = []
    for fmt, projs in context_projections.items():
        for p in projs:
            proj_rows.append({"format": fmt, **p})
    if proj_rows:
        fieldnames = list(proj_rows[0].keys())
        with open(proj_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(proj_rows)
    print(f"Projections CSV: {proj_csv_path}")

    # CSV: comment ablation
    abl_csv_path = output_dir / "comment_ablation.csv"
    if ablation_results:
        fieldnames = ["name", "format", "with_comments_tax",
                       "without_comments_tax", "tax_increase_without_comments",
                       "semantic_tokens_lost", "total_tokens_with",
                       "total_tokens_without"]
        with open(abl_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in ablation_results:
                writer.writerow({k: r[k] for k in fieldnames})
    print(f"Ablation CSV: {abl_csv_path}")

    # CSV: comment contribution
    comment_csv_path = output_dir / "comment_contribution.csv"
    if comment_analysis:
        fieldnames = ["name", "format", "comment_tokens",
                       "semantic_tokens", "comment_fraction_of_semantic"]
        with open(comment_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in comment_analysis:
                writer.writerow({k: r[k] for k in fieldnames})
    print(f"Comment CSV: {comment_csv_path}")

    print(f"\n{'=' * 70}")
    print("EXP-011 COMPLETE")
    print("=" * 70)

    return output


def main():
    run_experiment()


if __name__ == "__main__":
    main()
