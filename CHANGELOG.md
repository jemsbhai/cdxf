# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-05-25

### Fixed
- YAML bridge comment deduplication bug: ruamel.yaml stores the same
  CommentToken in both parent and child nodes; the bridge extracted
  from both, inflating comment count on each round-trip. Fixed by
  tracking CommentToken identity (`id()`) and skipping duplicates.
  Comments now preserved exactly: N in, N out, on unlimited round-trips.

### Added
- Project scaffolding and initial repository structure
- Literature survey covering existing binary serialization formats
- Gap analysis identifying the need for a universal binary interchange format
