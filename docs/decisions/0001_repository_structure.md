# ADR 0001: Canonical repository structure

- **Status:** Accepted
- **Date:** 2026-08-09

## Context

The repository mixed active code, duplicated model snapshots, generated outputs, paper builds, literature, and temporary analysis in top-level directories. It was unclear which copy of a model or manuscript was authoritative.

## Decision

Use `src/sura/` as the sole reusable-code location, `scripts/` for workflows, `paper/` for the manuscript, `experiments/results/` for reviewed metrics, and `archive/` for non-canonical history. Preserve the pre-renovation state on a dedicated branch.

## Consequences

New work has clear ownership and import paths. Historical material remains available but cannot silently become the source of a new result. Some legacy workflows require path migration before they can be rerun from the new layout.
