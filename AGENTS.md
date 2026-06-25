# AI_PIPELINE_4 — Agent Instructions

## Superpowers Integration

This project uses [superpowers](https://github.com/obra/superpowers) methodology.
Skills are located in `.superpowers/skills/`. Available skills:

- **brainstorming** — Refine requirements through guided questions
- **writing-plans** — Break work into small, precise tasks
- **executing-plans** — Execute planned tasks step by step
- **subagent-driven-development** — Dispatch specialized subagents
- **test-driven-development** — RED-GREEN-REFACTOR cycles
- **systematic-debugging** — Structured debugging approach
- **requesting-code-review** / **receiving-code-review** — Code review workflow
- **dispatching-parallel-agents** — Run multiple agents in parallel
- **using-git-worktrees** — Isolated development branches
- **finishing-a-development-branch** — Merge and cleanup
- **verification-before-completion** — Verify work before marking done

**Rule**: Before any non-trivial task, read the relevant skill file at `.superpowers/skills/<skill-name>/SKILL.md` and follow its methodology.

## Codegraph

This project has a `.codegraph/` index. Use codegraph to understand codebase structure before making changes.

## Project Overview

- **Language**: Python 3
- **Architecture**: Modular pipeline — structural engineering PDF → 3D model
  - `src/slab_v2/` — deterministic vector kernel + Gemini-as-selector
  - `src/` — legacy modules (floor, column, wall detection)
  - `app_v2.py` — main entry point (v2)
  - `tests/` — test suite
  - `debug_images/` — intermediate visual outputs (per-run hash folders)
  - `output/` — generated .rb (SketchUp Ruby), .csv, .log files

## Rules

- NEVER modify existing code without reading and understanding it first
- Always produce debug images for visual pipeline steps
- Keep modules isolated — each detector/extractor is self-contained
- When touching slab_v2, respect the vector-first approach: geometry from PDF vectors, AI only for selection/classification
- Do not commit secrets or API keys
