# Claude Code Instructions

Read and follow `AGENTS.md`; it is the canonical operating guide for this repository.

When you are already the active Claude Code agent, use the outer-agent `sample`/`submit` loop in
`AGENTS.md`; this is the direct autoresearch-style workflow and needs no nested Claude process.
When the user instead wants the controller to run unattended after this session exits, set
`AUTOEVOLVE_PROVIDER=claude_code`, run `uv run evolve.py doctor`, and launch
`uv run evolve.py run --iterations <N>`. That mode invokes a separately installed Claude Code CLI
in non-interactive, tool-disabled generation mode.

Do not directly edit `train.py` as a substitute for the controller loop. Do not modify
`prepare.py` or the evaluation contract.
