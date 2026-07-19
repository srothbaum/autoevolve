# Autoevolve Agent Guide

This repository is an executable mini AlphaEvolve system specialized to autoresearch.

## Contract

- `prepare.py` is immutable evaluation code.
- `train.py` is the only evolved program and must keep its metric summary.
- `program.md` is research context fed to every generation call.
- `evolve.py` owns parent selection, prompting, patching, evaluation, persistence, and export.
- Do not replace the controller with an ad hoc manual experiment loop.
- Do not edit the tracked `train.py` while a run is active. Winners are exported under
  `.autoevolve/best/`.

## Standard Workflow

1. Run `uv sync` when the environment is not prepared.
2. Run `uv run prepare.py` only when the autoresearch dataset/tokenizer cache is absent.
3. Run `uv run evolve.py doctor --skip-llm` and resolve evaluator prerequisites.
4. When you are the research model, repeat this exact loop for the requested number of attempts:
   - Run `uv run evolve.py sample`.
   - Read `.autoevolve/pending_prompt.md` and produce the requested SEARCH/REPLACE response.
   - Write that response to `.autoevolve/proposal.txt`.
   - Run `uv run evolve.py submit --response .autoevolve/proposal.txt --model-label <your-name>`.
5. Do not stop because a child crashes or is rejected; inspect the recorded feedback and sample again.
6. Inspect progress with `uv run evolve.py status`, `best`, and `events`.
7. Run `python -m unittest discover -s tests -v` after controller changes.

The outer-agent loop above is the direct equivalent of original autoresearch and requires no nested
model API. `uv run evolve.py run --iterations <N>` is a second option that lets the controller call
an HTTP or CLI model provider automatically.

The evaluator is intentionally single-GPU: `CUDA_VISIBLE_DEVICES` defaults to `0`, `train.py`
uses one CUDA device, and evaluator concurrency is `1`. Generation workers may overlap LLM calls
with that one evaluation slot; they do not consume additional GPUs.

## Model Providers

- `openai_compatible`: requires `OPENAI_API_KEY`, `AUTOEVOLVE_MODEL`, and optionally
  `OPENAI_BASE_URL`; the endpoint must implement `POST /v1/chat/completions`.
- `codex_cli`: uses the installed, authenticated Codex CLI through non-interactive `codex exec`.
- `claude_code`: uses the installed, authenticated Claude Code CLI through print mode.

Set `AUTOEVOLVE_PROVIDER` to choose a provider. For CLI providers, `AUTOEVOLVE_MODEL` is optional;
without it the CLI's configured/recommended model is used.

Generated candidate processes do not receive common LLM API-key environment variables. Do not
weaken that separation when changing the evaluator.
