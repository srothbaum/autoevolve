# autoevolve

`autoevolve` is a small, executable AlphaEvolve-style research loop built around the
[autoresearch](https://github.com/karpathy/autoresearch) contract:

- `prepare.py` is the immutable data and evaluation harness.
- `train.py` is the complete program genome.
- `program.md` tells the model what good research means for this task.
- `evolve.py` autonomously samples, edits, evaluates, selects, and checkpoints programs.

The implementation follows the public architecture in the
[AlphaEvolve paper](https://arxiv.org/abs/2506.13131): a prompt sampler selects a parent and
inspiration programs, an LLM ensemble proposes exact code edits, evaluators execute the child,
and an island-based MAP-Elites archive decides what remains available to future generations.
It also adopts the inspiration crossover and configurable exploration ideas described by
[CodeEvolve](https://arxiv.org/abs/2510.14150), while keeping the system small enough to read.

## What is implemented

- Exact `SEARCH/REPLACE` mutations, restricted to `EVOLVE-BLOCK` regions.
- Weighted OpenAI-compatible model ensembles.
- First-class Codex CLI and Claude Code backends using their existing authenticated sessions.
- Rich prompts containing the parent, diverse inspirations, metrics, and failure artifacts.
- Static validation followed by configurable cascade evaluation stages.
- Multiple scalar metrics with an explicit minimize/maximize objective.
- SQLite-backed lineage, prompts, responses, metrics, failures, and controller events.
- MAP-Elites feature cells inside independent islands, plus ring migration.
- An asynchronous controller that overlaps model generation with scarce evaluator capacity.
- Resume-by-default behavior and automatic export of the best program.

This is intentionally a mini implementation. It does not include a distributed scheduler,
embedding service, LLM-based evaluator, or a hardened container sandbox.

## Quick start

Requirements are Python 3.10+, `uv`, and one CUDA-capable GPU supported by the original
autoresearch training program. The evaluator exposes GPU `0` by default and serializes all training
runs, even when several LLM generation workers are active.

```bash
uv sync
uv run prepare.py

export OPENAI_API_KEY="..."
export AUTOEVOLVE_MODEL="your-openai-compatible-model"

uv run evolve.py doctor
uv run evolve.py run --iterations 100
```

## Give the repository to Codex or Claude Code

Both agents receive checked-in project instructions (`AGENTS.md` for Codex and `CLAUDE.md` for
Claude Code). Open this repository in either agent and ask it to run a research session. The agent
uses a model-free controller handshake:

```bash
uv run evolve.py doctor --skip-llm
uv run evolve.py sample
# Agent reads .autoevolve/pending_prompt.md and writes .autoevolve/proposal.txt
uv run evolve.py submit --response .autoevolve/proposal.txt --model-label codex
```

It repeats `sample`/`submit` for the requested number of experiments. Selection, exact patching,
training, result parsing, MAP-Elites insertion, migrations, and checkpoints still go through the
same controller. This is the closest equivalent to handing original autoresearch to a coding agent,
and it needs neither a nested CLI process nor a separate model API key.

For unattended runs after the outer agent exits, use `run` with one of the three providers below.

There are three interchangeable model-provider modes:

```bash
# OpenAI Chat Completions-compatible HTTP endpoint
export AUTOEVOLVE_PROVIDER="openai_compatible"
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://provider.example/v1"  # optional for OpenAI itself
export AUTOEVOLVE_MODEL="provider-model-id"

# Installed and authenticated OpenAI Codex CLI; model override is optional
export AUTOEVOLVE_PROVIDER="codex_cli"

# Installed and authenticated Claude Code CLI; model override is optional
export AUTOEVOLVE_PROVIDER="claude_code"
```

On PowerShell, use `$env:AUTOEVOLVE_PROVIDER = "codex_cli"` (and the same form for the other
variables). For a local HTTP endpoint without authentication, set `llm.api_key_env` to `null` in
`evolve.json`.

“OpenAI-compatible” here means the provider implements the standard
`POST /v1/chat/completions` request and response shape. An arbitrary key for a service that only
implements another API protocol is not sufficient.

Every invocation resumes the SQLite run in `.autoevolve/evolution.db`. The current winner is
written to `.autoevolve/best/train.py`; the tracked source `train.py` is never overwritten by
the controller.

## Commands

```bash
uv run evolve.py run --iterations 20
uv run evolve.py run --iterations 100 --target 0.95
uv run evolve.py sample
uv run evolve.py submit --response .autoevolve/proposal.txt --model-label claude-code
uv run evolve.py status
uv run evolve.py best
uv run evolve.py events --limit 30
uv run evolve.py doctor
```

Use `--config path/to/evolve.json` before the subcommand to run another configuration.

## Configuration

`evolve.json` controls the model ensemble, prompt budget, evaluator commands, objective,
islands, migration cadence, and MAP-Elites dimensions. Model entries are sampled by weight, so
a cheap model can generate most proposals while a stronger model is used occasionally.

Each model entry has a `provider`: `openai_compatible`, `codex_cli`, or `claude_code`.
`AUTOEVOLVE_PROVIDER` overrides it for quick switching. CLI providers use the current CLI login
and default model unless `AUTOEVOLVE_MODEL` or the entry's `name` supplies an override.

Evaluator commands are argument arrays with four placeholders:

- `{project_dir}`: this repository.
- `{candidate_dir}`: the isolated temporary candidate directory.
- `{program}`: the candidate `train.py` path.
- `{run_dir}`: the persistent `.autoevolve` directory.

Add earlier, cheaper stages to the `stages` array to create an evaluation cascade. A stage can
specify `threshold_metric`, `threshold`, and `threshold_direction`; candidates only proceed when
the threshold passes.

MAP-Elites feature names may refer to any numeric evaluator metric. The built-in `code_length`
and `novelty` dimensions are also available. The objective metric controls quality within each
cell; feature metrics preserve useful behavioral diversity.

## Safety and reproducibility

Candidate programs run in temporary directories and cannot modify the tracked `train.py` through
the normal workflow. This is process isolation, not a security boundary: generated code is still
code. Run autonomous evolution on a machine and account intended for this workload.

Common LLM credential variables are removed from the candidate training environment. CLI login
files still live on the host, so use an isolated machine/account when the model or task context is
not trusted.

Sampling is seeded and every prompt, response, result, and artifact is retained. Remote model
APIs can still be nondeterministic, so exact replay depends on the provider.

## Tests

The test suite uses a fake model and tiny Python evaluators, so it requires neither a GPU nor an
API key:

```bash
python -m unittest discover -s tests -v
```

## Prior art

The module boundaries and evaluator artifact handling were informed by
[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve). The smaller agent-role
split in [OpenAlpha_Evolve](https://github.com/shyamsaktawat/OpenAlpha_Evolve) was useful as a
contrast; this project keeps those responsibilities explicit without creating an agent class for
each one.

MIT license.
