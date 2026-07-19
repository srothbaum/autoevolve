# autoevolve

`autoevolve` is a readable, single-GPU implementation of the public
[AlphaEvolve](https://arxiv.org/abs/2506.13131) design applied to
[autoresearch](https://github.com/karpathy/autoresearch). It evolves the complete `train.py`
program while keeping `prepare.py`, the data, evaluation, and five-minute budget fixed.

Each iteration:

1. Samples a parent, donor inspirations, and a patch, mutable-block rewrite, or crossover operator.
2. Gives their code, metrics, failures, and periodically distilled research memory to an LLM.
3. Repairs malformed unattended proposals and rejects token-level near-duplicates before GPU evaluation.
4. Evaluates a novel child, assigns a parent-relative reward, and stores everything in SQLite.

The controller uses island-based MAP-Elites, migration, evaluation cascades, bounded proposal
repair, and UCB model selection. It resumes automatically and exports the best program without
overwriting tracked `train.py`. This mini version omits distributed scheduling, embedding or LLM
novelty judges, dynamic islands, prompt evolution, and container isolation.

## Setup

Requirements: Python 3.10+, `uv`, and one CUDA-capable GPU supported by `train.py`.

```bash
uv sync
uv run prepare.py
```

The default evaluator exposes only GPU `0` and runs one candidate at a time. LLM proposal workers
may run concurrently, but GPU training remains serialized.

## Coding-agent mode

Open the repository in Codex or Claude Code and ask it to run a research session. `AGENTS.md` and
`CLAUDE.md` define the same controller handshake:

```bash
uv run evolve.py doctor --skip-llm
uv run evolve.py sample
# Agent writes its response to .autoevolve/proposal.txt
uv run evolve.py submit --response .autoevolve/proposal.txt --model-label codex
```

The agent repeats `sample` and `submit`; the controller owns selection, patch validation,
evaluation, and persistence. This mode uses the outer agent's existing session and needs no nested
model process or separate API key.

## Unattended mode

Run the controller with an OpenAI-compatible HTTP endpoint, Codex CLI, or Claude Code CLI:

```bash
# OpenAI-compatible HTTP
export AUTOEVOLVE_PROVIDER="openai_compatible"
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://provider.example/v1"
export AUTOEVOLVE_MODEL="provider-model-id"
uv run evolve.py run --iterations 100

# Installed and authenticated CLI; AUTOEVOLVE_MODEL is optional
export AUTOEVOLVE_PROVIDER="codex_cli"       # or: claude_code
uv run evolve.py run --iterations 100
```

For OpenAI itself, omit `OPENAI_BASE_URL`; the default is `https://api.openai.com/v1`. Common HTTP
compatibility endpoints include:

| Service | `OPENAI_BASE_URL` |
| --- | --- |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Ollama | `http://localhost:11434/v1` |
| vLLM | `http://localhost:8000/v1` |
| OptiLLM | `http://localhost:8000/v1` |

Use the service's model ID and put its key in `OPENAI_API_KEY`. Ollama accepts a dummy key. For an
unauthenticated endpoint, set `llm.api_key_env` to `null` in `evolve.json`.

Compatibility means accepting `POST /chat/completions` under the configured base URL with OpenAI
Chat Completions request and response fields. Native Gemini or Anthropic endpoints are not accepted.
One run has one HTTP base URL and credential source; use a router to combine models hosted by
different HTTP providers.

On PowerShell, replace `export NAME="value"` with `$env:NAME = "value"`.

## Commands and state

```bash
uv run evolve.py doctor
uv run evolve.py run --iterations 20
uv run evolve.py run --iterations 100 --target 0.95
uv run evolve.py sample
uv run evolve.py submit --response .autoevolve/proposal.txt --model-label codex
uv run evolve.py status
uv run evolve.py best
uv run evolve.py events --limit 30
```

Put `--config path/to/evolve.json` before the subcommand to use another configuration. Runs resume
from `.autoevolve/evolution.db`; the winner is exported to `.autoevolve/best/train.py`. The
controller never overwrites the tracked `train.py`.

`evolve.json` defines models, evaluator stages, islands, operators, novelty, memory, and budgets.
Important controls:

- `llm.selection_strategy`: `ucb` adapts to measured model rewards; `weighted` preserves static
  weighted sampling. A model with weight `0` is disabled.
- `prompt.operator_weights`: probabilities for targeted patches, complete mutable-block rewrites,
  and donor crossover. Protected code and `EVOLVE-BLOCK` markers remain immutable.
- `prompt.proposal_retries`: repair or novelty-resampling attempts before recording a rejection.
- `prompt.novelty_threshold`: maximum archived token-shingle similarity; `0` disables the gate.
- `llm.max_calls`, `max_total_tokens`, `max_cost_usd`, and `run.max_wall_seconds`: optional run
  limits. Set per-model token prices to estimate cost; CLI token counts are estimated from text.

Evaluator stages can enforce metric thresholds before an expensive stage. MAP-Elites feature
dimensions may use numeric metrics, `code_length`, or `novelty`.

`proposal_retries` applies to unattended providers. In coding-agent mode, `submit` records the
rejection and the outer agent sees that feedback on the next `sample`, preserving the direct
autoresearch handshake without a nested model call.

## Safety and tests

Candidates execute in temporary directories, and common LLM API-key variables are removed from
their environment. This is process isolation, not a security boundary; generated Python still runs
on the host. Use an isolated machine or account for untrusted models or tasks.

Sampling is seeded, and prompts, retries, model usage, rewards, results, failures, and artifacts are
retained. Existing databases are migrated additively. Remote models may remain nondeterministic.

Tests use fake models and tiny evaluators, requiring neither a GPU nor an API key:

```bash
python -m unittest discover -s tests -v
```

Implementation references:
[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) and
[OpenAlpha_Evolve](https://github.com/shyamsaktawat/OpenAlpha_Evolve).

MIT License
