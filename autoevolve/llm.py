from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from .config import LLMConfig, ModelSpec
from .types import Generation


@dataclass(frozen=True)
class _ResolvedModel:
    name: str
    spec: ModelSpec


@dataclass
class _ModelStats:
    outcomes: int = 0
    reward: float = 0.0
    cost_usd: float = 0.0


class BudgetExceeded(RuntimeError):
    pass


SUPPORTED_PROVIDERS = {"openai_compatible", "codex_cli", "claude_code"}


class ModelEnsemble:
    """Budgeted adaptive ensemble spanning HTTP models, Codex CLI, and Claude Code."""

    def __init__(self, config: LLMConfig, seed: int = 42, cwd: Path | None = None):
        self.config = config
        self.random = random.Random(seed)
        self.cwd = None if cwd is None else cwd.resolve()
        self._stats: dict[str, _ModelStats] = {}
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0

    def _resolve_model(self, spec: ModelSpec) -> _ResolvedModel:
        provider = spec.resolved_provider()
        if provider not in SUPPORTED_PROVIDERS:
            raise RuntimeError(
                f"Unknown LLM provider {provider!r}; expected one of {sorted(SUPPORTED_PROVIDERS)}"
            )
        name = spec.name.strip()
        if not name and spec.name_env:
            name = os.environ.get(spec.name_env, "").strip()
        if not name and provider == "openai_compatible":
            env_hint = spec.name_env or "the model name in evolve.json"
            raise RuntimeError(f"No LLM model configured; set {env_hint}")
        return _ResolvedModel(name=name or f"{provider}-default", spec=spec)

    def _available_models(self) -> list[_ResolvedModel]:
        if not self.config.models:
            raise RuntimeError("No models configured")
        enabled = [model for model in self.config.models if model.weight > 0]
        weights = [max(0.0, model.weight) for model in self.config.models]
        if not any(weights):
            raise RuntimeError("At least one model weight must be positive")
        return [self._resolve_model(model) for model in enabled]

    def _choose_model(self) -> _ResolvedModel:
        models = self._available_models()
        if self.config.selection_strategy == "weighted" or len(models) == 1:
            return self.random.choices(
                models, weights=[item.spec.weight for item in models], k=1
            )[0]

        untried = [
            item
            for item in models
            if self._stats.get(item.name, _ModelStats()).outcomes == 0
        ]
        if untried:
            return self.random.choices(
                untried, weights=[item.spec.weight for item in untried], k=1
            )[0]

        total = sum(self._stats[item.name].outcomes for item in models)

        def score(item: _ResolvedModel) -> float:
            stats = self._stats[item.name]
            mean_reward = stats.reward / stats.outcomes
            mean_cost = stats.cost_usd / stats.outcomes
            exploration = self.config.ucb_exploration * math.sqrt(
                math.log(total + 1) / stats.outcomes
            )
            return mean_reward + exploration - self.config.cost_penalty * mean_cost

        scored = [(score(item), self.random.random(), item) for item in models]
        return max(scored, key=lambda entry: (entry[0], entry[1]))[2]

    def _model_by_name(self, name: str) -> _ResolvedModel:
        for model in self._available_models():
            if model.name == name:
                return model
        return self._choose_model()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def seed_usage(
        self, calls: int, input_tokens: int, output_tokens: int, cost_usd: float
    ) -> None:
        self._calls = max(0, int(calls))
        self._input_tokens = max(0, int(input_tokens))
        self._output_tokens = max(0, int(output_tokens))
        self._cost_usd = max(0.0, float(cost_usd))

    def seed_outcomes(self, outcomes: list[tuple[str, float, float]]) -> None:
        for model, reward, cost_usd in outcomes:
            self.record_outcome(model, reward, cost_usd)

    def record_outcome(self, model: str, reward: float, cost_usd: float = 0.0) -> None:
        stats = self._stats.setdefault(model, _ModelStats())
        stats.outcomes += 1
        stats.reward += float(reward)
        stats.cost_usd += max(0.0, float(cost_usd))

    def usage(self) -> dict[str, float | int]:
        return {
            "calls": self._calls,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cost_usd": self._cost_usd,
        }

    def _reserve_call(self) -> None:
        total_tokens = self._input_tokens + self._output_tokens
        if self.config.max_calls is not None and self._calls >= self.config.max_calls:
            raise BudgetExceeded(f"LLM call budget exhausted ({self.config.max_calls})")
        if (
            self.config.max_total_tokens is not None
            and total_tokens >= self.config.max_total_tokens
        ):
            raise BudgetExceeded(f"LLM token budget exhausted ({self.config.max_total_tokens})")
        if self.config.max_cost_usd is not None and self._cost_usd >= self.config.max_cost_usd:
            raise BudgetExceeded(f"LLM cost budget exhausted (${self.config.max_cost_usd:.4f})")
        self._calls += 1

    def _api_base(self) -> str:
        if self.config.api_base_env:
            override = os.environ.get(self.config.api_base_env)
            if override:
                return override.rstrip("/")
        return self.config.api_base.rstrip("/")

    def _http_request(self, model: _ResolvedModel, system: str, user: str) -> tuple[str, int, int]:
        base = self._api_base()
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing API key environment variable {self.config.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": model.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": model.spec.max_tokens,
        }
        if model.spec.temperature is not None:
            payload["temperature"] = model.spec.temperature

        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                api_request = request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with request.urlopen(api_request, timeout=self.config.timeout_seconds) as response:
                    data = json.load(response)
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "\n".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part)
                        for part in content
                    )
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("LLM returned an empty response")
                usage = data.get("usage", {})
                input_tokens = int(
                    usage.get("prompt_tokens", self._estimate_tokens(system + "\n" + user))
                )
                output_tokens = int(
                    usage.get("completion_tokens", self._estimate_tokens(content))
                )
                return content, input_tokens, output_tokens
            except (error.URLError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.config.retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error

    async def _cli_request(self, model: _ResolvedModel, system: str, user: str) -> str:
        provider = model.spec.resolved_provider()
        configured_name = model.name if not model.name.endswith("-default") else ""
        if provider == "codex_cli":
            command = [
                model.spec.executable or "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
            ]
            if configured_name:
                command.extend(["--model", configured_name])
            command.extend(model.spec.extra_args)
            command.append("-")
            prompt = f"{system}\n\n{user}"
        elif provider == "claude_code":
            command = [
                model.spec.executable or "claude",
                "--bare",
                "-p",
                "--no-session-persistence",
                "--tools",
                "",
                "--strict-mcp-config",
                "--output-format",
                "text",
                "--append-system-prompt",
                system,
            ]
            if configured_name:
                command.extend(["--model", configured_name])
            command.extend(model.spec.extra_args)
            prompt = user
        else:
            raise RuntimeError(f"Provider {provider!r} is not a CLI provider")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start {provider}: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"{provider} exceeded the {self.config.timeout_seconds}s generation timeout"
            ) from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"{provider} exited with code {process.returncode}: {detail}")
        output = stdout.decode("utf-8", errors="replace").strip()
        if not output:
            raise RuntimeError(f"{provider} returned an empty response")
        return output

    async def _generate(
        self, model: _ResolvedModel, system: str, user: str
    ) -> Generation:
        self._reserve_call()
        provider = model.spec.resolved_provider()
        if provider == "openai_compatible":
            text, input_tokens, output_tokens = await asyncio.to_thread(
                self._http_request, model, system, user
            )
        else:
            text = await self._cli_request(model, system, user)
            input_tokens = self._estimate_tokens(system + "\n" + user)
            output_tokens = self._estimate_tokens(text)
        cost_usd = (
            input_tokens * model.spec.input_cost_per_million
            + output_tokens * model.spec.output_cost_per_million
        ) / 1_000_000
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cost_usd += cost_usd
        return Generation(
            text=text,
            model=model.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    async def generate(self, system: str, user: str) -> Generation:
        return await self._generate(self._choose_model(), system, user)

    async def generate_with_model(self, system: str, user: str, model: str) -> Generation:
        return await self._generate(self._model_by_name(model), system, user)


OpenAICompatibleEnsemble = ModelEnsemble
