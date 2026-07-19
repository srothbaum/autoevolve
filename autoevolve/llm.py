from __future__ import annotations

import asyncio
import json
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


SUPPORTED_PROVIDERS = {"openai_compatible", "codex_cli", "claude_code"}


class ModelEnsemble:
    """Weighted ensemble spanning HTTP models, Codex CLI, and Claude Code."""

    def __init__(self, config: LLMConfig, seed: int = 42, cwd: Path | None = None):
        self.config = config
        self.random = random.Random(seed)
        self.cwd = None if cwd is None else cwd.resolve()

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

    def _choose_model(self) -> _ResolvedModel:
        if not self.config.models:
            raise RuntimeError("No models configured")
        weights = [max(0.0, model.weight) for model in self.config.models]
        if not any(weights):
            raise RuntimeError("At least one model weight must be positive")
        spec = self.random.choices(self.config.models, weights=weights, k=1)[0]
        return self._resolve_model(spec)

    def _api_base(self) -> str:
        if self.config.api_base_env:
            override = os.environ.get(self.config.api_base_env)
            if override:
                return override.rstrip("/")
        return self.config.api_base.rstrip("/")

    def _http_request(self, model: _ResolvedModel, system: str, user: str) -> str:
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
                return content
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

    async def generate(self, system: str, user: str) -> Generation:
        model = self._choose_model()
        provider = model.spec.resolved_provider()
        if provider == "openai_compatible":
            text = await asyncio.to_thread(self._http_request, model, system, user)
        else:
            text = await self._cli_request(model, system, user)
        return Generation(text=text, model=model.name)


OpenAICompatibleEnsemble = ModelEnsemble
