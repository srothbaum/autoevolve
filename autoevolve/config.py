from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import FeatureSpec


@dataclass(frozen=True)
class RunConfig:
    iterations: int = 100
    workers: int = 2
    seed: int = 42
    run_dir: Path = Path(".autoevolve")
    max_wall_seconds: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    provider: str = "openai_compatible"
    provider_env: str | None = "AUTOEVOLVE_PROVIDER"
    name: str = ""
    name_env: str | None = "AUTOEVOLVE_MODEL"
    weight: float = 1.0
    temperature: float | None = 0.8
    max_tokens: int = 12000
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    executable: str | None = None
    extra_args: list[str] = field(default_factory=list)

    def resolved_provider(self) -> str:
        if self.provider_env:
            import os

            override = os.environ.get(self.provider_env)
            if override:
                return override.strip()
        return self.provider


@dataclass(frozen=True)
class LLMConfig:
    api_base: str = "https://api.openai.com/v1"
    api_base_env: str | None = "OPENAI_BASE_URL"
    api_key_env: str | None = "OPENAI_API_KEY"
    timeout_seconds: float = 180.0
    retries: int = 2
    selection_strategy: str = "ucb"
    ucb_exploration: float = 0.7
    cost_penalty: float = 0.0
    max_calls: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    models: list[ModelSpec] = field(default_factory=lambda: [ModelSpec()])


@dataclass(frozen=True)
class PromptConfig:
    task_file: Path = Path("program.md")
    num_inspirations: int = 2
    max_prompt_chars: int = 140000
    artifact_chars: int = 6000
    failed_attempts: int = 2
    proposal_retries: int = 2
    novelty_threshold: float = 0.999
    novelty_shingle_size: int = 5
    memory_interval: int = 10
    memory_items: int = 5
    operator_weights: dict[str, float] = field(
        default_factory=lambda: {"patch": 0.6, "rewrite": 0.25, "crossover": 0.15}
    )


@dataclass(frozen=True)
class StageConfig:
    name: str
    command: list[str]
    timeout_seconds: float
    required_metrics: list[str] = field(default_factory=list)
    threshold_metric: str | None = None
    threshold: float | None = None
    threshold_direction: str = "maximize"


@dataclass(frozen=True)
class EvaluatorConfig:
    program_file: Path = Path("train.py")
    fixed_files: list[Path] = field(default_factory=lambda: [Path("prepare.py")])
    objective: str = "val_bpb"
    direction: str = "minimize"
    concurrency: int = 1
    cuda_visible_devices: str = "0"
    strip_env: list[str] = field(
        default_factory=lambda: [
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
        ]
    )
    stages: list[StageConfig] = field(default_factory=list)


@dataclass(frozen=True)
class DatabaseConfig:
    path: Path = Path(".autoevolve/evolution.db")
    population_size: int = 64
    num_islands: int = 4
    migration_interval: int = 20
    migration_count: int = 1
    exploitation_ratio: float = 0.70
    exploration_ratio: float = 0.20
    features: list[FeatureSpec] = field(default_factory=list)


@dataclass(frozen=True)
class AppConfig:
    project_dir: Path
    run: RunConfig
    llm: LLMConfig
    prompt: PromptConfig
    evaluator: EvaluatorConfig
    database: DatabaseConfig


def _resolve_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_dir / path).resolve()


def _model_from_dict(data: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        provider=str(data.get("provider", "openai_compatible")),
        provider_env=data.get("provider_env", "AUTOEVOLVE_PROVIDER"),
        name=str(data.get("name", "")),
        name_env=data.get("name_env", "AUTOEVOLVE_MODEL"),
        weight=float(data.get("weight", 1.0)),
        temperature=(
            None if data.get("temperature") is None else float(data["temperature"])
        ),
        max_tokens=int(data.get("max_tokens", 12000)),
        input_cost_per_million=max(0.0, float(data.get("input_cost_per_million", 0.0))),
        output_cost_per_million=max(0.0, float(data.get("output_cost_per_million", 0.0))),
        executable=data.get("executable"),
        extra_args=[str(item) for item in data.get("extra_args", [])],
    )


def load_config(path: str | Path = "evolve.json") -> AppConfig:
    config_path = Path(path).resolve()
    project_dir = config_path.parent
    data = json.loads(config_path.read_text(encoding="utf-8"))

    run_data = data.get("run", {})
    run = RunConfig(
        iterations=max(0, int(run_data.get("iterations", 100))),
        workers=max(1, int(run_data.get("workers", 2))),
        seed=int(run_data.get("seed", 42)),
        run_dir=_resolve_path(project_dir, run_data.get("run_dir", ".autoevolve")),
        max_wall_seconds=(
            None
            if run_data.get("max_wall_seconds") is None
            else max(0.0, float(run_data["max_wall_seconds"]))
        ),
    )

    llm_data = data.get("llm", {})
    models = [_model_from_dict(item) for item in llm_data.get("models", [{}])]
    llm = LLMConfig(
        api_base=str(llm_data.get("api_base", "https://api.openai.com/v1")),
        api_base_env=llm_data.get("api_base_env", "OPENAI_BASE_URL"),
        api_key_env=llm_data.get("api_key_env", "OPENAI_API_KEY"),
        timeout_seconds=float(llm_data.get("timeout_seconds", 180)),
        retries=max(0, int(llm_data.get("retries", 2))),
        selection_strategy=str(llm_data.get("selection_strategy", "ucb")),
        ucb_exploration=max(0.0, float(llm_data.get("ucb_exploration", 0.7))),
        cost_penalty=max(0.0, float(llm_data.get("cost_penalty", 0.0))),
        max_calls=(
            None if llm_data.get("max_calls") is None else max(0, int(llm_data["max_calls"]))
        ),
        max_total_tokens=(
            None
            if llm_data.get("max_total_tokens") is None
            else max(0, int(llm_data["max_total_tokens"]))
        ),
        max_cost_usd=(
            None
            if llm_data.get("max_cost_usd") is None
            else max(0.0, float(llm_data["max_cost_usd"]))
        ),
        models=models,
    )
    if llm.selection_strategy not in {"weighted", "ucb"}:
        raise ValueError("llm.selection_strategy must be 'weighted' or 'ucb'")

    prompt_data = data.get("prompt", {})
    operator_weights = {
        str(name): float(weight)
        for name, weight in prompt_data.get(
            "operator_weights", {"patch": 0.6, "rewrite": 0.25, "crossover": 0.15}
        ).items()
    }
    prompt = PromptConfig(
        task_file=_resolve_path(project_dir, prompt_data.get("task_file", "program.md")),
        num_inspirations=max(0, int(prompt_data.get("num_inspirations", 2))),
        max_prompt_chars=int(prompt_data.get("max_prompt_chars", 140000)),
        artifact_chars=int(prompt_data.get("artifact_chars", 6000)),
        failed_attempts=max(0, int(prompt_data.get("failed_attempts", 2))),
        proposal_retries=max(0, int(prompt_data.get("proposal_retries", 2))),
        novelty_threshold=float(prompt_data.get("novelty_threshold", 0.999)),
        novelty_shingle_size=max(1, int(prompt_data.get("novelty_shingle_size", 5))),
        memory_interval=max(0, int(prompt_data.get("memory_interval", 10))),
        memory_items=max(0, int(prompt_data.get("memory_items", 5))),
        operator_weights=operator_weights,
    )
    unknown_operators = set(prompt.operator_weights) - {"patch", "rewrite", "crossover"}
    if unknown_operators:
        raise ValueError(f"Unknown prompt operators: {sorted(unknown_operators)}")
    if any(weight < 0 for weight in prompt.operator_weights.values()) or not any(
        prompt.operator_weights.values()
    ):
        raise ValueError("prompt.operator_weights must contain at least one positive weight")
    if not 0.0 <= prompt.novelty_threshold <= 1.0:
        raise ValueError("prompt.novelty_threshold must be between 0 and 1")

    evaluator_data = data.get("evaluator", {})
    stages = [
        StageConfig(
            name=str(item["name"]),
            command=[str(part) for part in item["command"]],
            timeout_seconds=float(item.get("timeout_seconds", 900)),
            required_metrics=[str(metric) for metric in item.get("required_metrics", [])],
            threshold_metric=item.get("threshold_metric"),
            threshold=(None if item.get("threshold") is None else float(item["threshold"])),
            threshold_direction=str(item.get("threshold_direction", "maximize")),
        )
        for item in evaluator_data.get("stages", [])
    ]
    if not stages:
        raise ValueError("evaluator.stages must contain at least one command stage")
    for stage in stages:
        if stage.threshold_direction not in {"minimize", "maximize"}:
            raise ValueError(
                f"Stage {stage.name!r} threshold_direction must be 'minimize' or 'maximize'"
            )
    evaluator = EvaluatorConfig(
        program_file=_resolve_path(project_dir, evaluator_data.get("program_file", "train.py")),
        fixed_files=[
            _resolve_path(project_dir, item)
            for item in evaluator_data.get("fixed_files", ["prepare.py"])
        ],
        objective=str(evaluator_data.get("objective", "val_bpb")),
        direction=str(evaluator_data.get("direction", "minimize")),
        concurrency=max(1, int(evaluator_data.get("concurrency", 1))),
        cuda_visible_devices=str(evaluator_data.get("cuda_visible_devices", "0")),
        strip_env=[
            str(item)
            for item in evaluator_data.get(
                "strip_env",
                [
                    "OPENAI_API_KEY",
                    "CODEX_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "ANTHROPIC_AUTH_TOKEN",
                ],
            )
        ],
        stages=stages,
    )
    if evaluator.direction not in {"minimize", "maximize"}:
        raise ValueError("evaluator.direction must be 'minimize' or 'maximize'")

    database_data = data.get("database", {})
    features = [
        FeatureSpec(
            name=str(item["name"]),
            minimum=float(item["minimum"]),
            maximum=float(item["maximum"]),
            bins=int(item["bins"]),
        )
        for item in database_data.get("features", [])
    ]
    database = DatabaseConfig(
        path=_resolve_path(project_dir, database_data.get("path", ".autoevolve/evolution.db")),
        population_size=max(1, int(database_data.get("population_size", 64))),
        num_islands=max(1, int(database_data.get("num_islands", 4))),
        migration_interval=max(0, int(database_data.get("migration_interval", 20))),
        migration_count=max(1, int(database_data.get("migration_count", 1))),
        exploitation_ratio=float(database_data.get("exploitation_ratio", 0.70)),
        exploration_ratio=float(database_data.get("exploration_ratio", 0.20)),
        features=features,
    )
    if database.exploitation_ratio < 0 or database.exploration_ratio < 0:
        raise ValueError("Database sampling ratios cannot be negative")
    if database.exploitation_ratio + database.exploration_ratio > 1:
        raise ValueError("Database exploitation + exploration ratios cannot exceed 1")

    return AppConfig(
        project_dir=project_dir,
        run=run,
        llm=llm,
        prompt=prompt,
        evaluator=evaluator,
        database=database,
    )
