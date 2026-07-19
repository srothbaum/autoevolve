from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


SUCCESS_STATUSES = {"baseline", "success"}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    minimum: float
    maximum: float
    bins: int

    def coordinate(self, value: float) -> int:
        if self.bins < 1:
            raise ValueError(f"Feature {self.name!r} must have at least one bin")
        if self.maximum <= self.minimum:
            raise ValueError(f"Feature {self.name!r} maximum must exceed minimum")
        fraction = (value - self.minimum) / (self.maximum - self.minimum)
        return min(self.bins - 1, max(0, int(fraction * self.bins)))


@dataclass
class Program:
    id: str
    code: str
    parent_id: str | None = None
    inspiration_ids: list[str] = field(default_factory=list)
    island: int = 0
    generation: int = 0
    status: str = "pending"
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    prompt: str | None = None
    response: str | None = None
    error: str | None = None
    novelty: float = 0.0
    created_at: float = field(default_factory=time.time)
    sample_count: int = 0

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()

    @property
    def successful(self) -> bool:
        return self.status in SUCCESS_STATUSES

    @property
    def line_count(self) -> int:
        return len(self.code.splitlines())

    def feature_value(self, name: str) -> float:
        if name == "code_length":
            return float(self.line_count)
        if name == "novelty":
            return self.novelty
        return float(self.metrics.get(name, 0.0))

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "code_hash": self.code_hash,
            "parent_id": self.parent_id,
            "inspiration_ids": json.dumps(self.inspiration_ids),
            "island": self.island,
            "generation": self.generation,
            "status": self.status,
            "metrics": json.dumps(self.metrics, sort_keys=True),
            "artifacts": json.dumps(self.artifacts, sort_keys=True),
            "model": self.model,
            "prompt": self.prompt,
            "response": self.response,
            "error": self.error,
            "novelty": self.novelty,
            "created_at": self.created_at,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_row(cls, row: Any) -> "Program":
        return cls(
            id=row["id"],
            code=row["code"],
            parent_id=row["parent_id"],
            inspiration_ids=json.loads(row["inspiration_ids"] or "[]"),
            island=row["island"],
            generation=row["generation"],
            status=row["status"],
            metrics=json.loads(row["metrics"] or "{}"),
            artifacts=json.loads(row["artifacts"] or "{}"),
            model=row["model"],
            prompt=row["prompt"],
            response=row["response"],
            error=row["error"],
            novelty=row["novelty"],
            created_at=row["created_at"],
            sample_count=row["sample_count"],
        )


@dataclass(frozen=True)
class Sample:
    parent: Program
    inspirations: list[Program]
    island: int
    mode: str


@dataclass(frozen=True)
class Generation:
    text: str
    model: str


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    metrics: dict[str, float]
    artifacts: dict[str, Any]
    error: str | None = None


class Generator(Protocol):
    async def generate(self, system: str, user: str) -> Generation: ...


@dataclass(frozen=True)
class PendingProposal:
    iteration: int
    parent_id: str
    inspiration_ids: list[str]
    island: int
    mode: str
    system: str
    user: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "parent_id": self.parent_id,
            "inspiration_ids": self.inspiration_ids,
            "island": self.island,
            "mode": self.mode,
            "system": self.system,
            "user": self.user,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingProposal":
        return cls(
            iteration=int(data["iteration"]),
            parent_id=str(data["parent_id"]),
            inspiration_ids=[str(item) for item in data.get("inspiration_ids", [])],
            island=int(data["island"]),
            mode=str(data["mode"]),
            system=str(data["system"]),
            user=str(data["user"]),
            created_at=float(data.get("created_at", time.time())),
        )
