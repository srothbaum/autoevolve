from __future__ import annotations

import json
import random
from pathlib import Path

from .config import PromptConfig
from .types import Program


SYSTEM_MESSAGE = """You are the creative code-generation component of an evolutionary research system.
Propose one coherent, testable improvement to the current program. The program will be executed by an
external evaluator, so do not claim results and do not alter or bypass the evaluation contract.

Return targeted edits in this exact format, with enough unchanged context for every SEARCH to match once:

<<<<<<< SEARCH
exact text from the current program
=======
replacement text
>>>>>>> REPLACE

You may emit multiple blocks when one idea requires coordinated edits. SEARCH text must be copied exactly.
Only edit inside EVOLVE-BLOCK regions when they are present. Do not return a full file or a unified diff.
"""


SEARCH_STYLES = (
    "Favor a small, high-confidence improvement with a clear causal hypothesis.",
    "Explore a structurally different approach while keeping the implementation internally consistent.",
    "Look for a simplification that improves throughput, memory use, or optimization behavior.",
    "Combine one complementary idea from the inspirations with the strongest parts of the parent.",
)


class PromptBuilder:
    def __init__(self, config: PromptConfig, seed: int = 42):
        self.config = config
        self.random = random.Random(seed)
        self.task = Path(config.task_file).read_text(encoding="utf-8")

    @staticmethod
    def _metrics(program: Program) -> str:
        return json.dumps(program.metrics, sort_keys=True) if program.metrics else "{}"

    def _artifacts(self, program: Program) -> str:
        if not program.artifacts:
            return "(none)"
        rendered = json.dumps(program.artifacts, indent=2, sort_keys=True)
        return rendered[-self.config.artifact_chars :]

    def _program_section(self, heading: str, program: Program, *, include_artifacts: bool) -> str:
        parts = [
            f"## {heading}",
            f"id: {program.id}",
            f"generation: {program.generation}",
            f"metrics: {self._metrics(program)}",
            "```python",
            program.code,
            "```",
        ]
        if include_artifacts:
            parts.extend(["Evaluation feedback:", "```text", self._artifacts(program), "```"])
        return "\n".join(parts)

    def build(
        self,
        parent: Program,
        inspirations: list[Program],
        failed_attempts: list[Program],
        iteration: int,
        mode: str,
    ) -> tuple[str, str]:
        sections = [
            "# Research task",
            self.task,
            "",
            "# Evolution context",
            f"iteration: {iteration}",
            f"sampling mode: {mode}",
            self.random.choice(SEARCH_STYLES),
            "",
            self._program_section("Current parent", parent, include_artifacts=True),
        ]

        if inspirations:
            sections.extend(["", "# Prior programs selected as inspirations"])
            for index, program in enumerate(inspirations, 1):
                candidate = self._program_section(
                    f"Inspiration {index}", program, include_artifacts=False
                )
                projected = "\n".join(sections + [candidate])
                if len(projected) > self.config.max_prompt_chars:
                    break
                sections.append(candidate)

        if failed_attempts:
            sections.extend(["", "# Recent failed children of this parent"])
            for attempt in failed_attempts:
                failure = {
                    "status": attempt.status,
                    "error": attempt.error,
                    "artifacts": attempt.artifacts,
                }
                sections.append(json.dumps(failure, sort_keys=True)[-self.config.artifact_chars :])

        sections.extend(
            [
                "",
                "# Task",
                "Propose one new child program. Explain the hypothesis briefly, then provide only valid "
                "SEARCH/REPLACE blocks. The evaluator decides whether the idea survives.",
            ]
        )
        user = "\n".join(sections)
        if len(user) > self.config.max_prompt_chars:
            raise ValueError(
                "Parent program and task context exceed prompt.max_prompt_chars; increase that limit"
            )
        return SYSTEM_MESSAGE, user

