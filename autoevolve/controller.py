from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from pathlib import Path

from .config import AppConfig
from .database import ProgramDatabase
from .evaluator import Evaluator
from .patching import PatchError, apply_patch
from .prompt import PromptBuilder
from .types import EvaluationResult, Generation, Generator, PendingProposal, Program, Sample


ProgressCallback = Callable[[Program, Program | None, int], None]


class EvolutionController:
    """Asynchronous AlphaEvolve loop specialized to a single autoresearch file."""

    def __init__(
        self,
        config: AppConfig,
        database: ProgramDatabase,
        generator: Generator | None,
        evaluator: Evaluator,
        prompt_builder: PromptBuilder,
        on_result: ProgressCallback | None = None,
    ):
        self.config = config
        self.database = database
        self.generator = generator
        self.evaluator = evaluator
        self.prompt_builder = prompt_builder
        self.on_result = on_result
        self._database_lock = asyncio.Lock()
        self._stop = asyncio.Event()

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:10]

    @staticmethod
    def _novelty(parent: Program, child_code: str) -> float:
        parent_lines = {line.strip() for line in parent.code.splitlines() if line.strip()}
        child_lines = {line.strip() for line in child_code.splitlines() if line.strip()}
        union = parent_lines | child_lines
        return 0.0 if not union else 1.0 - len(parent_lines & child_lines) / len(union)

    async def ensure_baseline(self) -> Program:
        async with self._database_lock:
            if self.database.has_baseline():
                baseline = next(
                    program
                    for program in self.database.successful_programs()
                    if program.status == "baseline"
                )
                return baseline

        code = self.config.evaluator.program_file.read_text(encoding="utf-8")
        baseline = Program(id=self._new_id(), code=code, status="pending", island=0)
        result = await self.evaluator.evaluate(baseline)
        baseline.status = "baseline" if result.status == "success" else result.status
        baseline.metrics = result.metrics
        baseline.artifacts = result.artifacts
        baseline.error = result.error

        async with self._database_lock:
            memberships = list(range(self.config.database.num_islands)) if baseline.successful else []
            self.database.add_program(baseline, memberships=memberships)
            self.database.log_event(
                "baseline",
                {"id": baseline.id, "status": baseline.status, "metrics": baseline.metrics},
            )
        if not baseline.successful:
            raise RuntimeError(f"Baseline evaluation failed: {baseline.error}")
        self._export_best(baseline)
        return baseline

    async def run(self, iterations: int | None = None, target: float | None = None) -> Program:
        await self.ensure_baseline()
        total = self.config.run.iterations if iterations is None else max(0, iterations)
        if total == 0:
            best = self.database.best()
            if best is None:
                raise RuntimeError("No successful program exists")
            return best

        job_lock = asyncio.Lock()
        next_job = 0
        starting_iteration = self.database.completed_iterations

        async def reserve_job() -> int | None:
            nonlocal next_job
            async with job_lock:
                if self._stop.is_set() or next_job >= total:
                    return None
                next_job += 1
                return starting_iteration + next_job

        async def worker() -> None:
            while True:
                iteration = await reserve_job()
                if iteration is None:
                    return
                await self._evolve_once(iteration, target)

        worker_count = min(self.config.run.workers, total)
        await asyncio.gather(*(worker() for _ in range(worker_count)))
        best = self.database.best()
        if best is None:
            raise RuntimeError("Evolution completed without a successful program")
        self._export_best(best)
        return best

    async def _evolve_once(self, iteration: int, target: float | None) -> Program:
        pending, sample = await self._prepare_proposal(iteration)
        if self.generator is None:
            raise RuntimeError("No automatic model generator is configured")
        program_id = self._new_id()
        try:
            generation = await self.generator.generate(pending.system, pending.user)
        except Exception as exc:
            program = Program(
                id=program_id,
                code=sample.parent.code,
                parent_id=sample.parent.id,
                inspiration_ids=[item.id for item in sample.inspirations],
                island=sample.island,
                generation=sample.parent.generation + 1,
                status="generation_error",
                prompt=pending.user,
                error=str(exc),
            )
            return await self._record(program, iteration, target)
        return await self._evaluate_generation(pending, sample, generation, target, program_id)

    async def prepare_external_proposal(self) -> PendingProposal:
        await self.ensure_baseline()
        iteration = self.database.completed_iterations + 1
        pending, _sample = await self._prepare_proposal(iteration)
        return pending

    async def submit_external_proposal(
        self, pending: PendingProposal, response: str, model: str = "external-agent"
    ) -> Program:
        async with self._database_lock:
            parent = self.database.get(pending.parent_id)
            inspirations = [
                program
                for program_id in pending.inspiration_ids
                if (program := self.database.get(program_id)) is not None
            ]
        if parent is None or not parent.successful:
            raise RuntimeError(f"Pending parent {pending.parent_id!r} is missing or unsuccessful")
        sample = Sample(
            parent=parent,
            inspirations=inspirations,
            island=pending.island,
            mode=pending.mode,
        )
        generation = Generation(text=response, model=model)
        return await self._evaluate_generation(
            pending, sample, generation, target=None, program_id=self._new_id()
        )

    async def _prepare_proposal(self, iteration: int) -> tuple[PendingProposal, Sample]:
        async with self._database_lock:
            sample = self.database.sample(self.config.prompt.num_inspirations)
            failures = self.database.recent_failures(
                sample.parent.id, self.config.prompt.failed_attempts
            )
        system, user = self.prompt_builder.build(
            sample.parent,
            sample.inspirations,
            failures,
            iteration,
            sample.mode,
        )
        pending = PendingProposal(
            iteration=iteration,
            parent_id=sample.parent.id,
            inspiration_ids=[item.id for item in sample.inspirations],
            island=sample.island,
            mode=sample.mode,
            system=system,
            user=user,
        )
        return pending, sample

    async def _evaluate_generation(
        self,
        pending: PendingProposal,
        sample: Sample,
        generation: Generation,
        target: float | None,
        program_id: str,
    ) -> Program:
        try:
            child_code = apply_patch(sample.parent.code, generation.text)
        except PatchError as exc:
            program = Program(
                id=program_id,
                code=sample.parent.code,
                parent_id=sample.parent.id,
                inspiration_ids=[item.id for item in sample.inspirations],
                island=sample.island,
                generation=sample.parent.generation + 1,
                status="patch_error",
                model=generation.model,
                prompt=pending.user,
                response=generation.text,
                error=str(exc),
            )
            return await self._record(program, pending.iteration, target)

        candidate = Program(
            id=program_id,
            code=child_code,
            parent_id=sample.parent.id,
            inspiration_ids=[item.id for item in sample.inspirations],
            island=sample.island,
            generation=sample.parent.generation + 1,
            status="pending",
            model=generation.model,
            prompt=pending.user,
            response=generation.text,
            novelty=self._novelty(sample.parent, child_code),
        )

        async with self._database_lock:
            duplicate = self.database.find_success_by_hash(candidate.code_hash)
        if duplicate is not None:
            candidate.status = "duplicate"
            candidate.error = f"Program is identical to successful candidate {duplicate.id}"
            return await self._record(candidate, pending.iteration, target)

        result: EvaluationResult = await self.evaluator.evaluate(candidate)
        candidate.status = result.status
        candidate.metrics = result.metrics
        candidate.artifacts = result.artifacts
        candidate.error = result.error
        return await self._record(candidate, pending.iteration, target)

    async def _record(self, program: Program, iteration: int, target: float | None) -> Program:
        async with self._database_lock:
            self.database.add_program(program)
            completed = self.database.complete_iteration()
            self.database.maybe_migrate(completed)
            best = self.database.best()
            self.database.log_event(
                "evaluation",
                {
                    "iteration": iteration,
                    "completed": completed,
                    "id": program.id,
                    "parent_id": program.parent_id,
                    "island": program.island,
                    "status": program.status,
                    "metrics": program.metrics,
                    "model": program.model,
                },
            )
            if best is not None:
                self._export_best(best)
            if target is not None and best is not None and self._target_reached(best, target):
                self._stop.set()

        if self.on_result is not None:
            self.on_result(program, best, completed)
        return program

    def _target_reached(self, best: Program, target: float) -> bool:
        value = best.metrics.get(self.config.evaluator.objective)
        if value is None:
            return False
        if self.config.evaluator.direction == "minimize":
            return value <= target
        return value >= target

    def _export_best(self, best: Program) -> None:
        best_dir = self.config.run.run_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        program_path = best_dir / self.config.evaluator.program_file.name
        program_path.write_text(best.code, encoding="utf-8", newline="\n")
        (best_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": best.id,
                    "parent_id": best.parent_id,
                    "generation": best.generation,
                    "metrics": best.metrics,
                    "model": best.model,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
