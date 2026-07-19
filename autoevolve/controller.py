from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from pathlib import Path

from .config import AppConfig
from .database import ProgramDatabase
from .evaluator import Evaluator
from .llm import BudgetExceeded
from .novelty import code_similarity
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
        self._budget_reason: str | None = None
        if self.generator is not None:
            seed_usage = getattr(self.generator, "seed_usage", None)
            if callable(seed_usage):
                seed_usage(**self.database.usage_summary())
            seed_outcomes = getattr(self.generator, "seed_outcomes", None)
            if callable(seed_outcomes):
                seed_outcomes(self.database.model_outcomes())

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:10]

    @property
    def stop_reason(self) -> str | None:
        return self._budget_reason

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
        loop = asyncio.get_running_loop()
        deadline = (
            None
            if self.config.run.max_wall_seconds is None
            else loop.time() + self.config.run.max_wall_seconds
        )
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
                if deadline is not None and loop.time() >= deadline:
                    await self._stop_for_budget(
                        f"Wall-clock budget exhausted ({self.config.run.max_wall_seconds}s)"
                    )
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

    async def _evolve_once(self, iteration: int, target: float | None) -> Program | None:
        pending, sample = await self._prepare_proposal(iteration)
        if self.generator is None:
            raise RuntimeError("No automatic model generator is configured")
        program_id = self._new_id()
        try:
            generation = await self.generator.generate(pending.system, pending.user)
        except BudgetExceeded as exc:
            await self._stop_for_budget(str(exc))
            return None
        except Exception as exc:
            program = Program(
                id=program_id,
                code=sample.parent.code,
                parent_id=sample.parent.id,
                inspiration_ids=[item.id for item in sample.inspirations],
                island=sample.island,
                generation=sample.parent.generation + 1,
                status="generation_error",
                operator=pending.operator,
                prompt=pending.user,
                error=str(exc),
                reward=-0.25,
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
            memory = self.database.research_memory(
                self.config.prompt.memory_interval, self.config.prompt.memory_items
            )
        operator = self.prompt_builder.choose_operator(bool(sample.inspirations))
        system, user = self.prompt_builder.build(
            sample.parent,
            sample.inspirations,
            failures,
            iteration,
            sample.mode,
            operator,
            memory,
        )
        pending = PendingProposal(
            iteration=iteration,
            parent_id=sample.parent.id,
            inspiration_ids=[item.id for item in sample.inspirations],
            island=sample.island,
            mode=sample.mode,
            operator=operator,
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
        generations = [generation]
        while True:
            candidate = await self._build_candidate(
                pending, sample, generations[-1], generations, program_id
            )
            if candidate.status == "pending":
                result: EvaluationResult = await self.evaluator.evaluate(candidate)
                candidate.status = result.status
                candidate.metrics = result.metrics
                candidate.artifacts = result.artifacts
                candidate.error = result.error
                candidate.reward = self._reward(candidate, sample.parent)
                return await self._record(candidate, pending.iteration, target)

            repairable = candidate.status in {"patch_error", "duplicate", "not_novel"}
            retries_used = len(generations) - 1
            if (
                not repairable
                or self.generator is None
                or retries_used >= self.config.prompt.proposal_retries
            ):
                candidate.reward = self._reward(candidate, sample.parent)
                return await self._record(candidate, pending.iteration, target)

            async with self._database_lock:
                self.database.log_event(
                    "proposal_retry",
                    {
                        "iteration": pending.iteration,
                        "attempt": len(generations),
                        "status": candidate.status,
                        "error": candidate.error,
                        "model": generation.model,
                        "operator": pending.operator,
                    },
                )
            system, user = self.prompt_builder.repair(
                pending.user,
                generations[-1].text,
                candidate.error or candidate.status,
                pending.operator,
                len(generations) + 1,
            )
            try:
                generations.append(await self._generate_repair(system, user, generation.model))
            except BudgetExceeded as exc:
                await self._stop_for_budget(str(exc))
                candidate.reward = self._reward(candidate, sample.parent)
                return await self._record(candidate, pending.iteration, target)
            except Exception as exc:
                candidate.status = "generation_error"
                candidate.error = f"Proposal repair failed: {exc}"
                candidate.reward = self._reward(candidate, sample.parent)
                return await self._record(candidate, pending.iteration, target)

    async def _generate_repair(self, system: str, user: str, model: str) -> Generation:
        if self.generator is None:
            raise RuntimeError("No automatic model generator is configured")
        generate_with_model = getattr(self.generator, "generate_with_model", None)
        if callable(generate_with_model):
            return await generate_with_model(system, user, model)
        return await self.generator.generate(system, user)

    @staticmethod
    def _combined_response(generations: list[Generation]) -> str:
        if len(generations) == 1:
            return generations[0].text
        return "\n\n".join(
            f"## Proposal attempt {index}\n{item.text}"
            for index, item in enumerate(generations, 1)
        )

    async def _build_candidate(
        self,
        pending: PendingProposal,
        sample: Sample,
        generation: Generation,
        generations: list[Generation],
        program_id: str,
    ) -> Program:
        common = {
            "id": program_id,
            "parent_id": sample.parent.id,
            "inspiration_ids": [item.id for item in sample.inspirations],
            "island": sample.island,
            "generation": sample.parent.generation + 1,
            "model": generation.model,
            "operator": pending.operator,
            "attempts": len(generations),
            "input_tokens": sum(item.input_tokens for item in generations),
            "output_tokens": sum(item.output_tokens for item in generations),
            "cost_usd": sum(item.cost_usd for item in generations),
            "prompt": pending.user,
            "response": self._combined_response(generations),
        }
        try:
            child_code = apply_patch(sample.parent.code, generation.text)
        except PatchError as exc:
            return Program(
                code=sample.parent.code,
                status="patch_error",
                error=str(exc),
                **common,
            )

        candidate = Program(
            code=child_code,
            status="pending",
            **common,
        )

        async with self._database_lock:
            duplicate = self.database.find_success_by_hash(candidate.code_hash)
            archive = self.database.successful_programs()
        if duplicate is not None:
            candidate.status = "duplicate"
            candidate.error = f"Program is identical to successful candidate {duplicate.id}"
            return candidate

        if archive:
            similarities = [
                (
                    code_similarity(
                        candidate.code, program.code, self.config.prompt.novelty_shingle_size
                    ),
                    program,
                )
                for program in archive
            ]
            similarity, nearest = max(similarities, key=lambda item: item[0])
            candidate.novelty = 1.0 - similarity
            threshold = self.config.prompt.novelty_threshold
            if threshold > 0 and similarity >= threshold:
                candidate.status = "not_novel"
                candidate.error = (
                    f"Novelty gate: similarity {similarity:.6f} to archived program {nearest.id} "
                    f"meets threshold {threshold:.6f}"
                )
        else:
            candidate.novelty = 1.0
        return candidate

    def _reward(self, program: Program, parent: Program) -> float:
        if program.status != "success":
            return -0.10 if program.status in {"patch_error", "duplicate", "not_novel"} else -0.25
        parent_value = parent.metrics.get(self.config.evaluator.objective)
        child_value = program.metrics.get(self.config.evaluator.objective)
        if parent_value is None or child_value is None:
            return 0.0
        improvement = (
            parent_value - child_value
            if self.config.evaluator.direction == "minimize"
            else child_value - parent_value
        )
        return max(-1.0, min(1.0, improvement / max(abs(parent_value), 1e-12)))

    async def _stop_for_budget(self, reason: str) -> None:
        async with self._database_lock:
            if self._budget_reason is None:
                self._budget_reason = reason
                self.database.log_event("budget_stop", {"reason": reason})
        self._stop.set()

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
                    "operator": program.operator,
                    "attempts": program.attempts,
                    "reward": program.reward,
                    "cost_usd": program.cost_usd,
                },
            )
            if best is not None:
                self._export_best(best)
            if target is not None and best is not None and self._target_reached(best, target):
                self._stop.set()

        if self.on_result is not None:
            self.on_result(program, best, completed)
        record_outcome = None if self.generator is None else getattr(
            self.generator, "record_outcome", None
        )
        if callable(record_outcome) and program.model is not None:
            record_outcome(program.model, program.reward, program.cost_usd)
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
                    "operator": best.operator,
                    "reward": best.reward,
                    "novelty": best.novelty,
                    "attempts": best.attempts,
                    "input_tokens": best.input_tokens,
                    "output_tokens": best.output_tokens,
                    "cost_usd": best.cost_usd,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
