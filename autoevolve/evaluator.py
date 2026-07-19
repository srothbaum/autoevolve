from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from .config import EvaluatorConfig, StageConfig
from .types import EvaluationResult, Program


METRIC_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def parse_metrics(output: str) -> dict[str, float]:
    return {name: float(value) for name, value in METRIC_RE.findall(output)}


class Evaluator:
    """Static validation plus a configurable cascade of subprocess evaluators."""

    def __init__(self, config: EvaluatorConfig, project_dir: Path, run_dir: Path):
        self.config = config
        self.project_dir = project_dir.resolve()
        self.run_dir = run_dir.resolve()
        self.artifact_dir = self.run_dir / "artifacts"
        self.work_dir = self.run_dir / "work"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def evaluate(self, program: Program) -> EvaluationResult:
        try:
            compile(program.code, str(self.config.program_file), "exec")
        except SyntaxError as exc:
            return EvaluationResult(
                status="syntax_error",
                metrics={},
                artifacts={},
                error=f"{exc.msg} at line {exc.lineno}",
            )

        async with self._semaphore:
            return await self._evaluate_stages(program)

    async def _evaluate_stages(self, program: Program) -> EvaluationResult:
        metrics: dict[str, float] = {}
        log_paths: list[str] = []
        output_tails: dict[str, str] = {}

        with tempfile.TemporaryDirectory(prefix=f"{program.id}-", dir=self.work_dir) as temp_dir:
            candidate_dir = Path(temp_dir)
            candidate_path = candidate_dir / self.config.program_file.name
            candidate_path.write_text(program.code, encoding="utf-8", newline="\n")
            for fixed_file in self.config.fixed_files:
                if not fixed_file.exists():
                    return EvaluationResult(
                        status="evaluator_error",
                        metrics=metrics,
                        artifacts={"logs": log_paths},
                        error=f"Fixed evaluator file not found: {fixed_file}",
                    )
                shutil.copy2(fixed_file, candidate_dir / fixed_file.name)

            for stage in self.config.stages:
                try:
                    result = await self._run_stage(stage, program.id, candidate_dir, candidate_path)
                except (OSError, ValueError, KeyError) as exc:
                    return EvaluationResult(
                        status="evaluator_error",
                        metrics=metrics,
                        artifacts={"logs": log_paths, "output_tails": output_tails},
                        error=f"Could not start stage {stage.name!r}: {exc}",
                    )
                log_paths.append(str(result["log_path"]))
                output_tails[stage.name] = result["tail"]
                metrics.update(result["metrics"])
                artifacts = {"logs": log_paths, "output_tails": output_tails}

                if result["timed_out"]:
                    return EvaluationResult(
                        status="timeout",
                        metrics=metrics,
                        artifacts=artifacts,
                        error=f"Stage {stage.name!r} exceeded {stage.timeout_seconds}s",
                    )
                if result["returncode"] != 0:
                    return EvaluationResult(
                        status="crash",
                        metrics=metrics,
                        artifacts=artifacts,
                        error=f"Stage {stage.name!r} exited with code {result['returncode']}",
                    )
                missing = [name for name in stage.required_metrics if name not in metrics]
                if missing:
                    return EvaluationResult(
                        status="evaluator_error",
                        metrics=metrics,
                        artifacts=artifacts,
                        error=f"Stage {stage.name!r} omitted required metrics: {', '.join(missing)}",
                    )
                if not self._passes_threshold(stage, metrics):
                    return EvaluationResult(
                        status="rejected",
                        metrics=metrics,
                        artifacts=artifacts,
                        error=f"Stage {stage.name!r} did not pass its cascade threshold",
                    )

        if self.config.objective not in metrics:
            return EvaluationResult(
                status="evaluator_error",
                metrics=metrics,
                artifacts={"logs": log_paths, "output_tails": output_tails},
                error=f"Evaluation omitted objective metric {self.config.objective!r}",
            )
        return EvaluationResult(
            status="success",
            metrics=metrics,
            artifacts={"logs": log_paths, "output_tails": output_tails},
        )

    def _passes_threshold(self, stage: StageConfig, metrics: dict[str, float]) -> bool:
        if stage.threshold_metric is None or stage.threshold is None:
            return True
        if stage.threshold_metric not in metrics:
            return False
        value = metrics[stage.threshold_metric]
        if stage.threshold_direction == "minimize":
            return value <= stage.threshold
        return value >= stage.threshold

    async def _run_stage(
        self,
        stage: StageConfig,
        program_id: str,
        candidate_dir: Path,
        candidate_path: Path,
    ) -> dict[str, object]:
        replacements = {
            "project_dir": str(self.project_dir),
            "candidate_dir": str(candidate_dir),
            "program": str(candidate_path),
            "run_dir": str(self.run_dir),
        }
        command = [part.format(**replacements) for part in stage.command]
        log_path = self.artifact_dir / f"{program_id}-{stage.name}.log"
        environment = os.environ.copy()
        for name in self.config.strip_env:
            environment.pop(name, None)
        environment["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        creationflags = 0
        process_kwargs: dict[str, object] = {}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_kwargs["start_new_session"] = True

        timed_out = False
        with log_path.open("wb") as log_file:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=candidate_dir,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                env=environment,
                creationflags=creationflags,
                **process_kwargs,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=stage.timeout_seconds)
            except asyncio.TimeoutError:
                timed_out = True
                await self._terminate_process(process)

        output = log_path.read_text(encoding="utf-8", errors="replace")
        return {
            "returncode": process.returncode if process.returncode is not None else -1,
            "timed_out": timed_out,
            "metrics": parse_metrics(output),
            "tail": output[-12000:],
            "log_path": log_path,
        }

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=5)
