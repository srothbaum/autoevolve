from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import AppConfig, load_config
from .controller import EvolutionController
from .database import ProgramDatabase
from .evaluator import Evaluator
from .llm import ModelEnsemble
from .prompt import PromptBuilder
from .types import PendingProposal, Program


def _database(config: AppConfig) -> ProgramDatabase:
    return ProgramDatabase(
        config.database.path,
        config.database,
        config.evaluator.objective,
        config.evaluator.direction,
        config.run.seed,
    )


def _metric_text(program: Program | None, config: AppConfig) -> str:
    if program is None:
        return "n/a"
    value = program.metrics.get(config.evaluator.objective)
    return "n/a" if value is None else f"{value:.6f}"


def _progress(config: AppConfig):
    def report(program: Program, best: Program | None, completed: int) -> None:
        own = _metric_text(program, config)
        best_text = _metric_text(best, config)
        print(
            f"[{completed:04d}] {program.status:>16} island={program.island} "
            f"id={program.id} {config.evaluator.objective}={own} best={best_text}",
            flush=True,
        )

    return report


async def _run(config: AppConfig, iterations: int | None, target: float | None) -> int:
    database = _database(config)
    try:
        controller = EvolutionController(
            config=config,
            database=database,
            generator=ModelEnsemble(config.llm, config.run.seed, config.project_dir),
            evaluator=Evaluator(config.evaluator, config.project_dir, config.run.run_dir),
            prompt_builder=PromptBuilder(config.prompt, config.run.seed),
            on_result=_progress(config),
        )
        best = await controller.run(iterations=iterations, target=target)
        print(
            f"Best program: {best.id} | {config.evaluator.objective}="
            f"{_metric_text(best, config)} | exported to "
            f"{config.run.run_dir / 'best' / config.evaluator.program_file.name}"
        )
        return 0
    finally:
        database.close()


def _external_controller(config: AppConfig, database: ProgramDatabase) -> EvolutionController:
    return EvolutionController(
        config=config,
        database=database,
        generator=None,
        evaluator=Evaluator(config.evaluator, config.project_dir, config.run.run_dir),
        prompt_builder=PromptBuilder(config.prompt, config.run.seed),
        on_result=_progress(config),
    )


async def _sample_external(config: AppConfig) -> int:
    database = _database(config)
    try:
        pending = await _external_controller(config, database).prepare_external_proposal()
        config.run.run_dir.mkdir(parents=True, exist_ok=True)
        state_path = config.run.run_dir / "pending.json"
        prompt_path = config.run.run_dir / "pending_prompt.md"
        state_path.write_text(
            json.dumps(pending.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prompt_path.write_text(
            f"# System instructions\n\n{pending.system}\n\n# Evolution prompt\n\n{pending.user}\n",
            encoding="utf-8",
        )
        print(f"Proposal {pending.iteration} is ready: {prompt_path}")
        print("Write SEARCH/REPLACE output to a file, then run evolve.py submit --response <file>.")
        return 0
    finally:
        database.close()


async def _submit_external(config: AppConfig, response_path: Path, model_label: str) -> int:
    state_path = config.run.run_dir / "pending.json"
    prompt_path = config.run.run_dir / "pending_prompt.md"
    if not state_path.exists():
        raise RuntimeError("No pending proposal exists; run evolve.py sample first")
    pending = PendingProposal.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
    response = response_path.read_text(encoding="utf-8")
    database = _database(config)
    try:
        if pending.iteration != database.completed_iterations + 1:
            raise RuntimeError(
                "Pending proposal is stale because the database advanced; generate a new sample"
            )
        program = await _external_controller(config, database).submit_external_proposal(
            pending, response, model_label
        )
        state_path.unlink(missing_ok=True)
        prompt_path.unlink(missing_ok=True)
        print(
            f"Recorded {program.id}: status={program.status}, "
            f"{config.evaluator.objective}={_metric_text(program, config)}"
        )
        return 0 if program.status == "success" else 2
    finally:
        database.close()


def _doctor(config: AppConfig, skip_llm: bool = False) -> int:
    errors: list[str] = []
    if not config.evaluator.program_file.exists():
        errors.append(f"Program file does not exist: {config.evaluator.program_file}")
    else:
        try:
            compile(
                config.evaluator.program_file.read_text(encoding="utf-8"),
                str(config.evaluator.program_file),
                "exec",
            )
        except SyntaxError as exc:
            errors.append(f"Program file does not compile: {exc}")
    if not config.prompt.task_file.exists():
        errors.append(f"Task file does not exist: {config.prompt.task_file}")
    for fixed_file in config.evaluator.fixed_files:
        if not fixed_file.exists():
            errors.append(f"Fixed evaluator file does not exist: {fixed_file}")
    first_command = config.evaluator.stages[0].command[0]
    if "{" not in first_command and shutil.which(first_command) is None:
        errors.append(f"Evaluator executable is not on PATH: {first_command}")
    for model in [] if skip_llm else config.llm.models:
        provider = model.resolved_provider()
        if provider == "openai_compatible":
            if config.llm.api_key_env and not os.environ.get(config.llm.api_key_env):
                errors.append(f"LLM API key is not set: {config.llm.api_key_env}")
            if not model.name and model.name_env and not os.environ.get(model.name_env):
                errors.append(f"LLM model is not set: {model.name_env}")
        elif provider == "codex_cli":
            executable = model.executable or "codex"
            if shutil.which(executable) is None:
                errors.append(f"Codex CLI is not on PATH: {executable}")
            else:
                error = _cli_version_error(executable, "Codex CLI")
                if error:
                    errors.append(error)
        elif provider == "claude_code":
            executable = model.executable or "claude"
            if shutil.which(executable) is None:
                errors.append(f"Claude Code is not on PATH: {executable}")
            else:
                error = _cli_version_error(executable, "Claude Code")
                if error:
                    errors.append(error)
        else:
            errors.append(f"Unsupported LLM provider: {provider}")

    if errors:
        print("Configuration problems:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Configuration looks ready.")
    return 0


def _cli_version_error(executable: str, label: str) -> str | None:
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{label} could not execute: {exc}"
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-500:]
        return f"{label} --version failed with code {result.returncode}: {detail}"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini AlphaEvolve controller for the autoresearch training program"
    )
    parser.add_argument("--config", default="evolve.json", help="Path to evolve.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Evaluate the baseline and evolve new programs")
    run.add_argument("--iterations", type=int, default=None, help="Additional child attempts")
    run.add_argument("--target", type=float, default=None, help="Stop after reaching this objective")

    subparsers.add_parser("sample", help="Create a proposal prompt for an outer coding agent")
    submit = subparsers.add_parser("submit", help="Evaluate an outer agent's patch response")
    submit.add_argument("--response", type=Path, required=True, help="SEARCH/REPLACE response file")
    submit.add_argument("--model-label", default="external-agent")
    subparsers.add_parser("status", help="Show archive, island, and run state")
    subparsers.add_parser("best", help="Show the best program metadata")
    events = subparsers.add_parser("events", help="Show recent controller events")
    events.add_argument("--limit", type=int, default=20)
    doctor = subparsers.add_parser("doctor", help="Check local files, commands, and LLM environment")
    doctor.add_argument(
        "--skip-llm", action="store_true", help="Check the outer-agent sample/submit workflow"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "run":
            return asyncio.run(_run(config, args.iterations, args.target))
        if args.command == "sample":
            return asyncio.run(_sample_external(config))
        if args.command == "submit":
            return asyncio.run(_submit_external(config, args.response, args.model_label))
        if args.command == "doctor":
            return _doctor(config, args.skip_llm)

        database = _database(config)
        try:
            if args.command == "status":
                print(json.dumps(database.status(), indent=2, sort_keys=True))
            elif args.command == "best":
                best = database.best()
                if best is None:
                    print("No successful program has been evaluated yet.")
                    return 1
                print(
                    json.dumps(
                        {
                            "id": best.id,
                            "parent_id": best.parent_id,
                            "generation": best.generation,
                            "metrics": best.metrics,
                            "model": best.model,
                            "export": str(
                                config.run.run_dir / "best" / config.evaluator.program_file.name
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            elif args.command == "events":
                print(json.dumps(database.recent_events(args.limit), indent=2, sort_keys=True))
        finally:
            database.close()
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed evaluations remain in the database.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
