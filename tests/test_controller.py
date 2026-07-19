import tempfile
import unittest
from pathlib import Path

from autoevolve.config import (
    AppConfig,
    DatabaseConfig,
    EvaluatorConfig,
    LLMConfig,
    PromptConfig,
    RunConfig,
    StageConfig,
)
from autoevolve.controller import EvolutionController
from autoevolve.database import ProgramDatabase
from autoevolve.evaluator import Evaluator
from autoevolve.prompt import PromptBuilder
from autoevolve.types import EvaluationResult, Generation, Program


class FakeGenerator:
    async def generate(self, system, user):
        self.system = system
        self.user = user
        return Generation(
            model="fake-model",
            text="""Lower the synthetic objective.
<<<<<<< SEARCH
score = 1.0
=======
score = 0.5
>>>>>>> REPLACE
""",
        )


class RepairingGenerator:
    def __init__(self):
        self.calls = 0

    async def generate(self, system, user):
        self.calls += 1
        if self.calls == 1:
            return Generation(model="repair-model", text="This response forgot the patch blocks.")
        return Generation(
            model="repair-model",
            text=(
                "<<<<<<< SEARCH\nscore = 1.0\n=======\nscore = 0.25\n>>>>>>> REPLACE"
            ),
        )


class CountingEvaluator:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, program):
        self.calls += 1
        return EvaluationResult("success", {"val_bpb": 1.0}, {})


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_evolution_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train.py"
            train.write_text(
                "# EVOLVE-BLOCK-START\n"
                "score = 1.0\n"
                "print(f\"val_bpb: {score}\")\n"
                "print(\"size: 10\")\n"
                "# EVOLVE-BLOCK-END\n",
                encoding="utf-8",
            )
            task = root / "program.md"
            task.write_text("Minimize val_bpb.", encoding="utf-8")
            run_dir = root / "run"
            db_config = DatabaseConfig(
                path=run_dir / "evolution.db",
                population_size=8,
                num_islands=2,
                migration_interval=0,
                exploitation_ratio=1.0,
                exploration_ratio=0.0,
                features=[],
            )
            evaluator_config = EvaluatorConfig(
                program_file=train,
                fixed_files=[],
                objective="val_bpb",
                direction="minimize",
                concurrency=1,
                stages=[
                    StageConfig(
                        name="tiny",
                        command=[__import__("sys").executable, "{program}"],
                        timeout_seconds=5,
                        required_metrics=["val_bpb"],
                    )
                ],
            )
            config = AppConfig(
                project_dir=root,
                run=RunConfig(iterations=1, workers=1, seed=3, run_dir=run_dir),
                llm=LLMConfig(),
                prompt=PromptConfig(task_file=task, num_inspirations=1),
                evaluator=evaluator_config,
                database=db_config,
            )
            database = ProgramDatabase(
                db_config.path, db_config, "val_bpb", "minimize", seed=3
            )
            try:
                controller = EvolutionController(
                    config,
                    database,
                    FakeGenerator(),
                    Evaluator(evaluator_config, root, run_dir),
                    PromptBuilder(config.prompt, seed=3),
                )
                best = await controller.run()
                self.assertEqual(best.metrics["val_bpb"], 0.5)
                self.assertEqual(best.model, "fake-model")
                self.assertEqual(database.completed_iterations, 1)
                exported = run_dir / "best" / "train.py"
                self.assertIn("score = 0.5", exported.read_text(encoding="utf-8"))
                self.assertEqual(len(database.all_programs()), 2)
            finally:
                database.close()

    async def test_invalid_proposal_is_repaired_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train.py"
            train.write_text(
                "# EVOLVE-BLOCK-START\nscore = 1.0\n"
                'print(f"val_bpb: {score}")\n# EVOLVE-BLOCK-END\n',
                encoding="utf-8",
            )
            task = root / "program.md"
            task.write_text("Minimize val_bpb.", encoding="utf-8")
            run_dir = root / "run"
            db_config = DatabaseConfig(
                path=run_dir / "evolution.db",
                num_islands=1,
                migration_interval=0,
                exploitation_ratio=1.0,
                exploration_ratio=0.0,
            )
            evaluator_config = EvaluatorConfig(
                program_file=train,
                fixed_files=[],
                objective="val_bpb",
                direction="minimize",
                stages=[
                    StageConfig(
                        name="tiny",
                        command=[__import__("sys").executable, "{program}"],
                        timeout_seconds=5,
                        required_metrics=["val_bpb"],
                    )
                ],
            )
            config = AppConfig(
                project_dir=root,
                run=RunConfig(iterations=1, workers=1, run_dir=run_dir),
                llm=LLMConfig(),
                prompt=PromptConfig(task_file=task, proposal_retries=1),
                evaluator=evaluator_config,
                database=db_config,
            )
            database = ProgramDatabase(db_config.path, db_config, "val_bpb", "minimize")
            generator = RepairingGenerator()
            try:
                controller = EvolutionController(
                    config,
                    database,
                    generator,
                    Evaluator(evaluator_config, root, run_dir),
                    PromptBuilder(config.prompt),
                )
                best = await controller.run()
                child = database.get(best.id)
                self.assertEqual(generator.calls, 2)
                self.assertEqual(child.attempts, 2)
                self.assertEqual(child.metrics["val_bpb"], 0.25)
                self.assertEqual(
                    database.recent_events(5)[-2]["kind"], "proposal_retry"
                )
            finally:
                database.close()

    async def test_outer_agent_cosmetic_change_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train.py"
            source = (
                "# EVOLVE-BLOCK-START\nvalue = 1  # old note\n"
                'print(f"val_bpb: {value}")\n# EVOLVE-BLOCK-END\n'
            )
            train.write_text(source, encoding="utf-8")
            task = root / "program.md"
            task.write_text("Minimize val_bpb.", encoding="utf-8")
            run_dir = root / "run"
            db_config = DatabaseConfig(
                path=run_dir / "evolution.db",
                num_islands=1,
                migration_interval=0,
                exploitation_ratio=1.0,
                exploration_ratio=0.0,
            )
            evaluator_config = EvaluatorConfig(
                program_file=train,
                fixed_files=[],
                objective="val_bpb",
                direction="minimize",
            )
            config = AppConfig(
                project_dir=root,
                run=RunConfig(run_dir=run_dir),
                llm=LLMConfig(),
                prompt=PromptConfig(task_file=task, proposal_retries=2),
                evaluator=evaluator_config,
                database=db_config,
            )
            database = ProgramDatabase(db_config.path, db_config, "val_bpb", "minimize")
            baseline = Program(
                id="baseline",
                code=source,
                status="baseline",
                metrics={"val_bpb": 1.0},
            )
            database.add_program(baseline, memberships=[0])
            evaluator = CountingEvaluator()
            try:
                controller = EvolutionController(
                    config,
                    database,
                    None,
                    evaluator,
                    PromptBuilder(config.prompt),
                )
                pending = await controller.prepare_external_proposal()
                response = (
                    "<<<<<<< SEARCH\nvalue = 1  # old note\n=======\n"
                    "value = 1  # new note\n>>>>>>> REPLACE"
                )
                result = await controller.submit_external_proposal(pending, response, "codex")
                self.assertEqual(result.status, "not_novel")
                self.assertEqual(result.attempts, 1)
                self.assertEqual(evaluator.calls, 0)
            finally:
                database.close()


if __name__ == "__main__":
    unittest.main()
