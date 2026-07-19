import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from autoevolve.config import EvaluatorConfig, StageConfig
from autoevolve.evaluator import Evaluator, parse_metrics
from autoevolve.types import Program


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    async def asyncTearDown(self):
        self.temp.cleanup()

    def _config(self, stage):
        return EvaluatorConfig(
            program_file=self.root / "train.py",
            fixed_files=[],
            objective="val_bpb",
            direction="minimize",
            concurrency=1,
            stages=[stage],
        )

    async def test_executes_candidate_and_parses_metrics(self):
        stage = StageConfig(
            name="tiny",
            command=[sys.executable, "{program}"],
            timeout_seconds=5,
            required_metrics=["val_bpb", "peak_vram_mb"],
        )
        evaluator = Evaluator(self._config(stage), self.root, self.root / "run")
        program = Program(
            id="candidate",
            code='print("val_bpb: 0.750000")\nprint("peak_vram_mb: 12.5")\n',
        )
        result = await evaluator.evaluate(program)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.metrics["val_bpb"], 0.75)
        self.assertTrue(Path(result.artifacts["logs"][0]).exists())

    async def test_syntax_errors_do_not_spawn_a_process(self):
        stage = StageConfig(
            name="tiny", command=[sys.executable, "{program}"], timeout_seconds=5
        )
        evaluator = Evaluator(self._config(stage), self.root, self.root / "run")
        result = await evaluator.evaluate(Program(id="broken", code="value ="))
        self.assertEqual(result.status, "syntax_error")

    async def test_candidate_does_not_receive_llm_api_keys(self):
        stage = StageConfig(
            name="tiny",
            command=[sys.executable, "{program}"],
            timeout_seconds=5,
            required_metrics=["val_bpb", "credential_present", "visible_gpu"],
        )
        evaluator = Evaluator(self._config(stage), self.root, self.root / "run")
        code = (
            "import os\n"
            "print('val_bpb: 1.0')\n"
            "print(f\"credential_present: {int('OPENAI_API_KEY' in os.environ)}\")\n"
            "print(f\"visible_gpu: {os.environ['CUDA_VISIBLE_DEVICES']}\")\n"
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "should-not-leak"}):
            result = await evaluator.evaluate(Program(id="credentials", code=code))
        self.assertEqual(result.status, "success")
        self.assertEqual(result.metrics["credential_present"], 0.0)
        self.assertEqual(result.metrics["visible_gpu"], 0.0)

    def test_metric_parser_ignores_training_noise(self):
        output = "step 1 loss: 2.0\nval_bpb: 1.25e-1\nnum_steps: 42\n"
        self.assertEqual(parse_metrics(output), {"val_bpb": 0.125, "num_steps": 42.0})


if __name__ == "__main__":
    unittest.main()
