import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from autoevolve.config import LLMConfig, ModelSpec
from autoevolve.llm import BudgetExceeded, ModelEnsemble


class FakeProcess:
    def __init__(self, stdout=b"generated patch", stderr=b"", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.input = None

    async def communicate(self, data):
        self.input = data
        return self.stdout, self.stderr

    def kill(self):
        self.returncode = -1

    async def wait(self):
        return self.returncode


class LLMProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_cli_uses_stdin_and_read_only_mode(self):
        process = FakeProcess()
        create = AsyncMock(return_value=process)
        config = LLMConfig(
            api_key_env=None,
            models=[
                ModelSpec(
                    provider="codex_cli",
                    provider_env=None,
                    name="",
                    name_env=None,
                )
            ],
        )
        with patch("autoevolve.llm.asyncio.create_subprocess_exec", create):
            result = await ModelEnsemble(config, cwd=Path.cwd()).generate("system", "user")

        command = create.await_args.args
        self.assertEqual(command[:5], ("codex", "exec", "--ephemeral", "--sandbox", "read-only"))
        self.assertEqual(command[-1], "-")
        self.assertEqual(process.input, b"system\n\nuser")
        self.assertEqual(result.text, "generated patch")

    async def test_claude_code_disables_tools_and_accepts_system_prompt(self):
        process = FakeProcess()
        create = AsyncMock(return_value=process)
        config = LLMConfig(
            api_key_env=None,
            models=[
                ModelSpec(
                    provider="claude_code",
                    provider_env=None,
                    name="sonnet",
                    name_env=None,
                )
            ],
        )
        with patch("autoevolve.llm.asyncio.create_subprocess_exec", create):
            result = await ModelEnsemble(config, cwd=Path.cwd()).generate("system", "user")

        command = create.await_args.args
        self.assertEqual(command[0], "claude")
        self.assertIn("--tools", command)
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertEqual(command[command.index("--append-system-prompt") + 1], "system")
        self.assertEqual(command[command.index("--model") + 1], "sonnet")
        self.assertEqual(process.input, b"user")
        self.assertEqual(result.model, "sonnet")

    async def test_call_budget_is_hard(self):
        process = FakeProcess()
        create = AsyncMock(return_value=process)
        config = LLMConfig(
            api_key_env=None,
            max_calls=1,
            models=[ModelSpec(provider="codex_cli", provider_env=None, name_env=None)],
        )
        ensemble = ModelEnsemble(config, cwd=Path.cwd())
        with patch("autoevolve.llm.asyncio.create_subprocess_exec", create):
            await ensemble.generate("system", "user")
            with self.assertRaisesRegex(BudgetExceeded, "call budget"):
                await ensemble.generate("system", "user")
        self.assertEqual(create.await_count, 1)

    def test_ucb_prefers_model_with_better_observed_reward(self):
        config = LLMConfig(
            selection_strategy="ucb",
            ucb_exploration=0.0,
            models=[
                ModelSpec(provider="codex_cli", provider_env=None, name="strong", name_env=None),
                ModelSpec(provider="codex_cli", provider_env=None, name="weak", name_env=None),
            ],
        )
        ensemble = ModelEnsemble(config)
        ensemble.seed_outcomes([("strong", 0.5, 0.0), ("weak", -0.5, 0.0)])
        self.assertEqual(ensemble._choose_model().name, "strong")


if __name__ == "__main__":
    unittest.main()
