import tempfile
import unittest
from pathlib import Path

from autoevolve.config import DatabaseConfig
from autoevolve.database import ProgramDatabase
from autoevolve.types import FeatureSpec, Program


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        config = DatabaseConfig(
            path=Path(self.temp.name) / "evolution.db",
            population_size=4,
            num_islands=2,
            migration_interval=2,
            migration_count=1,
            exploitation_ratio=1.0,
            exploration_ratio=0.0,
            features=[FeatureSpec("size", 0, 100, 4)],
        )
        self.db = ProgramDatabase(config.path, config, "score", "maximize", seed=7)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def _program(self, program_id, score, size, island=0, parent_id=None):
        return Program(
            id=program_id,
            code=f"value = {score}\n",
            parent_id=parent_id,
            island=island,
            generation=0 if parent_id is None else 1,
            status="baseline" if parent_id is None else "success",
            metrics={"score": score, "size": size},
        )

    def test_map_elites_sampling_and_best(self):
        baseline = self._program("baseline", 1.0, 10)
        self.db.add_program(baseline, memberships=[0, 1])
        better_same_cell = self._program("better", 2.0, 12, parent_id="baseline")
        diverse = self._program("diverse", 1.5, 80, parent_id="baseline")
        self.db.add_program(better_same_cell)
        self.db.add_program(diverse)

        elites = self.db.cell_elites(0)
        self.assertEqual({program.id for program in elites.values()}, {"better", "diverse"})
        self.assertEqual(self.db.best().id, "better")
        sample = self.db.sample(num_inspirations=1)
        self.assertIn(sample.parent.id, {"better", "diverse"})
        self.assertEqual(sample.mode, "exploit")

    def test_ring_migration_moves_the_island_best(self):
        baseline = self._program("baseline", 1.0, 10)
        self.db.add_program(baseline, memberships=[0, 1])
        child = self._program("winner", 3.0, 50, island=0, parent_id="baseline")
        self.db.add_program(child)

        self.assertNotIn("winner", self.db.membership_ids(1))
        self.assertTrue(self.db.maybe_migrate(2))
        self.assertIn("winner", self.db.membership_ids(1))

    def test_failures_are_retained_but_not_sampled(self):
        baseline = self._program("baseline", 1.0, 10)
        self.db.add_program(baseline, memberships=[0, 1])
        failure = Program(
            id="failed",
            code="broken =",
            parent_id="baseline",
            island=0,
            generation=1,
            status="syntax_error",
            error="invalid syntax",
        )
        self.db.add_program(failure)
        self.assertEqual(self.db.recent_failures("baseline", 2)[0].id, "failed")
        self.assertNotIn("failed", self.db.membership_ids(0))


if __name__ == "__main__":
    unittest.main()

