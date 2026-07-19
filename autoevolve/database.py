from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path

from .config import DatabaseConfig
from .types import Program, SUCCESS_STATUSES, Sample


PROGRAM_COLUMNS = (
    "id, code, code_hash, parent_id, inspiration_ids, island, generation, status, "
    "metrics, artifacts, model, prompt, response, error, novelty, created_at, sample_count"
)


class ProgramDatabase:
    """Durable append-only trials plus an island-local MAP-Elites population."""

    def __init__(
        self,
        path: Path,
        config: DatabaseConfig,
        objective: str,
        direction: str,
        seed: int = 42,
    ):
        self.path = path.resolve()
        self.config = config
        self.objective = objective
        self.direction = direction
        self.random = random.Random(seed)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                parent_id TEXT,
                inspiration_ids TEXT NOT NULL,
                island INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                status TEXT NOT NULL,
                metrics TEXT NOT NULL,
                artifacts TEXT NOT NULL,
                model TEXT,
                prompt TEXT,
                response TEXT,
                error TEXT,
                novelty REAL NOT NULL,
                created_at REAL NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_programs_status ON programs(status);
            CREATE INDEX IF NOT EXISTS idx_programs_hash ON programs(code_hash);
            CREATE INDEX IF NOT EXISTS idx_programs_parent ON programs(parent_id);

            CREATE TABLE IF NOT EXISTS memberships (
                program_id TEXT NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
                island INTEGER NOT NULL,
                source TEXT NOT NULL DEFAULT 'birth',
                PRIMARY KEY (program_id, island)
            );
            CREATE INDEX IF NOT EXISTS idx_memberships_island ON memberships(island);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ProgramDatabase":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _get_meta_int(self, key: str, default: int = 0) -> int:
        row = self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return default if row is None else int(row["value"])

    def _set_meta(self, key: str, value: str | int) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    @property
    def completed_iterations(self) -> int:
        return self._get_meta_int("completed_iterations")

    def complete_iteration(self) -> int:
        completed = self.completed_iterations + 1
        self._set_meta("completed_iterations", completed)
        self.connection.commit()
        return completed

    def log_event(self, kind: str, payload: dict[str, object]) -> None:
        self.connection.execute(
            "INSERT INTO events(created_at, kind, payload) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(payload, sort_keys=True)),
        )
        self.connection.commit()

    def recent_events(self, limit: int = 10) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT created_at, kind, payload FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {
                "created_at": row["created_at"],
                "kind": row["kind"],
                "payload": json.loads(row["payload"]),
            }
            for row in reversed(rows)
        ]

    def add_program(self, program: Program, memberships: list[int] | None = None) -> None:
        row = program.to_row()
        columns = [part.strip() for part in PROGRAM_COLUMNS.split(",")]
        placeholders = ", ".join("?" for _ in columns)
        self.connection.execute(
            f"INSERT INTO programs ({PROGRAM_COLUMNS}) VALUES ({placeholders})",
            tuple(row[column] for column in columns),
        )
        if program.successful:
            target_islands = memberships if memberships is not None else [program.island]
            for island in target_islands:
                self.connection.execute(
                    "INSERT OR IGNORE INTO memberships(program_id, island, source) VALUES (?, ?, ?)",
                    (program.id, island % self.config.num_islands, "birth"),
                )
        self.connection.commit()
        if program.successful:
            for island in memberships if memberships is not None else [program.island]:
                self._prune_island(island % self.config.num_islands)

    def get(self, program_id: str) -> Program | None:
        row = self.connection.execute(
            f"SELECT {PROGRAM_COLUMNS} FROM programs WHERE id = ?", (program_id,)
        ).fetchone()
        return None if row is None else Program.from_row(row)

    def all_programs(self) -> list[Program]:
        rows = self.connection.execute(
            f"SELECT {PROGRAM_COLUMNS} FROM programs ORDER BY created_at"
        ).fetchall()
        return [Program.from_row(row) for row in rows]

    def successful_programs(self, island: int | None = None) -> list[Program]:
        statuses = tuple(SUCCESS_STATUSES)
        if island is None:
            rows = self.connection.execute(
                f"SELECT {PROGRAM_COLUMNS} FROM programs "
                "WHERE status IN (?, ?) ORDER BY created_at",
                statuses,
            ).fetchall()
        else:
            rows = self.connection.execute(
                f"SELECT {', '.join('p.' + item.strip() for item in PROGRAM_COLUMNS.split(','))} "
                "FROM programs p JOIN memberships m ON p.id = m.program_id "
                "WHERE m.island = ? AND p.status IN (?, ?) ORDER BY p.created_at",
                (island, *statuses),
            ).fetchall()
        return [Program.from_row(row) for row in rows]

    def recent_failures(self, parent_id: str, limit: int) -> list[Program]:
        if limit <= 0:
            return []
        rows = self.connection.execute(
            f"SELECT {PROGRAM_COLUMNS} FROM programs "
            "WHERE parent_id = ? AND status NOT IN (?, ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (parent_id, *tuple(SUCCESS_STATUSES), limit),
        ).fetchall()
        return [Program.from_row(row) for row in rows]

    def find_success_by_hash(self, code_hash: str) -> Program | None:
        row = self.connection.execute(
            f"SELECT {PROGRAM_COLUMNS} FROM programs "
            "WHERE code_hash = ? AND status IN (?, ?) ORDER BY created_at LIMIT 1",
            (code_hash, *tuple(SUCCESS_STATUSES)),
        ).fetchone()
        return None if row is None else Program.from_row(row)

    def has_baseline(self) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM programs WHERE status = 'baseline' LIMIT 1"
        ).fetchone()
        return row is not None

    def _fitness(self, program: Program) -> float:
        value = program.metrics.get(self.objective)
        if value is None:
            return float("-inf")
        return value if self.direction == "maximize" else -value

    def _better(self, left: Program, right: Program) -> bool:
        left_fitness = self._fitness(left)
        right_fitness = self._fitness(right)
        if left_fitness != right_fitness:
            return left_fitness > right_fitness
        if left.line_count != right.line_count:
            return left.line_count < right.line_count
        return left.created_at < right.created_at

    def feature_coordinates(self, program: Program) -> tuple[int, ...]:
        return tuple(
            feature.coordinate(program.feature_value(feature.name))
            for feature in self.config.features
        )

    def cell_elites(self, island: int | None = None) -> dict[tuple[int, ...], Program]:
        candidates = self.successful_programs(island)
        elites: dict[tuple[int, ...], Program] = {}
        for program in candidates:
            coordinates = self.feature_coordinates(program)
            incumbent = elites.get(coordinates)
            if incumbent is None or self._better(program, incumbent):
                elites[coordinates] = program
        return elites

    def best(self, island: int | None = None) -> Program | None:
        programs = self.successful_programs(island)
        return max(programs, key=self._fitness) if programs else None

    def top_programs(self, count: int, island: int | None = None) -> list[Program]:
        return sorted(self.successful_programs(island), key=self._fitness, reverse=True)[:count]

    def _increment_samples(self, programs: list[Program]) -> None:
        for program in programs:
            self.connection.execute(
                "UPDATE programs SET sample_count = sample_count + 1 WHERE id = ?", (program.id,)
            )
        self.connection.commit()

    @staticmethod
    def _code_distance(left: Program, right: Program) -> float:
        left_lines = {line.strip() for line in left.code.splitlines() if line.strip()}
        right_lines = {line.strip() for line in right.code.splitlines() if line.strip()}
        union = left_lines | right_lines
        return 0.0 if not union else 1.0 - len(left_lines & right_lines) / len(union)

    def sample(self, num_inspirations: int = 2) -> Sample:
        island = self._get_meta_int("next_island") % self.config.num_islands
        self._set_meta("next_island", (island + 1) % self.config.num_islands)
        candidates = self.successful_programs(island)
        if not candidates:
            candidates = self.successful_programs()
        if not candidates:
            raise RuntimeError("Cannot sample before a successful baseline exists")

        elites = list(self.cell_elites(island).values()) or candidates
        draw = self.random.random()
        if draw < self.config.exploitation_ratio:
            ranked = sorted(elites, key=self._fitness, reverse=True)
            parent = self.random.choices(
                ranked, weights=[1.0 / (index + 1) for index in range(len(ranked))], k=1
            )[0]
            mode = "exploit"
        elif draw < self.config.exploitation_ratio + self.config.exploration_ratio:
            underused = sorted(elites, key=lambda item: (item.sample_count, -item.novelty))
            pool_size = max(1, (len(underused) + 1) // 2)
            parent = self.random.choice(underused[:pool_size])
            mode = "explore"
        else:
            parent = self.random.choice(candidates)
            mode = "random"

        inspirations: list[Program] = []
        global_elites = list(self.cell_elites().values())
        global_best = self.best()
        if global_best is not None and global_best.id != parent.id:
            inspirations.append(global_best)
        diverse = sorted(
            (item for item in global_elites if item.id != parent.id and item not in inspirations),
            key=lambda item: (self._code_distance(parent, item), self._fitness(item)),
            reverse=True,
        )
        inspirations.extend(diverse[: max(0, num_inspirations - len(inspirations))])
        inspirations = inspirations[:num_inspirations]
        self._increment_samples([parent, *inspirations])
        self.connection.commit()
        return Sample(parent=parent, inspirations=inspirations, island=island, mode=mode)

    def _prune_island(self, island: int) -> None:
        members = self.successful_programs(island)
        if len(members) <= self.config.population_size:
            return
        elites = list(self.cell_elites(island).values())
        keep = sorted(elites, key=self._fitness, reverse=True)[: self.config.population_size]
        keep_ids = {program.id for program in keep}
        if len(keep) < self.config.population_size:
            remainder = sorted(
                (program for program in members if program.id not in keep_ids),
                key=self._fitness,
                reverse=True,
            )
            keep.extend(remainder[: self.config.population_size - len(keep)])
            keep_ids = {program.id for program in keep}
        placeholders = ",".join("?" for _ in keep_ids)
        self.connection.execute(
            f"DELETE FROM memberships WHERE island = ? AND program_id NOT IN ({placeholders})",
            (island, *sorted(keep_ids)),
        )
        self.connection.commit()

    def maybe_migrate(self, completed_iterations: int) -> bool:
        interval = self.config.migration_interval
        if interval <= 0 or completed_iterations == 0 or completed_iterations % interval:
            return False
        if self._get_meta_int("last_migration") == completed_iterations:
            return False

        migrations: list[dict[str, object]] = []
        selected = [
            self.top_programs(self.config.migration_count, island)
            for island in range(self.config.num_islands)
        ]
        for source, programs in enumerate(selected):
            destination = (source + 1) % self.config.num_islands
            for program in programs:
                cursor = self.connection.execute(
                    "INSERT OR IGNORE INTO memberships(program_id, island, source) VALUES (?, ?, ?)",
                    (program.id, destination, f"migration:{source}"),
                )
                if cursor.rowcount:
                    migrations.append(
                        {"program_id": program.id, "from": source, "to": destination}
                    )
        self._set_meta("last_migration", completed_iterations)
        self.connection.commit()
        for island in range(self.config.num_islands):
            self._prune_island(island)
        self.log_event("migration", {"iteration": completed_iterations, "moves": migrations})
        return True

    def membership_ids(self, island: int) -> set[str]:
        rows = self.connection.execute(
            "SELECT program_id FROM memberships WHERE island = ?", (island,)
        ).fetchall()
        return {row["program_id"] for row in rows}

    def status(self) -> dict[str, object]:
        counts = {
            row["status"]: row["count"]
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS count FROM programs GROUP BY status"
            ).fetchall()
        }
        best = self.best()
        islands = []
        for island in range(self.config.num_islands):
            island_best = self.best(island)
            islands.append(
                {
                    "island": island,
                    "members": len(self.membership_ids(island)),
                    "cells": len(self.cell_elites(island)),
                    "best_id": None if island_best is None else island_best.id,
                    "best_metrics": {} if island_best is None else island_best.metrics,
                }
            )
        return {
            "database": str(self.path),
            "completed_iterations": self.completed_iterations,
            "counts": counts,
            "best": None
            if best is None
            else {"id": best.id, "metrics": best.metrics, "generation": best.generation},
            "islands": islands,
            "recent_events": self.recent_events(8),
        }

