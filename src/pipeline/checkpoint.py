"""Scenario-level checkpointing for resumable pipeline runs."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CheckpointFingerprintError(RuntimeError):
    """Raised when a checkpoint belongs to an incompatible run."""


@dataclass
class CheckpointManager:
    checkpoint_dir: str | Path
    fingerprint: str
    resume: bool = True
    reset: bool = False
    completed_scenarios: set[str] = field(default_factory=set)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return Path(self.checkpoint_dir)

    @property
    def state_path(self) -> Path:
        return self.root / "run_state.json"

    @property
    def candidates_path(self) -> Path:
        return self.root / "candidates.jsonl"

    @property
    def verified_path(self) -> Path:
        return self.root / "verified.jsonl"

    def initialize(self) -> None:
        if self.reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

        if self.resume and self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("fingerprint") != self.fingerprint:
                raise CheckpointFingerprintError(
                    "Checkpoint fingerprint differs from current config/profile/tool catalog. "
                    "Use --reset-checkpoint to start a fresh run."
                )
            self.completed_scenarios = set(state.get("completed_scenarios", []))
            self.metrics = dict(state.get("metrics", {}))
            return

        self.completed_scenarios = set()
        self.metrics = {}
        self._write_state()
        _atomic_write_text(self.candidates_path, "")
        _atomic_write_text(self.verified_path, "")

    def should_skip_scenario(self, scenario_id: str) -> bool:
        return self.resume and scenario_id in self.completed_scenarios

    def append_candidates(self, scenario_id: str, candidates: list[dict]) -> None:
        self._append_jsonl(
            self.candidates_path,
            [{**candidate, "_checkpoint_scenario_id": scenario_id} for candidate in candidates],
        )

    def append_verified(self, samples: list[dict]) -> None:
        self._append_jsonl(self.verified_path, samples)

    def load_candidates(self) -> list[dict]:
        return self._read_jsonl(self.candidates_path)

    def load_verified(self) -> list[dict]:
        return self._read_jsonl(self.verified_path)

    def mark_scenario_completed(
        self,
        scenario_id: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.completed_scenarios.add(scenario_id)
        if metrics:
            self.metrics.update(metrics)
        self._write_state()

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics = dict(metrics)
        self._write_state()

    def _write_state(self) -> None:
        state = {
            "fingerprint": self.fingerprint,
            "completed_scenarios": sorted(self.completed_scenarios),
            "metrics": self.metrics,
        }
        _atomic_write_json(self.state_path, state)

    def _append_jsonl(self, path: Path, records: list[dict]) -> None:
        if not records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        content = existing
        if content and not content.endswith("\n"):
            content += "\n"
        content += "\n".join(lines) + "\n"
        _atomic_write_text(path, content)

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    record.pop("_checkpoint_scenario_id", None)
                    records.append(record)
        return records


def _atomic_write_json(path: Path, data: dict) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
