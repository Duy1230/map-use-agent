from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from src.llm_backend.llamacpp_backend import LlamaCppBackend
from src.pipeline.checkpoint import CheckpointFingerprintError, CheckpointManager
from src.pipeline.config import PipelineConfig
from src.pipeline.logging_utils import setup_logging
from src.pipeline.runner import run_pipeline


def test_checkpoint_manager_tracks_completed_scenarios_and_candidates(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, fingerprint="abc", resume=True)
    manager.initialize()

    manager.append_candidates("scenario_001", [{"id": "c1", "scenario_id": "scenario_001"}])
    manager.mark_scenario_completed("scenario_001", metrics={"candidates": 1})

    resumed = CheckpointManager(tmp_path, fingerprint="abc", resume=True)
    resumed.initialize()

    assert resumed.completed_scenarios == {"scenario_001"}
    assert resumed.load_candidates() == [{"id": "c1", "scenario_id": "scenario_001"}]
    assert resumed.should_skip_scenario("scenario_001")
    assert not resumed.should_skip_scenario("scenario_002")


def test_checkpoint_manager_rejects_changed_fingerprint(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, fingerprint="abc", resume=True)
    manager.initialize()
    manager.mark_scenario_completed("scenario_001")

    changed = CheckpointManager(tmp_path, fingerprint="different", resume=True)

    with pytest.raises(CheckpointFingerprintError):
        changed.initialize()


def test_checkpoint_manager_reset_removes_previous_state(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, fingerprint="abc", resume=True)
    manager.initialize()
    manager.append_candidates("scenario_001", [{"id": "c1"}])
    manager.mark_scenario_completed("scenario_001")

    reset = CheckpointManager(tmp_path, fingerprint="abc", resume=True, reset=True)
    reset.initialize()

    assert reset.completed_scenarios == set()
    assert reset.load_candidates() == []


def test_setup_logging_writes_to_file_and_respects_level(tmp_path: Path) -> None:
    log_file = tmp_path / "pipeline.log"
    setup_logging(level="DEBUG", log_file=log_file)

    logger = logging.getLogger("tests.runtime_qol")
    logger.debug("debug-visible")
    logger.info("info-visible")
    logging.shutdown()

    content = log_file.read_text(encoding="utf-8")
    assert "debug-visible" in content
    assert "info-visible" in content


def test_llamacpp_reconnect_recreates_client_and_retries() -> None:
    attempts = {"count": 0}
    created_clients: list[FakeHttpClient] = []

    def client_factory(timeout: float) -> FakeHttpClient:
        client = FakeHttpClient(attempts)
        created_clients.append(client)
        return client

    backend = LlamaCppBackend(
        base_url="http://localhost:8080",
        model="local",
        max_retries=3,
        retry_backoff_seconds=0,
        reconnect_on_failure=True,
        client_factory=client_factory,
    )

    response = backend.generate("hello", system="system")

    assert response.text == '{"ok": true}'
    assert attempts["count"] == 3
    assert len(created_clients) == 3
    assert created_clients[0].closed
    assert created_clients[1].closed


def test_llamacpp_retry_exhaustion_raises_clear_error() -> None:
    def client_factory(timeout: float) -> AlwaysFailClient:
        return AlwaysFailClient()

    backend = LlamaCppBackend(
        base_url="http://localhost:8080",
        model="local",
        max_retries=2,
        retry_backoff_seconds=0,
        reconnect_on_failure=True,
        client_factory=client_factory,
    )

    with pytest.raises(httpx.ConnectError):
        backend.generate("hello")


def test_run_pipeline_resumes_after_completed_scenario_without_duplicate_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    for idx in (1, 2):
        scenario = {
            "scenario_id": f"traffic_synthetic_{idx:03d}",
            "time": "2026-05-18T09:15:00+07:00",
            "objects": {"vehicles": [], "cameras": [], "polygons": [], "lines": []},
            "time_series": {},
        }
        (scenarios_dir / f"traffic_synthetic_{idx:03d}.json").write_text(
            json.dumps(scenario),
            encoding="utf-8",
        )

    cfg = PipelineConfig()
    cfg.scenarios_dir = str(scenarios_dir)
    cfg.datasets_dir = str(tmp_path / "datasets")
    cfg.reports_dir = str(tmp_path / "reports")
    cfg.runtime.checkpoint_dir = str(tmp_path / "reports" / "checkpoints")
    cfg.runtime.resume = True
    cfg.verification.run_semantic_verifier = False
    cfg.generation.blueprints_per_scenario = 1

    monkeypatch.setattr("src.pipeline.runner.create_backend", lambda **kwargs: object())
    monkeypatch.setattr(
        "src.pipeline.runner.generate_blueprints_for_scenario",
        lambda llm, scenario, **kwargs: [
                {
                    "task_type": f"show_object_info_{scenario['scenario_id']}",
                    "difficulty": "L1",
                    "target": {"reference_mode": "explicit_id", "expected_type": "vehicle"},
                    "required_semantic_steps": [],
                    "expected_final_state": [],
                    "_meta": {"scenario_id": scenario["scenario_id"]},
                }
            ],
        )
    monkeypatch.setattr("src.pipeline.runner.generate_gold_plan", lambda *args, **kwargs: [])
    monkeypatch.setattr("src.pipeline.runner.build_expected", lambda *args, **kwargs: {})
    monkeypatch.setattr("src.pipeline.runner.generate_negative_cases", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "src.pipeline.runner.generate_utterances",
        lambda llm, bp, **kwargs: [f"user request {bp['_meta']['scenario_id']}"],
    )
    monkeypatch.setattr("src.pipeline.runner.validate_schema", lambda sample: (True, []))
    monkeypatch.setattr("src.pipeline.runner.check_static", lambda *args, **kwargs: (True, []))
    monkeypatch.setattr(
        "src.pipeline.runner.check_execution",
        lambda *args, **kwargs: (True, [], {}),
    )
    monkeypatch.setattr("src.pipeline.runner.check_assertions", lambda *args, **kwargs: (True, []))
    monkeypatch.setattr("src.pipeline.runner.select_review_samples", lambda samples, **kwargs: [])
    monkeypatch.setattr("src.pipeline.runner.export_review_queue", lambda *args, **kwargs: None)

    exported: list[list[dict]] = []

    def capture_export(samples: list[dict], *args, **kwargs) -> dict[str, Path]:
        exported.append(samples)
        return {"train": tmp_path / "datasets" / "train.jsonl"}

    monkeypatch.setattr("src.pipeline.runner.export_dataset", capture_export)

    calls = {"count": 0}
    original_assemble = __import__("src.pipeline.runner", fromlist=["_assemble_sample"])._assemble_sample

    def crash_on_second_scenario(*args, **kwargs) -> dict:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated disconnect")
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr("src.pipeline.runner._assemble_sample", crash_on_second_scenario)
    with pytest.raises(RuntimeError):
        run_pipeline(cfg)

    monkeypatch.setattr("src.pipeline.runner._assemble_sample", original_assemble)
    run_pipeline(cfg)

    assert len(exported[-1]) == 2
    assert sorted(sample["scenario_id"] for sample in exported[-1]) == [
        "traffic_synthetic_001",
        "traffic_synthetic_002",
    ]


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://localhost")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("server error", request=request, response=response)

    def json(self) -> dict:
        return self._payload


class FakeHttpClient:
    def __init__(self, attempts: dict[str, int]) -> None:
        self.attempts = attempts
        self.closed = False

    def post(self, *args, **kwargs) -> FakeResponse:
        self.attempts["count"] += 1
        if self.attempts["count"] < 3:
            raise httpx.ConnectError("server down")
        return FakeResponse(
            {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "model": "local",
                "usage": {},
            }
        )

    def close(self) -> None:
        self.closed = True


class AlwaysFailClient:
    def post(self, *args, **kwargs) -> FakeResponse:
        raise httpx.ConnectError("server down")

    def close(self) -> None:
        pass
