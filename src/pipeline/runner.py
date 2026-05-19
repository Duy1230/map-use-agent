"""End-to-end pipeline orchestrator."""

from __future__ import annotations

import json
import logging
import random
import hashlib
import uuid
from pathlib import Path

from tqdm import tqdm

from src.blueprint_generator.generator import generate_blueprints_for_scenario
from src.dedup.dedup import deduplicate
from src.export.exporter import export_dataset
from src.gold_plan_generator.generator import build_expected, generate_gold_plan
from src.llm_backend.config import create_backend
from src.pipeline.checkpoint import CheckpointManager
from src.pipeline.context import PipelineContext
from src.negative_generator.generator import generate_negative_cases
from src.pipeline.config import PipelineConfig
from src.utterance_generator.generator import generate_utterances
from src.verification.assertion_checker import check_assertions
from src.verification.execution_checker import check_execution
from src.verification.human_review import export_review_queue, select_review_samples
from src.verification.schema_validator import validate_schema
from src.verification.semantic_verifier import verify_semantic
from src.verification.static_checker import check_static
from src.world_generator.generator import generate_worlds

logger = logging.getLogger(__name__)


class PipelineMetrics:
    def __init__(self) -> None:
        self.candidates = 0
        self.schema_pass = 0
        self.static_pass = 0
        self.exec_pass = 0
        self.assert_pass = 0
        self.semantic_pass = 0
        self.after_dedup = 0
        self.final = 0

    def report(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def print_report(self) -> None:
        r = self.report()
        print("\n=== Pipeline Metrics ===")
        for k, v in r.items():
            pct = f"  ({v * 100 / max(self.candidates, 1):.0f}%)" if self.candidates else ""
            print(f"  {k:25s} {v:>6d}{pct}")
        print()


def run_pipeline(cfg: PipelineConfig) -> dict[str, Path]:
    """Execute the full synthetic data generation pipeline."""
    context = PipelineContext.from_profile_path(cfg.profile)
    checkpoint = CheckpointManager(
        cfg.runtime.checkpoint_dir,
        fingerprint=_pipeline_fingerprint(cfg, context),
        resume=cfg.runtime.resume,
        reset=cfg.runtime.reset_checkpoint,
    )
    checkpoint.initialize()
    missing_handlers = context.validate_tool_registry()
    if missing_handlers:
        logger.warning("Tool catalog has no executable handlers for: %s", missing_handlers)

    llm = create_backend(
        backend=cfg.llm.backend,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        api_key=cfg.llm.api_key,
        timeout=cfg.llm.timeout,
        max_retries=cfg.llm.max_retries,
        retry_backoff_seconds=cfg.llm.retry_backoff_seconds,
        retry_backoff_max_seconds=cfg.llm.retry_backoff_max_seconds,
        reconnect_on_failure=cfg.llm.reconnect_on_failure,
    )
    metrics = PipelineMetrics()

    # --- Stage 1: Generate or load worlds ---------------------------------
    scenarios_dir = Path(cfg.scenarios_dir)
    scenario_files = sorted(scenarios_dir.glob("*.json"))
    if not scenario_files:
        logger.info("Generating %d synthetic worlds...", cfg.worlds.count)
        generate_worlds(
            cfg.worlds.count,
            output_dir=scenarios_dir,
            base_seed=cfg.worlds.base_seed,
            small_fraction=cfg.worlds.small_fraction,
            medium_fraction=cfg.worlds.medium_fraction,
        )
        scenario_files = sorted(scenarios_dir.glob("*.json"))

    scenarios = []
    for f in scenario_files:
        scenarios.append(json.loads(f.read_text(encoding="utf-8")))
    logger.info("Loaded %d scenarios", len(scenarios))

    # --- Stage 2-5: Generate per scenario ---------------------------------
    all_candidates: list[dict] = checkpoint.load_candidates()
    if all_candidates:
        logger.info("Loaded %d checkpointed candidates", len(all_candidates))

    for scenario in tqdm(scenarios, desc="Scenarios"):
        sid = scenario["scenario_id"]
        if checkpoint.should_skip_scenario(sid):
            logger.info("Skipping completed scenario from checkpoint: %s", sid)
            continue
        seed = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        scenario_candidates: list[dict] = []
        logger.info("Generating candidates for scenario %s", sid)

        # 2. Blueprints
        blueprints = generate_blueprints_for_scenario(
            llm,
            scenario,
            target_count=cfg.generation.blueprints_per_scenario,
            rng=rng,
            context=context,
        )

        # 3. Gold plans + expected
        for bp in blueprints:
            bp["_gold_trace"] = generate_gold_plan(bp, scenario, llm, context=context)
            bp["_expected"] = build_expected(bp, bp["_gold_trace"], context=context)

        # 4. Negative cases
        negatives = generate_negative_cases(
            blueprints,
            scenario,
            llm=llm,
            llm_adversarial_count=cfg.generation.llm_adversarial_per_scenario,
            rng=rng,
            context=context,
        )
        for neg in negatives:
            neg["_gold_trace"] = []
            neg["_expected"] = build_expected(neg, [], context=context)

        all_bps = blueprints + negatives

        # 5. Utterances + assemble samples
        for bp in all_bps:
            utterances = generate_utterances(
                llm,
                bp,
                temperature=cfg.generation.temperature_utterance,
                context=context,
            )
            if not utterances:
                utterances = [f"[auto] {bp.get('task_type', 'unknown')} task"]

            for utt in utterances:
                sample = _assemble_sample(bp, scenario, utt, context=context)
                scenario_candidates.append(sample)

        all_candidates.extend(scenario_candidates)
        if cfg.runtime.checkpoint_every_scenario:
            checkpoint.append_candidates(sid, scenario_candidates)
            checkpoint.mark_scenario_completed(
                sid,
                metrics={"candidates": len(all_candidates)},
            )
            logger.info(
                "Checkpointed scenario %s with %d candidates",
                sid,
                len(scenario_candidates),
            )

    metrics.candidates = len(all_candidates)
    logger.info("Total candidates: %d", metrics.candidates)

    # --- Stage 6: Verification pipeline -----------------------------------
    verified: list[dict] = []
    for sample in tqdm(all_candidates, desc="Verifying"):
        scenario = _find_scenario(scenarios, sample.get("scenario_id", ""))
        if scenario is None:
            continue

        # Gate 1: schema
        ok, errs = validate_schema(sample)
        if not ok:
            logger.debug("Schema fail %s: %s", sample["id"], errs)
            continue
        metrics.schema_pass += 1

        # Gate 2: static
        ok, errs = check_static(sample, scenario, context=context)
        if not ok:
            logger.debug("Static fail %s: %s", sample["id"], errs)
            continue
        metrics.static_pass += 1

        # Gate 3: execution
        ok, errs, final_state = check_execution(sample, scenario, context=context)
        if not ok:
            logger.debug("Exec fail %s: %s", sample["id"], errs)
            continue
        metrics.exec_pass += 1

        # Gate 4: assertions
        assertions = sample.get("expected", {}).get("final_state_assertions", [])
        is_neg = sample.get("expected", {}).get("should_ask_clarification", False)
        ok, errs = check_assertions(
            final_state,
            assertions,
            is_negative=is_neg,
            context=context,
        )
        if not ok:
            logger.debug("Assert fail %s: %s", sample["id"], errs)
            continue
        metrics.assert_pass += 1

        # Gate 5: semantic (LLM)
        if cfg.verification.run_semantic_verifier:
            ok, issues = verify_semantic(llm, sample)
            if not ok:
                logger.debug("Semantic fail %s: %s", sample["id"], issues)
                continue
        metrics.semantic_pass += 1

        sample["generation_metadata"]["verified_by_schema"] = True
        sample["generation_metadata"]["verified_by_execution"] = True
        sample["generation_metadata"]["verified_by_llm"] = cfg.verification.run_semantic_verifier
        verified.append(sample)
        checkpoint.append_verified([sample])

    # --- Stage 7: Dedup ---------------------------------------------------
    deduped = deduplicate(verified, context=context)
    metrics.after_dedup = len(deduped)

    # --- Stage 8: Human review sampling -----------------------------------
    review_samples = select_review_samples(deduped, review_rate=cfg.verification.human_review_rate)
    reports_dir = Path(cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    export_review_queue(review_samples, reports_dir / "review_queue.jsonl")

    # --- Stage 9: Export --------------------------------------------------
    metrics.final = len(deduped)
    paths = export_dataset(
        deduped,
        output_dir=cfg.datasets_dir,
        total_scenarios=len(scenarios),
    )

    # --- Metrics report ---------------------------------------------------
    metrics.print_report()
    metrics_path = reports_dir / "pipeline_metrics.json"
    metrics_path.write_text(json.dumps(metrics.report(), indent=2), encoding="utf-8")
    checkpoint.save_metrics(metrics.report())

    return paths


def _pipeline_fingerprint(cfg: PipelineConfig, context: PipelineContext) -> str:
    profile_path = Path(cfg.profile)
    profile_text = profile_path.read_text(encoding="utf-8") if profile_path.exists() else ""
    catalog_path = context.profile.tool_catalog.path
    catalog_text = catalog_path.read_text(encoding="utf-8") if catalog_path.exists() else ""
    payload = {
        "profile": str(profile_path),
        "profile_text": profile_text,
        "tool_catalog": str(catalog_path),
        "tool_catalog_text": catalog_text,
        "llm": {
            "backend": cfg.llm.backend,
            "base_url": cfg.llm.base_url,
            "model": cfg.llm.model,
        },
        "worlds": cfg.worlds.__dict__,
        "generation": cfg.generation.__dict__,
        "verification": cfg.verification.__dict__,
        "scenarios_dir": cfg.scenarios_dir,
        "schema_version": cfg.schema_version,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _assemble_sample(
    bp: dict,
    scenario: dict,
    utterance: str,
    *,
    context: PipelineContext | None = None,
) -> dict:
    """Turn a blueprint + utterance into a full TaskSample dict."""
    sample_id = f"mapagent_{uuid.uuid4().hex[:8]}"
    sid = scenario["scenario_id"]

    initial_state = bp.get("initial_state", {})
    if "selected" not in initial_state:
        slots = context.selection_slots if context else ["object", "line", "polygon"]
        initial_state["selected"] = {slot: None for slot in slots}
    if "active_layers" not in initial_state:
        initial_state["active_layers"] = (
            list(context.active_layers) if context else ["vehicles", "cameras", "roads", "zones"]
        )
    initial_state.setdefault("drawn_artifacts", [])
    initial_state["time"] = scenario.get("time", "")

    expected = bp.get("_expected", {})
    gold_trace = bp.get("_gold_trace", [])

    available_tools = list({step["tool"] for step in gold_trace}) if gold_trace else []

    difficulty = bp.get("difficulty", "L1")
    task_type = bp.get("task_type", "unknown")

    tags: list[str] = []
    ref_mode = bp.get("target", {}).get("reference_mode", "")
    if ref_mode:
        tags.append(ref_mode.replace(" ", "_"))
    etype = bp.get("target", {}).get("expected_type", "")
    if etype:
        tags.append(etype)
    if bp.get("negative_case"):
        tags.append("negative")
        neg_type = bp.get("_negative_type", "")
        if neg_type:
            tags.append(neg_type)
    if len(gold_trace) >= 3:
        tags.append("multi_tool")
    if bp.get("constraints", {}).get("time_range"):
        tags.append("temporal")

    return {
        "id": sample_id,
        "scenario_id": sid,
        "tool_catalog_version": (
            context.profile.tool_catalog.version if context else "map_tools_v0.1"
        ),
        "language": _detect_lang(utterance),
        "difficulty": difficulty,
        "task_type": task_type,
        "user_request": utterance,
        "initial_state": initial_state,
        "available_tools": available_tools,
        "gold_trace": gold_trace,
        "expected": expected,
        "tags": tags,
        "generation_metadata": {
            "generator_model": "",
            "blueprint_id": bp.get("_meta", {}).get("requested_family", ""),
            "verified_by_schema": False,
            "verified_by_execution": False,
            "verified_by_llm": False,
            "human_reviewed": False,
        },
    }


def _detect_lang(text: str) -> str:
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    return "en" if ascii_ratio > 0.9 else "vi"


def _find_scenario(scenarios: list[dict], scenario_id: str) -> dict | None:
    for s in scenarios:
        if s["scenario_id"] == scenario_id:
            return s
    return None
