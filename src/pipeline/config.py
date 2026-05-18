"""Pipeline configuration — loads from YAML or provides defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LLMConfig:
    backend: str = "api"
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout: float = 120.0


@dataclass
class WorldConfig:
    count: int = 100
    base_seed: int = 42
    small_fraction: float = 0.30
    medium_fraction: float = 0.50
    trajectory_duration_min: int = 30


@dataclass
class GenerationConfig:
    blueprints_per_scenario: int = 10
    utterances_per_blueprint: int = 6
    llm_adversarial_per_scenario: int = 3
    temperature_blueprint: float = 0.8
    temperature_utterance: float = 0.9
    temperature_verifier: float = 0.2


@dataclass
class TargetDistribution:
    """Target sample counts per difficulty level."""
    L0: int = 150
    L1: int = 150
    L2: int = 250
    L3: int = 200
    L4: int = 150
    L5: int = 100

    @property
    def total(self) -> int:
        return self.L0 + self.L1 + self.L2 + self.L3 + self.L4 + self.L5


@dataclass
class VerificationConfig:
    run_semantic_verifier: bool = True
    human_review_rate: float = 0.10


@dataclass
class PipelineConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    worlds: WorldConfig = field(default_factory=WorldConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    targets: TargetDistribution = field(default_factory=TargetDistribution)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    scenarios_dir: str = "scenarios"
    datasets_dir: str = "datasets"
    reports_dir: str = "reports"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data is None:
            return cls()

        cfg = cls()
        if "llm" in data:
            cfg.llm = LLMConfig(**data["llm"])
        if "worlds" in data:
            cfg.worlds = WorldConfig(**data["worlds"])
        if "generation" in data:
            cfg.generation = GenerationConfig(**data["generation"])
        if "targets" in data:
            cfg.targets = TargetDistribution(**data["targets"])
        if "verification" in data:
            cfg.verification = VerificationConfig(**data["verification"])
        for key in ("scenarios_dir", "datasets_dir", "reports_dir"):
            if key in data:
                setattr(cfg, key, data[key])
        return cfg
