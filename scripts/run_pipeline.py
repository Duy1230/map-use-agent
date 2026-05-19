#!/usr/bin/env python3
"""CLI entry-point for the synthetic data generation pipeline."""

import sys
from pathlib import Path
from typing import Optional

import typer

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.config import PipelineConfig  # noqa: E402
from src.pipeline.context import PipelineContext  # noqa: E402
from src.pipeline.logging_utils import setup_logging  # noqa: E402
from src.pipeline.runner import _generate_worlds_for_profile, run_pipeline  # noqa: E402

app = typer.Typer(help="Map Agent synthetic data pipeline")


@app.command()
def main(
    config: str = typer.Option(
        "pipeline_config.yaml", "--config", "-c", help="Path to YAML config"
    ),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="LLM backend override"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="LLM base URL override"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model override"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="LLM API key override"),
    num_worlds: Optional[int] = typer.Option(
        None, "--num-worlds", help="Number of worlds to generate"
    ),
    target_samples: Optional[int] = typer.Option(
        None, "--target-samples", help="Overall target sample count"
    ),
    stage: Optional[str] = typer.Option(
        None, "--stage", "-s", help="Run only a specific stage: worlds, full"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for datasets"
    ),
    profile: Optional[str] = typer.Option(None, "--profile", help="Domain profile YAML override"),
    resume: Optional[bool] = typer.Option(None, "--resume/--no-resume", help="Resume checkpointed runs"),
    reset_checkpoint: bool = typer.Option(
        False, "--reset-checkpoint", help="Delete checkpoint state before running"
    ),
    checkpoint_dir: Optional[str] = typer.Option(None, "--checkpoint-dir", help="Checkpoint directory"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Optional log file path"),
) -> None:
    """Run the synthetic data generation pipeline."""
    cfg_path = Path(config)
    if cfg_path.exists():
        cfg = PipelineConfig.from_yaml(cfg_path)
    else:
        typer.echo(f"Config {config} not found, using defaults")
        cfg = PipelineConfig()

    if backend:
        cfg.llm.backend = backend
    if base_url:
        cfg.llm.base_url = base_url
    if model:
        cfg.llm.model = model
    if api_key:
        cfg.llm.api_key = api_key
    if num_worlds:
        cfg.worlds.count = num_worlds
    if output_dir:
        cfg.datasets_dir = output_dir
    if profile:
        cfg.profile = profile
    if resume is not None:
        cfg.runtime.resume = resume
    if reset_checkpoint:
        cfg.runtime.reset_checkpoint = True
    if checkpoint_dir:
        cfg.runtime.checkpoint_dir = checkpoint_dir
    if log_level:
        cfg.runtime.log_level = log_level
    if log_file:
        cfg.runtime.log_file = log_file

    setup_logging(cfg.runtime.log_level, cfg.runtime.log_file)

    if stage == "worlds":
        typer.echo(f"Generating {cfg.worlds.count} worlds...")
        context = PipelineContext.from_profile_path(cfg.profile)
        scenarios_dir = Path(cfg.scenarios_dir)
        _generate_worlds_for_profile(context, cfg, scenarios_dir)
        paths = sorted(scenarios_dir.glob("*.json"))
        typer.echo(f"Generated {len(paths)} world files in {cfg.scenarios_dir}/")
        return

    typer.echo("Starting full pipeline...")
    result_paths = run_pipeline(cfg)
    typer.echo("\nExported datasets:")
    for name, path in result_paths.items():
        typer.echo(f"  {name}: {path}")


if __name__ == "__main__":
    app()
