#!/usr/bin/env python3
"""CLI entry-point for the synthetic data generation pipeline."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.config import PipelineConfig
from src.pipeline.runner import run_pipeline
from src.world_generator.generator import generate_worlds

app = typer.Typer(help="Map Agent synthetic data pipeline")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s  %(message)s",
)


@app.command()
def main(
    config: str = typer.Option("pipeline_config.yaml", "--config", "-c", help="Path to YAML config"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="LLM backend override"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="LLM base URL override"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model override"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="LLM API key override"),
    num_worlds: Optional[int] = typer.Option(None, "--num-worlds", help="Number of worlds to generate"),
    target_samples: Optional[int] = typer.Option(None, "--target-samples", help="Overall target sample count"),
    stage: Optional[str] = typer.Option(None, "--stage", "-s", help="Run only a specific stage: worlds, full"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory for datasets"),
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

    if stage == "worlds":
        typer.echo(f"Generating {cfg.worlds.count} worlds...")
        paths = generate_worlds(
            cfg.worlds.count,
            output_dir=cfg.scenarios_dir,
            base_seed=cfg.worlds.base_seed,
        )
        typer.echo(f"Generated {len(paths)} world files in {cfg.scenarios_dir}/")
        return

    typer.echo("Starting full pipeline...")
    result_paths = run_pipeline(cfg)
    typer.echo("\nExported datasets:")
    for name, path in result_paths.items():
        typer.echo(f"  {name}: {path}")


if __name__ == "__main__":
    app()
