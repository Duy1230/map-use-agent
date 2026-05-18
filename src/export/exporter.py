"""Dataset exporter — JSONL output with scenario-based train/dev/test splits."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def scenario_split(
    total_scenarios: int,
    *,
    train_frac: float = 0.80,
    dev_frac: float = 0.10,
) -> dict[str, list[int]]:
    """Return scenario index ranges for train / dev / test."""
    train_end = int(total_scenarios * train_frac)
    dev_end = train_end + int(total_scenarios * dev_frac)
    return {
        "train": list(range(1, train_end + 1)),
        "dev": list(range(train_end + 1, dev_end + 1)),
        "test": list(range(dev_end + 1, total_scenarios + 1)),
    }


def _scenario_index(scenario_id: str) -> int:
    """Extract the numeric index from a scenario_id like 'traffic_synthetic_042'."""
    parts = scenario_id.rsplit("_", 1)
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def export_dataset(
    samples: list[dict],
    output_dir: str | Path = "datasets",
    *,
    total_scenarios: int = 100,
) -> dict[str, Path]:
    """Split samples by scenario into train/dev/test and write JSONL files.

    Also creates difficulty-based sub-splits for the test set.
    Returns a dict mapping split name to file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits = scenario_split(total_scenarios)
    train_ids = set(splits["train"])
    dev_ids = set(splits["dev"])
    test_ids = set(splits["test"])

    buckets: dict[str, list[dict]] = {
        "train": [],
        "dev": [],
        "test_easy": [],
        "test_hard": [],
        "test_adversarial": [],
    }

    for s in samples:
        idx = _scenario_index(s.get("scenario_id", ""))
        if idx in train_ids:
            buckets["train"].append(s)
        elif idx in dev_ids:
            buckets["dev"].append(s)
        elif idx in test_ids:
            diff = s.get("difficulty", "L1")
            is_neg = s.get("expected", {}).get("should_ask_clarification", False)
            if is_neg or diff in ("L5",):
                buckets["test_adversarial"].append(s)
            elif diff in ("L0", "L1", "L2"):
                buckets["test_easy"].append(s)
            else:
                buckets["test_hard"].append(s)

    paths: dict[str, Path] = {}
    for name, data in buckets.items():
        if not data:
            continue
        p = out / f"{name}.jsonl"
        _write_jsonl(data, p)
        paths[name] = p
        logger.info("Exported %s: %d samples -> %s", name, len(data), p)

    return paths


def _write_jsonl(samples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            clean = {k: v for k, v in s.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
