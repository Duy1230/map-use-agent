"""Gate 6 — Human review sampling and export."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def select_review_samples(
    samples: list[dict],
    *,
    review_rate: float = 0.10,
    priority_tags: set[str] | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """Select samples for human review.

    Priority goes to negative cases, high-difficulty, and multi-turn samples.
    """
    r = rng or random.Random()
    priority_tags = priority_tags or {"negative", "adversarial", "L4", "L5", "multi_turn"}

    priority: list[dict] = []
    rest: list[dict] = []

    for s in samples:
        tags = set(s.get("tags", []))
        diff = s.get("difficulty", "")
        is_neg = s.get("expected", {}).get("should_ask_clarification", False)

        if is_neg or diff in ("L4", "L5") or tags & priority_tags:
            priority.append(s)
        else:
            rest.append(s)

    # Always include 50% of priority samples, plus fill to target
    target = max(1, int(len(samples) * review_rate))
    selected = r.sample(priority, min(len(priority), max(target // 2, 1)))

    remaining = target - len(selected)
    if remaining > 0 and rest:
        selected.extend(r.sample(rest, min(len(rest), remaining)))

    return selected


def export_review_queue(
    samples: list[dict],
    output_path: str | Path = "reports/review_queue.jsonl",
) -> Path:
    """Write selected samples to a JSONL file for human review."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            line = {
                "id": s.get("id"),
                "task_type": s.get("task_type"),
                "difficulty": s.get("difficulty"),
                "user_request": s.get("user_request"),
                "initial_state": s.get("initial_state"),
                "gold_trace": s.get("gold_trace"),
                "expected": s.get("expected"),
                "tags": s.get("tags"),
                "review_status": "pending",
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info("Exported %d samples for review to %s", len(samples), path)
    return path
