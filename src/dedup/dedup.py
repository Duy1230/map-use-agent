"""Deduplication pipeline — 5 levels of duplicate detection."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Lowercase, strip whitespace, remove punctuation, collapse spaces."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _trace_structure_key(gold_trace: list[dict]) -> str:
    """Hash the tool sequence ignoring argument values."""
    tools = [step.get("tool", "") for step in gold_trace]
    return _hash("|".join(tools))


def deduplicate(samples: list[dict]) -> list[dict]:
    """Remove duplicates using 5 dedup levels.

    Levels:
    1. Exact text hash of user_request
    2. Normalized text hash
    3. Composite key: task_type + target_object_id + time_range
    4. Gold trace structure hash (tool sequence only)
    5. (Scenario-level split enforcement is handled in the exporter)

    Returns the deduplicated list (order preserved).
    """
    seen_exact: set[str] = set()
    seen_norm: set[str] = set()
    seen_composite: set[str] = set()
    seen_trace: set[str] = set()

    kept: list[dict] = []
    stats = {"exact": 0, "norm": 0, "composite": 0, "trace": 0}

    for sample in samples:
        user_req = sample.get("user_request", "")

        # Level 1: exact text
        h1 = _hash(user_req)
        if h1 in seen_exact:
            stats["exact"] += 1
            continue
        seen_exact.add(h1)

        # Level 2: normalized text
        h2 = _hash(_normalize_text(user_req))
        if h2 in seen_norm:
            stats["norm"] += 1
            continue
        seen_norm.add(h2)

        # Level 3: composite key
        task_type = sample.get("task_type", "")
        target_id = ""
        initial = sample.get("initial_state", {})
        selected = initial.get("selected", {})
        for slot in ("object", "line", "polygon"):
            ent = selected.get(slot)
            if isinstance(ent, dict):
                target_id = ent.get("id", "")
                break
        time_range = ""
        for step in sample.get("gold_trace", []):
            tr = step.get("args", {}).get("time_range")
            if tr:
                time_range = tr
                break
        composite = f"{task_type}|{target_id}|{time_range}"
        h3 = _hash(composite)
        if h3 in seen_composite:
            stats["composite"] += 1
            continue
        seen_composite.add(h3)

        # Level 4: trace structure
        trace = sample.get("gold_trace", [])
        if trace:
            h4 = _trace_structure_key(trace)
            scenario_trace = f"{sample.get('scenario_id', '')}|{h4}|{target_id}"
            h4_full = _hash(scenario_trace)
            if h4_full in seen_trace:
                stats["trace"] += 1
                continue
            seen_trace.add(h4_full)

        kept.append(sample)

    total_removed = sum(stats.values())
    logger.info(
        "Dedup: %d -> %d (removed %d: exact=%d, norm=%d, composite=%d, trace=%d)",
        len(samples), len(kept), total_removed,
        stats["exact"], stats["norm"], stats["composite"], stats["trace"],
    )
    return kept
