"""Export Pydantic models to JSON Schema files in schemas/."""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import (  # noqa: E402
    FinalStateAssertion,
    MapState,
    Scenario,
    TaskSample,
    ToolCatalog,
)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
SCHEMA_DIR.mkdir(exist_ok=True)

EXPORTS = {
    "map_state.schema.json": MapState,
    "scenario.schema.json": Scenario,
    "tool_catalog.schema.json": ToolCatalog,
    "task.schema.json": TaskSample,
    "final_state_assertion.schema.json": FinalStateAssertion,
}


def main() -> None:
    for filename, model in EXPORTS.items():
        path = SCHEMA_DIR / filename
        schema = model.model_json_schema()
        path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
