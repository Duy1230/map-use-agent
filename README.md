# State-Aware Map Tool-Use Agent — Synthetic Data Pipeline

Automated pipeline for generating, validating, and exporting synthetic
evaluation and training data for a state-aware map tool-use agent.

The agent has no vision — it reasons solely over structured map state,
user requests in natural language, and a canonical set of map tools.
This pipeline produces high-quality datasets by combining **deterministic
geometry/validation** with **LLM-driven generation**, then filtering
every candidate through a 6-gate verification process.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | >= 3.11 |
| An LLM backend (one of the options below) | |

**Supported LLM backends:**

| Backend | When to use |
|---------|-------------|
| **vLLM** | You have a local GPU server running `vllm serve` |
| **llama.cpp** | You are running a GGUF model via `llama-server` |
| **OpenAI-compatible API** | OpenAI, Together, Groq, or any `/v1/chat/completions` endpoint |

---

## Installation

### Option A — pip (local)

```bash
# Clone the repo
git clone <your-repo-url> map-use-agent
cd map-use-agent

# Install in editable mode (creates the `map-pipeline` CLI)
pip install -e ".[dev]"
```

### Option B — Docker

```bash
docker build -t map-use-agent .

# Run with a vLLM backend on the host network
docker run --rm \
  -e LLM_BACKEND=vllm \
  -e LLM_BASE_URL=http://host.docker.internal:8000/v1 \
  -e LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/scenarios:/app/scenarios \
  -v $(pwd)/reports:/app/reports \
  map-use-agent

# Generate only the synthetic worlds
docker run --rm -v $(pwd)/scenarios:/app/scenarios \
  map-use-agent --stage worlds --num-worlds 100
```

---

## Configuration

The pipeline reads from `pipeline_config.yaml` by default.
Every field can be overridden via CLI flags or environment variables.

```yaml
# pipeline_config.yaml — key sections

llm:
  backend: "vllm"                        # vllm | llamacpp | api
  base_url: "http://localhost:8000/v1"
  model: "meta-llama/Llama-3.1-70B-Instruct"
  api_key: "EMPTY"

worlds:
  count: 100          # number of synthetic map scenarios
  base_seed: 42       # reproducibility

generation:
  blueprints_per_scenario: 10
  utterances_per_blueprint: 6
  llm_adversarial_per_scenario: 3

targets:              # target sample count per difficulty
  L0: 150             # clarification / no-tool
  L1: 150             # single tool, explicit ID
  L2: 250             # selected-reference resolution
  L3: 200             # multi-step tool chain
  L4: 150             # multi-turn / artifact-aware
  L5: 100             # adversarial / ambiguity

verification:
  run_semantic_verifier: true
  human_review_rate: 0.10

profile: "profiles/traffic_map.yaml"     # domain vocabulary and pipeline wiring
schema_version: "legacy"                 # legacy | flexible
```

**Environment variable overrides** (take precedence over the config file
when passed to the CLI via `--backend`, `--base-url`, etc.):

| Variable | Description |
|----------|-------------|
| `LLM_BACKEND` | `vllm`, `llamacpp`, or `api` |
| `LLM_BASE_URL` | e.g. `http://localhost:8000/v1` |
| `LLM_MODEL` | Model name or path |
| `LLM_API_KEY` | API key (use `EMPTY` for local servers) |

---

## Usage

### 1. Generate synthetic map worlds (no LLM needed)

```bash
python scripts/run_pipeline.py --stage worlds --num-worlds 100
```

This writes 100 JSON scenario files to `scenarios/`.
Each world contains vehicles, cameras, polygon zones, road segments,
and timestamped vehicle trajectories — all generated deterministically
from a seed.

### 2. Run the full pipeline

```bash
# Using a vLLM server
python scripts/run_pipeline.py \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-70B-Instruct

# Using an OpenAI-compatible API (e.g. OpenAI, Together, Groq)
python scripts/run_pipeline.py \
  --backend api \
  --model gpt-4o \
  --api-key sk-...

# Using a llama.cpp server
python scripts/run_pipeline.py \
  --backend llamacpp \
  --base-url http://localhost:8080
```

The pipeline will:

1. Load (or generate) synthetic worlds from `scenarios/`
2. Generate task blueprints via the LLM
3. Produce gold tool-call traces (deterministic + LLM)
4. Generate user utterance variants (Vietnamese, English, code-mixed)
5. Create negative / adversarial test cases
6. Run the 6-gate verification pipeline
7. Deduplicate and split into train / dev / test
8. Export JSONL datasets to `datasets/`
9. Write metrics to `reports/pipeline_metrics.json`

### 3. CLI reference

```
python scripts/run_pipeline.py --help

Options:
  -c, --config TEXT          Path to YAML config [default: pipeline_config.yaml]
  -b, --backend TEXT         LLM backend: vllm | llamacpp | api
      --base-url TEXT        LLM server URL
  -m, --model TEXT           Model name or path
      --api-key TEXT         API key
      --num-worlds INTEGER   Number of worlds to generate
      --target-samples INT   Overall target sample count
  -s, --stage TEXT           Run a single stage: worlds | full
  -o, --output-dir TEXT      Output directory for datasets
      --profile TEXT         Domain profile YAML override
```

---

## Domain Profiles

The current traffic-map setup is defined in `profiles/traffic_map.yaml`.
The profile is the source of truth for domain vocabulary and generation
wiring that used to be spread across Python modules:

- object types, object collections, geometry expectations, and selection slots
- active layers, time ranges, task families, difficulty guidance, and prompt rules
- assertion names, negative-case strategies, and adversarial prompt patterns
- declarative gold-plan templates for common task families
- the versioned tool catalog used for valid tool names and parameter enums

Use a profile override to run the same pipeline with a different domain:

```bash
python scripts/run_pipeline.py --profile profiles/traffic_map.yaml
```

### Customize without code

Common changes only require editing YAML/JSON:

- Add an object type by adding an `object_types` entry with `collection`,
  `geometry`, and `selection_slot`.
- Add a task family by adding `task_families` plus a `plan_templates` entry
  that calls existing tools.
- Change valid time windows by editing `time_ranges`; validators and prompts
  consume the profile at runtime.
- Change tool names, descriptions, and parameter enums in `tool_catalogs/*.json`;
  blueprint prompts and static validation read the catalog.

New executable behavior still requires Python: implement the handler in
`src/mock_environment/tools.py`, register it in `TOOL_REGISTRY`, then reference
that tool from the catalog/profile.

---

## Pipeline stages

```
Canonical Tool Catalog
        |
Synthetic World Generator        (deterministic geometry + trajectories)
        |
Blueprint Generator              (LLM: task type, target, difficulty)
        |
Gold Plan Generator              (deterministic templates + LLM fallback)
        |
Mock Environment Execution       (deterministic: validate tool traces)
        |
User Utterance Generator         (LLM: VI/EN/code-mixed paraphrases)
        |
Negative / Adversarial Generator (deterministic mutations + LLM)
        |
Verification Pipeline            (6 gates — see below)
        |
Deduplication + Difficulty Label
        |
Final Dataset Export (JSONL)
```

### Verification gates

| Gate | Type | What it checks |
|------|------|----------------|
| 1. Schema | deterministic | JSON structure, required fields |
| 2. Static consistency | deterministic | Object IDs exist in scenario, types match, valid time ranges |
| 3. Execution | deterministic | Gold trace runs without error in the mock environment |
| 4. Final-state assertions | deterministic | Artifacts, highlights, popups match expectations |
| 5. Semantic verification | LLM | User request matches blueprint intent, no hallucinations |
| 6. Human review | manual | Sampled subset exported to `reports/review_queue.jsonl` |

---

## Output datasets

After the pipeline completes, `datasets/` contains:

| File | Contents |
|------|----------|
| `train.jsonl` | Training split (scenarios 1–80) |
| `dev.jsonl` | Dev split (scenarios 81–90) |
| `test_easy.jsonl` | L0–L2 test samples (scenarios 91–100) |
| `test_hard.jsonl` | L3–L4 test samples |
| `test_adversarial.jsonl` | L5 + negative cases |

Each line is a self-contained JSON object:

```json
{
  "id": "mapagent_a1b2c3d4",
  "scenario_id": "traffic_synthetic_042",
  "tool_catalog_version": "map_tools_v0.1",
  "language": "vi",
  "difficulty": "L2",
  "task_type": "draw_vehicle_trajectory",
  "user_request": "Ve quy dao cua xe nay trong 10 phut gan nhat",
  "initial_state": { "time": "...", "selected": {...}, ... },
  "available_tools": ["get_vehicle_trajectory", "draw_polyline", ...],
  "gold_trace": [ {"tool": "...", "args": {...}}, ... ],
  "expected": { "final_state_assertions": [...], ... },
  "tags": ["selected_reference", "vehicle", "trajectory"],
  "generation_metadata": { "verified_by_schema": true, ... }
}
```

---

## Repository layout

```
map-use-agent/
  pyproject.toml               Project metadata and dependencies
  requirements.txt             Pinned dependency list
  pipeline_config.yaml         Default pipeline configuration
  Dockerfile                   Container build

  profiles/                    Domain profiles (default: traffic_map.yaml)
  schemas/                     JSON Schemas (auto-generated from Pydantic)
  tool_catalogs/               Versioned canonical tool definitions
  scenarios/                   Generated synthetic map worlds
  datasets/                    Final exported JSONL splits
  reports/                     Pipeline metrics and human review queue

  src/
    schemas.py                 Pydantic models (source of truth)
    domain/                    Profile loading and generic scenario access
    llm_backend/               LLM abstraction (vLLM, llama.cpp, API)
    world_generator/           Deterministic world + trajectory generation
    blueprint_generator/       LLM-driven task blueprint generation
    gold_plan_generator/       Canonical tool trace generation
    mock_environment/          Executable sandbox (state, database, tools)
    utterance_generator/       LLM-driven multilingual paraphrasing
    negative_generator/        Adversarial / negative case generation
    verification/              6-gate verification pipeline
    dedup/                     5-level deduplication
    export/                    JSONL dataset exporter
    pipeline/                  Orchestrator + YAML config loader

  scripts/
    run_pipeline.py            CLI entry-point
    export_schemas.py          Regenerate JSON Schemas from Pydantic models
```

---

## Extending

### Adding a new tool to the catalog

1. Add the tool definition to `tool_catalogs/map_tools_v0_2.json`
2. Implement the handler in `src/mock_environment/tools.py` and register it in `TOOL_REGISTRY`
3. Reference the tool catalog from a profile
4. Add a declarative `plan_templates` entry if an existing task family can use it

Static validation reads tool names and parameter enums from the catalog, so no
separate `VALID_TOOL_NAMES` edit is needed.

### Adding a new LLM backend

1. Create `src/llm_backend/my_backend.py` implementing `LLMBackend`
2. Register it in `src/llm_backend/config.py` `create_backend()`

### Connecting to a production map API

Write an adapter that maps canonical tool calls to your real frontend/backend
API. The canonical tool catalog is deliberately decoupled from any specific
map library (Leaflet, Mapbox, OpenLayers, etc.).

---

## License

See [LICENSE](LICENSE) for details.
