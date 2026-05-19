# Map Use Agent Data Generation Manual

This manual explains how to use, configure, and extend the synthetic data
generation pipeline for state-aware map tool-use agents.

The pipeline creates datasets for agents that do not see the map visually. The
agent receives only structured map state, natural-language user requests, and a
catalog of available map tools. The pipeline combines deterministic synthetic
world generation with LLM-generated tasks and language, then validates every
candidate before exporting JSONL datasets.

## 1. What The Pipeline Produces

The final output is a set of JSONL dataset files under `datasets/`.

Typical output files:

| File | Purpose |
|---|---|
| `train.jsonl` | Training split |
| `dev.jsonl` | Development split |
| `test_easy.jsonl` | Easier L0-L2 evaluation cases |
| `test_hard.jsonl` | Harder L3-L4 cases |
| `test_adversarial.jsonl` | L5 and negative/clarification cases |

Each dataset line is a complete task sample with:

- scenario ID
- user request
- initial map state
- available tools
- gold tool-call trace
- expected final-state assertions
- tags and generation metadata

## 2. Repository Layout

Important paths:

| Path | Description |
|---|---|
| `pipeline_config.yaml` | Main pipeline configuration |
| `profiles/traffic_map.yaml` | Default domain profile |
| `tool_catalogs/map_tools_v0_1.json` | Canonical tool definitions |
| `src/schemas.py` | Pydantic source of truth for schemas |
| `schemas/` | Generated JSON Schema files |
| `src/pipeline/runner.py` | End-to-end pipeline orchestrator |
| `src/world_generator/` | Deterministic synthetic world generator |
| `src/mock_environment/` | Executable mock map environment |
| `src/verification/` | Validation gates |
| `scripts/run_pipeline.py` | CLI entry point |
| `scripts/export_schemas.py` | Regenerate JSON Schemas |

Generated artifacts:

| Path | Description |
|---|---|
| `scenarios/` | Synthetic world JSON files |
| `datasets/` | Exported JSONL datasets |
| `reports/` | Metrics and human-review queue |

## 3. Installation

Use Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

This installs runtime dependencies plus development tools such as `pytest` and
`ruff`.

You can also build the Docker image:

```bash
docker build -t map-use-agent .
```

## 4. Configuration Overview

The pipeline reads `pipeline_config.yaml` by default.

Key sections:

```yaml
llm:
  backend: "vllm"              # vllm | llamacpp | api
  base_url: "http://localhost:8000/v1"
  model: "meta-llama/Llama-3.1-70B-Instruct"
  api_key: "EMPTY"
  max_retries: 3
  retry_backoff_seconds: 1.0
  retry_backoff_max_seconds: 30.0
  reconnect_on_failure: true

worlds:
  count: 100
  base_seed: 42

generation:
  blueprints_per_scenario: 10
  utterances_per_blueprint: 6
  llm_adversarial_per_scenario: 3

verification:
  run_semantic_verifier: true
  human_review_rate: 0.10

scenarios_dir: "scenarios"
datasets_dir: "datasets"
reports_dir: "reports"
profile: "profiles/traffic_map.yaml"
schema_version: "legacy"

runtime:
  log_level: "INFO"
  log_file: null
  checkpoint_dir: "reports/checkpoints"
  resume: true
  reset_checkpoint: false
  checkpoint_every_scenario: true
```

CLI flags override the config file for common options:

```bash
python scripts/run_pipeline.py --backend api --model gpt-4o --api-key <key>
```

Supported LLM backends:

| Backend | Use When |
|---|---|
| `vllm` | You have a local vLLM OpenAI-compatible server |
| `llamacpp` | You have a local llama.cpp server |
| `api` | You use an OpenAI-compatible hosted API |

Environment variables can also be used by the backend factory:

| Variable | Description |
|---|---|
| `LLM_BACKEND` | `vllm`, `llamacpp`, or `api` |
| `LLM_BASE_URL` | OpenAI-compatible base URL |
| `LLM_MODEL` | Model name or local model identifier |
| `LLM_API_KEY` | API key, or `EMPTY` for local servers |

## 5. Quick Start

### Generate Worlds Only

This stage does not require an LLM.

```bash
python scripts/run_pipeline.py --stage worlds --num-worlds 5
```

Output:

```text
scenarios/traffic_synthetic_001.json
scenarios/traffic_synthetic_002.json
...
```

Use this first to confirm local setup is working.

### Run The Full Pipeline

With vLLM:

```bash
python scripts/run_pipeline.py \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-70B-Instruct
```

With an OpenAI-compatible API:

```bash
python scripts/run_pipeline.py \
  --backend api \
  --model gpt-4o \
  --api-key <your-api-key>
```

With llama.cpp:

```bash
python scripts/run_pipeline.py \
  --backend llamacpp \
  --base-url http://localhost:8080
```

## 6. Pipeline Stages

The full pipeline runs these stages:

1. Load or generate synthetic map worlds.
2. Generate structured task blueprints with an LLM.
3. Generate gold tool-call traces.
4. Generate negative and adversarial cases.
5. Generate natural-language user utterances.
6. Validate candidates through deterministic and optional LLM gates.
7. Deduplicate accepted samples.
8. Export train/dev/test JSONL splits.
9. Write reports and metrics.

The key design principle is:

```text
LLMs propose candidates; deterministic validators decide what is accepted.
```

## 7. Verification Gates

Every candidate can pass through these gates:

| Gate | Type | Purpose |
|---|---|---|
| Schema | deterministic | Validate required fields and JSON Schema |
| Static consistency | deterministic | Check tool names, object IDs, object types, time ranges |
| Execution | deterministic | Execute gold trace in the mock map environment |
| Final-state assertions | deterministic | Check artifacts, highlights, popups, and clarification behavior |
| Semantic verifier | LLM | Check user request and expected behavior match |
| Human review | manual | Export sampled cases for inspection |

Disable the LLM semantic verifier when you want a cheaper or faster run:

```yaml
verification:
  run_semantic_verifier: false
```

## 8. Domain Profiles

Domain profiles make the pipeline configurable without editing scattered Python
constants. The default profile is:

```text
profiles/traffic_map.yaml
```

A profile defines:

- object types and their scenario collections
- selection slots
- active layers
- valid time ranges
- task families
- difficulty guidance
- prompt rules
- utterance styles
- valid assertion types
- negative-case strategies
- tool type constraints
- declarative gold-plan templates
- the tool catalog path

Run with a profile:

```bash
python scripts/run_pipeline.py --profile profiles/traffic_map.yaml
```

Or set it in `pipeline_config.yaml`:

```yaml
profile: "profiles/traffic_map.yaml"
```

## 9. Customizing Without Code

Most domain changes should start in the profile or tool catalog.

### Add An Object Type

Add a new entry under `object_types`:

```yaml
object_types:
  sensor:
    collection: sensors
    geometry: Point
    selection_slot: object
    aliases: ["sensor", "device"]
```

Then your scenarios may include:

```json
{
  "objects_by_type": {
    "sensor": [
      {
        "id": "sensor_001",
        "type": "sensor",
        "name": "Sensor 001",
        "geometry": {"type": "Point", "coordinates": [106.7, 10.7]}
      }
    ]
  }
}
```

Legacy grouped objects such as `objects.vehicles` are still supported.

### Add A Time Range

Edit `time_ranges`:

```yaml
time_ranges:
  last_15_minutes: 15
```

Validators and database queries will use the configured value.

If the time range is also exposed as a tool parameter enum, update the tool
catalog too.

### Add A Task Family

Add a task family:

```yaml
task_families:
  - name: show_sensor_info
    difficulty: ["L1", "L2"]
    description: Show information about a sensor.
```

Then add a plan template if existing tools can satisfy it:

```yaml
plan_templates:
  show_sensor_info:
    - tool: get_current_map_state
      args: {}
    - tool: get_object_info
      args:
        object_id: "{{ target.object_id }}"
    - tool: show_popup
      args:
        target_id: "{{ target.object_id }}"
        content: "Info for {{ target.object_id }}"
```

Template values support blueprint paths:

```yaml
"{{ target.object_id }}"
"{{ constraints.time_range|last_10_minutes }}"
```

The value after `|` is the default when the blueprint path is missing.

### Add Or Change Tool Metadata

Edit `tool_catalogs/*.json`.

The tool catalog is the source of truth for:

- valid tool names
- tool descriptions
- parameter names
- parameter enums
- return schemas
- side-effect metadata

Static validation and blueprint prompts read from the selected catalog.

## 10. When Python Changes Are Still Needed

Profiles avoid many manual code edits, but executable behavior still needs
Python.

You need Python changes when you add:

- a brand-new tool behavior
- custom spatial logic
- a new world generator
- a custom assertion checker beyond the generic built-ins
- a new LLM backend

### Add A New Executable Tool

1. Add the tool metadata in `tool_catalogs/*.json`.
2. Implement the handler in `src/mock_environment/tools.py`.
3. Register the handler in `TOOL_REGISTRY`.
4. Reference the catalog from your profile.
5. Optionally add a `plan_templates` entry that uses the tool.
6. Add tests for the tool and any validation behavior.

## 11. Gold Plan Templates

Gold plans are canonical tool traces.

The profile can define plan templates for common task families:

```yaml
plan_templates:
  draw_vehicle_trajectory:
    - tool: get_current_map_state
      args: {}
    - tool: get_vehicle_trajectory
      args:
        vehicle_id: "{{ target.object_id }}"
        time_range: "{{ constraints.time_range|last_10_minutes }}"
    - tool: draw_polyline
      args:
        points: "$get_vehicle_trajectory.points"
        label: "Trajectory of {{ target.object_id }}"
        source_object_id: "{{ target.object_id }}"
```

Two kinds of references are supported:

| Syntax | Meaning |
|---|---|
| `{{ target.object_id }}` | Read from the blueprint |
| `$get_vehicle_trajectory.points` | Read from a previous tool result during execution |

Conditional steps are supported:

```yaml
- when:
    path: target.reference_mode
    equals: selected_object
  tool: get_selected_entity
  args:
    entity_type: object
```

If a task family has no declarative template, the system can fall back to
existing Python planners or LLM planning.

## 12. Dataset Schema Notes

Current schemas are generated from `src/schemas.py`.

Regenerate schemas after model changes:

```bash
python scripts/export_schemas.py
```

The schema policy is backward compatible:

- existing legacy scenarios are still valid
- generic `objects_by_type` scenarios are supported
- final-state assertion `type` is a string so profiles can define assertion names

Do not hand-edit generated schema files unless you are deliberately overriding
the generated output.

## 13. Quality Checks

Run these before committing changes:

```bash
python -m pytest -q
python -m ruff check src scripts tests
python scripts/run_pipeline.py --stage worlds --num-worlds 2
```

If you changed schemas:

```bash
python scripts/export_schemas.py
```

If `run_pipeline.py --stage worlds` creates temporary `scenarios/` files during
development, remove them unless you intend to commit generated scenarios.

## 14. Common Workflows

### Create A Small Deterministic World Sample

```bash
python scripts/run_pipeline.py --stage worlds --num-worlds 5
```

Use this to inspect scenario shape and object generation.

### Run A Small Full Pipeline Experiment

Edit `pipeline_config.yaml`:

```yaml
worlds:
  count: 2

generation:
  blueprints_per_scenario: 2
  utterances_per_blueprint: 2
  llm_adversarial_per_scenario: 1

verification:
  run_semantic_verifier: false
```

Then run:

```bash
python scripts/run_pipeline.py --backend api --model <model> --api-key <key>
```

### Use A Different Output Directory

```bash
python scripts/run_pipeline.py --output-dir datasets_experiment
```

### Use A Different Profile

```bash
python scripts/run_pipeline.py --profile profiles/my_domain.yaml
```

## 15. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run commands from the repository root, or install the project:

```bash
python -m pip install -e ".[dev]"
```

### No LLM Server Is Available

Use worlds-only mode:

```bash
python scripts/run_pipeline.py --stage worlds --num-worlds 5
```

The full pipeline requires an LLM backend.

### Static Validation Rejects A Tool

Check that the selected profile points to the intended tool catalog:

```yaml
tool_catalog: ../tool_catalogs/map_tools_v0_1.json
```

Then confirm the tool exists in that catalog and, if executable, in
`TOOL_REGISTRY`.

### Static Validation Rejects A Time Range

Make sure the time range exists in both places when needed:

- `profiles/<profile>.yaml` under `time_ranges`
- `tool_catalogs/*.json` as the parameter enum for tools that expose it

### Execution Fails With Unknown Tool

The tool exists in the catalog but has no Python handler. Implement and register
the handler in `src/mock_environment/tools.py`.

### Final-State Assertion Is Unknown

Add the assertion name to the profile `assertions` list. If it requires custom
logic, implement the checker before using it in accepted samples.

### Full Pipeline Produces Few Or No Samples

Check `reports/pipeline_metrics.json`. Low pass rates usually point to:

- schema mismatch
- LLM-generated object IDs that do not exist
- invalid tool arguments
- final-state assertions that do not match execution
- semantic verifier rejecting utterances

Temporarily disabling `run_semantic_verifier` can help isolate deterministic
pipeline issues from LLM semantic-review issues.

## 17. Checkpoints, Resume, And Reconnect

Long full-pipeline runs are resumable. By default, the runner writes
scenario-level checkpoints to:

```text
reports/checkpoints/
```

Checkpoint files:

| File | Description |
|---|---|
| `run_state.json` | Fingerprint, completed scenarios, metrics snapshot |
| `candidates.jsonl` | Generated candidates saved after each completed scenario |
| `verified.jsonl` | Verified samples appended during verification |

If the process stops after scenario 20, rerun the same command and scenarios
1-20 will be skipped. The runner checks a fingerprint built from the config,
profile, and tool catalog. If those inputs changed, resume is rejected so old
and new candidates are not mixed.

Useful flags:

```bash
python scripts/run_pipeline.py --no-resume
python scripts/run_pipeline.py --reset-checkpoint
python scripts/run_pipeline.py --checkpoint-dir reports/checkpoints_experiment
```

Logging flags:

```bash
python scripts/run_pipeline.py --log-level DEBUG
python scripts/run_pipeline.py --log-file reports/pipeline.log
```

For llama.cpp, transient server disconnects are retried automatically. The
backend catches connection, read, timeout, protocol, and transient 5xx failures,
then recreates the HTTP client before retrying when `reconnect_on_failure` is
enabled.

Configure retry behavior:

```yaml
llm:
  backend: "llamacpp"
  base_url: "http://localhost:8080"
  max_retries: 5
  retry_backoff_seconds: 1.0
  retry_backoff_max_seconds: 30.0
  reconnect_on_failure: true
```

## 16. Best Practices

- Start with worlds-only generation before running the full pipeline.
- Keep profile changes small and test them with a tiny run.
- Treat `src/schemas.py` as the schema source of truth.
- Regenerate schemas after schema model changes.
- Prefer profile and catalog edits over Python edits for domain vocabulary.
- Add tests before changing validators, mock tools, or schema behavior.
- Do not commit API keys, private URLs, generated datasets, or temporary reports
  unless the project explicitly needs them.
