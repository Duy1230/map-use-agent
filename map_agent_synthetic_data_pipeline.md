# Quy trình sinh dữ liệu synthetic cho State-Aware Map Tool-Use Agent

## 1. Bối cảnh và mục tiêu

Dự án cần đánh giá các mô hình LLM cho bài toán **state-aware map tool-use agent**: Agent không có vision, không điều khiển bản đồ như người dùng thật, mà chỉ quan sát được:

- Yêu cầu tự nhiên của người dùng.
- Trạng thái bản đồ hiện tại ở dạng có cấu trúc.
- Danh sách tool hiện có và schema của chúng.

Agent cần hiểu intent, resolve đối tượng từ state hoặc ID/tên do người dùng cung cấp, gọi tool phù hợp, và tạo ra trạng thái bản đồ cuối đúng.

Vì hiện tại chưa có bản đồ production và chưa biết API frontend/backend thật của team dev, dataset nên được thiết kế theo hướng **canonical semantic tools** thay vì phụ thuộc vào API thật. Sau này có thể viết adapter từ canonical tools sang API production.

Mục tiêu của pipeline:

1. Sinh dữ liệu đánh giá chất lượng cao cho các model hiện tại.
2. Sinh dữ liệu synthetic quy mô lớn hơn để phục vụ SFT/RL/data augmentation sau này.
3. Đảm bảo dữ liệu có thể kiểm chứng bằng simulator/mock environment.
4. Giữ format tương thích với API thật trong tương lai thông qua adapter.

---

## 2. Nguyên tắc thiết kế chính

Không nên sinh dữ liệu trực tiếp theo dạng:

```text
user prompt -> expected answer
```

Thay vào đó, nên sinh theo pipeline có thể thực thi:

```text
synthetic world
-> task blueprint
-> gold semantic plan
-> mock tool execution
-> final state assertions
-> user utterances
-> verification
-> final dataset
```

Nguyên tắc cốt lõi:

> LLM sinh ý tưởng, task, paraphrase và biến thể ngôn ngữ; simulator và validator quyết định dữ liệu có đúng hay không.

LLM không nên là nguồn sự thật cuối cùng cho:

- Object ID có tồn tại hay không.
- Object có đúng type không.
- Polygon có chứa object không.
- Trajectory có hợp lệ không.
- Tool call có execute được không.
- Final map state có đúng không.

Các phần này phải được kiểm tra bằng code deterministic.

---

## 3. Kiến trúc tổng thể của pipeline

```text
Canonical Tool Catalog
        ↓
Synthetic Map World Generator
        ↓
Task Blueprint Generator
        ↓
Gold Plan / Gold State Generator
        ↓
Mock Environment Execution
        ↓
User Utterance Generator
        ↓
Negative / Adversarial Case Generator
        ↓
Verification Pipeline
        ↓
Deduplication + Difficulty Labeling
        ↓
Human Review Sampling
        ↓
Final Dataset Export
```

---

## 4. Canonical Tool Catalog

Vì chưa biết API thật của dev, cần định nghĩa một bộ tool chuẩn ở tầng semantic, gọi là **Canonical Map Interaction API**.

Các tool này không phụ thuộc vào Leaflet, Mapbox, OpenLayers, Redux hay frontend cụ thể nào.

Ví dụ nhóm tool MVP:

### 4.1. State tools

```python
get_current_map_state()
get_selected_entity(entity_type: str | None)
resolve_entity(reference: str, expected_type: str | None)
```

### 4.2. Object tools

```python
get_object_info(object_id: str)
get_vehicle_trajectory(vehicle_id: str, time_range: str)
get_nearby_objects(object_id: str, object_type: str, radius_meters: float)
```

### 4.3. Polygon/line tools

```python
get_polygon_area(polygon_id: str)
get_line_length(line_id: str)
get_objects_inside_polygon(polygon_id: str, object_type: str)
get_objects_crossing_line(line_id: str, object_type: str, time_range: str)
```

### 4.4. Map action tools

```python
highlight_objects(object_ids: list[str])
draw_polyline(points: list, label: str, source_object_id: str | None = None)
draw_polygon(geometry: dict, label: str)
draw_marker(coordinates: list[float], label: str)
show_popup(target_id: str, content: str)
clear_artifacts(scope: str)
```

Tool catalog cần được version hóa:

```text
map_tools_v0.1
map_tools_v0.2
map_tools_v1.0
```

Sau này khi team dev cung cấp API thật, chỉ cần viết adapter:

```text
canonical tool -> production frontend/backend API
```

---

## 5. Synthetic Map World Generator

Synthetic world là môi trường bản đồ giả lập. Một world nên chứa:

- Vehicles.
- Cameras.
- Road segments hoặc lines.
- Zones hoặc polygons.
- Vehicle trajectories theo thời gian.
- Object aliases.
- Active layers.
- Possible selected entities.
- Drawn artifacts ban đầu nếu cần.

Ví dụ world file:

```json
{
  "scenario_id": "traffic_synthetic_001",
  "coordinate_system": "EPSG:4326",
  "time": "2026-05-18T09:15:00+07:00",
  "objects": {
    "vehicles": [
      {
        "id": "vehicle_102",
        "type": "vehicle",
        "name": "Truck 102",
        "aliases": ["xe 102", "truck 102", "51A-102.88"],
        "geometry": {
          "type": "Point",
          "coordinates": [106.7001, 10.7762]
        },
        "properties": {
          "speed_kmh": 42,
          "status": "moving",
          "heading_deg": 35
        }
      }
    ],
    "cameras": [
      {
        "id": "camera_03",
        "type": "camera",
        "name": "Camera 03",
        "aliases": ["cam 03", "camera số 3"],
        "geometry": {
          "type": "Point",
          "coordinates": [106.7010, 10.7770]
        }
      }
    ],
    "polygons": [
      {
        "id": "zone_07",
        "type": "polygon",
        "name": "Zone 07",
        "geometry": {
          "type": "Polygon",
          "coordinates": [
            [
              [106.699, 10.775],
              [106.703, 10.775],
              [106.703, 10.779],
              [106.699, 10.779],
              [106.699, 10.775]
            ]
          ]
        }
      }
    ]
  },
  "time_series": {
    "vehicle_102": {
      "trajectory": [
        {
          "timestamp": "2026-05-18T09:05:00+07:00",
          "coordinates": [106.7001, 10.7762]
        },
        {
          "timestamp": "2026-05-18T09:10:00+07:00",
          "coordinates": [106.7005, 10.7769]
        },
        {
          "timestamp": "2026-05-18T09:15:00+07:00",
          "coordinates": [106.7012, 10.7775]
        }
      ]
    }
  }
}
```

Khuyến nghị:

- Geometry và trajectory nên sinh bằng code deterministic.
- LLM có thể hỗ trợ sinh metadata, aliases, mô tả scenario, nhưng không nên tự bịa geometry phức tạp.
- Dùng GeoJSON cho geometry.
- Coordinate order thống nhất: `[longitude, latitude]`.
- Thời gian dùng ISO 8601.
- Distance dùng meters.
- Area dùng square meters.

---

## 6. Task Blueprint Generator

LLM không nên sinh thẳng user request cuối cùng. Trước hết cần sinh **task blueprint**.

Blueprint là biểu diễn có cấu trúc của task, gồm:

- Task type.
- Difficulty.
- Target entity.
- Reference mode.
- Constraints.
- Required semantic steps.
- Expected final state.
- Negative/positive flag.
- Clarification requirement nếu thiếu context.

Ví dụ blueprint:

```json
{
  "task_type": "draw_vehicle_trajectory",
  "difficulty": "L2",
  "target": {
    "reference_mode": "selected_object",
    "expected_type": "vehicle",
    "object_id": "vehicle_102"
  },
  "constraints": {
    "time_range": "last_10_minutes"
  },
  "required_semantic_steps": [
    "resolve_selected_vehicle",
    "retrieve_vehicle_trajectory",
    "draw_trajectory_polyline"
  ],
  "expected_final_state": [
    {
      "type": "artifact_exists",
      "artifact_type": "polyline",
      "source_object_id": "vehicle_102"
    }
  ],
  "should_ask_clarification": false,
  "negative_case": false
}
```

Các nhóm blueprint nên có:

1. Explicit ID reference.
2. Selected object / selected line / selected polygon.
3. Missing reference.
4. Type mismatch.
5. Multi-step spatial query.
6. Temporal query.
7. Artifact-aware query.
8. Multi-turn query.
9. Unsupported hoặc dangerous action.

---

## 7. Gold Plan / Gold State Generator

Từ blueprint, sinh gold plan ở dạng canonical tool trace.

Ví dụ:

```json
[
  {
    "tool": "get_current_map_state",
    "args": {}
  },
  {
    "tool": "get_vehicle_trajectory",
    "args": {
      "vehicle_id": "vehicle_102",
      "time_range": "last_10_minutes"
    }
  },
  {
    "tool": "draw_polyline",
    "args": {
      "points": "$get_vehicle_trajectory.points",
      "label": "Trajectory of vehicle_102",
      "source_object_id": "vehicle_102"
    }
  }
]
```

Tuy nhiên, gold trace không nên là oracle duy nhất. Model có thể gọi thêm tool hợp lệ mà vẫn đúng.

Vì vậy expected section nên gồm:

```json
{
  "required_semantic_steps": [
    "resolve_target_vehicle",
    "retrieve_vehicle_trajectory",
    "draw_polyline_for_trajectory"
  ],
  "allowed_extra_tools": [
    "get_object_info",
    "resolve_entity"
  ],
  "forbidden_tools": [
    "delete_object",
    "clear_all_layers"
  ],
  "final_state_assertions": [
    {
      "type": "artifact_exists",
      "artifact_type": "polyline",
      "source_object_id": "vehicle_102"
    }
  ]
}
```

Final state assertions nên là tiêu chuẩn chính để chấm task success.

---

## 8. Mock Environment Execution

Mock environment là thành phần kiểm chứng dữ liệu.

Nó cần execute được canonical tools và cập nhật state.

Ví dụ skeleton:

```python
class MockMapEnvironment:
    def __init__(self, scenario, initial_state):
        self.scenario = scenario
        self.state = initial_state

    def get_current_map_state(self):
        return self.state.visible_to_agent()

    def get_vehicle_trajectory(self, vehicle_id, time_range):
        # Validate vehicle_id
        # Read trajectory from hidden scenario database
        # Filter by time_range
        # Return points
        ...

    def draw_polyline(self, points, label, source_object_id=None):
        artifact_id = self.state.add_artifact(
            type="polyline",
            geometry={
                "type": "LineString",
                "coordinates": points
            },
            label=label,
            source_object_id=source_object_id
        )
        return {"artifact_id": artifact_id}

    def highlight_objects(self, object_ids):
        # Validate object IDs
        self.state.highlighted_objects.update(object_ids)
        return {"highlighted": object_ids}
```

Mỗi candidate sample phải được chạy qua mock environment. Nếu execution lỗi, cần discard hoặc đưa lại cho LLM sửa.

Mock environment giúp phát hiện:

- Object ID không tồn tại.
- Tool args sai schema.
- Type mismatch.
- Gold plan không execute được.
- Final state không khớp expected assertions.

---

## 9. User Utterance Generator

Sau khi blueprint và gold plan đã valid, mới sinh user utterance.

Input cho LLM:

```json
{
  "task_type": "draw_vehicle_trajectory",
  "target_reference_mode": "selected_object",
  "target_type": "vehicle",
  "time_range": "last_10_minutes",
  "language": "vi",
  "style": "casual"
}
```

Output:

```json
{
  "utterances": [
    "Vẽ quỹ đạo của xe này trong 10 phút gần nhất",
    "Cho tôi xem đường đi của xe này 10 phút vừa rồi",
    "Xe này vừa chạy như thế nào trong 10 phút qua?",
    "Draw trajectory cho xe này trong last 10 minutes"
  ]
}
```

Nên sinh nhiều style:

- Vietnamese formal.
- Vietnamese casual.
- English.
- Vietnamese-English code-mixed.
- Short command.
- Long natural request.
- Noisy/typo variants.

Quy tắc quan trọng:

- Nếu `reference_mode = selected_object`, user utterance nên dùng “xe này”, “vùng này”, “đường này”, không nên tự thêm ID.
- Không được thêm constraint mới ngoài blueprint.
- Không được đổi task intent.
- Không được tạo reference mà state không resolve được, trừ khi đó là negative case.

---

## 10. Negative và adversarial cases

Negative cases là phần bắt buộc để kiểm tra model có biết không gọi tool khi thiếu context hay không.

### 10.1. Missing selected object

```json
{
  "user_request": "Vẽ trajectory của xe này",
  "initial_state": {
    "selected": {
      "object": null,
      "line": null,
      "polygon": null
    }
  },
  "expected": {
    "should_ask_clarification": true,
    "forbidden_tools": ["get_vehicle_trajectory", "draw_polyline"]
  }
}
```

### 10.2. Type mismatch

```json
{
  "user_request": "Vẽ trajectory của xe này",
  "initial_state": {
    "selected": {
      "object": {
        "id": "camera_03",
        "type": "camera"
      }
    }
  },
  "expected": {
    "should_ask_clarification_or_refuse": true,
    "forbidden_tool_calls": [
      {
        "tool": "get_vehicle_trajectory",
        "args_contains": {
          "vehicle_id": "camera_03"
        }
      }
    ]
  }
}
```

### 10.3. Multiple selected objects

```json
{
  "user_request": "Vẽ trajectory của xe này",
  "initial_state": {
    "selected": {
      "objects": [
        {"id": "vehicle_102", "type": "vehicle"},
        {"id": "vehicle_205", "type": "vehicle"}
      ]
    }
  },
  "expected": {
    "should_ask_clarification": true
  }
}
```

### 10.4. Unsupported visual reference

```json
{
  "user_request": "Highlight cái xe màu đỏ bên trái",
  "initial_state": {
    "selected": {
      "object": null
    }
  },
  "expected": {
    "should_ask_clarification": true,
    "reason": "agent_has_no_vision"
  }
}
```

Các nhóm negative/adversarial nên có:

- Missing selected entity.
- Wrong selected type.
- Multiple selected entities.
- Object ID không tồn tại.
- Visual reference không thể resolve.
- Tool confusion.
- Unsupported action.
- Dangerous action cần xác nhận hoặc từ chối.

---

## 11. Verification Pipeline

Candidate data cần đi qua nhiều gate kiểm chứng.

### Gate 1 — Schema validation

Kiểm tra JSON và schema:

- Task có đủ field không?
- Initial state đúng schema không?
- Tool args đúng schema không?
- Expected assertions hợp lệ không?

### Gate 2 — Static consistency check

Kiểm tra logic không cần execute:

- Object ID có tồn tại trong scenario không?
- Selected object có đúng type không?
- Time range có thuộc enum không?
- Polygon ID có phải polygon không?
- Line ID có phải line không?
- Task type có khớp expected tools không?

### Gate 3 — Execution check

Chạy gold plan trong mock environment:

- Tool có execute thành công không?
- Final state có artifact/highlight/popup đúng không?
- Có tool nào trả lỗi không?

### Gate 4 — Final state assertion check

So sánh final state với expected assertions:

- `artifact_exists`
- `object_highlighted`
- `objects_highlighted`
- `popup_shown`
- `artifact_removed`
- `polyline_source_matches_vehicle`
- `objects_inside_polygon_highlighted`

### Gate 5 — LLM semantic verifier

Sau các check deterministic, dùng LLM verifier để kiểm tra semantic.

Input:

```text
user_request
initial_state
tool trace
final_state
expected behavior
```

Output mong muốn:

```json
{
  "valid": true,
  "issues": [],
  "suggested_fix": null
}
```

LLM verifier chỉ là lớp bổ sung, không phải gate duy nhất.

### Gate 6 — Human review sampling

Khuyến nghị:

- Evaluation data: review thủ công 30–50% ở giai đoạn đầu, sau đó 10–20% khi pipeline ổn định.
- Training data: review 1–5%, ưu tiên sample khó, negative, multi-turn.

---

## 12. Multi-turn synthetic data

Sau khi single-turn ổn định mới mở rộng multi-turn.

Ví dụ multi-turn sample:

```json
{
  "id": "mapagent_mt_001",
  "scenario_id": "traffic_synthetic_001",
  "turns": [
    {
      "user": "Vẽ trajectory của xe này",
      "expected_steps": [
        "resolve_selected_vehicle",
        "draw_trajectory"
      ]
    },
    {
      "user": "Bây giờ chỉ giữ đoạn 5 phút gần nhất",
      "expected_steps": [
        "resolve_previous_trajectory_artifact",
        "retrieve_vehicle_trajectory_last_5_minutes",
        "replace_polyline"
      ]
    },
    {
      "user": "Highlight các camera nó đã đi qua",
      "expected_steps": [
        "resolve_source_vehicle_from_artifact",
        "query_crossed_cameras",
        "highlight_cameras"
      ]
    }
  ],
  "initial_state": {
    "selected": {
      "object": {
        "id": "vehicle_102",
        "type": "vehicle"
      }
    }
  },
  "final_state_assertions": [
    {
      "type": "artifact_exists",
      "artifact_type": "polyline",
      "source_object_id": "vehicle_102",
      "time_range": "last_5_minutes"
    },
    {
      "type": "objects_highlighted",
      "object_type": "camera"
    }
  ]
}
```

Multi-turn cũng nên theo nguyên tắc:

```text
blueprint first -> dialogue later -> execution verification -> final state check
```

---

## 13. Dataset export format

Mỗi final sample nên là một dòng JSONL.

Ví dụ single-turn sample:

```json
{
  "id": "mapagent_000137",
  "scenario_id": "traffic_synthetic_001",
  "tool_catalog_version": "map_tools_v0.1",
  "language": "vi",
  "difficulty": "L2",
  "task_type": "draw_vehicle_trajectory",
  "user_request": "Vẽ quỹ đạo của xe này trong 10 phút gần nhất",
  "initial_state": {
    "time": "2026-05-18T09:15:00+07:00",
    "selected": {
      "object": {
        "id": "vehicle_102",
        "type": "vehicle",
        "name": "Truck 102"
      },
      "line": null,
      "polygon": null
    },
    "active_layers": ["vehicles", "roads"],
    "drawn_artifacts": []
  },
  "available_tools": [
    "get_current_map_state",
    "get_vehicle_trajectory",
    "draw_polyline"
  ],
  "gold_trace": [
    {
      "tool": "get_current_map_state",
      "args": {}
    },
    {
      "tool": "get_vehicle_trajectory",
      "args": {
        "vehicle_id": "vehicle_102",
        "time_range": "last_10_minutes"
      }
    },
    {
      "tool": "draw_polyline",
      "args": {
        "points": "$get_vehicle_trajectory.points",
        "label": "Trajectory of vehicle_102",
        "source_object_id": "vehicle_102"
      }
    }
  ],
  "expected": {
    "required_semantic_steps": [
      "resolve_selected_vehicle",
      "retrieve_vehicle_trajectory",
      "draw_trajectory_polyline"
    ],
    "allowed_extra_tools": [
      "get_object_info",
      "resolve_entity"
    ],
    "forbidden_tools": [],
    "final_state_assertions": [
      {
        "type": "artifact_exists",
        "artifact_type": "polyline",
        "source_object_id": "vehicle_102"
      }
    ]
  },
  "tags": [
    "selected_reference",
    "vehicle",
    "trajectory",
    "temporal",
    "multi_tool"
  ],
  "generation_metadata": {
    "generator_model": "your_llm_name",
    "blueprint_id": "bp_000137",
    "verified_by_schema": true,
    "verified_by_execution": true,
    "verified_by_llm": true,
    "human_reviewed": false
  }
}
```

---

## 14. Difficulty levels

Đề xuất 6 level:

```text
L0 — Clarification/no-tool task
L1 — Single tool with explicit ID
L2 — Selected reference resolution
L3 — Multi-step tool chain
L4 — Multi-turn or artifact-aware interaction
L5 — Ambiguity, type mismatch, adversarial phrasing
```

Phân bổ MVP khoảng 1,000 samples:

| Level | Số lượng | Mục tiêu |
|---|---:|---|
| L0 | 150 | Biết hỏi lại, không hallucinate |
| L1 | 150 | Gọi đúng tool với ID rõ |
| L2 | 250 | Resolve “này/đó” từ state |
| L3 | 200 | Tool chaining |
| L4 | 150 | Multi-turn + artifact state |
| L5 | 100 | Robustness, type mismatch, ambiguity |

Phân bổ theo task type:

| Task type | Tỷ lệ gợi ý |
|---|---:|
| draw_vehicle_trajectory | 20% |
| show_object_info | 10% |
| highlight_selected_entity | 10% |
| objects_inside_polygon | 15% |
| objects_crossing_line | 10% |
| polygon_area / line_length | 10% |
| nearby_cameras / nearby_objects | 10% |
| artifact modification / deletion | 10% |
| unsupported / ambiguity | 5% |

---

## 15. Deduplication và leakage control

Synthetic data rất dễ bị trùng pattern. Cần dedup ở nhiều mức:

1. Exact text duplicate.
2. Same normalized user request.
3. Same `task_type + target + time_range`.
4. Same gold trace structure.
5. Same scenario leakage across train/dev/test.

Nên split theo scenario, không split ngẫu nhiên theo sample:

```text
train_scenarios: traffic_synthetic_001 -> 080
dev_scenarios:   traffic_synthetic_081 -> 090
test_scenarios:  traffic_synthetic_091 -> 100
```

Không nên để cùng một synthetic world xuất hiện ở cả train và test.

---

## 16. Metrics cho data generation pipeline

Cần đo chất lượng pipeline sinh dữ liệu, không chỉ đo model.

Các metric nên log:

```text
Candidate acceptance rate
Schema pass rate
Static validation pass rate
Execution pass rate
Final-state assertion pass rate
Semantic verifier pass rate
Human review pass rate
Duplicate rate
Negative case correctness rate
Average tool steps per task
Distribution by task type
Distribution by difficulty
Distribution by language/style
```

Ví dụ report:

```text
Generated candidates:        10,000
Schema valid:                 9,400  94%
Static valid:                 8,700  87%
Execution valid:              7,900  79%
Semantic verified:            7,100  71%
After dedup:                  5,600  56%
Human review pass:              92%
```

Nếu execution pass rate thấp, generator đang bịa tool args hoặc object IDs. Nếu semantic verifier pass rate thấp, blueprint và utterance thường không khớp.

---

## 17. Prompt mẫu cho LLM sinh blueprint

```text
You are generating synthetic evaluation blueprints for a state-aware map tool-use agent.

The agent has no vision. It can only use:
1. User request
2. Structured map state
3. Available tools

Generate a task blueprint, not a natural language user request yet.

Rules:
- Do not invent object IDs outside the provided scenario.
- If the task uses "this vehicle", the initial state must contain a selected vehicle.
- If the task uses "this polygon", the initial state must contain a selected polygon.
- For negative cases, explicitly specify why the agent must ask for clarification.
- Every task must have final_state_assertions.
- Prefer semantic correctness over linguistic diversity.

Return JSON only.

Scenario:
{scenario_json}

Available tools:
{tool_catalog_json}

Requested task family:
{task_family}

Difficulty:
{difficulty}
```

---

## 18. Prompt mẫu cho LLM sinh user utterance

```text
Generate natural user requests for the following map-agent task blueprint.

Rules:
- Preserve the exact intent.
- Do not add new constraints.
- Do not mention object ID if reference_mode = selected_object.
- Use deictic expressions such as "xe này", "vùng này", "đường này" when appropriate.
- Generate Vietnamese, English, and Vietnamese-English code-mixed variants.
- Some variants can be short commands.
- Some variants can be natural conversational requests.
- Return JSON only.

Blueprint:
{blueprint_json}
```

---

## 19. Prompt mẫu cho LLM semantic verifier

```text
You are a strict verifier for a synthetic map tool-use dataset.

The agent has no vision and must not infer objects from visual descriptions.
It can only use structured state and tools.

Check whether the candidate sample is semantically valid.

Evaluate:
1. Does the user request match the blueprint?
2. Is the target object resolvable from the initial state or explicit ID/name?
3. Are the required semantic steps correct?
4. Are final_state_assertions sufficient?
5. Are there any hallucinated object IDs, geometries, tools, or constraints?
6. For negative cases, should the agent ask clarification instead of calling tools?

Return JSON:
{
  "valid": true/false,
  "issues": [...],
  "suggested_fix": ...
}

Candidate:
{candidate_json}
```

---

## 20. Quy trình triển khai MVP

### Week 1

- Định nghĩa `map_tools_v0.1`.
- Viết `map_state.schema.json`.
- Viết `task.schema.json`.
- Viết 3 synthetic worlds bằng code.
- Viết mock environment tối giản.

### Week 2

- Viết blueprint generator prompt.
- Sinh 200 candidate blueprints.
- Viết static validator.
- Viết execution validator.
- Sinh 100 sample evaluation đầu tiên.

### Week 3

- Thêm user utterance generator.
- Thêm negative/adversarial cases.
- Thêm LLM verifier.
- Mở rộng lên 500–700 samples.

### Week 4

- Chạy benchmark nhiều model.
- Phân tích lỗi theo task type.
- Điều chỉnh tool schema và state schema.
- Chuẩn hóa báo cáo benchmark.

---

## 21. Cấu trúc repo đề xuất

```text
map-agent-benchmark/
  README.md

  schemas/
    map_state.schema.json
    tool_catalog.schema.json
    task.schema.json
    final_state_assertion.schema.json

  tool_catalogs/
    map_tools_v0_1.json
    map_tools_v0_2.json

  scenarios/
    traffic_synthetic_001.json
    traffic_synthetic_002.json
    traffic_synthetic_003.json

  datasets/
    dev.jsonl
    test_easy.jsonl
    test_hard.jsonl
    test_multiturn.jsonl
    test_adversarial.jsonl

  generation/
    generate_worlds.py
    generate_blueprints.py
    generate_utterances.py
    generate_negative_cases.py

  mock_environment/
    environment.py
    tools.py
    state.py
    database.py

  evaluator/
    run_eval.py
    metrics.py
    assertions.py
    validators.py

  adapters/
    openai_function_calling.py
    anthropic_tool_use.py
    local_vllm_openai_compatible.py

  reports/
    model_results_template.md
```

---

## 22. Kết luận

Pipeline nên được thiết kế như một **executable synthetic data generation system**, không phải một bộ prompt/answer thủ công.

Format cốt lõi của mỗi sample:

```text
Task = {
  user_request,
  initial_map_state,
  available_tools,
  scenario_id,
  hidden/mock database,
  gold_trace,
  required_semantic_steps,
  final_state_assertions,
  response_assertions,
  tags,
  generation_metadata
}
```

Các năng lực cần đánh giá:

- Intent understanding.
- Reference resolution từ selected state hoặc ID/name.
- Không hallucinate object khi thiếu context.
- Tool selection correctness.
- Argument correctness.
- Multi-step tool chaining.
- Final map state correctness.
- Clarification behavior.
- Robustness trước ambiguity/type mismatch/adversarial prompts.

Thiết kế này giúp bạn đánh giá model ngay cả khi chưa có frontend thật, đồng thời vẫn giữ khả năng tích hợp sau này thông qua adapter từ canonical tools sang API production.
