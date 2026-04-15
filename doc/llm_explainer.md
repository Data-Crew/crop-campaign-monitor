# LLM Explainer — Implementation Reference

> Deep-dive into `src/explain.py`, `src/explain_prompts.py`, `src/explain_schema.py`,
> and `src/llm.py`. For operational details (enabling the step, GPU profiles, cache)
> see [doc/03-runtime-stack.md § LLM Explain Step](03-runtime-stack.md#llm-explain-step).

---

## Overview

The explain step is the last step in Phase 3. It reads the scored parcels produced
by `src/score.py`, generates a structured agronomic explanation per parcel using a
local Hugging Face model, and writes results to `data/output/parcel_explanations.parquet`.

Entry points:

- **Pipeline run**: called by `run_monitor.sh` as `python -m src.explain`. Exits
  immediately with no model loaded when `llm.enabled: false` in `config/monitor.yaml`.
- **Streamlit on-demand**: the "Regenerate explanation" button in the Parcel Details
  tab runs `python -m src.explain --config config/monitor.yaml llm.enabled=true
  --parcel-id <id>` for a single selected parcel.

---

## Data Flow

```
data/output/parcel_scores.parquet
          │
          ▼
  build_payload()          ← one dict per parcel row
          │
          ▼
  USER_PROMPT_TEMPLATE     ← payload serialized as JSON into the user message
  build_system_prompt()    ← rules + JSON schema + few-shot example
          │
          ▼
  tokenizer.apply_chat_template()   ← formats messages into model-native prompt string
  tokenizer()                       ← encodes prompt string → input_ids tensor
          │
          ▼
  model.generate()         ← autoregressive inference on GPU
  tokenizer.decode()       ← decodes new tokens → raw text
          │
          ▼
  extract_json_object()    ← strips markdown fences, isolates first { … }
  validate_explanation()   ← Pydantic parse + status consistency check
          │
          ├── validation OK  → ParcelExplanation
          ├── validation fail → retry with higher temperature
          └── retry fail     → fallback_explanation()
          │
          ▼
  data/output/parcel_explanations.parquet
```

---

## Step 1 — Payload construction (`build_payload`)

**Source**: `src/explain.py :: build_payload(row, cfg, thresholds)`

Each row from `parcel_scores.parquet` is converted into a structured dict
that becomes the LLM's input. The payload contains only information that is
already computed by upstream steps — the LLM never receives raw satellite data.

```python
{
    "parcel_id": "16tgk_00_00_42",
    "crop_name": "soybeans",
    "health_score": 0.0069,        # float distance from reference profile; lower = healthier
    "status": "GREEN",             # GREEN / YELLOW / RED / GRAY
    "n_observations": 15,          # number of Sentinel-2 dates with valid data
    "max_deviation_date": null,    # date of peak distance from profile, or null
    "distance_summary": {          # statistics over the full distance trajectory
        "mean": 0.004,
        "max": 0.012,
        "min": 0.001,
        "std": 0.003,
        "n_points": 15
    },
    "trajectory_flags": [],        # derived flags (see below)
    "data_quality_flags": [],      # data coverage flags (see below)
    "context": {
        "season_start": "2024-04-15",
        "season_end": "2024-10-31",
        "cadence_days": 16,
        "scoring_method": "adaptive",
        "green_threshold": 0.058,
        "red_threshold": 0.096
    }
}
```

### Trajectory flags

Derived from `distance_trajectory` (list of per-observation distances stored in
`parcel_scores.parquet`):

| Flag | Condition |
|------|-----------|
| `early_deviation` | max distance in the first third of observations exceeds `green_threshold` |
| `persistent_deviation` | all of the last 3 observations exceed `green_threshold` |
| `worsening_trend` | more than 60% of consecutive observation pairs show increasing distance |

### Data quality flags

| Flag | Condition |
|------|-----------|
| `few_observations` | fewer than `max(3, scoring.min_observations)` observations |
| `insufficient_data` | parcel status is `GRAY` |

When either quality flag is set, the system prompt's **Rule 3** requires the LLM
to output `"status": "insufficient_data"`.

### Payload hash

A SHA-256 hash of the serialized payload is computed and stored in
`parcel_explanations.parquet`. When `llm.skip_if_unchanged: true`, any parcel
whose hash matches the stored value is skipped without re-running the model.

---

## Step 2 — Prompt construction (`src/explain_prompts.py`)

The prompt is a two-message chat conversation: a **system message** and a
**user message**.

### System prompt (`SYSTEM_PROMPT` + `FEW_SHOT_BLOCK`)

The system prompt has two parts concatenated by `build_system_prompt(include_few_shot=True)`:

**Part 1 — Rules and JSON schema**

Eight numbered rules constrain model behavior (use hedged language, never invent
data, status must align with scoring, etc.) followed by the exact JSON schema
the model must output:

```json
{
  "status": "normal | warning | critical | insufficient_data",
  "summary": "string (1-3 sentences)",
  "possible_causes": ["hypothesis 1", "hypothesis 2"],
  "confidence": "low | medium | high",
  "recommended_action": "string",
  "evidence_used": ["evidence 1", "evidence 2"],
  "consistency_check": "consistent | weakly_supported | unsupported"
}
```

**Part 2 — Few-shot example (`FEW_SHOT_BLOCK`)**

A single complete input/output pair (RED corn parcel) is appended to the system
message. This shows the model the expected reasoning style and output format
without consuming user-turn tokens.

### User prompt (`USER_PROMPT_TEMPLATE`)

Minimal wrapper — the serialized payload JSON is inserted verbatim:

```
Analyze this parcel and produce your JSON explanation:

{payload_json}
```

The user message intentionally contains no instructions; all constraints are in
the system message.

---

## Step 3 — Model loading (`src/llm.py`)

### Profile selection (`select_llm_profile`)

Called once before the parcel loop. Scans all CUDA devices via
`src.gpu.discover_gpus()`, finds the GPU with the most total VRAM, and compares
against `llm.vram_threshold_gb` (default 8 GB):

```
best VRAM ≥ threshold → profile "high"   (Phi-3.5-mini fp16, ~7.6 GB)
best VRAM <  threshold → profile "low"   (Qwen2.5-1.5B 4-bit, ~1.2 GB)
```

The index of that GPU is also returned and passed to `load_model_and_tokenizer`
so all model layers land on a single device.

### Model loading (`load_model_and_tokenizer`)

Uses HuggingFace `transformers` standard classes:

```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
```

`trust_remote_code` is **not** used. Both default models (Phi-3.5-mini,
Qwen2.5) are natively implemented in `transformers >= 4.40` and do not require
custom model code from the HuggingFace Hub. This keeps the model cache
predictable — only weights (`.safetensors` shards) are stored, not executable
Python fetched from the model repo.

**kwargs** set based on profile config:

| Config | kwargs effect |
|--------|---------------|
| `quantization: "4bit"` + CUDA | `BitsAndBytesConfig(load_in_4bit=True)` + `device_map` |
| `quantization: "4bit"` + CPU | `dtype=torch.float16` (4-bit not supported on CPU) |
| `dtype: "float16"` | `dtype=torch.float16` + `device_map` |
| no CUDA | loaded in float16 on CPU, `device_map` removed |

`device_map` is always set to `f"cuda:{target_gpu}"` (the specific GPU with most
VRAM) rather than `"auto"`, to prevent accelerate from distributing model layers
across multiple GPUs including small or already-occupied ones.

---

## Step 4 — Chat template and tokenization (`generate_json_response`)

### Chat template (`tokenizer.apply_chat_template`)

The `messages` list (system + user dicts) is formatted into a single prompt
string using the model's built-in chat template:

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,           # return string, not token IDs
    add_generation_prompt=True,  # append the model's "start of response" marker
)
```

Each model family has its own template baked into its tokenizer config:
- **Phi-3.5-mini**: wraps messages in `<|system|>`, `<|user|>`, `<|assistant|>` tags
- **Qwen2.5**: uses `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant` tags

`add_generation_prompt=True` appends the assistant-start marker so the model
begins generating a response immediately rather than continuing the conversation.

If `apply_chat_template` fails (e.g. the tokenizer has no template), the code
falls back to plain concatenation: `f"{system_prompt}\n\n{user_text}\n"`.

### Tokenization

The formatted prompt string is encoded into input tensors and moved to the model's device:

```python
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
```

### Inference (`model.generate`)

```python
out = model.generate(
    **inputs,
    max_new_tokens=512,          # from profile config
    pad_token_id=...,
    do_sample=True,              # when temperature > 0
    temperature=0.1,             # low for deterministic JSON
    top_p=0.9,
)
```

`temperature=0.1` keeps the output close to greedy decoding, which is important
for JSON format compliance. The model is in `eval()` mode and wrapped in
`torch.no_grad()`.

### Decoding

Only the newly generated tokens are decoded (prompt tokens are sliced off):

```python
gen  = out[0, inputs["input_ids"].shape[1]:]
text = tokenizer.decode(gen, skip_special_tokens=True)
```

---

## Step 5 — Output extraction and validation

### JSON extraction (`extract_json_object`)

The raw decoded text may include markdown code fences (` ```json … ``` `) or
leading/trailing prose. `extract_json_object` strips fences and extracts the
first `{ … }` substring by finding the outermost braces.

### Pydantic validation (`validate_explanation`)

The extracted JSON string is parsed and validated against `ParcelExplanation`
(defined in `src/explain_schema.py`):

```python
class ParcelExplanation(BaseModel):
    status:              ExplainStatus       # normal | warning | critical | insufficient_data
    summary:             str                 # max 500 chars
    possible_causes:     list[str]           # max 5 items
    confidence:          Confidence          # low | medium | high
    recommended_action:  str                 # max 300 chars
    evidence_used:       list[str]           # max 8 items
    consistency_check:   ConsistencyCheck    # consistent | weakly_supported | unsupported
```

Before Pydantic validation, `_normalize_data` maps common LLM alias values:

| LLM output | Normalized to |
|------------|---------------|
| `"GREEN"`, `"green"` | `"normal"` |
| `"RED"` | `"critical"` |
| `"GRAY"`, `"grey"` | `"insufficient_data"` |
| `"very high"` | `"high"` |
| `"moderate"` | `"medium"` |
| `"weakly supported"` | `"weakly_supported"` |

After Pydantic validation, `status` is checked against the upstream scoring
status (GREEN/YELLOW/RED/GRAY). If they mismatch, `consistency_check` is
overridden to `"unsupported"` and a warning is logged.

### Retry on failure

If extraction or validation fails, the step retries once with a higher
temperature (`llm.retry_temperature`, default 0.35) to encourage a different
output format. If the retry also fails, `fallback_explanation()` produces a
minimal valid `ParcelExplanation` with `confidence: low` and a message
indicating the failure reason.

---

## Output — `parcel_explanations.parquet`

One row per explained parcel. Written to `data/output/parcel_explanations.parquet`.

| Column | Type | Description |
|--------|------|-------------|
| `parcel_id` | str | Parcel identifier |
| `explanation_json` | str | JSON-serialized `ParcelExplanation` |
| `llm_status` | str | `ExplainStatus` value |
| `llm_confidence` | str | `Confidence` value |
| `llm_model` | str | HuggingFace model ID used (or `"mock"`) |
| `profile` | str | Profile name: `"high"`, `"low"`, or `"mock"` |
| `generated_at` | str | ISO 8601 UTC timestamp |
| `payload_hash` | str | SHA-256 of the input payload (for change detection) |

When the step is called for a subset of parcel IDs (e.g. from Streamlit), the
existing parquet file is read, the relevant rows are replaced, and the merged
result is written back.

---

## Mock mode

When `llm.use_mock: true` or `--mock` is passed, the step runs without loading
any model. `_mock_explanation_json` produces a deterministic `ParcelExplanation`
derived purely from the scoring status. This is useful for testing the full
pipeline end-to-end without GPU or model availability.

---

## Configuration reference (`config/monitor.yaml`)

```yaml
llm:
  enabled: false              # true to run the step in pipeline mode
  use_mock: false             # true = skip model load, produce deterministic output
  vram_threshold_gb: 8        # VRAM cutoff between high and low profiles
  profile_override: null      # force "high" or "low"; null = auto by VRAM
  batch_size: 10              # future batching (currently unused in inference loop)
  max_parcels: null           # limit parcel count for testing; null = all
  temperature: 0.1            # sampling temperature (low = more deterministic)
  top_p: 0.9                  # nucleus sampling probability mass
  retry_temperature: 0.35     # temperature for the single retry on validation failure
  skip_if_unchanged: false    # skip parcels whose payload hash matches stored value

  profiles:
    high:
      model_id: "microsoft/Phi-3.5-mini-instruct"
      dtype: "float16"
      quantization: null        # no quantization; full fp16 (~7.6 GB VRAM)
      max_new_tokens: 512
    low:
      model_id: "Qwen/Qwen2.5-1.5B-Instruct"
      dtype: "float16"
      quantization: "4bit"      # BitsAndBytes 4-bit; ~1.2 GB VRAM
      max_new_tokens: 384
```

To swap a model, change `model_id` and adjust `quantization`/`dtype` as needed.
No code changes are required. See
[doc/01-workflow.md § Step 10](01-workflow.md#step-10--explain-srcexplainpy)
for the full profile table and examples.
