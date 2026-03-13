# Design: Token Limit, Auto Template Generation, ICL Example Images

## Overview

Three features added to the text-extraction-pipeline Streamlit app:

1. **Token limit enforcement** — Prevent OOM/hangs on large inputs
2. **Auto template generation** — Generate JSON templates from natural language descriptions
3. **ICL example images** — Support images in few-shot examples on the Image tab

## 1. Token Limit Enforcement

**Constant:** `MAX_INPUT_TOKENS = 10_000`

**Behavior:** In `extract()`, after tokenizing the input, check `inputs["input_ids"].shape[1]` against `MAX_INPUT_TOKENS`. If exceeded, raise a `ValueError` instead of calling `model.generate()`.

**UI handling:**
- Text/Image tabs: catch the error and display via `st.error` with a message indicating the input is too long.
- CSV tab: rows that exceed the limit are treated as extraction failures (`result = None`), counted in the "Failed" metric, and a warning lists which rows were skipped.

## 2. Auto Template Generation (Dual-Mode Template Field)

**UI change:** The existing sidebar template text area remains. When JSON parsing fails, a "Generate Template" button appears below the error message. Clicking it calls `generate_template()` and replaces the field content with the generated JSON.

**New function in `utils.py`:**

```python
def generate_template(description: str, model, processor, device: str) -> tuple[dict | None, str | None]:
```

- Constructs a prompt asking the NuExtract model to produce a JSON extraction template from the natural language description.
- Runs inference with a lower `max_new_tokens` (templates are small).
- Parses the result as JSON.
- Returns `(template_dict, None)` on success, `(None, error_message)` on failure.

**Flow:**
1. User types natural language in the template field (e.g., "extract person name, age, and job")
2. `validate_template()` fails — error shown, "Generate Template" button appears
3. User clicks button — `generate_template()` is called
4. Generated JSON replaces the template field content
5. User reviews/edits the template, then clicks Extract

## 3. ICL Example Images (Image Tab, URL References)

**Extended examples format:** The `input` field in examples can be either a string (text, existing behavior) or a dict for images:

```json
[
  {
    "input": {"type": "image", "image": "https://example.com/photo.png"},
    "output": "{\"name\": \"John\"}"
  }
]
```

**`parse_examples()` updates:** Accepts both string and dict `input` formats. For dict format, validates that `type` is `"image"` and `image` key contains a URL string.

**New function in `utils.py`:**

```python
def process_all_vision_info(messages: list, examples: list | None = None) -> list | None:
```

Adapted from the HF demo's `utils.py`. Extracts images from both ICL examples and the user message, returns them in correct order (example images first, then input images). Returns `None` if no images found.

**`extract()` updates:** When `image` is provided and examples contain image inputs, call `process_all_vision_info()` instead of the current direct `process_vision_info()` call to gather all images.

**Scope:** Only the Image tab uses image examples. Text and CSV tabs continue with text-only examples.

## 4. New `utils.py` Module

Contains two functions:
- `generate_template()` — template generation from natural language
- `process_all_vision_info()` — image extraction from examples + messages

Imported by `streamlit_app.py`. No other structural changes.

## 5. Testing

**New tests:**

- Token limit: input under limit passes, input over limit raises `ValueError`
- Token limit on CSV tab: over-limit rows counted as failures
- `generate_template`: valid description produces a template dict, failure returns error tuple
- `process_all_vision_info`: examples with images returns correct order, no images returns `None`, batch ordering
- `parse_examples`: new image input format accepted, invalid image format rejected
- `extract` with image examples: images passed in correct order via `process_all_vision_info`

**Existing tests:** All 32 existing tests remain unchanged and passing.
