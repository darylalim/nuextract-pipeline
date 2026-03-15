# Batch Vision Support Design

## Overview

Add batch vision processing to the NuExtract pipeline, enabling multi-image extraction through a new Image Batch tab and extending the CSV tab with optional image column support. Inspired by the NuMind demo's batch `process_all_vision_info` pattern.

## Requirements

- Multi-image upload with per-image extraction using the same template
- CSV rows with image URLs or file paths alongside text
- True batched inference (multiple inputs per `model.generate()` call)
- Shared context with optional per-image overrides
- Results displayed as individual cards + summary table with CSV download

## Architecture

### Approach: New tab + CSV extension, unified `extract_batch()`

- New "Image Batch" tab for standalone multi-image upload
- Existing CSV tab extended with optional image column selector
- `extract_batch()` becomes the single inference function
- `extract()` becomes a thin wrapper calling `extract_batch()` with a single item
- `process_all_vision_info` gains batch support (NuMind pattern)

## Component Design

### 1. `process_all_vision_info` batch support (`utils.py`)

**Input normalization:**
- Detects single vs. batch input by checking if `messages[0]` is a list
- Same detection for `examples` parameter
- Single inputs normalized to batch-of-one internally

**Batch processing:**
- Iterates over each item in the batch
- Collects images in interleaved order: `[item1_example_imgs, item1_message_imgs, item2_example_imgs, item2_message_imgs, ...]`
- Raises `ValueError` if batched `examples` length doesn't match batched `messages` length

**Backward compatibility:**
- Signature unchanged: `process_all_vision_info(messages, examples=None) -> list | None`
- Single inputs produce identical output to current behavior

### 2. `extract_batch()` function (`streamlit_app.py`)

**Signature:**
```python
extract_batch(
    inputs: list[dict],   # [{"text": str|None, "image": PIL|None, "context": str|None}, ...]
    model, processor, device,
    template, examples,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
) -> list[tuple[dict|None, bool]]
```

**Processing:**
- Builds messages per item (text-only, image-only, or image+context)
- Calls `apply_chat_template` per item (tokenizer requirement)
- Passes all formatted texts to processor at once: `text=[formatted1, formatted2, ...]`
- Calls batched `process_all_vision_info` for all images in correct interleaved order
- Checks each item's token count against `MAX_INPUT_TOKENS`, raises `ValueError` listing which items exceeded
- Single `model.generate()` call with left-padded batch
- Trims, decodes, and parses JSON per item independently
- Returns list of `(dict|None, bool)` tuples

**`extract()` wrapper:**
```python
def extract(input_content, model, processor, device, template, examples, image=None, max_new_tokens=...):
    batch_input = {"text": input_content, "image": image, "context": None}
    results = extract_batch([batch_input], model, processor, device, template, examples, max_new_tokens)
    return results[0]
```

### 3. Image Batch tab (new 4th tab)

**UI layout:**
- Tab order: Text | Image | CSV Batch | Image Batch
- Multi-image uploader: `st.file_uploader` with `accept_multiple_files=True` (png, jpg, jpeg, webp)
- Shared context: `st.text_area` for context applying to all images
- Per-image overrides: `st.expander("Per-image context")` with text input per uploaded image (keyed by filename), defaults to empty (falls back to shared context)
- Batch size slider: 1-8, default 4
- Extract button

**Results display:**
- Individual cards: image thumbnail + extracted JSON in `st.expander` (first 3 expanded, rest collapsed)
- Summary table: dataframe with columns `filename` + template fields
- Metrics row: Total / Extracted / Failed
- Download button: CSV export

**Processing:**
1. Build input dicts: `{"text": None, "image": pil_image, "context": per_image_context or shared_context}`
2. Call `extract_batch()` in chunks
3. Progress bar updates per chunk

### 4. CSV tab image column extension

**UI changes:**
- New optional selectbox after text column selector: `"Select image column (optional)"` with options `["None"] + df.columns.tolist()`
- Batch size slider appears when image column is selected

**Image loading:**
- URLs (starts with `http://` or `https://`): fetched via `fetch_image` from `qwen_vl_utils`
- File paths: loaded via `PIL.Image.open`
- Invalid paths/URLs: row treated as text-only with warning

**Processing:**
- With image column: `{"text": text_value, "image": loaded_image, "context": None}` — text becomes context
- Without image column: `{"text": text_value, "image": None, "context": None}` — identical to current behavior
- Batched via `extract_batch()` in chunks

### 5. Batching strategy and memory management

**Chunked batching:**
- `extract_batch()` processes inputs in chunks (default size 4, configurable parameter)
- Each chunk is a true batched forward pass via `model.generate()`
- Progress bar updates per chunk

**Chunk size control:**
- Image Batch tab: slider (1-8, default 4)
- CSV tab: slider appears when image column is selected; text-only stays sequential per-row

**Error handling:**
- OOM (`RuntimeError`) on a chunk: fall back to sequential processing for that chunk's items
- Token limit (`ValueError`) items: skipped, marked as `None`, row numbers reported in warning

## Testing Strategy

### `tests/test_utils.py` — batched `process_all_vision_info`
- Batch of 2 message lists with examples — verifies interleaved image order
- Batch messages with single (non-batched) examples — normalized correctly
- Mismatched batch lengths — raises `ValueError`
- Single input backward compatibility

### `tests/test_streamlit_app.py` — `extract_batch()`
- Batch of 2 text-only inputs — returns list of 2 results
- Batch with mixed text and image inputs — correct vision processing
- Batch with context per item — context passed correctly
- Token limit exceeded on one item — that item skipped, others succeed
- Truncation detection per item — independent `was_truncated` flags
- Chunk fallback on OOM — items in failed chunk processed sequentially
- Single-item batch — same result as old `extract()`

### `tests/test_streamlit_app.py` — `extract()` wrapper
- Verify `extract()` delegates to `extract_batch()` and returns same result format

### No UI tests
- Consistent with existing approach (current tests don't test UI rendering)

## Files Modified

- `utils.py` — `process_all_vision_info` batch support
- `streamlit_app.py` — `extract_batch()`, `extract()` wrapper, Image Batch tab, CSV tab extension
- `tests/test_utils.py` — new batch vision tests
- `tests/test_streamlit_app.py` — new `extract_batch()` and wrapper tests
- `presets.json` — no changes
- `CLAUDE.md` — update architecture section after implementation
