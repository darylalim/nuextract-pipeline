# Batch Vision Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add batch vision processing with a new Image Batch tab and CSV image column support, unified under a single `extract_batch()` inference function.

**Architecture:** `extract_batch()` becomes the core inference function handling chunked batched forward passes. `extract()` becomes a thin wrapper. `process_all_vision_info` gains batch support. New Image Batch tab for multi-image upload. CSV tab extended with optional image column.

**Tech Stack:** PyTorch, Streamlit, Transformers, qwen_vl_utils, PIL

**Spec:** `docs/superpowers/specs/2026-03-15-batch-vision-support-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `utils.py` | Modify | `process_all_vision_info` batch support |
| `streamlit_app.py` | Modify | `_clear_device_cache`, `extract_batch`, `extract` wrapper, Image Batch tab, CSV tab extension |
| `tests/test_utils.py` | Modify | New batch `process_all_vision_info` tests |
| `tests/test_streamlit_app.py` | Modify | New `extract_batch`, wrapper, CSV image loading tests; update fixture for 4 tabs |
| `CLAUDE.md` | Modify | Update architecture docs |

---

## Chunk 1: `process_all_vision_info` batch support

### Task 1: Add batch `process_all_vision_info` tests

**Files:**
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests for batch `process_all_vision_info`**

Add these tests at the end of `tests/test_utils.py`:

```python
# --- process_all_vision_info batch support ---


def test_process_all_vision_info_batch_two_items_with_examples():
    from utils import process_all_vision_info

    fake_ex_img1 = MagicMock(name="ex_img1")
    fake_ex_img2 = MagicMock(name="ex_img2")
    fake_msg_img1 = MagicMock(name="msg_img1")
    fake_msg_img2 = MagicMock(name="msg_img2")

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://example.com/input1.png"}
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://example.com/input2.png"}
                ],
            }
        ],
    ]
    examples = [
        [
            {
                "input": {"type": "image", "image": "https://example.com/ex1.png"},
                "output": '{"name": "A"}',
            }
        ],
        [
            {
                "input": {"type": "image", "image": "https://example.com/ex2.png"},
                "output": '{"name": "B"}',
            }
        ],
    ]

    call_count = [0]
    example_images = [fake_ex_img1, fake_ex_img2]

    def mock_fetch_image(inp):
        img = example_images[call_count[0]]
        call_count[0] += 1
        return img

    msg_images = [[fake_msg_img1], [fake_msg_img2]]
    pvi_call_count = [0]

    def mock_process_vision_info(msgs):
        imgs = msg_images[pvi_call_count[0]]
        pvi_call_count[0] += 1
        return (imgs, None)

    with (
        patch("utils.process_vision_info", side_effect=mock_process_vision_info),
        patch("utils.fetch_image", side_effect=mock_fetch_image),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_ex_img1, fake_msg_img1, fake_ex_img2, fake_msg_img2]


def test_process_all_vision_info_batch_single_examples_broadcast():
    from utils import process_all_vision_info

    fake_ex_img = MagicMock(name="ex_img")
    fake_msg_img1 = MagicMock(name="msg_img1")
    fake_msg_img2 = MagicMock(name="msg_img2")

    messages = [
        [{"role": "user", "content": [{"type": "image", "image": "https://example.com/1.png"}]}],
        [{"role": "user", "content": [{"type": "image", "image": "https://example.com/2.png"}]}],
    ]
    examples = [
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"name": "A"}',
        }
    ]

    msg_images = [[fake_msg_img1], [fake_msg_img2]]
    pvi_call_count = [0]

    def mock_process_vision_info(msgs):
        imgs = msg_images[pvi_call_count[0]]
        pvi_call_count[0] += 1
        return (imgs, None)

    with (
        patch("utils.process_vision_info", side_effect=mock_process_vision_info),
        patch("utils.fetch_image", return_value=fake_ex_img),
    ):
        result = process_all_vision_info(messages, examples)

    # Single examples list is broadcast to each batch item
    assert result == [fake_ex_img, fake_msg_img1, fake_ex_img, fake_msg_img2]


def test_process_all_vision_info_batch_mismatched_lengths():
    from utils import process_all_vision_info

    messages = [
        [{"role": "user", "content": "text1"}],
        [{"role": "user", "content": "text2"}],
    ]
    examples = [
        [{"input": "ex1", "output": "out1"}],
        [{"input": "ex2", "output": "out2"}],
        [{"input": "ex3", "output": "out3"}],
    ]

    with pytest.raises(ValueError, match="length"):
        process_all_vision_info(messages, examples)


def test_process_all_vision_info_single_input_backward_compat():
    from utils import process_all_vision_info

    fake_message_img = MagicMock(name="message_img")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]

    with patch(
        "utils.process_vision_info",
        return_value=([fake_message_img], None),
    ):
        result = process_all_vision_info(messages)

    assert result == [fake_message_img]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py::test_process_all_vision_info_batch_two_items_with_examples tests/test_utils.py::test_process_all_vision_info_batch_single_examples_broadcast tests/test_utils.py::test_process_all_vision_info_batch_mismatched_lengths tests/test_utils.py::test_process_all_vision_info_single_input_backward_compat -v`

Expected: FAIL — batch detection not implemented yet.

### Task 2: Implement batch `process_all_vision_info`

**Files:**
- Modify: `utils.py:156-173`

- [ ] **Step 3: Implement batch support**

Replace the `process_all_vision_info` function in `utils.py` (lines 156-173):

```python
def process_all_vision_info(messages, examples=None):
    """Extract images from both ICL examples and user messages.

    Supports single input (messages is a list of dicts) or batch input
    (messages is a list of lists of dicts). Returns a flat list of images
    in per-item order (example images then message images for each item),
    or None if no images found.
    """
    # Detect single vs batch: single messages is [{"role": ...}, ...],
    # batch is [[{"role": ...}, ...], ...]
    is_batch = messages and isinstance(messages[0], list)
    messages_batch = messages if is_batch else [messages]

    # Normalize examples
    if examples is None:
        examples_batch = [None] * len(messages_batch)
    elif isinstance(examples, list) and examples and isinstance(examples[0], list):
        # Batched examples: [[ex1, ex2], [ex3, ex4]]
        examples_batch = examples
    else:
        # Single examples list: broadcast to each batch item
        examples_batch = [examples] * len(messages_batch)

    if len(examples_batch) != len(messages_batch):
        raise ValueError(
            f"Examples batch length ({len(examples_batch)}) must match "
            f"messages batch length ({len(messages_batch)})."
        )

    all_images = []
    for msg_group, ex_group in zip(messages_batch, examples_batch):
        if ex_group:
            for ex in ex_group:
                inp = ex.get("input")
                if isinstance(inp, dict) and inp.get("type") == "image":
                    all_images.append(fetch_image(inp))

        message_images = process_vision_info(msg_group)[0] or []
        all_images.extend(message_images)

    return all_images if all_images else None
```

- [ ] **Step 4: Run new batch tests to verify they pass**

Run: `uv run pytest tests/test_utils.py::test_process_all_vision_info_batch_two_items_with_examples tests/test_utils.py::test_process_all_vision_info_batch_single_examples_broadcast tests/test_utils.py::test_process_all_vision_info_batch_mismatched_lengths tests/test_utils.py::test_process_all_vision_info_single_input_backward_compat -v`

Expected: PASS

- [ ] **Step 5: Run all existing `process_all_vision_info` tests to verify backward compatibility**

Run: `uv run pytest tests/test_utils.py -k "process_all_vision_info" -v`

Expected: All 10 tests PASS (6 existing + 4 new).

- [ ] **Step 6: Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: add batch support to process_all_vision_info"
```

---

## Chunk 2: `extract_batch()` and `extract()` wrapper

### Task 3: Add `_clear_device_cache` helper

**Files:**
- Modify: `streamlit_app.py` (add after `_convert_template_if_needed`, before `# --- Streamlit UI ---`)
- Test: `tests/test_streamlit_app.py`

- [ ] **Step 7: Write failing tests for `_clear_device_cache`**

Add at end of `tests/test_streamlit_app.py`:

```python
# --- _clear_device_cache ---


def test_clear_device_cache_cuda(app):
    with patch("streamlit_app.torch") as mock_torch:
        mock_torch.cuda = MagicMock()
        app._clear_device_cache("cuda")
        mock_torch.cuda.empty_cache.assert_called_once()


def test_clear_device_cache_mps(app):
    with patch("streamlit_app.torch") as mock_torch:
        mock_torch.mps = MagicMock()
        app._clear_device_cache("mps")
        mock_torch.mps.empty_cache.assert_called_once()


def test_clear_device_cache_cpu_is_noop(app):
    with patch("streamlit_app.torch") as mock_torch:
        app._clear_device_cache("cpu")
        mock_torch.cuda.empty_cache.assert_not_called()
```

- [ ] **Step 8: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::test_clear_device_cache_cuda tests/test_streamlit_app.py::test_clear_device_cache_mps tests/test_streamlit_app.py::test_clear_device_cache_cpu_is_noop -v`

Expected: FAIL — `_clear_device_cache` not defined.

- [ ] **Step 9: Implement `_clear_device_cache`**

Add after line 260 in `streamlit_app.py` (after `_convert_template_if_needed`):

```python
def _clear_device_cache(device):
    """Clear device memory cache after OOM errors."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::test_clear_device_cache_cuda tests/test_streamlit_app.py::test_clear_device_cache_mps tests/test_streamlit_app.py::test_clear_device_cache_cpu_is_noop -v`

Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add _clear_device_cache helper for OOM recovery"
```

### Task 4: Add `extract_batch()` tests

**Files:**
- Test: `tests/test_streamlit_app.py`

- [ ] **Step 12: Add `make_batch_mocks` helper and tests**

Add `make_batch_mocks` after the existing `make_mocks` function in `tests/test_streamlit_app.py`:

```python
def make_batch_mocks(decode_outputs, input_lengths=None):
    """Create mock model and processor for batch inference.

    decode_outputs: list of decoded strings, one per batch item.
    input_lengths: list of non-padded token counts per item. Defaults to [5]*n.
    """
    n = len(decode_outputs)
    if input_lengths is None:
        input_lengths = [5] * n
    max_len = max(input_lengths)
    gen_tokens = 3

    processor = MagicMock()

    # Left-padded input_ids and attention_mask
    input_ids = torch.ones(n, max_len, dtype=torch.long)
    attention_mask = torch.zeros(n, max_len, dtype=torch.long)
    for i, length in enumerate(input_lengths):
        pad = max_len - length
        attention_mask[i, pad:] = 1

    inputs_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    proc_result = MagicMock()
    proc_result.to.return_value = inputs_dict
    processor.return_value = proc_result
    processor.tokenizer.apply_chat_template.return_value = "formatted prompt"
    processor.batch_decode.return_value = decode_outputs

    model = MagicMock()
    output = torch.cat(
        [input_ids, torch.ones(n, gen_tokens, dtype=torch.long) * 10], dim=1
    )
    model.generate.return_value = output

    return model, processor
```

Then add the `extract_batch` tests:

```python
# --- extract_batch ---


def test_extract_batch_two_text_items(app):
    outputs = [
        json.dumps({"company": "Acme", "revenue": "$1B"}),
        json.dumps({"company": "Beta", "revenue": "$2B"}),
    ]
    model, processor = make_batch_mocks(outputs)
    inputs = [
        {"text": "Acme text", "image": None, "context": None},
        {"text": "Beta text", "image": None, "context": None},
    ]
    results = app.extract_batch(
        inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert len(results) == 2
    assert results[0][0] == {"company": "Acme", "revenue": "$1B"}
    assert results[1][0] == {"company": "Beta", "revenue": "$2B"}


def test_extract_batch_empty_inputs(app):
    model, processor = make_batch_mocks([])
    results = app.extract_batch(
        [], model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert results == []
    model.generate.assert_not_called()


def test_extract_batch_with_image_and_context(app):
    outputs = [json.dumps({"company": "Acme"})]
    model, processor = make_batch_mocks(outputs)
    fake_image = MagicMock()

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=[MagicMock()],
    ):
        inputs = [{"text": None, "image": fake_image, "context": "some context"}]
        results = app.extract_batch(
            inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
        )

    assert results[0][0] == {"company": "Acme"}
    # Verify message was built with image + context
    call_args = processor.tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    content = messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[1] == {"type": "text", "text": "some context"}


def test_extract_batch_image_without_context(app):
    outputs = [json.dumps({"company": "Acme"})]
    model, processor = make_batch_mocks(outputs)
    fake_image = MagicMock()

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=[MagicMock()],
    ):
        inputs = [{"text": None, "image": fake_image, "context": None}]
        results = app.extract_batch(
            inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
        )

    assert results[0][0] == {"company": "Acme"}
    call_args = processor.tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    content = messages[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "image"


def test_extract_batch_token_limit_skips_item(app):
    outputs = [
        json.dumps({"company": "Acme"}),
        json.dumps({"company": "Beta"}),
    ]
    # Item 2 has 10_001 tokens, exceeding MAX_INPUT_TOKENS
    model, processor = make_batch_mocks(outputs, input_lengths=[5, 10_001])

    inputs = [
        {"text": "short", "image": None, "context": None},
        {"text": "very long text", "image": None, "context": None},
    ]
    results = app.extract_batch(
        inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES, chunk_size=1
    )
    assert len(results) == 2
    assert results[0][0] == {"company": "Acme"}
    assert results[1] == (None, False)


def test_extract_batch_truncation_detection(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_batch_mocks([output])
    # make_batch_mocks generates 3 output tokens. max_new_tokens=3 triggers truncation.
    results = app.extract_batch(
        [{"text": "text", "image": None, "context": None}],
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        max_new_tokens=3,
    )
    assert results[0][1] is True


def test_extract_batch_no_truncation(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_batch_mocks([output])
    results = app.extract_batch(
        [{"text": "text", "image": None, "context": None}],
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        max_new_tokens=100,
    )
    assert results[0][1] is False


def test_extract_batch_oom_falls_back_to_sequential(app):
    outputs = [
        json.dumps({"company": "Acme"}),
        json.dumps({"company": "Beta"}),
    ]
    model, processor = make_batch_mocks(outputs)

    call_count = [0]
    single_output_a = torch.tensor([[1, 2, 3, 4, 5, 10, 20, 30]])
    single_output_b = torch.tensor([[1, 2, 3, 4, 5, 11, 21, 31]])

    def generate_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("CUDA out of memory")
        if call_count[0] == 2:
            return single_output_a
        return single_output_b

    model.generate.side_effect = generate_side_effect

    # Need single-item processor results for fallback
    single_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    single_inputs = {
        "input_ids": single_input_ids,
        "attention_mask": torch.ones_like(single_input_ids),
    }
    single_proc = MagicMock()
    single_proc.to.return_value = single_inputs

    proc_call_count = [0]
    original_proc = processor.return_value

    def proc_side_effect(*args, **kwargs):
        proc_call_count[0] += 1
        if proc_call_count[0] == 1:
            return original_proc
        return single_proc

    processor.side_effect = proc_side_effect
    processor.batch_decode.side_effect = [
        [json.dumps({"company": "Acme"})],
        [json.dumps({"company": "Beta"})],
    ]

    with patch("streamlit_app._clear_device_cache"):
        inputs = [
            {"text": "text1", "image": None, "context": None},
            {"text": "text2", "image": None, "context": None},
        ]
        results = app.extract_batch(
            inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES, chunk_size=2
        )

    assert len(results) == 2
    assert results[0][0] == {"company": "Acme"}
    assert results[1][0] == {"company": "Beta"}


def test_extract_batch_single_item(app):
    output = json.dumps({"company": "Acme", "revenue": "$1B"})
    model, processor = make_batch_mocks([output])
    inputs = [{"text": "some text", "image": None, "context": None}]
    results = app.extract_batch(
        inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert len(results) == 1
    assert results[0][0] == {"company": "Acme", "revenue": "$1B"}


def test_extract_batch_progress_callback(app):
    outputs = [
        json.dumps({"company": "A"}),
        json.dumps({"company": "B"}),
        json.dumps({"company": "C"}),
    ]
    model, processor = make_batch_mocks(outputs)
    callback = MagicMock()

    inputs = [
        {"text": "t1", "image": None, "context": None},
        {"text": "t2", "image": None, "context": None},
        {"text": "t3", "image": None, "context": None},
    ]
    app.extract_batch(
        inputs,
        model,
        processor,
        "cpu",
        TEST_TEMPLATE,
        TEST_EXAMPLES,
        chunk_size=2,
        progress_callback=callback,
    )
    # 3 items, chunk_size=2 -> 2 chunks (2 items + 1 item)
    assert callback.call_count == 2
    callback.assert_any_call(2, 3)
    callback.assert_any_call(3, 3)


def test_extract_batch_mixed_text_and_image(app):
    outputs = [
        json.dumps({"company": "Acme"}),
        json.dumps({"company": "Beta"}),
    ]
    model, processor = make_batch_mocks(outputs)
    fake_image = MagicMock()

    with patch(
        "streamlit_app.process_all_vision_info",
        return_value=[MagicMock()],
    ):
        inputs = [
            {"text": "text only input", "image": None, "context": None},
            {"text": None, "image": fake_image, "context": "image context"},
        ]
        results = app.extract_batch(
            inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
        )

    assert len(results) == 2
    assert results[0][0] == {"company": "Acme"}
    assert results[1][0] == {"company": "Beta"}


def test_extract_batch_different_input_lengths(app):
    outputs = [
        json.dumps({"company": "Acme"}),
        json.dumps({"company": "Beta"}),
    ]
    # Items with different token counts: 3 and 7
    model, processor = make_batch_mocks(outputs, input_lengths=[3, 7])

    inputs = [
        {"text": "short", "image": None, "context": None},
        {"text": "much longer text input", "image": None, "context": None},
    ]
    results = app.extract_batch(
        inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    assert len(results) == 2
    assert results[0][0] == {"company": "Acme"}
    assert results[1][0] == {"company": "Beta"}


def test_extract_batch_passes_template_and_examples(app):
    output = json.dumps({"company": "Acme"})
    model, processor = make_batch_mocks([output])
    inputs = [{"text": "text", "image": None, "context": None}]
    app.extract_batch(
        inputs, model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
    )
    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["template"] == TEST_TEMPLATE
    assert call_args[1]["examples"] == TEST_EXAMPLES
```

- [ ] **Step 13: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py -k "extract_batch" -v`

Expected: FAIL — `extract_batch` not defined.

### Task 5: Implement `extract_batch()`

**Files:**
- Modify: `streamlit_app.py` (add after `_clear_device_cache`, before `# --- Streamlit UI ---`)

- [ ] **Step 14: Implement `extract_batch`**

Add after `_clear_device_cache` in `streamlit_app.py`:

```python
def extract_batch(
    inputs,
    model,
    processor,
    device,
    template,
    examples,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    chunk_size=4,
    progress_callback=None,
):
    if not inputs:
        return []

    all_results = []
    total = len(inputs)

    for chunk_start in range(0, total, chunk_size):
        chunk = inputs[chunk_start : chunk_start + chunk_size]
        chunk_results = _process_chunk(
            chunk, model, processor, device, template, examples, max_new_tokens
        )
        all_results.extend(chunk_results)
        if progress_callback:
            progress_callback(len(all_results), total)

    return all_results


def _build_message(item):
    """Build a chat message from a batch input dict."""
    if item.get("image") is not None:
        content = [{"type": "image", "image": item["image"]}]
        if item.get("context"):
            content.append({"type": "text", "text": item["context"]})
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": item["text"]}]


def _process_chunk(chunk, model, processor, device, template, examples, max_new_tokens):
    """Process a single chunk of inputs as a batched forward pass."""
    all_messages = []
    formatted_texts = []
    results = [(None, False)] * len(chunk)

    for i, item in enumerate(chunk):
        messages = _build_message(item)
        formatted = processor.tokenizer.apply_chat_template(
            messages,
            template=template,
            examples=examples,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_messages.append(messages)
        formatted_texts.append(formatted)

    try:
        batch_results = _run_batch_inference(
            all_messages,
            formatted_texts,
            model,
            processor,
            device,
            examples,
            max_new_tokens,
        )
        for i, res in enumerate(batch_results):
            results[i] = res
    except ValueError:
        if len(chunk) == 1:
            # Single item (from extract() wrapper): propagate ValueError
            raise
        # Multi-item batch: fall back to one-at-a-time
        for i, (messages, formatted) in enumerate(
            zip(all_messages, formatted_texts)
        ):
            try:
                results[i] = _run_batch_inference(
                    [messages],
                    [formatted],
                    model,
                    processor,
                    device,
                    examples,
                    max_new_tokens,
                )[0]
            except ValueError:
                results[i] = (None, False)
            except RuntimeError as e2:
                if "out of memory" not in str(e2).lower():
                    raise
                _clear_device_cache(device)
                results[i] = (None, False)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        _clear_device_cache(device)
        # Fallback: process each item sequentially
        for i, (messages, formatted) in enumerate(
            zip(all_messages, formatted_texts)
        ):
            try:
                results[i] = _run_batch_inference(
                    [messages],
                    [formatted],
                    model,
                    processor,
                    device,
                    examples,
                    max_new_tokens,
                )[0]
            except (ValueError, RuntimeError) as e2:
                if isinstance(e2, RuntimeError) and "out of memory" not in str(e2).lower():
                    raise
                if isinstance(e2, RuntimeError):
                    _clear_device_cache(device)
                results[i] = (None, False)

    return results


def _run_batch_inference(
    all_messages, formatted_texts, model, processor, device, examples, max_new_tokens
):
    """Run batched inference on pre-formatted texts. Returns list of (dict|None, bool).

    Raises ValueError if any item exceeds MAX_INPUT_TOKENS (checked post-processor,
    matching the original extract() behavior).
    """
    # Collect images
    batched_messages = list(all_messages)
    image_inputs = process_all_vision_info(batched_messages, examples)

    inputs = processor(
        text=formatted_texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Use padded input length for trimming (same for all items after left-padding)
    padded_input_len = inputs["input_ids"].shape[1]

    # Per-item token counts from attention mask for limit checking
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    over_limit = [
        (i, int(l)) for i, l in enumerate(input_lens) if int(l) > MAX_INPUT_TOKENS
    ]
    if over_limit:
        if len(formatted_texts) == 1:
            _, count = over_limit[0]
            raise ValueError(
                f"Input too long: {count} tokens (limit: {MAX_INPUT_TOKENS})."
            )
        raise ValueError(
            f"Items exceed {MAX_INPUT_TOKENS} token limit: {over_limit}"
        )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )

    results = []
    for i in range(len(formatted_texts)):
        trimmed = output[i, padded_input_len:]
        generated_len = trimmed.shape[0]
        was_truncated = generated_len == max_new_tokens
        decoded = processor.batch_decode(
            trimmed.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            parsed = json.loads(decoded)
            results.append((parsed, was_truncated))
        except (json.JSONDecodeError, IndexError):
            results.append((None, was_truncated))

    return results
```

- [ ] **Step 15: Run `extract_batch` tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -k "extract_batch" -v`

Expected: All 11 tests PASS.

- [ ] **Step 16: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add extract_batch with chunked batched inference"
```

### Task 6: Refactor `extract()` as wrapper and verify backward compatibility

**Files:**
- Modify: `streamlit_app.py:186-239`
- Test: `tests/test_streamlit_app.py`

- [ ] **Step 17: Add wrapper delegation test**

Add at end of `tests/test_streamlit_app.py`:

```python
# --- extract() wrapper ---


def test_extract_delegates_to_extract_batch(app):
    output = json.dumps({"company": "Acme", "revenue": "$1B"})
    model, processor = make_mocks(output)
    with patch.object(app, "extract_batch", return_value=[({"company": "Acme", "revenue": "$1B"}, False)]) as mock_eb:
        result, was_truncated = app.extract(
            "some text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES
        )
    mock_eb.assert_called_once()
    call_args = mock_eb.call_args
    batch_input = call_args[0][0][0]
    assert batch_input == {"text": "some text", "image": None, "context": None}
    assert call_args[1]["chunk_size"] == 1
    assert result == {"company": "Acme", "revenue": "$1B"}
    assert was_truncated is False


def test_extract_wrapper_image_maps_to_context(app):
    model, processor = make_mocks(json.dumps({"company": "Acme"}))
    fake_image = MagicMock()
    with patch.object(app, "extract_batch", return_value=[({"company": "Acme"}, False)]) as mock_eb:
        app.extract(
            "context text", model, processor, "cpu", TEST_TEMPLATE, TEST_EXAMPLES,
            image=fake_image,
        )
    batch_input = mock_eb.call_args[0][0][0]
    assert batch_input["text"] is None
    assert batch_input["image"] is fake_image
    assert batch_input["context"] == "context text"
```

- [ ] **Step 18: Replace `extract()` with wrapper**

Replace the `extract` function in `streamlit_app.py` (lines 186-239):

```python
def extract(
    input_content,
    model,
    processor,
    device,
    template,
    examples,
    image=None,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
):
    if image is not None:
        batch_input = {"text": None, "image": image, "context": input_content}
    else:
        batch_input = {"text": input_content, "image": None, "context": None}
    results = extract_batch(
        [batch_input],
        model,
        processor,
        device,
        template,
        examples,
        max_new_tokens=max_new_tokens,
        chunk_size=1,
    )
    return results[0]
```

- [ ] **Step 19: Run ALL existing tests to verify backward compatibility**

Run: `uv run pytest tests/ -v`

Expected: All tests PASS (64 existing streamlit_app tests + 31 existing utils tests + new tests).

- [ ] **Step 20: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "refactor: extract() delegates to extract_batch wrapper"
```

---

## Chunk 3: Image Batch tab and CSV tab extension

### Task 7: Update test fixture for 4 tabs

**Files:**
- Modify: `tests/test_streamlit_app.py:27`

- [ ] **Step 21: Update tab mocks from 3 to 4**

In the `app` fixture in `tests/test_streamlit_app.py`, change line 27:

```python
    tab_mocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
```

- [ ] **Step 22: Run all tests to verify fixture change is safe**

Run: `uv run pytest tests/ -v`

Expected: All tests PASS.

- [ ] **Step 23: Commit**

```bash
git add tests/test_streamlit_app.py
git commit -m "test: update fixture for 4-tab layout"
```

### Task 8: Add Image Batch tab

**Files:**
- Modify: `streamlit_app.py:366` (tab creation) and add new tab block

- [ ] **Step 24: Update tab creation and add Image Batch tab**

In `streamlit_app.py`, change line 366 from:

```python
text_tab, image_tab, csv_tab = st.tabs(["Text", "Image", "CSV Batch"])
```

to:

```python
text_tab, image_tab, image_batch_tab, csv_tab = st.tabs(
    ["Text", "Image", "Image Batch", "CSV Batch"]
)
```

Then add the Image Batch tab block after the `image_tab` block (after line 449) and before the `csv_tab` block:

```python
with image_batch_tab:
    uploaded_images = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="image_batch_upload",
    )
    shared_context = st.text_area(
        "Shared context (optional)", height=100, key="image_batch_context"
    )

    if uploaded_images:
        with st.expander("Per-image context"):
            per_image_contexts = {}
            for img_file in uploaded_images:
                per_image_contexts[img_file.name] = st.text_input(
                    f"Context for {img_file.name}",
                    value="",
                    key=f"ctx_{img_file.name}",
                )

        batch_size = st.slider(
            "Batch size",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            key="image_batch_size",
            help="Images per inference batch. Lower if running out of memory.",
        )

        if st.button("Extract", type="primary", key="image_batch_extract"):
            if not _has_config_errors(template_error, examples_error, template_parsed):
                converted = _convert_template_if_needed(json_str, source_format)
                if converted:
                    template_str = converted

                pil_images = []
                filenames = []
                for img_file in uploaded_images:
                    pil_images.append(Image.open(img_file))
                    filenames.append(img_file.name)

                batch_inputs = []
                for i, pil_img in enumerate(pil_images):
                    ctx = per_image_contexts.get(filenames[i], "").strip()
                    if not ctx:
                        ctx = shared_context.strip() if shared_context else None
                    batch_inputs.append(
                        {"text": None, "image": pil_img, "context": ctx or None}
                    )

                progress_bar = st.progress(0, text="Starting...")

                def update_progress(completed, total):
                    progress_bar.progress(
                        completed / total,
                        text=f"Processing {completed} of {total} images...",
                    )

                with st.spinner("Extracting..."):
                    try:
                        results = extract_batch(
                            batch_inputs,
                            model,
                            processor,
                            device,
                            template_str,
                            examples_parsed,
                            max_new_tokens=max_new_tokens,
                            chunk_size=batch_size,
                            progress_callback=update_progress,
                        )
                    except RuntimeError as e:
                        st.error(f"Runtime error: {e}. Try reducing batch size.")
                        results = None

                if results is not None:
                    progress_bar.progress(1.0, text="Done.")
                    truncated_items = []

                    # Individual cards
                    for i, (pil_img, (result, was_truncated)) in enumerate(
                        zip(pil_images, results)
                    ):
                        if was_truncated:
                            truncated_items.append(filenames[i])
                        with st.expander(filenames[i], expanded=(i < 3)):
                            st.image(pil_img, width=300)
                            if result is not None:
                                st.json(result)
                            else:
                                st.error("Extraction failed.")

                    if truncated_items:
                        st.warning(
                            f"Possibly truncated: {truncated_items}"
                        )

                    # Summary table
                    fields = list(template_parsed.keys())
                    table_data = {"filename": filenames}
                    for field in fields:
                        table_data[field] = [
                            r.get(field, "") if r is not None else ""
                            for r, _ in results
                        ]
                    result_df = pd.DataFrame(table_data)

                    st.write("Summary")
                    st.dataframe(result_df, width="stretch")

                    total = len(results)
                    failed = sum(1 for r, _ in results if r is None)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total", total)
                    col2.metric("Extracted", total - failed)
                    col3.metric("Failed", failed)

                    st.download_button(
                        label="Download",
                        data=result_df.to_csv(index=False),
                        file_name="image_batch_extract.csv",
                        mime="text/csv",
                        key="image_batch_download",
                    )
    else:
        st.info("Upload one or more images.")
```

- [ ] **Step 25: Run all tests to verify nothing is broken**

Run: `uv run pytest tests/ -v`

Expected: All tests PASS.

- [ ] **Step 26: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Image Batch tab for multi-image extraction"
```

### Task 9: Add CSV image column support and tests

**Files:**
- Modify: `streamlit_app.py` (CSV tab block)
- Test: `tests/test_streamlit_app.py`

- [ ] **Step 27: Add CSV image loading tests**

Add at end of `tests/test_streamlit_app.py`:

```python
# --- CSV image loading ---


def test_load_csv_image_url(app):
    fake_img = MagicMock(name="fetched_img")
    with patch("qwen_vl_utils.fetch_image", return_value=fake_img) as mock_fetch:
        result = app._load_csv_image("https://example.com/img.png")
    assert result is fake_img
    mock_fetch.assert_called_once_with({"image": "https://example.com/img.png"})


def test_load_csv_image_http_url(app):
    fake_img = MagicMock(name="fetched_img")
    with patch("qwen_vl_utils.fetch_image", return_value=fake_img) as mock_fetch:
        result = app._load_csv_image("http://example.com/img.png")
    assert result is fake_img
    mock_fetch.assert_called_once_with({"image": "http://example.com/img.png"})


def test_load_csv_image_file_path(app, tmp_path):
    from PIL import Image as PILImage
    img_path = tmp_path / "test.png"
    PILImage.new("RGB", (10, 10)).save(img_path)
    result = app._load_csv_image(str(img_path))
    assert result is not None
    assert hasattr(result, "size")


def test_load_csv_image_invalid_path(app):
    result = app._load_csv_image("/nonexistent/path/image.png")
    assert result is None


def test_load_csv_image_empty_string(app):
    result = app._load_csv_image("")
    assert result is None


def test_load_csv_image_nan(app):
    result = app._load_csv_image("nan")
    assert result is None


def test_load_csv_image_none(app):
    result = app._load_csv_image(None)
    assert result is None
```

- [ ] **Step 28: Add `_load_csv_image` helper and update CSV tab**

Add `_load_csv_image` in `streamlit_app.py` after `_clear_device_cache`:

```python
def _load_csv_image(value):
    """Load an image from a URL or file path. Returns PIL Image or None."""
    if not value or not str(value).strip():
        return None
    value = str(value).strip()
    if value.lower() == "nan":
        return None
    if value.startswith(("http://", "https://")):
        try:
            from qwen_vl_utils import fetch_image
            return fetch_image({"image": value})
        except (OSError, ValueError, RuntimeError):
            return None
    try:
        return Image.open(value)
    except (OSError, ValueError):
        return None
```

Note: `fetch_image` is imported locally from `qwen_vl_utils` (not through `utils.py`) to avoid coupling to an internal import detail. No changes needed to `utils.py` or the existing imports in `streamlit_app.py`.

Then update the CSV tab block. Replace the section starting at `selected_column = st.selectbox(` through the existing processing loop. The key changes:

1. After the text column selector, add an image column selector
2. When image column is selected, show batch size slider and use `extract_batch` with image loading
3. When no image column, keep existing `extract()` loop unchanged

Replace the CSV tab's inner content (from `selected_column` through the end of the processing block):

```python
            selected_column = st.selectbox(
                "Select text column", options=df.columns.tolist()
            )

            image_col_options = ["None"] + [
                c for c in df.columns.tolist() if c != selected_column
            ]
            selected_image_column = st.selectbox(
                "Select image column (optional)",
                options=image_col_options,
                key="csv_image_column",
            )
            use_images = selected_image_column != "None"

            if use_images:
                csv_batch_size = st.slider(
                    "Batch size",
                    min_value=1,
                    max_value=8,
                    value=4,
                    step=1,
                    key="csv_batch_size",
                    help="Images per inference batch. Lower if running out of memory.",
                )

            if st.button("Extract", type="primary", key="csv_extract"):
                if not _has_config_errors(
                    template_error, examples_error, template_parsed
                ):
                    converted = _convert_template_if_needed(json_str, source_format)
                    if converted:
                        template_str = converted

                    if use_images:
                        # Batched image+text extraction
                        batch_inputs = []
                        invalid_image_rows = []
                        texts = df[selected_column].astype(str).tolist()
                        img_vals = df[selected_image_column].astype(str).tolist()
                        for i, (text_val, img_val) in enumerate(
                            zip(texts, img_vals)
                        ):
                            loaded_image = _load_csv_image(img_val)
                            if loaded_image is not None:
                                batch_inputs.append(
                                    {
                                        "text": None,
                                        "image": loaded_image,
                                        "context": text_val,
                                    }
                                )
                            else:
                                if img_val.strip() and img_val.lower() != "nan":
                                    invalid_image_rows.append(i + 1)
                                batch_inputs.append(
                                    {
                                        "text": text_val,
                                        "image": None,
                                        "context": None,
                                    }
                                )

                        if invalid_image_rows:
                            st.warning(
                                f"Could not load images for rows: {invalid_image_rows}. "
                                "Falling back to text-only for those rows."
                            )

                        progress_bar = st.progress(0, text="Starting...")
                        results = None
                        skipped_rows = []
                        truncated_rows = []

                        def update_csv_progress(completed, total):
                            progress_bar.progress(
                                completed / total,
                                text=f"Processing {completed} of {total} rows...",
                            )

                        with st.spinner("Extracting..."):
                            try:
                                results_tuples = extract_batch(
                                    batch_inputs,
                                    model,
                                    processor,
                                    device,
                                    template_str,
                                    examples_parsed,
                                    max_new_tokens=max_new_tokens,
                                    chunk_size=csv_batch_size,
                                    progress_callback=update_csv_progress,
                                )
                                results = [r for r, _ in results_tuples]
                                truncated_rows = [
                                    i + 1
                                    for i, (_, trunc) in enumerate(results_tuples)
                                    if trunc
                                ]
                            except RuntimeError as e:
                                st.error(
                                    f"Runtime error: {e}. Try reducing batch size."
                                )

                        if results is not None:
                            progress_bar.progress(1.0, text="Done.")
                            if skipped_rows:
                                st.warning(
                                    f"Rows skipped (input too long): {skipped_rows}"
                                )
                            if truncated_rows:
                                st.warning(
                                    f"Rows possibly truncated: {truncated_rows}"
                                )

                            fields = list(template_parsed.keys())
                            for field in fields:
                                df[field] = [
                                    r.get(field, "")
                                    if r is not None
                                    else ""
                                    for r in results
                                ]

                            st.write("Preview")
                            st.dataframe(
                                df[[selected_column] + fields].head(),
                                width="stretch",
                            )

                            total = len(df)
                            failed = sum(1 for r in results if r is None)

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total Rows", total)
                            col2.metric("Extracted", total - failed)
                            col3.metric("Failed", failed)

                            base_name = uploaded_file.name.rsplit(".", 1)[0]
                            st.download_button(
                                label="Download",
                                data=df.to_csv(index=False),
                                file_name=f"{base_name}_extract.csv",
                                mime="text/csv",
                            )
                    else:
                        # Text-only: existing extract() loop
                        results = []
                        progress_bar = st.progress(0, text="Starting...")
                        skipped_rows = []
                        truncated_rows = []
                        for i, text in enumerate(
                            df[selected_column].astype(str)
                        ):
                            try:
                                result, was_truncated = extract(
                                    text,
                                    model,
                                    processor,
                                    device,
                                    template_str,
                                    examples_parsed,
                                    max_new_tokens=max_new_tokens,
                                )
                                if was_truncated:
                                    truncated_rows.append(i + 1)
                            except (ValueError, RuntimeError):
                                result = None
                                skipped_rows.append(i + 1)
                            results.append(result)
                            progress_bar.progress(
                                (i + 1) / len(df),
                                text=f"Processing row {i + 1} of {len(df)}",
                            )

                        progress_bar.progress(1.0, text="Done.")
                        if skipped_rows:
                            st.warning(
                                f"Rows skipped (input too long): {skipped_rows}"
                            )
                        if truncated_rows:
                            st.warning(
                                f"Rows possibly truncated: {truncated_rows}"
                            )

                        fields = list(template_parsed.keys())
                        for field in fields:
                            df[field] = [
                                r.get(field, "")
                                if r is not None
                                else ""
                                for r in results
                            ]

                        st.write("Preview")
                        st.dataframe(
                            df[[selected_column] + fields].head(),
                            width="stretch",
                        )

                        total = len(df)
                        failed = sum(1 for r in results if r is None)

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Rows", total)
                        col2.metric("Extracted", total - failed)
                        col3.metric("Failed", failed)

                        base_name = uploaded_file.name.rsplit(".", 1)[0]
                        st.download_button(
                            label="Download",
                            data=df.to_csv(index=False),
                            file_name=f"{base_name}_extract.csv",
                            mime="text/csv",
                        )
```

- [ ] **Step 29: Run all tests**

Run: `uv run pytest tests/ -v`

Expected: All tests PASS.

- [ ] **Step 30: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add CSV image column support and _load_csv_image helper"
```

---

## Chunk 4: Final verification and docs

### Task 10: Lint, format, type check

- [ ] **Step 31: Run linter**

Run: `uv run ruff check .`

Expected: No errors. If there are, fix them.

- [ ] **Step 32: Run formatter**

Run: `uv run ruff format .`

Expected: Files formatted.

- [ ] **Step 33: Run type checker**

Run: `uv run ty check`

Expected: No new errors.

- [ ] **Step 34: Run full test suite**

Run: `uv run pytest tests/ -v`

Expected: All tests PASS.

- [ ] **Step 35: Commit any fixes**

```bash
git add streamlit_app.py utils.py tests/test_streamlit_app.py tests/test_utils.py
git commit -m "chore: lint and format batch vision support"
```

### Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 36: Update CLAUDE.md architecture section**

Update the architecture section to document:
- `extract_batch()` function signature and behavior
- `_clear_device_cache()` and `_load_csv_image()` helpers
- `_build_message()` and `_process_chunk()` and `_run_batch_inference()` internal functions
- Image Batch tab description
- CSV tab image column extension
- Updated `process_all_vision_info` batch support description
- `fetch_image` re-export from utils

- [ ] **Step 37: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for batch vision support"
```
