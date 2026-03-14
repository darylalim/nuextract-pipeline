import json
from unittest.mock import MagicMock, patch

import torch


def _make_mocks(decode_output):
    """Create mock model and processor that produce the given decode output."""
    processor = MagicMock()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    inputs_dict = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    proc_result = MagicMock()
    proc_result.to.return_value = inputs_dict
    processor.return_value = proc_result
    processor.tokenizer.apply_chat_template.return_value = "formatted prompt"
    processor.batch_decode.return_value = [decode_output]

    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 10, 20, 30]])

    return model, processor


# --- generate_template ---


def test_generate_template_returns_dict_on_valid_output():
    from utils import generate_template

    output = json.dumps({"name": "string", "age": "integer"})
    model, processor = _make_mocks(output)
    result, error = generate_template("extract name and age", model, processor, "cpu")
    assert result == {"name": "string", "age": "integer"}
    assert error is None


def test_generate_template_passes_template_none():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = _make_mocks(output)
    generate_template("extract name", model, processor, "cpu")

    call_args = processor.tokenizer.apply_chat_template.call_args
    assert call_args[1]["template"] is None


def test_generate_template_passes_description_as_message():
    from utils import generate_template

    output = json.dumps({"name": "string"})
    model, processor = _make_mocks(output)
    generate_template("extract the person's name", model, processor, "cpu")

    call_args = processor.tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert messages == [{"role": "user", "content": "extract the person's name"}]


def test_generate_template_invalid_output_returns_error():
    from utils import generate_template

    model, processor = _make_mocks("not valid json {{{")
    result, error = generate_template("extract stuff", model, processor, "cpu")
    assert result is None
    assert error is not None


def test_generate_template_model_error_propagates():
    import pytest
    from utils import generate_template

    model, processor = _make_mocks(json.dumps({"name": "string"}))
    model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        generate_template("extract name", model, processor, "cpu")


# --- process_all_vision_info ---


def test_process_all_vision_info_example_images_and_message_image():
    from utils import process_all_vision_info

    fake_example_img = MagicMock(name="example_img")
    fake_message_img = MagicMock(name="message_img")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]
    examples = [
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"name": "John"}',
        }
    ]

    with (
        patch(
            "utils.process_vision_info",
            return_value=([fake_message_img], None),
        ),
        patch("utils.fetch_image", return_value=fake_example_img),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_example_img, fake_message_img]


def test_process_all_vision_info_no_images_returns_none():
    from utils import process_all_vision_info

    messages = [{"role": "user", "content": "just text"}]

    with patch(
        "utils.process_vision_info",
        return_value=(None, None),
    ):
        result = process_all_vision_info(messages, None)

    assert result is None


def test_process_all_vision_info_text_examples_ignored():
    from utils import process_all_vision_info

    fake_message_img = MagicMock(name="message_img")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]
    examples = [
        {"input": "just text", "output": '{"name": "John"}'},
    ]

    with patch(
        "utils.process_vision_info",
        return_value=([fake_message_img], None),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_message_img]


def test_process_all_vision_info_mixed_examples():
    from utils import process_all_vision_info

    fake_example_img = MagicMock(name="example_img")
    fake_message_img = MagicMock(name="message_img")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "https://example.com/input.png"}],
        }
    ]
    examples = [
        {"input": "just text", "output": '{"name": "Alice"}'},
        {
            "input": {"type": "image", "image": "https://example.com/ex.png"},
            "output": '{"name": "Bob"}',
        },
    ]

    with (
        patch(
            "utils.process_vision_info",
            return_value=([fake_message_img], None),
        ),
        patch("utils.fetch_image", return_value=fake_example_img),
    ):
        result = process_all_vision_info(messages, examples)

    assert result == [fake_example_img, fake_message_img]
