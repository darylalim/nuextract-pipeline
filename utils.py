import json

import torch
import yaml
from qwen_vl_utils import fetch_image, process_vision_info

DEFAULT_MAX_NEW_TOKENS = 2048


def detect_and_convert_template(template_str):
    """Detect template format and convert to JSON.

    Returns (json_str, source_format, error) where source_format is
    "json", "yaml", "pydantic", "pydantic_with_unknown", or None.
    """
    if not template_str or not template_str.strip():
        return None, None, "Template must not be empty."

    # 1. Try JSON
    try:
        parsed = json.loads(template_str)
        if isinstance(parsed, dict):
            if not parsed:
                return None, None, "Template must not be empty."
            return template_str.strip(), "json", None
        else:
            return None, None, "Template must be a JSON object."
    except json.JSONDecodeError:
        pass

    # 2. Try YAML
    try:
        parsed = yaml.safe_load(template_str)
        if isinstance(parsed, dict) and parsed:
            return json.dumps(parsed, indent=2), "yaml", None
    except yaml.YAMLError:
        pass

    # 3. Try Pydantic (implemented in Task 6)

    # No format matched — treat as natural language
    return None, None, None


def generate_template(description, model, processor, device):
    """Generate a JSON extraction template from a natural language description.

    Uses the NuExtract model's native template generation mode by passing
    template=None to apply_chat_template.

    Returns (dict, None) on success, (None, error_message) on failure.
    """
    messages = [{"role": "user", "content": description}]
    formatted = processor.tokenizer.apply_chat_template(
        messages,
        template=None,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[formatted], images=None, padding=True, return_tensors="pt"
    ).to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
        )

    trimmed = output[:, input_len:]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    try:
        parsed = json.loads(decoded[0])
        return parsed, None
    except (json.JSONDecodeError, IndexError) as e:
        return None, f"Could not parse generated template: {e}"


def process_all_vision_info(messages, examples=None):
    """Extract images from both ICL examples and user messages.

    Returns a flat list of images in correct order (example images first,
    then message images), or None if no images found.
    """
    all_images = []

    if examples:
        for ex in examples:
            inp = ex.get("input")
            if isinstance(inp, dict) and inp.get("type") == "image":
                all_images.append(fetch_image(inp))

    message_images = process_vision_info(messages)[0] or []
    all_images.extend(message_images)

    return all_images if all_images else None
