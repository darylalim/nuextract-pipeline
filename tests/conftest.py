import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch

_TESTS_DIR = str(Path(__file__).resolve().parent)
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def make_mocks(decode_output):
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


def make_batch_mocks(decode_outputs, input_lengths=None):
    """Create mock model and processor for batch inference.

    decode_outputs: list of decoded strings, one per batch item.
    input_lengths: list of non-padded token counts per item. Defaults to [5]*n.
    """
    n = len(decode_outputs)
    if input_lengths is None:
        input_lengths = [5] * n
    max_len = max(input_lengths) if input_lengths else 0
    gen_tokens = 3

    processor = MagicMock()

    # Left-padded input_ids and attention_mask
    input_ids = (
        torch.ones(n, max_len, dtype=torch.long)
        if n > 0
        else torch.ones(0, 0, dtype=torch.long)
    )
    attention_mask = (
        torch.zeros(n, max_len, dtype=torch.long)
        if n > 0
        else torch.zeros(0, 0, dtype=torch.long)
    )
    for i, length in enumerate(input_lengths):
        pad = max_len - length
        attention_mask[i, pad:] = 1

    # Track which call we're on (for chunk_size=1 scenarios)
    call_counter = [0]

    def make_proc_result(text_arg):
        """Return processor result sized to the number of texts passed."""
        batch_n = len(text_arg) if isinstance(text_arg, list) else n
        if batch_n == n:
            ids = input_ids
            mask = attention_mask
        else:
            # Map call to correct rows for single-item chunks
            start = call_counter[0]
            end = min(start + batch_n, n)
            ids = input_ids[start:end]
            mask = attention_mask[start:end]
            if ids.shape[0] < batch_n:
                ids = input_ids[:batch_n]
                mask = attention_mask[:batch_n]
            call_counter[0] = end
        result_dict = {"input_ids": ids, "attention_mask": mask}
        proc_mock = MagicMock()
        proc_mock.to.return_value = result_dict
        return proc_mock

    def proc_side_effect(*args, **kwargs):
        text_arg = kwargs.get("text", args[0] if args else [])
        return make_proc_result(text_arg)

    processor.side_effect = proc_side_effect
    processor.tokenizer.apply_chat_template.return_value = "formatted prompt"
    processor.batch_decode.return_value = decode_outputs

    model = MagicMock()
    if n > 0:
        output = torch.cat(
            [input_ids, torch.ones(n, gen_tokens, dtype=torch.long) * 10], dim=1
        )
    else:
        output = torch.ones(0, 0, dtype=torch.long)
    model.generate.return_value = output

    return model, processor
