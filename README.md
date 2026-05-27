# NuExtract Pipeline

Extract structured information from documents, convert images to Markdown, and generate templates with the NuMind NuExtract3 model on Apple Silicon with MLX. Mirrors the [official NuExtract3 Hugging Face Space](https://huggingface.co/spaces/numind/NuExtract3) but runs entirely locally via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), no GPU or external API required.

## Features

- **Three modes** — structured JSON extraction, document-to-markdown conversion, and natural-language → template generation
- **Multimodal input** — upload an image (screenshot, scan, photo) and/or paste text
- **Typed template system** — `verbatim-string`, `string`, `integer`, `number`, `date`, `boolean`, enums, multi-enums, and more (see [TYPES.md on Hugging Face](https://huggingface.co/numind/NuExtract3/blob/main/TYPES.md))
- **Streaming output** — results stream in token-by-token
- **Optional reasoning mode** — model emits `<think>...</think>` traces shown in a dedicated pane
- **Download buttons** — save the JSON, markdown, or generated template to disk
- **Local Apple Silicon inference** — no API key, no network calls during extraction

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- ~6 GB free disk for the model (downloaded on first run)
- 16 GB unified memory recommended (more is better for long-context inference)

## Installation

```bash
uv sync
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

First run downloads the ~5 GB model. Subsequent runs use the local cache.

## Modes

| Button | Inputs | Output |
|---|---|---|
| **Extract JSON** | Image and/or text + JSON template | Structured JSON matching the template |
| **Convert to Markdown** | Image (required) | Clean markdown with HTML tables and embedded structure |
| **Generate template** | Image or text describing the document | A JSON template you can paste back into the editor |

## Testing

```bash
uv run pytest                              # Run all tests
uv run python scripts/probe_mlx_vlm.py     # End-to-end model probe (downloads + extracts)
```

## Project Structure

```
streamlit_app.py                    # UI: two-column layout, three buttons, streaming output
nuextract.py                        # mlx-vlm wrapper: load, render prompt, stream extraction
scripts/
  probe_mlx_vlm.py                  # Verifies model + template kwargs flow-through end-to-end
tests/
  conftest.py                       # sys.path setup
  test_nuextract.py                 # Wrapper tests (40)
  test_streamlit_app.py             # App helper tests (24)
  test_streamlit_app_apptest.py     # End-to-end UI tests via Streamlit AppTest (20)
```
