# NuExtract Pipeline

Structured extraction pipeline using [NuExtract-1.5-MLX-8bit](https://huggingface.co/mlx-community/numind-NuExtract-1.5-MLX-8bit). Accepts text with a user-defined template to extract structured data. Streamlit web UI with text and CSV batch processing tabs. Optimized for Apple Silicon via MLX.

## Features

- **Text extraction** — paste text, define a template, get structured JSON output
- **CSV batch processing** — extract from every row in a CSV file with progress tracking
- **Multi-format templates** — accepts JSON, YAML, or Pydantic model definitions
- **Extraction presets** — 5 built-in presets (Person, Job Posting, Invoice, Product, Scientific Paper)
- **Configurable output length** — inline slider for max new tokens (64–4096, default 2048)
- **Token limit** — enforces a 4,096 input token limit to prevent memory issues
- **Multi-language** — supports English, French, Spanish, German, Portuguese, Italian

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+

## Installation

```bash
uv sync
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

The model (~4 GB) is downloaded automatically on first run.

## Testing

```bash
uv run pytest
```

## Project Structure

```
streamlit_app.py          # Main app — UI, model loading, extraction
utils.py                  # Template format detection and conversion
presets.json              # 5 built-in extraction presets
tests/
  conftest.py             # Shared test fixtures
  test_streamlit_app.py   # App tests (48 tests)
  test_utils.py           # Utility tests (16 tests)
  data/csv/               # Sample test data
```
