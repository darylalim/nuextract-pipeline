# Text Extraction Pipeline

General-purpose structured extraction pipeline using [NuExtract-2.0-4B](https://huggingface.co/numind/NuExtract-2.0-4B). Accepts text or images with a user-defined JSON template and optional in-context learning (ICL) examples to extract structured data. Streamlit web UI with text, image, and CSV batch processing tabs.

## Features

- **Text extraction** — paste text, define a JSON template, get structured output
- **Image extraction** — upload an image with optional context text
- **CSV batch processing** — extract from every row in a CSV file
- **Auto template generation** — describe what to extract in plain English and generate a JSON template
- **ICL examples** — provide input/output examples (text or image URLs) to guide extraction
- **Token limit** — enforces a 10,000 input token limit to prevent OOM errors

## Installation

```bash
uv sync
```

Optionally set your Hugging Face token for faster downloads:

```bash
export HF_TOKEN=hf_...
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

## Testing

```bash
uv run pytest
```

Sample test data is in `tests/data/csv/sample_persons.csv` (30 rows of synthetic person descriptions).

## Project Structure

```
streamlit_app.py          # Main app — UI, validation, extraction
utils.py                  # Utilities — template generation, vision processing
tests/
  test_streamlit_app.py   # Tests for main app (45 tests)
  test_utils.py           # Tests for utilities (13 tests)
  data/csv/               # Sample test data
```
