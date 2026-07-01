# Claude Code hooks (team-shared)

These hooks run automatically inside every teammate's Claude Code session (they
do **not** run in CI). They enforce the same gates CI does, at edit time.

Wired up in [`../settings.json`](../settings.json).

## Prerequisites

| Tool | Why | Install |
| --- | --- | --- |
| [`uv`](https://docs.astral.sh/uv/) | runs `ruff` / `ty` / `pytest` via `uv run` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [`jq`](https://jqlang.github.io/jq/) | parses the tool-call JSON on stdin | `brew install jq` |

If `jq` is missing the convenience hooks (`py-checks`, `run-tests`) **degrade to
a silent no-op** (they won't error or spam) — you lose their benefit until you
install it. `block-secrets` is the exception: it falls back to a jq-free path
check so secret protection never silently disappears.

## The hooks

| Script | Event | Behavior |
| --- | --- | --- |
| [`py-checks.sh`](py-checks.sh) | `PostToolUse` (Write/Edit) | On a `.py` edit: `ruff format` → `ruff check --fix` (silent), then reports any unfixable lint + `ty check` errors back to Claude. Format + lint + type-check are one ordered script to avoid a parallel-hook race on the file. |
| [`block-secrets.sh`](block-secrets.sh) | `PreToolUse` (Write/Edit) | Denies edits to `.env*` and `.streamlit/secrets.toml` (case-insensitive). **Accident guardrail, not a security boundary** — it only covers Write/Edit/MultiEdit, so a Bash write (`echo x > .env`) is not intercepted by design. |
| [`run-tests.sh`](run-tests.sh) | `Stop` | Runs `uv run pytest -q` when the turn changed any `.py` file. Loop-guarded via `stop_hook_active`; failures are fed back so Claude fixes regressions before stopping. |

## Notes

- **Activation:** Claude Code snapshots hooks at session start. After pulling
  changes to these files, restart the session (or run `/hooks` to confirm they
  loaded) before they take effect.
- **Personal overrides** go in `.claude/settings.local.json`, which stays
  gitignored — use it for machine-specific tweaks without touching the shared set.
