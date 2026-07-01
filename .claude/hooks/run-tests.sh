#!/usr/bin/env bash
# Stop hook: run the pytest suite when Claude finishes a turn — but only when
# Python files actually changed, and never in a runaway loop.
#
# The suite never loads the real ~5GB model, so it is fast. Failures are fed
# back via exit code 2 so Claude fixes regressions before handing control back.
set -uo pipefail

# Team-shared: if a teammate lacks jq, degrade to a silent no-op instead of
# erroring on every turn (see .claude/hooks/README.md for prerequisites).
command -v jq >/dev/null 2>&1 || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# Loop guard: if this Stop was itself triggered by a prior Stop-hook block,
# stand down so we force at most one fix cycle instead of looping forever.
stop_active="$(jq -r '.stop_hook_active // false')"
[ "$stop_active" = "true" ] && exit 0

# Skip pure-conversation turns: only run when there are uncommitted .py changes.
py_changed="$(git status --porcelain 2>/dev/null | grep -E '\.py$' || true)"
[ -z "$py_changed" ] && exit 0

if ! test_out="$(uv run pytest -q 2>&1)"; then
  {
    echo "pytest failed after your changes — fix the failures before stopping:"
    echo
    printf '%s\n' "$test_out" | tail -c 4000
  } >&2
  exit 2
fi
exit 0
