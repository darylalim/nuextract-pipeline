#!/usr/bin/env bash
# PostToolUse hook (Write|Edit|MultiEdit): format, lint-fix, and type-check an
# edited Python file — in that order — so the working tree stays CI-green.
#
# Runs the same ruff + ty gates CI enforces. ruff format and `ruff check --fix`
# mutate the file in place; anything unfixable (undefined names, type errors) is
# fed back to Claude via exit code 2 so it gets corrected immediately.
set -uo pipefail

# Team-shared: if a teammate lacks jq, degrade to a silent no-op instead of
# erroring on every edit (see .claude/hooks/README.md for prerequisites).
command -v jq >/dev/null 2>&1 || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# The edited file path arrives in the tool-call JSON on stdin.
file_path="$(jq -r '.tool_input.file_path // empty')"
[ -z "$file_path" ] && exit 0
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;   # only Python files are formatted/type-checked
esac
[ -f "$file_path" ] || exit 0

# 1. Format + auto-fix (these rewrite the file in place; noise suppressed).
uv run ruff format "$file_path" >/dev/null 2>&1

# 2. Collect anything the auto-fixer could not resolve, plus type errors.
problems=""
if ! fix_out="$(uv run ruff check --fix "$file_path" 2>&1)"; then
  problems+="ruff (unfixable lint issues):\n${fix_out}\n\n"
fi
# ty checks the whole project (its unit of analysis), not just this file.
if ! ty_out="$(uv run ty check 2>&1)"; then
  problems+="ty (type errors):\n${ty_out}\n"
fi

if [ -n "$problems" ]; then
  printf 'Post-edit checks failed for %s:\n\n%b' "$file_path" "$problems" >&2
  exit 2   # surface the failures to Claude as actionable feedback
fi
exit 0
