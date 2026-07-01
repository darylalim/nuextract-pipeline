#!/usr/bin/env bash
# PreToolUse hook (Write|Edit|MultiEdit): refuse to modify secrets files.
#
# .env* and .streamlit/secrets.toml are gitignored credential stores. Blocking
# them here means an accidental Write/Edit can never clobber them; edit by hand
# if genuinely intended.
#
# SCOPE — this is an accident guardrail, NOT a hardened security boundary. It
# only covers the file-editing tools it is matched on (Write/Edit/MultiEdit); a
# Bash write such as `echo x > .env` is deliberately NOT intercepted, since a
# command-string blocklist (printf/tee/cp/sed -i/python ...) is trivially evaded
# and would give false confidence. Treat it as defense-in-depth.
#
# Unlike the convenience hooks, this one does NOT fail open when jq is absent:
# it falls back to a jq-free path extraction so secret protection never silently
# disappears on a machine without jq.
set -uo pipefail
shopt -s nocasematch 2>/dev/null || true  # also catch .ENV on case-insensitive filesystems

payload="$(cat)"
if command -v jq >/dev/null 2>&1; then
  file_path="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // empty')"
elif [[ "$payload" =~ \"file_path\"[[:space:]]*:[[:space:]]*\"([^\"]*)\" ]]; then
  # jq-free fallback using only bash builtins — no external tools required, so
  # the security hook cannot fail open just because jq/sed/head are absent.
  file_path="${BASH_REMATCH[1]}"
else
  file_path=""
fi

case "$file_path" in
  .env|.env.*|*/.env|*/.env.* | \
  .streamlit/secrets.toml|*/.streamlit/secrets.toml)
    echo "Blocked by project hook: refusing to edit secrets file '$file_path'." \
         "(.env* / .streamlit/secrets.toml hold credentials and are gitignored." \
         "Edit it manually if this is intended.)" >&2
    exit 2   # exit 2 on PreToolUse denies the tool call, stderr shown to Claude
    ;;
esac
exit 0
