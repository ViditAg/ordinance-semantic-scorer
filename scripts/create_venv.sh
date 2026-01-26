#!/usr/bin/env bash
set -euo pipefail

# Creates a local virtual environment and installs dependencies.
# Default env dir name is descriptive and repo-specific.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR_DEFAULT=".venv-ordinance-semantic-scorer"

VENV_DIR="${1:-$VENV_DIR_DEFAULT}"
REQ_FILE="${2:-requirements.txt}"

cd "$REPO_ROOT"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "requirements file not found: $REQ_FILE" >&2
  exit 1
fi

if [[ -d "$VENV_DIR" ]]; then
  echo "venv already exists: $VENV_DIR" >&2
  echo "Remove it first or pass a different name, e.g.:" >&2
  echo "  $0 .venv-ordinance-semantic-scorer-2" >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$REQ_FILE"

echo
printf "Created venv: %s\n" "$VENV_DIR"
printf "Activate with: source %s/bin/activate\n" "$VENV_DIR"
