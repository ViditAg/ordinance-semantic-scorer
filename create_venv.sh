#!/usr/bin/env bash
# Creates a local virtual environment and installs all dependencies.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-.venv-ordinance-semantic-scorer}"

cd "$REPO_ROOT"

if [[ -d "$VENV_DIR" ]]; then
  echo "venv already exists: $VENV_DIR" >&2
  echo "Remove it first or pass a different name, e.g.:" >&2
  echo "  $0 .my-venv" >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -r requirements.txt

echo
printf "Created venv: %s\n" "$VENV_DIR"
printf "Activate with: source %s/bin/activate\n" "$VENV_DIR"
printf "Run tests with: pytest\n"