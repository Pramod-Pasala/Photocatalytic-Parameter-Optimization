#!/usr/bin/env bash
set -e

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "uv version:"
uv --version

echo "Python version:"
python --version

echo "Installing dependencies with uv..."
if [ -f "uv.lock" ] || [ -f "pyproject.toml" ]; then
  uv sync
else
  echo "No pyproject.toml or uv.lock found — skipping dependency install"
fi

echo "Setup complete."
