# Luxical Project Justfile
# Use `just <command>` to run commands

# Allow passing args starting with '-' without '--'
set positional-arguments := true

# The default command is `help`
default: help

# Show available commands
help:
    @just --list

# Set up local R&D Python environment for running examples
setup-dev:
    uv sync --group misc

# Run tests
test:
    @echo "Running Python tests..."
    uv run pytest tests/ -v
    # @echo "Running Rust tests..."  # Currently no rust tests.
    # cd arrow_tokenize && cargo test
    @echo "✅ All tests passed!"

# Clean build artifacts
clean:
    @echo "Cleaning build artifacts..."
    rm -rf dist/
    rm -rf arrow_tokenize/dist/
    rm -rf arrow_tokenize/target/
    rm -rf build/
    rm -rf src/luxical.egg-info/
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -delete
    @echo "✅ Clean complete!"

# Lint, format, and type check (one command)
lint:
    # Auto-format
    uv run ruff format .
    # Lint and auto-fix where safe
    uv run ruff check --fix .
    # Type check
    uv run pyright


## Arrow Tokenize Compilation (macos and Linux only for now) ##

# macOS wheel targets (x86_64 just in case)
MACOS_TARGETS := "aarch64-apple-darwin x86_64-apple-darwin"

# Build arrow_tokenize wheels for macOS locally across CPython 3.11–3.13
build-arrow-tokenize-macos-local:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building wheels for macOS locally (3.11, 3.12, 3.13)..."
    cd arrow_tokenize
    for PY in 3.11 3.12 3.13; do
    for TARGET in {{MACOS_TARGETS}}; do
    echo "==> Building for Python $PY on $TARGET"
    uv run \
      -p "$PY" \
      --group build \
      maturin pep517 build-wheel \
      --target="$TARGET" \
      --compatibility off
    done
    done
    echo "✅ MacOS wheels built for 3.11–3.13!"


# Build Linux wheels for multiple architectures using Docker
build-arrow-tokenize-linux-cross:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🐧 Building Linux wheels for x86_64 and aarch64 via ghcr.io/pyo3/maturin docker image..."
    for PY in 3.11 3.12 3.13; do
      echo "==> Building for Python ${PY} (x86_64)..."
      docker run --rm -v $(pwd)/arrow_tokenize:/io --platform linux/amd64 ghcr.io/pyo3/maturin build --release -i python${PY} --target x86_64-unknown-linux-gnu
      echo "==> Building for Python ${PY} (aarch64)..."
      docker run --rm -v $(pwd)/arrow_tokenize:/io --platform linux/arm64 ghcr.io/pyo3/maturin build --release -i python${PY} --target aarch64-unknown-linux-gnu
    done
    echo "✅ Linux cross-compilation complete! Wheels in arrow_tokenize/target/wheels/"



## Luxical/combined Compilation ##

# Build the luxical wheel
build-luxical:
    @echo "Building luxical locally..."
    uv build --wheel
    @echo "✅ Luxical local wheels built!"


# Build all arrow_tokenize wheels
build-arrow-tokenize: build-arrow-tokenize-macos-local build-arrow-tokenize-linux-cross

# Build arrow_tokenize and luxical wheels for local macos usage
build-local: build-arrow-tokenize-macos-local build-luxical

# Build all wheels
build: build-arrow-tokenize build-luxical

# Publish arrow tokenize wheels to PyPI
publish-wheel-arrow-tokenize: clean
    @echo "Publishing arrow_tokenize wheels to PyPI..."
    uv publish --wheel arrow_tokenize/target/wheels/*
    @echo "✅ arrow_tokenize wheels published!"

# Publish luxical wheel to PyPI
publish-wheel-luxical: clean
    @echo "Publishing luxical wheel to PyPI..."
    uv publish --wheel dist/*
    @echo "✅ luxical wheel published!"
