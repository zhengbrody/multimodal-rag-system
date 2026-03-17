# Contributing Guide

Thanks for your interest in contributing! This project focuses on a clean, reproducible RAG demo suitable for portfolios and interview discussions.

## Quick Start (Local)

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements_simple.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Formatter: `black`
- Linting: `ruff`
- Type checks: `mypy` (best-effort, not required for all modules)

```bash
black src/ tests/
ruff check src/ tests/
```

## Pull Requests

1. Create a feature branch from `main`.
2. Keep changes focused and scoped.
3. Add tests for new behavior.
4. Update documentation if behavior changes.
5. Open a PR with a clear summary and testing notes.

## Reporting Issues

Please include:

- What you expected vs what happened
- Steps to reproduce
- Logs or error output if available
- OS and Python version

