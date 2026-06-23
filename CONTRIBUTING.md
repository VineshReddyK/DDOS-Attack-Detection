# Contributing to DDoS Attack Detection System

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository and clone your fork
2. Ensure you have Python 3.9+, pip, and Docker installed
3. Install dependencies: `pip install -r requirements.txt`
4. Download the CIC-DDoS2019 dataset and place it under `data/raw/` (see README for details)
5. Run tests: `pytest`

## Branching

- Branch from `main`
- Use descriptive branch names: `feat/new-model`, `fix/cnn-lstm-overfitting`, `chore/upgrade-sklearn`

## Making Changes

- Keep changes focused — one feature or fix per PR
- Add tests for any new model logic or API changes
- If adding or modifying a model, include updated performance metrics in your PR description (accuracy, F1, inference latency)
- Run `pre-commit run --all-files` before opening a PR (pre-commit hooks are configured)

## Pull Request Guidelines

- Write a clear PR title and description explaining the **why**
- Link related issues with `Closes #<issue>`
- Model changes should include a brief comparison against the baseline metrics in the README

## Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Keep model training scripts reproducible — set random seeds explicitly

## Reporting Issues

Open a GitHub Issue with:
- Clear title and description
- Steps to reproduce (for bugs)
- Python version, OS, and relevant package versions (`pip freeze`)
- Expected vs actual behavior

## Questions

Reach out via [email](mailto:vineshreddyy.k@gmail.com) or open a Discussion on GitHub.
