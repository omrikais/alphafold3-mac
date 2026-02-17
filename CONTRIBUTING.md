# Contributing to AlphaFold 3 for Mac

Thank you for your interest in contributing! We welcome bug fixes, documentation
improvements, and feature suggestions.

## Getting Started

1. **Fork** the repository and clone your fork.
2. Run the installer to set up your development environment:
   ```bash
   ./scripts/install.sh
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. Create a branch for your change:
   ```bash
   git checkout -b my-fix
   ```

## Development

- Use `python3` (not `python`) for all commands.
- Run tests before submitting:
  ```bash
  PYTHONPATH=src pytest tests/unit/ -v
  ```
- Build the docs locally to check your changes:
  ```bash
  mkdocs build --strict
  ```
- Build the frontend if you changed UI code:
  ```bash
  cd frontend && npm run build
  ```

## Submitting a Pull Request

1. Keep PRs focused — one logical change per PR.
2. Write a clear title and description explaining **what** changed and **why**.
3. Ensure all CI checks pass (docs build, frontend tests, path-leak-check).
4. All PRs require review from [@omrikais](https://github.com/omrikais).
5. Signed commits are required on `main`. Configure your signing key:
   ```bash
   git config commit.gpgsign true
   ```

## Reporting Bugs

Use the [Bug Report](https://github.com/omrikais/alphafold3-mac/issues/new?template=bug_report.yml)
issue template. Include:

- macOS version and chip (e.g., M4 Max, 128 GB)
- Python and MLX versions
- Steps to reproduce
- Input JSON (if applicable, with sensitive sequences redacted)

## Feature Requests

Use the [Feature Request](https://github.com/omrikais/alphafold3-mac/issues/new?template=feature_request.yml)
issue template.

## Style

- Follow existing code conventions — no need to reformat unrelated code.
- Add tests for bug fixes when feasible.
- Keep commit messages concise and use [Conventional Commits](https://www.conventionalcommits.org/) format.

## License

By contributing, you agree that your contributions will be licensed under the
same [CC-BY-NC-SA 4.0](LICENSE) license that covers the project.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please
read it before participating.
