# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| latest  | Yes                |

We only support the latest version on the `main` branch. Please ensure you are
using the most recent code before reporting a vulnerability.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.
**Do not open a public issue.**

Use GitHub's **Private Vulnerability Reporting** to submit a report:

1. Go to the [Security tab](https://github.com/omrikais/alphafold3-mac/security)
   of this repository.
2. Click **Report a vulnerability**.
3. Fill in a description, steps to reproduce, potential impact, and a suggested
   fix if you have one.

You should receive an acknowledgement within 72 hours. We will work with you to
understand the issue and coordinate a fix before any public disclosure.

## Scope

This project runs locally on the user's machine. The primary security concerns
are:

- **Model weight handling** — weights are loaded from local disk; ensure file
  permissions are appropriate.
- **Web API** — the FastAPI server binds to `127.0.0.1` by default (not
  externally accessible). If you expose it to a network, apply appropriate
  access controls.
- **Input parsing** — malformed JSON or mmCIF inputs should not cause code
  execution or memory corruption.
- **Dependencies** — we monitor upstream advisories for MLX, FastAPI, uvicorn,
  and other dependencies.

## Dependency Updates

We use Dependabot and periodic manual review to keep dependencies current. If
you notice a vulnerable dependency, please open an issue or email us.
