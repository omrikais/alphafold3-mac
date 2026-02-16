Use Conventional Commits format:

<type>: <description>

[body]

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation only
- refactor: Code refactoring (no functional change)
- test: Adding or updating tests
- chore: Maintenance tasks, dependencies
- perf: Performance improvements
- breaking: Breaking change (or use feat!: with bang)

Rules:
- Lowercase type prefix
- Imperative mood: "Add feature" not "Added feature"
- First line under 72 characters
- Be specific: "fix: Resolve null pointer in user auth" not "fix: Bug fix"
- Use HEREDOC for multi-line messages

If no changes exist, report that and don't create empty commits.