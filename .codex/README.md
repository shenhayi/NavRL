# Workspace Codex Agents

This workspace has a local multi-agent setup in [config.toml](./config.toml).

Recommended order
- `explorer`
- `planner`
- `test-writer`
- `reviewer`

Suggested prompt
```text
Use the workspace agents in this order: explorer first, then planner, then test-writer, and finally reviewer.
```

Suggested coding-task prompt
```text
Use the workspace agents in this order: explorer first to inspect the codebase, planner second to propose the implementation plan, test-writer third to implement the change, and reviewer last to review the result and conclude.
```

Notes
- This workspace-local setup makes the agents available for `~/projects/NavRL/NavRL-code/NavRL`.
- It does not by itself force automatic agent spawning on every task.
- The prompt above is the practical way to trigger the intended workflow consistently.
