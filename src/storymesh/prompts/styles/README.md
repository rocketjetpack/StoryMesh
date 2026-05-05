Prompt styles live in this directory.

- `default/` is the canonical style.
- `slim/` contains selective overrides for experiments with shorter prompts.

Prompt resolution order for style `X`:

1. `styles/X/<prompt>.yaml`
2. `styles/default/<prompt>.yaml`

This lets a style override only the files it wants to change.
