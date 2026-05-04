Prompt styles live in this directory.

- `default/` is the canonical style. When a prompt file is not present there,
  the loader falls back to the legacy root prompt file under
  `src/storymesh/prompts/`.
- `slim/` contains selective overrides for experiments with shorter prompts.

Prompt resolution order for style `X`:

1. `styles/X/<prompt>.yaml`
2. `styles/default/<prompt>.yaml`
3. `<prompt>.yaml` at the prompts package root

This lets a style override only the files it wants to change.
