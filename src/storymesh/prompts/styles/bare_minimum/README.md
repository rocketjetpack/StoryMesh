Bare-minimum prompt style.

Use this as:

- a very short prompt style for experiments
- a starter template for creating a custom style

Run it with:

`storymesh generate "your prompt" --prompt-style bare_minimum`

What is included here:

- the main creative and editorial prompts
- short instructions with the same placeholders and output schemas as the
  default style

What is intentionally not included here:

- some lower-level utility prompts such as `genre_normalizer`,
  `genre_inference`, and `book_ranker`

Any prompt file missing from this folder automatically falls back to:

1. `styles/default/<prompt>.yaml`

How to make your own style:

1. Copy this folder to a new name, for example `styles/my_style/`
2. Edit only the prompt files you want to change
3. Run with `--prompt-style my_style`

Rule of thumb:

- Keep the schema and placeholders unchanged
- Change the instruction text freely
- If you delete a prompt file from your style, StoryMesh will inherit it from
  the default style
