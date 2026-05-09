# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the StoryMesh repository.

## Overview

This is a single repository for an agentic ai pipeline that will take a user supplied prompt and output a novel fictional plot synopsis.

### Components

For components review the README.md in the root of the repository. README.md is maintained as a best-effort record of the current pipeline state; treat it as a starting point, not a source of truth. When README.md conflicts with the code, the code takes precedence. If a significant discrepancy is found, flag it and update the README as part of the work.

### Core Concents

Communication is controlled by tightly binding Pydantic contracts at each step.  
Input and output from each stage of the pipeline is stored as JSON to provide a mechanism for auditing the entire pipeline.  
For version data about dependencies always review pyproject.toml in the root of the repository.  

### Technical Details

Prompts should ALWAYS reside in dedicated data files and never be embedded in agent code.  
Voice profiles (`src/storymesh/prompts/voice_profiles/*.yaml`) are prompt-adjacent data — they contain overlay text and exemplars that are injected into system prompts at runtime. They follow the same "data files only" rule.  
Model selection and temperature should be active decisions to balance accuracy against cost.  

### Python Best Practices

1. Follow the PEP 8 style guide at all times
2. Always use static typing hings and validate with MyPy
3. Ensure stylistic compliance with Ruff
4. Proper exception handling at all times
5. Aim for 100% coverage of tests and evaluate any non-covered code
6. Have accurate and descriptive docstrings

## Specific Instructions

ALWAYS propose code and discuss the plan. The primary goal of this project is to become familiar with best practices and strong development patters for agentic workflows.  
NEVER modify or create files without explicit consent.  
Do NOT treat README.md as a bible. If a better implementation, tool, architecture, or other topics conflict with README.md ALWAYS highlight this as a topic for discussion.

When engagint with frontend design always append your system message with the contents of the DISTILLED_AESTHETICS_PROMPT.

DISTILLED_AESTHETICS_PROMPT = """
<frontend_aesthetics>
You tend to converge toward generic, "on distribution" outputs. In frontend design, this creates what users call the "AI slop" aesthetic. Avoid this: make creative, distinctive frontends that surprise and delight. Focus on:

Typography: Choose fonts that are beautiful, unique, and interesting. Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics.

Color & Theme: Commit to a cohesive aesthetic. Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes. Draw from IDE themes and cultural aesthetics for inspiration.

Motion: Use animations for effects and micro-interactions. Prioritize CSS-only solutions for HTML. Use Motion library for React when available. Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions.

Backgrounds: Create atmosphere and depth rather than defaulting to solid colors. Layer CSS gradients, use geometric patterns, or add contextual effects that match the overall aesthetic.

Avoid generic AI-generated aesthetics:
- Overused font families (Inter, Roboto, Arial, system fonts)
- Clichéd color schemes (particularly purple gradients on white backgrounds)
- Predictable layouts and component patterns
- Cookie-cutter design that lacks context-specific character

Interpret creatively and make unexpected choices that feel genuinely designed for the context. Vary between light and dark themes, different fonts, different aesthetics. You still tend to converge on common choices (Space Grotesk, for example) across generations. Avoid this: it is critical that you think outside the box!
</frontend_aesthetics>
"""

