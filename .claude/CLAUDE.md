# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the StoryMesh repository.

## Overview

This is a single repository for an agentic ai pipeline that will take a user supplied prompt and output a novel fictional plot synopsis.

### Components

For components review the README.md in the root of the repository. This file is always up to date and has information related to the list of agents and responsibility scope for each agent. 

### Core Concents

Communication is controlled by tightly binding Pydantic contracts at each step.  
Input and output from each stage of the pipeline is stored as JSON to provide a mechanism for auditing the entire pipeline.  
For version data about dependencies always review pyproject.toml in the root of the repository.  

### Technical Details

Prompts should ALWAYS reside in dedicated data files and never be embedded in agent code.  
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
