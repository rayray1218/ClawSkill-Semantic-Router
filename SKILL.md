---
name: semantic-model-router
description: Routes AI queries to the optimal LLM tier (Elite/Balanced/Basic) using semantic similarity. Saves cost by sending simple queries to cheap models and complex ones to powerful models. Use this skill before any LLM call to pick the right model.
version: 1.2.0
author: Ray
tags: [llm-ops, routing, cost-saving, efficiency, orchestration]
homepage: https://github.com/rayray1218/ClawSkill-Semantic-Router
files: ["scripts/*"]
dependencies:
  - sentence-transformers>=2.2.2
  - numpy>=1.24.0
requires:
  env: []
---

# Semantic Model Orchestrator

An intelligent middleware layer that analyzes user query complexity and routes it to the most cost-effective LLM tier — without sacrificing response quality.

## When to Use This Skill

Use this skill **before every main LLM call** to determine which model tier is most appropriate. The agent should:

1. Call `ModelRouter.route(query)` with the user's query.
2. Read the returned `tier` and `model` fields.
3. Forward the query to that model.

## How It Works

```python
from scripts.model_router import ModelRouter

router = ModelRouter()
result = router.route("Design a distributed caching layer for a fintech platform.")
# Returns: {"tier": "ELITE", "model": "anthropic/claude-3-5-sonnet-latest", "confidence": 0.97}
```

## Model Tiers

| Tier | Default Model | Trigger Examples |
|---|---|---|
| **ELITE** | `anthropic/claude-3-5-sonnet-latest` | Architecture design, complex algorithms, security audits, implement X |
| **BALANCED** | `openai/gpt-4o-mini` | Summarization, translation, email drafting, explain X |
| **BASIC** | `deepseek/deepseek-chat` | Greetings, simple math, status checks, small talk |

## Core Script: `scripts/model_router.py`

The entry point for this skill. Import and call `ModelRouter.route(query: str)`.

**Returns:**
```json
{
  "tier": "ELITE",
  "model": "anthropic/claude-3-5-sonnet-latest",
  "confidence": 0.97
}
```

## Dynamic Keyword Expansion (Rolling Adjustment)

Add new routing signals at runtime:

```python
router.add_keywords("ELITE", ["blockchain audit", "zero-knowledge proof"])
```

Every query is logged to `query_history.json` for offline refinement.

## Overriding Model Tiers

```python
router = ModelRouter(
    elite_model="anthropic/claude-opus-4-5-20251101",
    balanced_model="openai/gpt-4o",
    basic_model="deepseek/deepseek-chat"
)
```

## Security and Privacy

- This skill does **not** make any external network calls.
- Query logs are stored locally in `query_history.json` only.
- No API keys are required by this skill itself.

## External Endpoints

None. This skill operates fully locally.

## Trust Statement

This skill performs read-only analysis of input text and writes only to a local `query_history.json` log file. It does not execute external requests or modify system state.
