---
name: Semantic Model Orchestrator
description: Intelligently routes AI queries to the optimal LLM tier (Elite/Balanced/Basic) using semantic analysis. Saves cost by sending simple queries to cheaper models and complex ones to powerful models.
version: 1.2.0
author: Ray
tags: [llm-ops, routing, cost-saving, efficiency, orchestration]
entry_point: scripts/model_router.py
---

# Semantic Model Orchestrator

An intelligent middleware layer that analyzes user query complexity and routes it to the most cost-effective LLM — without sacrificing quality.

## How It Works

The router embeds the incoming query using `sentence-transformers` and computes cosine similarity against a curated **Intent Matrix** to classify it into one of three tiers:

| Tier | Model | Use Case |
|---|---|---|
| **ELITE** | `anthropic/claude-3-5-sonnet-latest` | Architecture design, complex algorithms, security audits |
| **BALANCED** | `openai/gpt-4o-mini` | Summarization, translation, email drafting, explanations |
| **BASIC** | `deepseek/deepseek-chat` | Greetings, simple math, status checks, small talk |

## Features

- **Semantic Intent Recognition** — Embedding-based similarity matching across 3 tiers
- **Keyword Fallback** — Rule-based routing when `sentence-transformers` is unavailable
- **Rolling Adjustment API** — `add_keywords()` and `refine_keywords()` allow dynamic retraining from query history
- **Query Logging** — Every query is logged to `query_history.json` for future refinement
- **Zero Hard Dependencies** — Degrades gracefully without `torch` or `sentence-transformers`

## Usage

```python
from scripts.model_router import ModelRouter

router = ModelRouter()

result = router.route("Design a distributed caching layer for a fintech platform.")
# {"tier": "ELITE", "model": "anthropic/claude-3-5-sonnet-latest", "confidence": 0.97}

result = router.route("Translate this email to Spanish.")
# {"tier": "BALANCED", "model": "openai/gpt-4o-mini", "confidence": 0.96}

result = router.route("Hello!")
# {"tier": "BASIC", "model": "deepseek/deepseek-chat", "confidence": 0.99}
```

## Dynamic Keyword Expansion

Add new routing signals at runtime without restarting:

```python
router.add_keywords("ELITE", ["blockchain audit", "zero-knowledge proof", "Rust systems programming"])
```

## Model Tiers (Configurable)

Override any tier at initialization:

```python
router = ModelRouter(
    elite_model="anthropic/claude-opus-4-5-20251101",
    balanced_model="openai/gpt-4o",
    basic_model="deepseek/deepseek-chat"
)
```

## Requirements

```
sentence-transformers>=2.2.2
numpy>=1.24.0
```

> `sentence-transformers` and `torch` are optional — the skill falls back to keyword matching if they are not installed.
